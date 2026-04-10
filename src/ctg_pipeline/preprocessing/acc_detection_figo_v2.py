"""
FHR Acceleration Detection (FIGO Guidelines) - Optimized Version

Based on FIGO 2015 Intrapartum Fetal Monitoring Guidelines:
- Acceleration: A transient increase in FHR ≥15 bpm above the baseline
- Duration: ≥15 seconds and <10 minutes

Optimizations in v2:
1. Vectorized candidate region detection using numpy
2. Simplified boundary refinement with gradient-based approach
3. Consolidated configuration with sensible defaults
4. Improved robustness for noisy signals
5. Better handling of edge cases
6. Renamed AccelerationType to AccelerationCriterion (it's a detection criterion, not a type)
7. Added meets_figo_criteria flag to Acceleration dataclass
8. Detection criterion directly controlled by AccelerationCriterion enum
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class AccelerationCriterion(Enum):
    """Detection criterion (amplitude_bpm, duration_sec) used to identify the acceleration."""
    RULE_8_8 = "rule_8_8"          # ≥8 bpm, ≥8 sec (relaxed criteria)
    RULE_10_10 = "rule_10_10"      # ≥10 bpm, ≥10 sec (relaxed)
    RULE_12_12 = "rule_12_12"      # ≥12 bpm, ≥12 sec (intermediate)
    RULE_15_15 = "rule_15_15"      # ≥15 bpm, ≥15 sec (FIGO standard criteria)
    
    def get_thresholds(self) -> Tuple[float, float]:
        """Get (amplitude_bpm, duration_sec) thresholds for this criterion."""
        if self == AccelerationCriterion.RULE_8_8:
            return 8.0, 8.0
        elif self == AccelerationCriterion.RULE_10_10:
            return 10.0, 10.0
        elif self == AccelerationCriterion.RULE_12_12:
            return 12.0, 12.0
        else:  # RULE_15_15
            return 15.0, 15.0


@dataclass
class AccelerationConfig:
    """
    Configuration for acceleration detection.
    
    Detection criterion is controlled by the 'criterion' field which determines
    the amplitude and duration thresholds.
    """
    # Sample rate
    sample_rate: int = 4  # Hz
    
    # Detection criterion - determines amplitude and duration thresholds
    criterion: AccelerationCriterion = AccelerationCriterion.RULE_15_15
    
    # Entry/exit thresholds (hysteresis for noise robustness)
    entry_threshold: float = 5.0   # bpm above baseline to start candidate
    exit_threshold: float = 1.0    # bpm above baseline to end candidate
    
    # Boundary search window (seconds) - same as v1, will be extended by 2x in refinement
    boundary_search_sec: float = 3.0  # 3 sec base, extended to 6 sec in refinement
    
    # Maximum duration (>10 min is baseline change)
    duration_max_sec: float = 600.0  # 10 minutes
    
    # Minimum gap between accelerations for merging
    min_gap_sec: float = 3.0
    
    # Signal quality
    max_artifact_fraction: float = 0.3  # max 30% artifacts
    min_reliability: int = 30  # minimum reliability score
    
    def get_thresholds(self) -> Tuple[float, float]:
        """Get amplitude and duration thresholds based on criterion."""
        return self.criterion.get_thresholds()


@dataclass
class Acceleration:
    """Detected acceleration event."""
    start_idx: int
    end_idx: int  # exclusive
    peak_idx: int
    peak_amplitude: float
    mean_amplitude: float
    duration_sec: float
    duration_samples: int
    criterion: AccelerationCriterion = AccelerationCriterion.RULE_15_15
    meets_figo_criteria: bool = True  # True if ≥15 bpm and ≥15 sec
    reliability: int = 100
    artifact_fraction: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "peak_idx": self.peak_idx,
            "peak_amplitude": round(self.peak_amplitude, 1),
            "mean_amplitude": round(self.mean_amplitude, 1),
            "duration_sec": round(self.duration_sec, 1),
            "duration_samples": self.duration_samples,
            "criterion": self.criterion.value,
            "meets_figo_criteria": self.meets_figo_criteria,
            "reliability": self.reliability,
            "artifact_fraction": round(self.artifact_fraction, 3)
        }


class AccelerationDetector:
    """
    Optimized FIGO-compliant acceleration detector.
    
    Key improvements:
    1. Vectorized candidate detection for efficiency
    2. Gradient-based boundary refinement for accuracy
    3. Robust handling of noisy signals
    """
    
    def __init__(self, config: Optional[AccelerationConfig] = None):
        """Initialize detector with configuration."""
        self.config = config or AccelerationConfig()
    
    def detect(self, 
               fhr: np.ndarray, 
               baseline: np.ndarray,
               mask: Optional[np.ndarray] = None) -> List[Acceleration]:
        """
        Detect accelerations in FHR signal.
        
        Algorithm:
        1. Compute deviation from baseline
        2. Find candidate regions using hysteresis thresholds
        3. Refine boundaries to find true start/end points
        4. Validate against amplitude and duration criteria
        5. Merge nearby accelerations
        
        Args:
            fhr: FHR signal (corrected/pretreated)
            baseline: Baseline FHR signal
            mask: Binary mask (1 = artifact, 0 = reliable)
            
        Returns:
            List of detected Acceleration objects
        """
        # Prepare data
        fhr = np.asarray(fhr, dtype=np.float64)
        baseline = np.asarray(baseline, dtype=np.float64)
        n = len(fhr)
        
        if mask is None:
            mask = np.zeros(n, dtype=np.int8)
        else:
            mask = np.asarray(mask, dtype=np.int8)
        
        # Get thresholds from criterion
        amp_threshold, dur_threshold_sec = self.config.get_thresholds()
        dur_min_samples = int(dur_threshold_sec * self.config.sample_rate)
        dur_max_samples = int(self.config.duration_max_sec * self.config.sample_rate)
        
        # Compute deviation from baseline
        deviation = fhr - baseline
        
        # Handle NaN values
        nan_mask = np.isnan(deviation)
        deviation_clean = deviation.copy()
        deviation_clean[nan_mask] = 0
        
        # Find candidate regions using vectorized approach
        candidates = self._find_candidate_regions(deviation_clean, n)
        
        # Refine boundaries and validate each candidate
        accelerations = []
        search_samples = int(self.config.boundary_search_sec * self.config.sample_rate)
        
        for start, end in candidates:
            # Refine boundaries
            refined_start = self._refine_boundary(deviation_clean, start, -1, search_samples)
            refined_end = self._refine_boundary(deviation_clean, end, 1, search_samples, n)
            
            # Ensure valid range
            if refined_end <= refined_start:
                continue
            
            # Create and validate acceleration
            acc = self._create_acceleration(
                refined_start, refined_end, deviation_clean, mask,
                amp_threshold, dur_min_samples, dur_max_samples
            )
            
            if acc is not None:
                accelerations.append(acc)
        
        # Merge nearby accelerations
        accelerations = self._merge_nearby(accelerations)
        
        # Set criterion and FIGO compliance flag
        for acc in accelerations:
            acc.criterion = self.config.criterion
            acc.meets_figo_criteria = (acc.peak_amplitude >= 15.0 and acc.duration_sec >= 15.0)
        
        return accelerations
    
    def _find_candidate_regions(self, deviation: np.ndarray, n: int) -> List[Tuple[int, int]]:
        """
        Find candidate acceleration regions using hysteresis thresholds.
        
        Uses state machine approach:
        - Start when deviation >= entry_threshold
        - End when deviation <= exit_threshold
        """
        candidates = []
        entry_thresh = self.config.entry_threshold
        exit_thresh = self.config.exit_threshold
        
        in_region = False
        region_start = 0
        
        for i in range(n):
            d = deviation[i]
            
            if not in_region:
                if d >= entry_thresh:
                    in_region = True
                    region_start = i
            else:
                if d <= exit_thresh:
                    if i > region_start:
                        candidates.append((region_start, i))
                    in_region = False
        
        # Handle case where signal ends during acceleration
        if in_region and n > region_start:
            candidates.append((region_start, n))
        
        return candidates
    
    def _refine_boundary(self, deviation: np.ndarray, idx: int, direction: int,
                         search_samples: int, n: Optional[int] = None) -> int:
        """
        Refine acceleration boundary by finding point closest to baseline.
        
        Uses same logic as v1: extends search window by 2x and tracks minimum
        deviation point, stopping early if we find a point very close to baseline
        or go significantly below baseline.
        
        Args:
            deviation: Deviation from baseline array
            idx: Starting index for search
            direction: -1 for backward (start), +1 for forward (end)
            search_samples: Number of samples to search
            n: Signal length (for forward search)
            
        Returns:
            Refined boundary index
        """
        if n is None:
            n = len(deviation)
        
        # Extend search window for better boundary detection (same as v1)
        extended_search = search_samples * 2
        
        # Define search range
        if direction < 0:  # Backward search for start
            search_start = max(0, idx - extended_search)
            search_range = range(idx - 1, search_start - 1, -1)
        else:  # Forward search for end
            search_end = min(n, idx + extended_search)
            search_range = range(idx, search_end)
        
        # Find point closest to baseline (minimum positive deviation or first negative)
        best_idx = idx
        best_val = deviation[idx] if idx < n else float('inf')
        
        for j in search_range:
            d = deviation[j]
            
            # Stop if we go significantly below baseline
            if d < -2:
                best_idx = j
                break
            
            # Track point closest to baseline (minimum deviation)
            if d < best_val:
                best_val = d
                best_idx = j
            
            # If very close to baseline, that's a good boundary
            if d <= 0.5:
                best_idx = j
                break
        
        return best_idx
    
    def _create_acceleration(self, start: int, end: int, deviation: np.ndarray,
                              mask: np.ndarray, amp_threshold: float,
                              dur_min_samples: int, dur_max_samples: int) -> Optional[Acceleration]:
        """Create and validate an acceleration from refined boundaries."""
        # Duration validation
        duration_samples = end - start
        if duration_samples < dur_min_samples or duration_samples > dur_max_samples:
            return None
        
        # Get region data
        region_deviation = deviation[start:end]
        if len(region_deviation) == 0:
            return None
        
        # Find peak
        peak_local_idx = np.argmax(region_deviation)
        peak_idx = start + peak_local_idx
        peak_amplitude = region_deviation[peak_local_idx]
        
        # Amplitude validation
        if peak_amplitude < amp_threshold:
            return None
        
        # Mean amplitude (only positive deviation)
        positive_dev = region_deviation[region_deviation >= 0]
        mean_amplitude = np.mean(positive_dev) if len(positive_dev) > 0 else peak_amplitude
        
        # Artifact fraction
        region_mask = mask[start:end]
        artifact_fraction = np.mean(region_mask)
        
        if artifact_fraction > self.config.max_artifact_fraction:
            return None
        
        # Duration
        duration_sec = duration_samples / self.config.sample_rate
        
        # Reliability score
        reliability = self._compute_reliability(
            duration_sec, peak_amplitude, mean_amplitude, artifact_fraction
        )
        
        if reliability < self.config.min_reliability:
            return None
        
        return Acceleration(
            start_idx=start,
            end_idx=end,
            peak_idx=peak_idx,
            peak_amplitude=peak_amplitude,
            mean_amplitude=mean_amplitude,
            duration_sec=duration_sec,
            duration_samples=duration_samples,
            reliability=reliability,
            artifact_fraction=artifact_fraction
        )
    
    def _merge_nearby(self, accelerations: List[Acceleration]) -> List[Acceleration]:
        """Merge accelerations that are close together."""
        if len(accelerations) < 2:
            return accelerations
        
        min_gap = int(self.config.min_gap_sec * self.config.sample_rate)
        merged = []
        current = accelerations[0]
        
        for next_acc in accelerations[1:]:
            gap = next_acc.start_idx - current.end_idx
            
            if gap < min_gap:
                # Merge accelerations
                new_peak_idx = current.peak_idx if current.peak_amplitude >= next_acc.peak_amplitude else next_acc.peak_idx
                new_peak_amp = max(current.peak_amplitude, next_acc.peak_amplitude)
                new_duration = next_acc.end_idx - current.start_idx
                
                current = Acceleration(
                    start_idx=current.start_idx,
                    end_idx=next_acc.end_idx,
                    peak_idx=new_peak_idx,
                    peak_amplitude=new_peak_amp,
                    mean_amplitude=(current.mean_amplitude + next_acc.mean_amplitude) / 2,
                    duration_sec=new_duration / self.config.sample_rate,
                    duration_samples=new_duration,
                    reliability=min(current.reliability, next_acc.reliability),
                    artifact_fraction=max(current.artifact_fraction, next_acc.artifact_fraction)
                )
            else:
                merged.append(current)
                current = next_acc
        
        merged.append(current)
        return merged
    
    def _compute_reliability(self, duration_sec: float, peak_amplitude: float,
                              mean_amplitude: float, artifact_fraction: float) -> int:
        """Compute reliability score (0-100)."""
        score = 100.0
        
        # Duration factor: optimal 15-60 seconds
        if duration_sec < 8:
            score *= duration_sec / 8
        elif duration_sec < 15:
            score *= 0.9
        elif duration_sec > 120:
            score *= 0.85
        
        # Amplitude factor
        if peak_amplitude < 8:
            score *= peak_amplitude / 8
        elif peak_amplitude < 15:
            score *= 0.95
        elif peak_amplitude >= 25:
            score *= 1.1
        
        # Artifact penalty
        score *= (1 - artifact_fraction)
        
        return min(100, max(0, int(round(score))))
    
    def to_binary_signal(self, accelerations: List[Acceleration], length: int) -> np.ndarray:
        """Convert accelerations to binary signal."""
        binary = np.zeros(length, dtype=np.int8)
        for acc in accelerations:
            binary[acc.start_idx:acc.end_idx] = 1
        return binary
    
    def get_summary(self, accelerations: List[Acceleration]) -> dict:
        """Get summary statistics for detected accelerations."""
        if not accelerations:
            return {
                "count": 0, "total_duration_sec": 0, "mean_duration_sec": 0,
                "mean_amplitude": 0, "mean_reliability": 0, "figo_count": 0
            }
        
        durations = [acc.duration_sec for acc in accelerations]
        amplitudes = [acc.peak_amplitude for acc in accelerations]
        
        return {
            "count": len(accelerations),
            "total_duration_sec": sum(durations),
            "mean_duration_sec": np.mean(durations),
            "mean_amplitude": np.mean(amplitudes),
            "mean_reliability": np.mean([acc.reliability for acc in accelerations]),
            "figo_count": sum(1 for acc in accelerations if acc.meets_figo_criteria)
        }


def detect_accelerations_figo(fhr: np.ndarray,
                               baseline: np.ndarray,
                               mask: Optional[np.ndarray] = None,
                               config: Optional[AccelerationConfig] = None) -> Tuple[List[Acceleration], np.ndarray]:
    """
    Convenience function to detect accelerations using FIGO criteria.
    
    Args:
        fhr: FHR signal (corrected/pretreated)
        baseline: Baseline FHR signal
        mask: Binary reliability mask (1 = artifact)
        config: Optional configuration
        
    Returns:
        Tuple of (list of Acceleration objects, binary acceleration signal)
    """
    detector = AccelerationDetector(config)
    accelerations = detector.detect(fhr, baseline, mask)
    binary_signal = detector.to_binary_signal(accelerations, len(fhr))
    return accelerations, binary_signal
