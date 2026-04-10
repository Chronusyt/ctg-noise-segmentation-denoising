"""
FHR Deceleration Detection (FIGO Guidelines) - Optimized Version

Based on FIGO 2015 Intrapartum Fetal Monitoring Guidelines:
- Deceleration: A transient decrease in FHR ≥15 bpm below the baseline
- Duration: ≥15 seconds and <10 minutes
- Prolonged deceleration: ≥15 bpm below baseline, 2-10 minutes

Classification based on relationship with uterine contractions:
- Early Deceleration (ED): Gradual onset, nadir coincides with UC peak
- Late Deceleration (LD): Gradual onset, nadir after UC peak (lag >20 sec)
- Variable Deceleration (VD): Rapid onset (<30 sec), variable relationship with UC
- Prolonged Deceleration (PD): Duration 2-10 minutes

Optimizations over v1:
- Removed RULE_10_10 from DecelerationType (it's a criterion, not a type)
- Added meets_figo_criteria flag to Deceleration dataclass
- Simplified filtering: uses amplitude, duration, and artifact thresholds only
- Removed complex reliability calculation (unnecessary with proper thresholds)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class DecelerationType(Enum):
    """Type of deceleration based on FIGO criteria and UC relationship."""
    EARLY = "early"           # ED: Nadir coincides with UC peak
    LATE = "late"             # LD: Nadir after UC peak (lag > 20 sec)
    VARIABLE = "variable"     # VD: Rapid onset, variable relationship
    PROLONGED = "prolonged"   # PD: 2-10 minutes duration


class DecelerationCriterion(Enum):
    """Detection criterion (amplitude_bpm, duration_sec) used to identify the deceleration."""
    RULE_8_8 = "rule_8_8"          # ≥8 bpm, ≥8 sec (relaxed criteria)
    RULE_10_10 = "rule_10_10"      # ≥10 bpm, ≥10 sec (relaxed)
    RULE_12_12 = "rule_12_12"      # ≥12 bpm, ≥12 sec (intermediate)
    RULE_15_15 = "rule_15_15"      # ≥15 bpm, ≥15 sec (FIGO criteria)
    
    def get_thresholds(self) -> Tuple[float, float]:
        """Get (amplitude_bpm, duration_sec) thresholds for this criterion."""
        if self == DecelerationCriterion.RULE_8_8:
            return 8.0, 8.0
        elif self == DecelerationCriterion.RULE_10_10:
            return 10.0, 10.0
        elif self == DecelerationCriterion.RULE_12_12:
            return 12.0, 12.0
        else:  # RULE_15_15
            return 15.0, 15.0


@dataclass
class DecelerationConfig:
    """
    Configuration for deceleration detection.
    
    FIGO 2015 Guidelines:
    - Standard: ≥15 bpm below baseline, ≥15 seconds, <10 minutes
    - Prolonged: ≥15 bpm below baseline, 2-10 minutes
    
    Detection criterion is controlled by the 'criterion' field which determines
    the amplitude and duration thresholds.
    """
    # Sample rate
    sample_rate: int = 4  # Hz
    
    # Detection criterion - determines amplitude and duration thresholds
    criterion: DecelerationCriterion = DecelerationCriterion.RULE_15_15
    
    # Entry/exit thresholds (hysteresis for noise robustness)
    entry_threshold: float = 5.0   # bpm below baseline to enter candidate
    exit_threshold: float = 1.0    # bpm below baseline to exit candidate
    
    # Boundary refinement
    boundary_search_sec: float = 3.0  # seconds to search for true boundary
    
    # Duration limits
    duration_max_sec: float = 600.0  # 10 minutes max
    prolonged_threshold_sec: float = 120.0  # 2 minutes = prolonged
    
    # Merging and quality
    min_gap_sec: float = 3.0  # merge if closer than this
    max_artifact_fraction: float = 0.3  # max 30% artifacts
    
    # Variable deceleration classification
    rapid_onset_sec: float = 30.0  # <30 sec onset = variable
    
    def get_thresholds(self) -> Tuple[float, float]:
        """Get (amplitude_bpm, duration_sec) thresholds based on criterion."""
        return self.criterion.get_thresholds()
    
    @property
    def amp_threshold(self) -> float:
        """Amplitude threshold based on criterion."""
        return self.criterion.get_thresholds()[0]
    
    @property
    def dur_threshold_sec(self) -> float:
        """Duration threshold based on criterion."""
        return self.criterion.get_thresholds()[1]


@dataclass
class Deceleration:
    """
    Detected deceleration event.
    
    Attributes:
        start_idx: Start index in signal
        end_idx: End index in signal (exclusive)
        nadir_idx: Index of minimum FHR below baseline
        nadir_amplitude: Maximum depth below baseline (positive value, bpm)
        mean_amplitude: Mean depth below baseline during deceleration
        duration_sec: Duration in seconds
        duration_samples: Duration in samples
        dec_type: Type of deceleration (EARLY, LATE, VARIABLE, PROLONGED)
        criterion: Detection criterion used (RULE_8_8, RULE_10_10, etc.)
        meets_figo_criteria: True if ≥15 bpm and ≥15 sec (FIGO compliant)
        reliability: Confidence score (0-100)
        artifact_fraction: Fraction of samples with artifacts
        onset_time_sec: Time from start to nadir
        recovery_time_sec: Time from nadir to end
        area: Integral area below baseline
        associated_uc_idx: Index of associated uterine contraction (-1 if none)
    """
    start_idx: int
    end_idx: int
    nadir_idx: int
    nadir_amplitude: float
    mean_amplitude: float
    duration_sec: float
    duration_samples: int
    dec_type: DecelerationType = DecelerationType.VARIABLE
    criterion: DecelerationCriterion = DecelerationCriterion.RULE_15_15
    meets_figo_criteria: bool = True
    reliability: int = 100
    artifact_fraction: float = 0.0
    onset_time_sec: float = 0.0
    recovery_time_sec: float = 0.0
    area: float = 0.0
    associated_uc_idx: int = -1
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "nadir_idx": self.nadir_idx,
            "nadir_amplitude": round(self.nadir_amplitude, 1),
            "mean_amplitude": round(self.mean_amplitude, 1),
            "duration_sec": round(self.duration_sec, 1),
            "duration_samples": self.duration_samples,
            "dec_type": self.dec_type.value,
            "criterion": self.criterion.value,
            "meets_figo_criteria": self.meets_figo_criteria,
            "reliability": self.reliability,
            "artifact_fraction": round(self.artifact_fraction, 3),
            "onset_time_sec": round(self.onset_time_sec, 1),
            "recovery_time_sec": round(self.recovery_time_sec, 1),
            "area": round(self.area, 1),
            "associated_uc_idx": self.associated_uc_idx
        }


class DecelerationDetector:
    """
    FIGO-compliant deceleration detector (optimized version).
    
    Usage:
        detector = DecelerationDetector()
        decelerations = detector.detect(fhr, baseline, mask)
        binary_dec = detector.to_binary_signal(decelerations, len(fhr))
    """
    
    def __init__(self, config: Optional[DecelerationConfig] = None):
        """Initialize detector with configuration."""
        self.config = config or DecelerationConfig()
        
        # Pre-compute sample-based thresholds
        self._dur_min_samples = int(self.config.dur_threshold_sec * self.config.sample_rate)
        self._dur_max_samples = int(self.config.duration_max_sec * self.config.sample_rate)
        self._search_samples = int(self.config.boundary_search_sec * self.config.sample_rate)
        self._min_gap_samples = int(self.config.min_gap_sec * self.config.sample_rate)
    
    def detect(self, 
               fhr: np.ndarray, 
               baseline: np.ndarray,
               mask: Optional[np.ndarray] = None,
               uc_models: Optional[List[dict]] = None) -> List[Deceleration]:
        """
        Detect decelerations in FHR signal.
        
        Args:
            fhr: FHR signal (corrected/pretreated)
            baseline: Baseline FHR signal
            mask: Binary mask (0 = reliable, 1 = artifact/unreliable)
            uc_models: List of uterine contraction models for type classification
            
        Returns:
            List of detected Deceleration objects
        """
        # Prepare data
        fhr = np.asarray(fhr, dtype=np.float64)
        baseline = np.asarray(baseline, dtype=np.float64)
        n = len(fhr)
        
        if mask is None:
            mask = np.zeros(n, dtype=np.int8)  # 0 = all reliable
        else:
            mask = np.asarray(mask, dtype=np.int8)
        
        if uc_models is None:
            uc_models = []
        
        # Compute deviation from baseline (negative = below baseline)
        deviation = fhr - baseline
        deviation = np.nan_to_num(deviation, nan=0.0)
        
        # Find candidate regions using vectorized approach
        candidates = self._find_candidates_vectorized(deviation)
        
        # Process each candidate
        decelerations = []
        for start, end in candidates:
            dec = self._process_candidate(start, end, deviation, fhr, baseline, mask)
            if dec is not None:
                decelerations.append(dec)
        
        # Merge nearby decelerations
        decelerations = self._merge_nearby(decelerations)
        
        # Classify types, set criterion and FIGO compliance flag
        for dec in decelerations:
            dec.dec_type = self._classify_type(dec, uc_models)
            dec.criterion = self.config.criterion
            dec.meets_figo_criteria = (dec.nadir_amplitude >= 15.0 and dec.duration_sec >= 15.0)
        
        return decelerations
    
    def _find_candidates_vectorized(self, deviation: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find candidate deceleration regions using state machine with optimized iteration.
        
        Uses hysteresis thresholds:
        - Enter when deviation <= -entry_threshold
        - Exit when deviation >= -exit_threshold
        
        This matches v1 behavior exactly but is optimized for typical CTG signals.
        """
        entry_thresh = -self.config.entry_threshold
        exit_thresh = -self.config.exit_threshold
        
        candidates = []
        n = len(deviation)
        in_region = False
        region_start = 0
        
        # Use same logic as v1 but with optimized numpy access
        for i in range(n):
            d = deviation[i]
            
            if not in_region:
                if d <= entry_thresh:  # FHR drops below baseline - entry_threshold
                    in_region = True
                    region_start = i
            else:
                if d >= exit_thresh:  # FHR returns to near baseline
                    if i > region_start:
                        candidates.append((region_start, i))
                    in_region = False
        
        # Handle case where signal ends during deceleration.
        # Candidate end indexes are exclusive, so an event that continues to the
        # last sample should end at n instead of n - 1.
        if in_region and n > region_start:
            candidates.append((region_start, n))
        
        return candidates
    
    def _process_candidate(self, start: int, end: int, 
                           deviation: np.ndarray,
                           fhr: np.ndarray, 
                           baseline: np.ndarray,
                           mask: np.ndarray) -> Optional[Deceleration]:
        """Process a candidate region and create a validated Deceleration."""
        n = len(deviation)
        
        # Refine boundaries using gradient-based approach
        refined_start = self._refine_start_boundary(deviation, start)
        refined_end = self._refine_end_boundary(deviation, end, n)
        
        if refined_end <= refined_start:
            return None
        
        # Duration validation
        duration_samples = refined_end - refined_start
        if duration_samples < self._dur_min_samples or duration_samples > self._dur_max_samples:
            return None
        
        # Extract region
        region_dev = deviation[refined_start:refined_end]
        if len(region_dev) == 0:
            return None
        
        # Find nadir (minimum = most negative)
        nadir_local = np.argmin(region_dev)
        nadir_idx = refined_start + nadir_local
        nadir_amplitude = abs(region_dev[nadir_local])
        
        # Amplitude validation
        if nadir_amplitude < self.config.amp_threshold:
            return None
        
        # Compute metrics
        negative_mask = region_dev < 0
        mean_amplitude = abs(np.mean(region_dev[negative_mask])) if np.any(negative_mask) else nadir_amplitude
        
        # Artifact fraction (mask convention: 0=reliable, 1=unreliable)
        region_mask = mask[refined_start:refined_end]
        artifact_fraction = float(np.mean(region_mask))
        
        if artifact_fraction > self.config.max_artifact_fraction:
            return None
        
        # Duration and timing
        duration_sec = duration_samples / self.config.sample_rate
        onset_time_sec = (nadir_idx - refined_start) / self.config.sample_rate
        recovery_time_sec = (refined_end - nadir_idx) / self.config.sample_rate
        
        # Area below baseline
        area = abs(np.sum(region_dev[negative_mask])) if np.any(negative_mask) else 0.0
        
        return Deceleration(
            start_idx=refined_start,
            end_idx=refined_end,
            nadir_idx=nadir_idx,
            nadir_amplitude=nadir_amplitude,
            mean_amplitude=mean_amplitude,
            duration_sec=duration_sec,
            duration_samples=duration_samples,
            reliability=100,  # Default reliability (no calculation)
            artifact_fraction=artifact_fraction,
            onset_time_sec=onset_time_sec,
            recovery_time_sec=recovery_time_sec,
            area=area
        )
    
    def _refine_start_boundary(self, deviation: np.ndarray, start: int) -> int:
        """
        Refine start boundary by finding point closest to baseline.
        Uses gradient information for robust boundary detection.
        """
        search_range = self._search_samples * 2
        search_start = max(0, start - search_range)
        
        if search_start >= start:
            return start
        
        # Search backward from start
        region = deviation[search_start:start + 1]
        
        # Find point closest to zero (baseline)
        best_idx = start
        best_val = deviation[start]
        
        for i in range(len(region) - 1, -1, -1):
            d = region[i]
            
            # Stop if we go significantly above baseline
            if d > 2.0:
                best_idx = search_start + i
                break
            
            # Track point closest to baseline
            if d > best_val:
                best_val = d
                best_idx = search_start + i
            
            # If very close to baseline, good boundary
            if d >= -0.5:
                best_idx = search_start + i
                break
        
        return best_idx
    
    def _refine_end_boundary(self, deviation: np.ndarray, end: int, n: int) -> int:
        """
        Refine end boundary by finding point closest to baseline.
        """
        search_range = self._search_samples * 2
        search_end = min(n, end + search_range)
        
        if end >= search_end:
            return min(end, n)
        
        # Search forward from end
        region = deviation[end:search_end]
        
        best_idx = end
        best_val = deviation[end] if end < n else float('-inf')
        
        for i in range(len(region)):
            d = region[i]
            
            # Stop if we go significantly above baseline
            if d > 2.0:
                best_idx = end + i
                break
            
            # Track point closest to baseline
            if d > best_val:
                best_val = d
                best_idx = end + i
            
            # If very close to baseline, good boundary
            if d >= -0.5:
                best_idx = end + i
                break
        
        return best_idx
    
    def _merge_nearby(self, decelerations: List[Deceleration]) -> List[Deceleration]:
        """Merge decelerations that are close together."""
        if len(decelerations) < 2:
            return decelerations
        
        merged = []
        current = decelerations[0]
        
        for next_dec in decelerations[1:]:
            gap = next_dec.start_idx - current.end_idx
            
            if gap < self._min_gap_samples:
                # Merge
                new_nadir_idx = current.nadir_idx if current.nadir_amplitude >= next_dec.nadir_amplitude else next_dec.nadir_idx
                new_duration = next_dec.end_idx - current.start_idx
                
                current = Deceleration(
                    start_idx=current.start_idx,
                    end_idx=next_dec.end_idx,
                    nadir_idx=new_nadir_idx,
                    nadir_amplitude=max(current.nadir_amplitude, next_dec.nadir_amplitude),
                    mean_amplitude=(current.mean_amplitude + next_dec.mean_amplitude) / 2,
                    duration_sec=new_duration / self.config.sample_rate,
                    duration_samples=new_duration,
                    reliability=min(current.reliability, next_dec.reliability),
                    artifact_fraction=max(current.artifact_fraction, next_dec.artifact_fraction),
                    onset_time_sec=(new_nadir_idx - current.start_idx) / self.config.sample_rate,
                    recovery_time_sec=(next_dec.end_idx - new_nadir_idx) / self.config.sample_rate,
                    area=current.area + next_dec.area
                )
            else:
                merged.append(current)
                current = next_dec
        
        merged.append(current)
        return merged
    
    def _classify_type(self, dec: Deceleration, uc_models: List[dict]) -> DecelerationType:
        """
        Classify deceleration type based on FIGO criteria.
        
        Priority:
        1. Prolonged: duration ≥ 2 minutes
        2. Variable: rapid onset (<30 sec to nadir)
        3. Early: nadir coincides with UC peak (within ±20 sec)
        4. Late: nadir after UC peak (lag > 20 sec)
        5. Default: Variable
        """
        # Prolonged: duration ≥ 2 minutes
        if dec.duration_sec >= self.config.prolonged_threshold_sec:
            return DecelerationType.PROLONGED
        
        # Variable: rapid onset
        if dec.onset_time_sec < self.config.rapid_onset_sec:
            return DecelerationType.VARIABLE
        
        # Find associated UC for early/late classification
        associated_uc = self._find_associated_uc(dec, uc_models)
        
        if associated_uc is None:
            return DecelerationType.VARIABLE
        
        # Get UC peak
        uc_peak_idx = associated_uc.get('peakIndex', associated_uc.get('peak_index', 0))
        
        # Calculate lag
        lag_sec = (dec.nadir_idx - uc_peak_idx) / self.config.sample_rate
        
        # Early: nadir coincides with UC peak (within ±20 sec)
        if abs(lag_sec) <= 20:
            return DecelerationType.EARLY
        
        # Late: nadir after UC peak
        if lag_sec > 20:
            return DecelerationType.LATE
        
        return DecelerationType.VARIABLE
    
    def _find_associated_uc(self, dec: Deceleration, uc_models: List[dict]) -> Optional[dict]:
        """Find the uterine contraction most associated with this deceleration."""
        if not uc_models:
            return None
        
        best_uc = None
        best_overlap = 0
        
        for uc in uc_models:
            uc_start = uc.get('startIndex', uc.get('start_index', 0))
            uc_end = uc.get('endIndex', uc.get('end_index', 0))
            
            overlap = max(0, min(dec.end_idx, uc_end) - max(dec.start_idx, uc_start))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_uc = uc
        
        # Return UC if meaningful overlap (≥25% of deceleration)
        if best_overlap >= dec.duration_samples * 0.25:
            return best_uc
        
        return None
    
    def to_binary_signal(self, decelerations: List[Deceleration], length: int) -> np.ndarray:
        """Convert decelerations to binary signal."""
        binary = np.zeros(length, dtype=np.int8)
        for dec in decelerations:
            binary[dec.start_idx:dec.end_idx] = 1
        return binary
    
    def get_summary(self, decelerations: List[Deceleration]) -> dict:
        """Get summary statistics for detected decelerations."""
        if not decelerations:
            return {
                "count": 0,
                "figo_count": 0,
                "total_duration_sec": 0,
                "mean_duration_sec": 0,
                "mean_amplitude": 0,
                "mean_reliability": 0,
                "by_type": {"early": 0, "late": 0, "variable": 0, "prolonged": 0}
            }
        
        durations = [dec.duration_sec for dec in decelerations]
        amplitudes = [dec.nadir_amplitude for dec in decelerations]
        figo_count = sum(1 for dec in decelerations if dec.meets_figo_criteria)
        
        by_type = {
            "early": sum(1 for d in decelerations if d.dec_type == DecelerationType.EARLY),
            "late": sum(1 for d in decelerations if d.dec_type == DecelerationType.LATE),
            "variable": sum(1 for d in decelerations if d.dec_type == DecelerationType.VARIABLE),
            "prolonged": sum(1 for d in decelerations if d.dec_type == DecelerationType.PROLONGED)
        }
        
        return {
            "count": len(decelerations),
            "figo_count": figo_count,
            "total_duration_sec": sum(durations),
            "mean_duration_sec": float(np.mean(durations)),
            "mean_amplitude": float(np.mean(amplitudes)),
            "mean_reliability": float(np.mean([dec.reliability for dec in decelerations])),
            "by_type": by_type
        }


def detect_decelerations_figo(fhr: np.ndarray,
                               baseline: np.ndarray,
                               mask: Optional[np.ndarray] = None,
                               uc_models: Optional[List[dict]] = None,
                               config: Optional[DecelerationConfig] = None) -> Tuple[List[Deceleration], np.ndarray]:
    """
    Convenience function to detect decelerations using FIGO criteria.
    
    Args:
        fhr: FHR signal (corrected/pretreated)
        baseline: Baseline FHR signal
        mask: Binary reliability mask (1 = artifact, 0 = reliable)
        uc_models: List of uterine contraction models for type classification
        config: Optional configuration
        
    Returns:
        Tuple of (list of Deceleration objects, binary deceleration signal)
    """
    detector = DecelerationDetector(config)
    decelerations = detector.detect(fhr, baseline, mask, uc_models)
    binary_signal = detector.to_binary_signal(decelerations, len(fhr))
    return decelerations, binary_signal
