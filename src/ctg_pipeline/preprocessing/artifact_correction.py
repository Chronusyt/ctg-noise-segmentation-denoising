"""
Artifact Correction Module for FHR Signals (Optimized)

This module corrects artifacts detected in FHR signals based on the reliability mask
from signal_quality.py.

IMPORTANT: This module expects the mask from signal_quality.py which already includes:
- Boundary saturation (values ≤50 or ≥220, which covers dropout markers 0 and 255)
- Extreme jumps (>30 bpm between consecutive samples)
- Merged nearby artifact regions

Processing Pipeline:
1. Mark all masked (unreliable) samples as NaN
2. Detect and remove fixed/constant value runs (≥20 samples) in reliable regions
3. Apply correction based on gap duration:
   - Linear interpolation for short gaps (< 30 seconds)
   - Neighbor mean fill for long gaps (≥ 30 seconds)
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


class CorrectionMethod(Enum):
    """Available correction methods."""
    LINEAR = "linear"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    NEIGHBOR_MEAN = "neighbor_mean"
    AUTO = "auto"


@dataclass
class CorrectionConfig:
    """
    Configuration for artifact correction.
    
    Attributes:
        gap_threshold: Maximum samples for linear interpolation (default: 120 = 30 seconds)
        fixed_value_min_run: Minimum run length to detect as fixed value artifact (default: 20)
        smoothing_window: Window size for post-correction smoothing (default: 5)
        min_valid_neighbors: Minimum valid samples needed on each side for interpolation
        neighbor_search_window: Window to search for valid neighbors for mean calculation (default: 480 = 2 min)
        valid_range: Valid FHR range for fixed-value detection (default: (50, 220))
    """
    gap_threshold: int = 120  # 30 seconds at 4 Hz
    fixed_value_min_run: int = 20  # 5 seconds at 4 Hz
    smoothing_window: int = 5
    min_valid_neighbors: int = 4  # 1 second of valid data needed
    neighbor_search_window: int = 480  # 2 minutes at 4 Hz for finding mean
    valid_range: Tuple[int, int] = (50, 220)


class ArtifactCorrector:
    """
    Correct artifacts in FHR signals based on the reliability mask.
    
    Expects mask from signal_quality.py which already includes dropout detection,
    boundary saturation, and extreme jumps.
    """
    
    def __init__(self, config: Optional[CorrectionConfig] = None):
        """
        Initialize the artifact corrector.
        
        Args:
            config: Correction configuration (uses defaults if None)
        """
        self.config = config if config is not None else CorrectionConfig()
    
    def correct(self, 
                fhr: np.ndarray, 
                mask: np.ndarray,
                method: CorrectionMethod = CorrectionMethod.AUTO) -> Tuple[np.ndarray, dict]:
        """
        Correct artifacts in FHR signal.
        
        Args:
            fhr: Raw FHR signal
            mask: Reliability mask from signal_quality.py (0=reliable, 1=unreliable)
            method: Correction method to use
            
        Returns:
            Tuple of (corrected_fhr, correction_stats)
        """
        # Make a copy to avoid modifying original
        corrected = fhr.astype(np.float64).copy()
        
        # Statistics
        stats = {
            'total_regions': 0,
            'linear_corrected': 0,
            'fill_corrected': 0,
            'samples_corrected': 0,
            'mask_invalidated': 0,
            'fixed_removed': 0,
        }
        
        # Step 1: Mark all unreliable samples (from mask) as NaN
        mask_invalid = (mask == 1)
        stats['mask_invalidated'] = int(np.sum(mask_invalid))
        corrected[mask_invalid] = np.nan
        
        # Step 2: Detect and remove fixed/constant value runs in reliable regions
        fixed_removed = self._remove_fixed_values(corrected)
        stats['fixed_removed'] = fixed_removed
        
        # Step 3: Create combined mask (original mask + fixed values)
        combined_mask = mask.copy()
        combined_mask[np.isnan(corrected)] = 1
        
        # Step 4: Find all artifact regions
        regions = self._find_artifact_regions(combined_mask)
        stats['total_regions'] = len(regions)
        
        # Step 5: Process each region
        for start, end in regions:
            gap_length = end - start
            
            # Determine method for this region
            if method == CorrectionMethod.AUTO:
                region_method = self._select_method(gap_length, start, end, len(fhr))
            else:
                region_method = method
            
            # Apply correction
            if region_method == CorrectionMethod.LINEAR:
                self._linear_interpolate(corrected, start, end)
                stats['linear_corrected'] += 1
                stats['samples_corrected'] += gap_length
                
            elif region_method == CorrectionMethod.FORWARD_FILL:
                self._forward_fill(corrected, start, end)
                stats['fill_corrected'] += 1
                stats['samples_corrected'] += gap_length
                
            elif region_method == CorrectionMethod.BACKWARD_FILL:
                self._backward_fill(corrected, start, end)
                stats['fill_corrected'] += 1
                stats['samples_corrected'] += gap_length
            
            elif region_method == CorrectionMethod.NEIGHBOR_MEAN:
                self._neighbor_mean_fill(corrected, start, end, combined_mask)
                stats['fill_corrected'] += 1
                stats['samples_corrected'] += gap_length
        
        return corrected, stats
    
    def _remove_fixed_values(self, fhr: np.ndarray) -> int:
        """
        Detect and remove fixed (constant) value runs in the FHR signal.
        
        Returns:
            Number of samples marked as fixed artifacts
        """
        n = len(fhr)
        min_run = self.config.fixed_value_min_run
        valid_min, valid_max = self.config.valid_range
        removed_count = 0
        
        i = 0
        while i < n - 1:
            val = fhr[i]
            
            # Skip if already NaN or outside valid range
            if np.isnan(val) or val < valid_min or val > valid_max:
                i += 1
                continue
            
            # Find run of identical values
            run_end = i + 1
            while run_end < n and fhr[run_end] == val:
                run_end += 1
            
            run_length = run_end - i
            
            # Mark as artifact if run is too long
            if run_length >= min_run:
                fhr[i:run_end] = np.nan
                removed_count += run_length
            
            i = run_end
        
        return removed_count
    
    def _find_artifact_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find continuous artifact regions from mask."""
        padded = np.concatenate([[0], mask, [0]])
        diff = np.diff(padded)
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        return list(zip(starts, ends))
    
    def _select_method(self, gap_length: int, start: int, end: int, total_length: int) -> CorrectionMethod:
        """Automatically select the best correction method based on gap characteristics."""
        # Check if we have enough valid neighbors
        has_valid_before = start >= self.config.min_valid_neighbors
        has_valid_after = end <= total_length - self.config.min_valid_neighbors
        
        # Edge cases - use fill methods
        if not has_valid_before and not has_valid_after:
            return CorrectionMethod.NEIGHBOR_MEAN
        elif not has_valid_before:
            return CorrectionMethod.BACKWARD_FILL
        elif not has_valid_after:
            return CorrectionMethod.FORWARD_FILL
        
        # Interior gaps - select based on length
        if gap_length <= self.config.gap_threshold:
            return CorrectionMethod.LINEAR
        else:
            return CorrectionMethod.NEIGHBOR_MEAN
    
    def _linear_interpolate(self, fhr: np.ndarray, start: int, end: int) -> None:
        """Apply linear interpolation to fill the gap."""
        if start == 0 or end >= len(fhr):
            return
        
        left_val = fhr[start - 1]
        right_val = fhr[end]
        
        if np.isnan(left_val) or np.isnan(right_val):
            return
        
        gap_length = end - start
        t = np.linspace(0, 1, gap_length + 2)[1:-1]
        fhr[start:end] = left_val + t * (right_val - left_val)
    
    def _forward_fill(self, fhr: np.ndarray, start: int, end: int) -> None:
        """Fill with the last valid value before the gap."""
        if start == 0:
            self._backward_fill(fhr, start, end)
            return
        
        fill_value = fhr[start - 1]
        if np.isnan(fill_value):
            return
        
        fhr[start:end] = fill_value
        
        # Smooth transition to next value
        if end < len(fhr):
            transition_len = min(4, end - start)
            if transition_len > 1:
                next_val = fhr[end]
                if not np.isnan(next_val):
                    t = np.linspace(0, 1, transition_len)
                    fhr[end-transition_len:end] = fill_value + t * (next_val - fill_value)
    
    def _backward_fill(self, fhr: np.ndarray, start: int, end: int) -> None:
        """Fill with the first valid value after the gap."""
        if end >= len(fhr):
            return
        
        fill_value = fhr[end]
        if np.isnan(fill_value):
            return
        
        fhr[start:end] = fill_value
        
        # Smooth transition from previous value
        if start > 0:
            transition_len = min(4, end - start)
            if transition_len > 1:
                prev_val = fhr[start - 1]
                if not np.isnan(prev_val):
                    t = np.linspace(0, 1, transition_len)
                    fhr[start:start+transition_len] = prev_val + t * (fill_value - prev_val)
    
    def _neighbor_mean_fill(self, fhr: np.ndarray, start: int, end: int, mask: np.ndarray) -> None:
        """Fill with mean of valid neighbors within a search window."""
        search_window = self.config.neighbor_search_window
        n = len(fhr)
        valid_min, valid_max = self.config.valid_range
        
        # Collect valid samples from before and after the gap
        valid_samples = []
        
        # Search before the gap
        search_start = max(0, start - search_window)
        for i in range(search_start, start):
            if mask[i] == 0 and not np.isnan(fhr[i]) and valid_min < fhr[i] < valid_max:
                valid_samples.append(fhr[i])
        
        # Search after the gap
        search_end = min(n, end + search_window)
        for i in range(end, search_end):
            if mask[i] == 0 and not np.isnan(fhr[i]) and valid_min < fhr[i] < valid_max:
                valid_samples.append(fhr[i])
        
        if len(valid_samples) > 0:
            fill_value = np.mean(valid_samples)
        else:
            # Fallback: use global mean or default baseline value
            valid_global = fhr[(mask == 0) & (~np.isnan(fhr)) & (fhr > valid_min) & (fhr < valid_max)]
            fill_value = np.mean(valid_global) if len(valid_global) > 0 else 140.0
        
        fhr[start:end] = fill_value
        
        # Apply smooth transitions at boundaries
        gap_length = end - start
        transition_len = min(8, gap_length // 4) if gap_length > 8 else 0
        
        if transition_len > 1:
            if start > 0 and not np.isnan(fhr[start - 1]):
                prev_val = fhr[start - 1]
                t = np.linspace(0, 1, transition_len)
                fhr[start:start+transition_len] = prev_val + t * (fill_value - prev_val)
            
            if end < n and not np.isnan(fhr[end]):
                next_val = fhr[end]
                t = np.linspace(0, 1, transition_len)
                fhr[end-transition_len:end] = fill_value + t * (next_val - fill_value)
    
    def smooth_transitions(self, fhr: np.ndarray, mask: np.ndarray, 
                          window: Optional[int] = None) -> np.ndarray:
        """
        Apply smoothing to corrected regions to reduce transition artifacts.
        
        Args:
            fhr: Corrected FHR signal
            mask: Original reliability mask
            window: Smoothing window size (default from config)
            
        Returns:
            Smoothed FHR signal
        """
        if window is None:
            window = self.config.smoothing_window
        
        smoothed = fhr.copy()
        regions = self._find_artifact_regions(mask)
        
        for start, end in regions:
            smooth_start = max(0, start - window)
            smooth_end = min(len(fhr), end + window)
            
            if not np.any(np.isnan(smoothed[smooth_start:smooth_end])):
                smoothed[smooth_start:smooth_end] = uniform_filter1d(
                    smoothed[smooth_start:smooth_end], 
                    size=window, 
                    mode='nearest'
                )
        
        return smoothed


def correct_artifacts(fhr: np.ndarray, 
                      mask: np.ndarray,
                      method: CorrectionMethod = CorrectionMethod.AUTO,
                      config: Optional[CorrectionConfig] = None) -> Tuple[np.ndarray, dict]:
    """
    Correct artifacts in FHR signal.
    
    Args:
        fhr: Raw FHR signal
        mask: Reliability mask from signal_quality.py (0=reliable, 1=unreliable)
        method: Correction method (default: AUTO)
        config: Optional correction configuration
        
    Returns:
        Tuple of (corrected_fhr, correction_stats)
    """
    corrector = ArtifactCorrector(config)
    return corrector.correct(fhr, mask, method)


def get_correction_summary(stats: dict) -> str:
    """Generate a human-readable summary of correction statistics."""
    lines = [
        "Artifact Correction Summary",
        "=" * 40,
        f"Input:",
        f"  Samples marked by mask: {stats.get('mask_invalidated', 0)}",
        f"  Fixed value runs removed: {stats.get('fixed_removed', 0)}",
        "-" * 40,
        f"Gap Correction:",
        f"  Total artifact regions: {stats['total_regions']}",
        f"  Linear interpolation: {stats['linear_corrected']} regions",
        f"  Fill methods: {stats['fill_corrected']} regions",
        "-" * 40,
        f"Samples corrected: {stats['samples_corrected']}",
    ]
    return "\n".join(lines)


# =============================================================================
# Deep Learning Optimized Functions
# =============================================================================

def correct_artifacts_for_dl(
    fhr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    valid_range: Tuple[int, int] = (50, 220),
    max_jump: float = 25.0,
    fixed_value_threshold: int = 12,
) -> Tuple[np.ndarray, dict]:
    """
    Correct artifacts optimized for deep learning model input.
    
    Key guarantees for DL:
    - NO NaN values in output
    - NO zeros in output  
    - ALL values in valid_range
    - Uses only linear interpolation (consistent, predictable)
    
    Args:
        fhr: Raw FHR signal
        mask: Optional pre-computed artifact mask. If None, will auto-detect.
        valid_range: Valid FHR range (min, max)
        max_jump: Maximum jump to consider valid (bpm)
        fixed_value_threshold: Min run of identical values to flag
        
    Returns:
        Tuple of (corrected_fhr, stats_dict)
    """
    fhr = np.asarray(fhr, dtype=np.float64).flatten()
    n = len(fhr)
    valid_min, valid_max = valid_range
    
    if n == 0:
        return np.array([], dtype=np.float64), {'total_samples': 0}
    
    # Create mask if not provided
    if mask is None:
        mask = np.zeros(n, dtype=np.uint8)
        
        # Out of range
        mask[(fhr <= valid_min) | (fhr >= valid_max)] = 1
        
        # Extreme jumps
        if n >= 2:
            diff = np.abs(np.diff(fhr))
            for idx in np.where(diff > max_jump)[0]:
                if mask[idx] == 0 and mask[idx + 1] == 0:
                    mask[idx] = 1
                    mask[idx + 1] = 1
        
        # Fixed value runs
        if n >= fixed_value_threshold:
            diff_vals = np.diff(fhr)
            change_points = np.concatenate([[0], np.where(diff_vals != 0)[0] + 1, [n]])
            for i in range(len(change_points) - 1):
                start, end = change_points[i], change_points[i + 1]
                if end - start >= fixed_value_threshold:
                    val = fhr[start]
                    if valid_min < val < valid_max and np.mean(mask[start:end]) < 0.5:
                        mask[start:end] = 1
    
    # Fill artifacts with linear interpolation
    fhr_clean = fhr.copy()
    valid_indices = np.where(mask == 0)[0]
    
    if len(valid_indices) == 0:
        # All invalid - use default baseline
        fhr_clean[:] = 140.0
    else:
        # Find artifact regions
        padded = np.concatenate([[0], mask, [0]])
        diff = np.diff(padded.astype(np.int8))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            # Find boundaries
            left_val = fhr_clean[start - 1] if start > 0 else None
            right_val = fhr_clean[end] if end < n else None
            
            if left_val is not None and right_val is not None:
                # Linear interpolation
                t = np.linspace(0, 1, end - start + 2)[1:-1]
                fhr_clean[start:end] = left_val + t * (right_val - left_val)
            elif left_val is not None:
                fhr_clean[start:end] = left_val
            elif right_val is not None:
                fhr_clean[start:end] = right_val
            else:
                fhr_clean[start:end] = np.mean(fhr_clean[valid_indices])
    
    # Final clip
    fhr_clean = np.clip(fhr_clean, valid_min, valid_max)
    
    # Stats
    stats = {
        'total_samples': n,
        'artifact_samples': int(np.sum(mask)),
        'artifact_percent': float(np.sum(mask) / n * 100),
        'output_has_nan': bool(np.any(np.isnan(fhr_clean))),
        'output_has_zeros': bool(np.any(fhr_clean == 0)),
        'output_all_valid': bool(np.all((fhr_clean >= valid_min) & (fhr_clean <= valid_max))),
    }
    
    return fhr_clean, stats

