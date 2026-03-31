"""
Optimized FHR Baseline Analysis Module

This is a simplified version of the FHR baseline analysis that takes advantage
of artifact-free pretreated signals. Since artifacts have already been detected
and corrected, the baseline computation can be streamlined.

Key simplifications:
1. No need to handle 0 or 255 values (already corrected)
2. Two-pass approach with full FIGO ACC/DEC detection
3. Cleaner signal allows for simpler processing

The algorithm:
1. Two-pass baseline estimation:
   a. First pass: Use median for rough baseline estimate
   b. Second pass: Exclude ACC/DEC using full FIGO detector, take median
2. Interpolate baseline to full resolution
3. Apply final smoothing

Key insight: Baseline should represent "resting" FHR, so we exclude both:
- Accelerations: Regions >= 15 bpm above baseline for >= 15 seconds
- Decelerations: Regions >= 15 bpm below baseline for >= 15 seconds
Then take median of remaining samples for a stable baseline.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Optional, Tuple, Union
from dataclasses import dataclass

# Import full FIGO detectors
from .acc_detection_figo_v2 import AccelerationDetector, AccelerationConfig
from .dec_detection_figo_v2 import DecelerationDetector, DecelerationConfig


@dataclass
class BaselineConfig:
    """
    Configuration for baseline analysis.
    
    Attributes:
        window_size: Sliding window size in samples (default: 1920 = 8 min)
        window_step: Step size for sliding window (default: 240 = 1 min)
        smoothing_window: Final smoothing window size (default: 240 = 60 sec)
        variability_threshold: Exclude windows with extreme variability (default: 25 bpm)
        min_valid_ratio: Minimum valid sample ratio to use direct estimation (default: 0.5)
    """
    window_size: int = 1920
    window_step: int = 240
    smoothing_window: int = 240
    variability_threshold: float = 25.0
    min_valid_ratio: float = 0.5  # 50% minimum valid samples


@dataclass
class BaselineDiagnostics:
    """Diagnostics from baseline computation."""
    baseline: np.ndarray
    valid_ratios_pass2: np.ndarray
    window_centers: np.ndarray
    valid_ratios_pass2_full: np.ndarray


def analyse_baseline_optimized(fhr_clean: np.ndarray,
                                config: Optional[BaselineConfig] = None,
                                mask: Optional[np.ndarray] = None,
                                sample_rate: float = 4.0) -> np.ndarray:
    """
    Compute FHR baseline from artifact-free (cleaned) signal.
    
    Uses 2-pass FIGO-style algorithm:
    - Pass 1: Rough baseline using median
    - Pass 2: Exclude ACC/DEC (>=15 bpm for >=15 sec), take median
    """
    if config is None:
        config = BaselineConfig()
    
    has_nan = np.any(np.isnan(fhr_clean))
    if has_nan:
        fhr_interp = interpolate_nan(fhr_clean)
    else:
        fhr_interp = fhr_clean.astype(np.float64)

    if mask is None:
        mask = np.zeros(len(fhr_clean), dtype=bool)
    else:
        mask = (mask != 0)
    
    baseline_points, baseline_indices = compute_window_baseline_two_pass_figo(
        fhr_interp, config.window_size, config.window_step,
        mask, config.variability_threshold, config.min_valid_ratio,
        sample_rate=sample_rate,
    )
    
    baseline = interpolate_baseline(baseline_points, baseline_indices, len(fhr_clean))
    baseline = uniform_filter1d(baseline, size=config.smoothing_window, mode='nearest')
    baseline = np.round(baseline).astype(np.int32)
    
    return baseline


def analyse_baseline_with_diagnostics(fhr_clean: np.ndarray,
                                       config: Optional[BaselineConfig] = None,
                                       mask: Optional[np.ndarray] = None,
                                       sample_rate: float = 4.0) -> BaselineDiagnostics:
    """
    Compute FHR baseline with diagnostics for debugging/visualization.
    
    Uses 2-pass FIGO-style algorithm with diagnostic output.
    """
    if config is None:
        config = BaselineConfig()
    
    has_nan = np.any(np.isnan(fhr_clean))
    if has_nan:
        fhr_interp = interpolate_nan(fhr_clean)
    else:
        fhr_interp = fhr_clean.astype(np.float64)

    if mask is None:
        mask = np.zeros(len(fhr_clean), dtype=bool)
    else:
        mask = (mask != 0)
    
    baseline_points, baseline_indices, valid_ratios_pass2, window_centers = compute_window_baseline_two_pass_figo(
        fhr_interp, config.window_size, config.window_step,
        mask, config.variability_threshold, config.min_valid_ratio,
        return_diagnostics=True,
        sample_rate=sample_rate,
    )
    
    baseline = interpolate_baseline(baseline_points, baseline_indices, len(fhr_clean))
    
    n = len(fhr_clean)
    valid_ratios_pass2_full = np.interp(np.arange(n), window_centers, valid_ratios_pass2)
    
    baseline = uniform_filter1d(baseline, size=config.smoothing_window, mode='nearest')
    baseline = np.round(baseline).astype(np.int32)
    
    return BaselineDiagnostics(
        baseline=baseline,
        valid_ratios_pass2=np.array(valid_ratios_pass2),
        window_centers=np.array(window_centers),
        valid_ratios_pass2_full=valid_ratios_pass2_full
    )


def interpolate_nan(fhr: np.ndarray) -> np.ndarray:
    """Interpolate NaN values using linear interpolation."""
    result = fhr.astype(np.float64).copy()
    nan_mask = np.isnan(result)
    
    if not np.any(nan_mask):
        return result
    
    valid_indices = np.where(~nan_mask)[0]
    nan_indices = np.where(nan_mask)[0]
    
    if len(valid_indices) < 2:
        mean_val = np.nanmean(fhr) if np.any(~nan_mask) else 140.0
        result[nan_mask] = mean_val
        return result
    
    result[nan_indices] = np.interp(nan_indices, valid_indices, result[valid_indices])
    return result


def detect_acc_dec_figo(
    fhr: np.ndarray,
    baseline: np.ndarray,
    sample_rate: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect ACC/DEC using full FIGO detector (AccelerationDetector and DecelerationDetector).
    
    This function uses the same detectors as the final ACC/DEC detection in the
    analysis pipeline, ensuring consistent results between baseline estimation
    and final detection.
    
    Uses FIGO RULE_15_15 criterion: >= 15 bpm for >= 15 sec
    
    Args:
        fhr: FHR signal
        baseline: Baseline signal
        
    Returns:
        Tuple of (acc_mask, dec_mask) - boolean arrays marking ACC/DEC regions
    """
    n = len(fhr)
    acc_mask = np.zeros(n, dtype=bool)
    dec_mask = np.zeros(n, dtype=bool)
    
    # Use full FIGO detectors with RULE_15_15 criterion at runtime sample rate
    sr = max(1, int(round(sample_rate)))
    acc_detector = AccelerationDetector(AccelerationConfig(sample_rate=sr))
    dec_detector = DecelerationDetector(DecelerationConfig(sample_rate=sr))
    
    # Detect accelerations
    accelerations = acc_detector.detect(fhr, baseline, mask=None)
    for acc in accelerations:
        acc_mask[acc.start_idx:acc.end_idx] = True
    
    # Detect decelerations
    decelerations = dec_detector.detect(fhr, baseline, mask=None)
    for dec in decelerations:
        dec_mask[dec.start_idx:dec.end_idx] = True
    
    return acc_mask, dec_mask


def compute_window_baseline_two_pass_figo(
        fhr: np.ndarray,
        window_size: int,
        step: int,
        mask: Optional[np.ndarray] = None,
        variability_threshold: float = 25.0,
        min_valid_ratio: float = 0.4,
    return_diagnostics: bool = False,
    sample_rate: float = 4.0,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute baseline using 2-pass FIGO algorithm.
    
    Two-pass approach:
    1. First pass: Compute rough baseline using median (50th percentile)
    2. Second pass: Exclude ACC/DEC using full FIGO detector, take median
    
    If valid_ratio < min_valid_ratio, use linear interpolation from neighbors.
    
    Args:
        fhr: FHR signal
        window_size: Window size in samples
        step: Step size in samples
        mask: Optional mask for invalid samples
        variability_threshold: Exclude windows with std > this (bpm)
        min_valid_ratio: Minimum valid ratio (below this, use interpolation)
        return_diagnostics: If True, also return valid_ratios and window_centers
        
    Returns:
        If return_diagnostics=False: Tuple of (baseline_points, baseline_indices)
        If return_diagnostics=True: Tuple of (baseline_points, baseline_indices, valid_ratios, window_centers)
    """
    n = len(fhr)
    if mask is None:
        mask = np.zeros(n, dtype=bool)
    else:
        mask = (mask != 0)
        
    if n < window_size:
        valid = fhr[~np.isnan(fhr)]
        if len(valid) > 0:
            if return_diagnostics:
                return (np.array([np.median(valid)]), np.array([n // 2]), 
                        np.array([1.0]), np.array([n // 2]))
            return np.array([np.median(valid)]), np.array([n // 2])
        if return_diagnostics:
            return np.array([140.0]), np.array([n // 2]), np.array([0.0]), np.array([n // 2])
        return np.array([140.0]), np.array([n // 2])
    
    # ==================== FIRST PASS: Rough baseline ====================
    rough_baseline_points = []
    rough_baseline_indices = []
    
    for i in range(0, n - window_size + 1, step):
        window_data = fhr[i:i + window_size]
        window_mask = mask[i:i + window_size]
        valid_idx = ~np.isnan(window_data) & (~window_mask)
        valid_window = window_data[valid_idx]
        local_std = np.nanstd(valid_window) if len(valid_window) > 0 else np.nan
        min_valid = max(4, window_size // 8)
        
        if len(valid_window) > min_valid and (np.isnan(local_std) or local_std <= variability_threshold):
            baseline_val = np.median(valid_window)
        else:
            baseline_val = np.nan
        
        center = i + window_size // 2
        rough_baseline_points.append(baseline_val)
        rough_baseline_indices.append(center)
    
    rough_points = np.array(rough_baseline_points)
    rough_indices = np.array(rough_baseline_indices)
    valid_mask = ~np.isnan(rough_points)
    valid_fraction = np.sum(valid_mask) / len(rough_points) if len(rough_points) > 0 else 0
    
    # Fallback if too few valid windows (relax variability constraint)
    if valid_fraction < 0.5:
        rough_baseline_points = []
        rough_baseline_indices = []
        for i in range(0, n - window_size + 1, step):
            window_data = fhr[i:i + window_size]
            window_mask = mask[i:i + window_size]
            valid_idx = ~np.isnan(window_data) & (~window_mask)
            valid_window = window_data[valid_idx]
            min_valid = max(4, window_size // 8)
            if len(valid_window) > min_valid:
                baseline_val = np.median(valid_window)
            else:
                baseline_val = np.nan
            center = i + window_size // 2
            rough_baseline_points.append(baseline_val)
            rough_baseline_indices.append(center)
        rough_points = np.array(rough_baseline_points)
        rough_indices = np.array(rough_baseline_indices)
        valid_mask = ~np.isnan(rough_points)
    
    if not np.any(valid_mask):
        if return_diagnostics:
            return (np.array([140.0, 140.0]), np.array([0, n-1]), 
                    np.array([0.0, 0.0]), np.array([0, n-1]))
        return np.array([140.0, 140.0]), np.array([0, n-1])
    
    # Interpolate rough baseline to full resolution
    rough_baseline_full = np.interp(np.arange(n), rough_indices[valid_mask], rough_points[valid_mask])
    
    # ==================== FIGO ACC/DEC detection ====================
    acc_mask_full, dec_mask_full = detect_acc_dec_figo(
        fhr, rough_baseline_full, sample_rate=sample_rate
    )
    
    # ==================== SECOND PASS: Refine using FIGO detection ====================
    pass2_baseline_points = []
    pass2_baseline_indices = []
    pass2_valid_ratios = []
    pass2_valid_flags = []
    window_centers = []
    
    min_valid_samples = max(4, int(window_size * min_valid_ratio))
    
    for i in range(0, n - window_size + 1, step):
        window_data = fhr[i:i + window_size]
        window_mask = mask[i:i + window_size]
        window_acc = acc_mask_full[i:i + window_size]
        window_dec = dec_mask_full[i:i + window_size]

        # Exclude ACC, DEC, masked, and NaN
        nan_mask = np.isnan(window_data) | window_mask
        valid_mask_local = (~window_acc) & (~window_dec) & (~nan_mask)
        
        valid_count = np.sum(valid_mask_local)
        valid_ratio = valid_count / window_size
        pass2_valid_ratios.append(valid_ratio)
        
        center = i + window_size // 2
        window_centers.append(center)
        
        if valid_count >= min_valid_samples:
            valid_samples = window_data[valid_mask_local]
            baseline_val = np.median(valid_samples)
            pass2_valid_flags.append(True)
        else:
            baseline_val = np.nan
            pass2_valid_flags.append(False)

        pass2_baseline_points.append(baseline_val)
        pass2_baseline_indices.append(center)
    
    # Convert to arrays
    pass2_baseline_points = np.array(pass2_baseline_points, dtype=np.float64)
    pass2_valid_flags = np.array(pass2_valid_flags)
    pass2_baseline_indices = np.array(pass2_baseline_indices)
    
    # Linear interpolation for windows with valid_ratio < min_valid_ratio
    if not np.all(pass2_valid_flags) and np.any(pass2_valid_flags):
        valid_idx = np.where(pass2_valid_flags)[0]
        invalid_idx = np.where(~pass2_valid_flags)[0]
        pass2_baseline_points[invalid_idx] = np.interp(
            invalid_idx, valid_idx, pass2_baseline_points[valid_idx]
        )
    elif not np.any(pass2_valid_flags):
        # No valid windows at all, fall back to rough baseline
        for i, center in enumerate(window_centers):
            pass2_baseline_points[i] = rough_baseline_full[center]
    
    # Convert to lists for edge point handling
    baseline_points = pass2_baseline_points.tolist()
    baseline_indices = pass2_baseline_indices.tolist()
    
    if len(baseline_indices) == 0:
        if return_diagnostics:
            return (np.array([140.0, 140.0]), np.array([0, n-1]),
                    np.array([0.0, 0.0]), np.array([0, n-1]))
        return np.array([140.0, 140.0]), np.array([0, n-1])
    
    # Add edge points if needed
    if baseline_indices[0] > 0:
        baseline_indices.insert(0, 0)
        baseline_points.insert(0, baseline_points[0])
        if return_diagnostics:
            pass2_valid_ratios.insert(0, pass2_valid_ratios[0])
            window_centers.insert(0, 0)
    
    if baseline_indices[-1] < n - 1:
        baseline_indices.append(n - 1)
        baseline_points.append(baseline_points[-1])
        if return_diagnostics:
            pass2_valid_ratios.append(pass2_valid_ratios[-1])
            window_centers.append(n - 1)
    
    if return_diagnostics:
        return (np.array(baseline_points), np.array(baseline_indices),
                np.array(pass2_valid_ratios), np.array(window_centers))
    return np.array(baseline_points), np.array(baseline_indices)


def interpolate_baseline(points: np.ndarray, indices: np.ndarray, length: int) -> np.ndarray:
    """Interpolate baseline points to full signal length."""
    all_indices = np.arange(length)
    baseline = np.interp(all_indices, indices, points)
    return baseline
