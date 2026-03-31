"""
TOCO Baseline Estimation v2 - Robust Baseline for Denoised TOCO Signals

This module implements a robust TOCO baseline estimation algorithm designed
for denoised TOCO signals. The algorithm produces smooth, physiologically
meaningful baselines that accurately track the resting uterine tone.

Key improvements over v1:
1. Morphological filtering for robust minimum estimation
2. Percentile-based baseline (not just histogram mode)
3. Gaussian smoothing for smooth output
4. Adaptive window sizing based on signal quality
5. Proper handling of invalid regions (quality mask)
6. No complex resampling - works directly at signal sample rate

Algorithm Overview:
1. Apply morphological opening (erosion + dilation) to find local minima envelope
2. Use rolling percentile (e.g., 10th percentile) within sliding windows
3. Apply Gaussian smoothing for final smooth baseline
4. Optionally incorporate quality mask to handle invalid regions
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import minimum_filter1d, maximum_filter1d, uniform_filter1d, gaussian_filter1d
from scipy.signal import savgol_filter


@dataclass
class TocoBaselineConfig:
    """Configuration for TOCO baseline estimation."""
    
    # Sample rate
    sample_rate: int = 4  # Hz
    
    # Window sizes (in seconds)
    # Increased window sizes for slower, more stable baseline tracking
    erosion_window_sec: float = 80.0  # Window for morphological erosion
    dilation_window_sec: float = 140.0 # Window for morphological dilation 
    percentile_window_sec: float = 400.0 # Window for percentile calculation 
    
    # Percentile for baseline estimation (lower = more aggressive minimum tracking)
    baseline_percentile: float = 15.0  # 15th percentile
    
    # Smoothing parameters
    smooth_window_sec: float = 90.0  # Gaussian smoothing window
    smooth_sigma_sec: float = 40.0  # Gaussian sigma
    
    # Minimum signal length for full processing (seconds)
    min_signal_length_sec: float = 120.0  # 2 minutes
    
    # Output range
    output_min: float = 0.0
    output_max: float = 100.0
    
    # Use quality mask if available
    use_quality_mask: bool = True


@dataclass
class TocoBaselineResult:
    """Result of TOCO baseline estimation."""
    
    # Estimated baseline
    baseline: np.ndarray
    
    # Intermediate results (for debugging/visualization)
    erosion_result: Optional[np.ndarray] = None
    dilation_result: Optional[np.ndarray] = None
    percentile_result: Optional[np.ndarray] = None
    
    # Statistics
    mean_baseline: float = 0.0
    std_baseline: float = 0.0
    min_baseline: float = 0.0
    max_baseline: float = 0.0


def morphological_opening(signal: np.ndarray, 
                          erosion_size: int, 
                          dilation_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply morphological opening (erosion followed by dilation).
    
    This operation finds the lower envelope of the signal by:
    1. Erosion: Replace each point with minimum in window (shrinks peaks)
    2. Dilation: Replace each point with maximum in window (restores baseline)
    
    The result tracks the "floor" of the signal, ignoring peaks (contractions).
    
    Args:
        signal: Input signal
        erosion_size: Window size for erosion (samples)
        dilation_size: Window size for dilation (samples)
        
    Returns:
        Tuple of (opened_signal, erosion_result, dilation_result)
    """
    # Erosion: local minimum
    eroded = minimum_filter1d(signal, size=erosion_size, mode='nearest')
    
    # Dilation: local maximum of eroded signal
    opened = maximum_filter1d(eroded, size=dilation_size, mode='nearest')
    
    return opened, eroded, opened


def rolling_percentile(signal: np.ndarray, 
                       window_size: int, 
                       percentile: float = 15.0) -> np.ndarray:
    """
    Compute rolling percentile of signal.
    
    Uses a sliding window approach to compute percentile at each position.
    More robust than simple minimum for baseline estimation.
    
    Args:
        signal: Input signal
        window_size: Window size in samples
        percentile: Percentile to compute (0-100)
        
    Returns:
        Rolling percentile array
    """
    n = len(signal)
    result = np.zeros(n, dtype=np.float32)
    half_window = window_size // 2
    
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window_data = signal[start:end]
        result[i] = np.percentile(window_data, percentile)
    
    return result


def rolling_percentile_fast(signal: np.ndarray,
                            window_size: int,
                            percentile: float = 15.0,
                            step: int = 10) -> np.ndarray:
    """
    Fast rolling percentile using strided computation.
    
    Computes percentile at regular intervals and interpolates between.
    Much faster for long signals.
    
    Args:
        signal: Input signal
        window_size: Window size in samples
        percentile: Percentile to compute (0-100)
        step: Step size for computation (interpolate between)
        
    Returns:
        Rolling percentile array
    """
    n = len(signal)
    half_window = window_size // 2
    
    # Compute at sparse points
    indices = list(range(0, n, step))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    
    values = np.zeros(len(indices), dtype=np.float32)
    
    for idx, i in enumerate(indices):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window_data = signal[start:end]
        values[idx] = np.percentile(window_data, percentile)
    
    # Interpolate to full resolution
    result = np.interp(np.arange(n), indices, values)
    
    return result.astype(np.float32)


def apply_quality_mask(signal: np.ndarray,
                       baseline: np.ndarray,
                       quality_mask: np.ndarray) -> np.ndarray:
    """
    Apply quality mask to baseline estimation.
    
    For invalid regions (quality_mask > 0), interpolate baseline
    from neighboring valid regions.
    
    Args:
        signal: Original signal
        baseline: Estimated baseline
        quality_mask: Quality mask (0=valid, 1=invalid, 2=suspicious)
        
    Returns:
        Corrected baseline with invalid regions interpolated
    """
    result = baseline.copy()
    invalid_mask = quality_mask > 0
    
    if not np.any(invalid_mask):
        return result
    
    # Find valid indices
    valid_indices = np.where(~invalid_mask)[0]
    
    if len(valid_indices) == 0:
        # All invalid - return signal mean
        return np.full_like(baseline, np.mean(signal))
    
    if len(valid_indices) == len(baseline):
        return result
    
    # Interpolate invalid regions from valid baseline values
    invalid_indices = np.where(invalid_mask)[0]
    valid_baseline_values = baseline[valid_indices]
    
    # Linear interpolation
    result[invalid_indices] = np.interp(
        invalid_indices,
        valid_indices,
        valid_baseline_values
    )
    
    return result


def estimate_baseline(toco: np.ndarray,
                      quality_mask: Optional[np.ndarray] = None,
                      config: Optional[TocoBaselineConfig] = None) -> TocoBaselineResult:
    """
    Estimate TOCO baseline using robust morphological and percentile methods.
    
    Algorithm:
    1. Apply morphological opening to find lower envelope
    2. Compute rolling percentile for robust minimum estimation
    3. Apply Gaussian smoothing for smooth output
    4. Handle invalid regions using quality mask
    
    Args:
        toco: Denoised TOCO signal (numpy array, values 0-100)
        quality_mask: Optional quality mask (0=valid, 1=invalid, 2=suspicious)
        config: Configuration parameters. If None, uses defaults.
        
    Returns:
        TocoBaselineResult with estimated baseline and diagnostics
    """
    if config is None:
        config = TocoBaselineConfig()
    
    # Convert to float for processing
    signal = np.asarray(toco, dtype=np.float64)
    n = len(signal)
    
    # Handle short signals
    min_samples = int(config.min_signal_length_sec * config.sample_rate)
    if n < min_samples:
        # For very short signals, use simple percentile
        baseline_value = np.percentile(signal, config.baseline_percentile)
        baseline = np.full(n, baseline_value, dtype=np.float32)
        return TocoBaselineResult(
            baseline=baseline,
            mean_baseline=baseline_value,
            std_baseline=0.0,
            min_baseline=baseline_value,
            max_baseline=baseline_value,
        )
    
    # Convert window sizes to samples
    erosion_size = max(3, int(config.erosion_window_sec * config.sample_rate))
    dilation_size = max(3, int(config.dilation_window_sec * config.sample_rate))
    percentile_window = max(11, int(config.percentile_window_sec * config.sample_rate))
    smooth_sigma = max(1, int(config.smooth_sigma_sec * config.sample_rate))
    
    # Ensure odd sizes for symmetric windows
    if erosion_size % 2 == 0:
        erosion_size += 1
    if dilation_size % 2 == 0:
        dilation_size += 1
    if percentile_window % 2 == 0:
        percentile_window += 1
    
    # Step 1: Morphological opening
    # This finds the lower envelope by removing peaks (contractions)
    opened, erosion_result, dilation_result = morphological_opening(
        signal, erosion_size, dilation_size
    )
    
    # Step 2: Rolling percentile on opened signal
    # Use fast version for efficiency
    step = max(1, config.sample_rate)  # Compute every second
    percentile_result = rolling_percentile_fast(
        opened, percentile_window, config.baseline_percentile, step
    )
    
    # Step 3: Gaussian smoothing for smooth output
    baseline = gaussian_filter1d(percentile_result, sigma=smooth_sigma, mode='nearest')
    
    # Step 4: Apply quality mask if provided
    if quality_mask is not None and config.use_quality_mask:
        baseline = apply_quality_mask(signal, baseline, quality_mask)
        # Re-smooth after interpolation
        baseline = gaussian_filter1d(baseline, sigma=smooth_sigma // 2, mode='nearest')
    
    # Clip to valid range
    baseline = np.clip(baseline, config.output_min, config.output_max)
    
    # Convert to float32 for output
    baseline = baseline.astype(np.float32)
    
    # Statistics
    mean_baseline = float(np.mean(baseline))
    std_baseline = float(np.std(baseline))
    min_baseline = float(np.min(baseline))
    max_baseline = float(np.max(baseline))
    
    return TocoBaselineResult(
        baseline=baseline,
        erosion_result=erosion_result.astype(np.float32),
        dilation_result=dilation_result.astype(np.float32),
        percentile_result=percentile_result,
        mean_baseline=mean_baseline,
        std_baseline=std_baseline,
        min_baseline=min_baseline,
        max_baseline=max_baseline,
    )


def estimate_baseline_simple(toco: np.ndarray,
                             sample_rate: int = 4) -> np.ndarray:
    """
    Simple wrapper for baseline estimation.
    
    Args:
        toco: Denoised TOCO signal
        sample_rate: Sample rate in Hz (default 4)
        
    Returns:
        Estimated baseline array
    """
    config = TocoBaselineConfig(sample_rate=sample_rate)
    result = estimate_baseline(toco, config=config)
    return result.baseline


# Alias for compatibility
analyse_baseline_v2 = estimate_baseline
