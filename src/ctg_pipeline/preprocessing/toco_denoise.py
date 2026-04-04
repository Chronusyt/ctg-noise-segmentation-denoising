"""
TOCO Signal Denoising Module

This module provides denoising functionality for raw TOCO (tocodynamometer) signals.
Based on artifact analysis of CTG data, the following artifact types are addressed:

1. Spikes - Rapid jumps (derivative > threshold)
2. Dropouts - Very low values indicating signal loss (< 5)
3. Plateaus - Consecutive identical values (stuck sensor)
4. Saturation - Values at maximum (255) indicating disconnection/saturation
5. High-frequency noise - Smoothed with low-pass filtering

Design Philosophy:
- Mark artifacts rather than immediately replace, allowing multi-stage processing
- Use median filtering to remove spikes while preserving UC peaks
- Interpolate short gaps, mark long gaps as invalid
- Apply gentle low-pass filtering to reduce high-frequency noise
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy import signal as scipy_signal


@dataclass 
class TocoDenoiseConfig:
    """Configuration for TOCO signal denoising."""
    
    # Sample rate
    sample_rate: int = 4  # Hz
    
    # Spike detection (derivative threshold)
    spike_threshold: float = 20.0  # Units per sample (rapid jump)
    spike_window: int = 5  # Samples to consider for spike repair
    
    # Dropout detection
    dropout_threshold: float = 3.0  # Values below this are dropout
    dropout_max_interpolate: int = 20  # Max samples to interpolate (5 sec)
    
    # Plateau detection (stuck sensor)
    plateau_min_length: int = 12  # Min consecutive identical values (3 sec)
    plateau_max_interpolate: int = 40  # Max plateau to interpolate (10 sec)
    
    # Saturation detection
    saturation_threshold: float = 250.0  # Values above this are saturation
    saturation_low_threshold: float = 5.0  # Very low flat values (sensor issue)
    
    # Low-pass filter
    lowpass_cutoff: float = 0.1  # Hz (very gentle, preserves UCs)
    lowpass_order: int = 2  # Filter order
    
    # Median filter for spike removal
    median_window: int = 5  # Window size for median filter
    
    # Output range
    output_min: float = 0.0
    output_max: float = 100.0


@dataclass
class TocoDenoiseResult:
    """Result of TOCO signal denoising."""
    
    # Denoised signal
    signal: np.ndarray
    
    # Quality mask (0=valid, 1=invalid/unrepairable)
    quality_mask: np.ndarray
    
    # Artifact masks (for analysis)
    spike_mask: np.ndarray
    dropout_mask: np.ndarray
    plateau_mask: np.ndarray
    saturation_mask: np.ndarray
    
    # Statistics
    n_spikes: int = 0
    n_dropouts: int = 0
    n_plateaus: int = 0
    n_saturated: int = 0
    pct_valid: float = 0.0


def detect_spikes(signal: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """
    Detect spikes based on first derivative magnitude.
    
    A spike is defined as a sudden large change in signal value.
    
    Args:
        signal: Input TOCO signal
        threshold: Derivative magnitude threshold for spike detection
        
    Returns:
        Boolean mask where True indicates spike location
    """
    deriv = np.diff(signal, prepend=signal[0])
    spike_mask = np.abs(deriv) > threshold
    return spike_mask


def detect_dropouts(signal: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect dropouts (very low values indicating signal loss).
    
    Args:
        signal: Input TOCO signal
        threshold: Values below this are considered dropouts
        
    Returns:
        Boolean mask where True indicates dropout
    """
    return signal < threshold


def detect_saturation(signal: np.ndarray, 
                      high_threshold: float = 250.0,
                      low_threshold: float = 5.0) -> np.ndarray:
    """
    Detect saturation (values at max or stuck at very low).
    
    High saturation: sensor at maximum (often 255)
    Low saturation: sensor stuck at very low value (e.g., 9-11)
    
    Args:
        signal: Input TOCO signal  
        high_threshold: Values above this are high saturation
        low_threshold: Used in combination with plateau for low saturation
        
    Returns:
        Boolean mask where True indicates saturation
    """
    return signal >= high_threshold


def detect_plateaus(signal: np.ndarray, 
                    min_length: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect plateaus (consecutive identical values - stuck sensor).
    
    Args:
        signal: Input TOCO signal
        min_length: Minimum consecutive identical values to be a plateau
        
    Returns:
        Tuple of (plateau_mask, plateau_lengths) 
        plateau_mask: Boolean mask where True indicates plateau
        plateau_lengths: Array of plateau lengths at each plateau point
    """
    n = len(signal)
    plateau_mask = np.zeros(n, dtype=bool)
    plateau_lengths = np.zeros(n, dtype=np.int32)
    
    i = 0
    while i < n:
        # Find run of identical values
        j = i + 1
        while j < n and signal[j] == signal[i]:
            j += 1
        
        run_length = j - i
        if run_length >= min_length:
            plateau_mask[i:j] = True
            plateau_lengths[i:j] = run_length
        
        i = j
    
    return plateau_mask, plateau_lengths


def interpolate_gaps(signal: np.ndarray, 
                     gap_mask: np.ndarray,
                     max_gap_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate short gaps in signal using linear interpolation.
    
    Long gaps (> max_gap_length) are left as-is and marked invalid.
    
    Args:
        signal: Input signal
        gap_mask: Boolean mask where True indicates gap (invalid data)
        max_gap_length: Maximum gap length to interpolate
        
    Returns:
        Tuple of (interpolated_signal, remaining_invalid_mask)
    """
    output = signal.copy()
    remaining_mask = np.zeros(len(signal), dtype=bool)
    n = len(signal)
    
    # Find gap runs
    i = 0
    while i < n:
        if not gap_mask[i]:
            i += 1
            continue
        
        # Find gap end
        j = i
        while j < n and gap_mask[j]:
            j += 1
        
        gap_length = j - i
        
        if gap_length <= max_gap_length:
            # Interpolate this gap
            # Find valid points before and after
            before_idx = i - 1 if i > 0 else 0
            after_idx = j if j < n else n - 1
            
            # Handle edge cases
            if i == 0:
                # Gap at start - use first valid value
                if j < n:
                    output[i:j] = signal[j]
                else:
                    remaining_mask[i:j] = True
            elif j >= n:
                # Gap at end - use last valid value
                output[i:j] = signal[before_idx]
            else:
                # Normal case - linear interpolation
                before_val = signal[before_idx]
                after_val = signal[after_idx]
                interp_vals = np.linspace(before_val, after_val, gap_length + 2)[1:-1]
                output[i:j] = interp_vals
        else:
            # Gap too long - mark as remaining invalid
            remaining_mask[i:j] = True
        
        i = j
    
    return output, remaining_mask


def apply_median_filter(signal: np.ndarray, 
                        spike_mask: np.ndarray,
                        window: int = 5) -> np.ndarray:
    """
    Apply targeted median filter to remove spikes while preserving signal.
    
    Only applies median filter around detected spikes to preserve UC shapes.
    
    Args:
        signal: Input signal
        spike_mask: Boolean mask of spike locations
        window: Median filter window size
        
    Returns:
        Filtered signal
    """
    output = signal.copy()
    
    # Dilate spike mask to get repair region
    kernel = np.ones(window, dtype=bool)
    expanded_mask = np.convolve(spike_mask.astype(int), kernel, mode='same') > 0
    
    # Apply median filter to full signal
    filtered = median_filter(signal, size=window)
    
    # Only replace at spike locations
    output[expanded_mask] = filtered[expanded_mask]
    
    return output


def apply_lowpass_filter(signal: np.ndarray,
                         cutoff: float = 0.1,
                         sample_rate: float = 4.0,
                         order: int = 2) -> np.ndarray:
    """
    Apply gentle low-pass filter to reduce high-frequency noise.
    
    Uses zero-phase Butterworth filter to avoid phase distortion.
    
    Args:
        signal: Input signal
        cutoff: Cutoff frequency in Hz
        sample_rate: Sample rate in Hz
        order: Filter order
        
    Returns:
        Filtered signal
    """
    # Normalize cutoff frequency
    nyq = sample_rate / 2
    normalized_cutoff = cutoff / nyq
    
    # Ensure cutoff is valid
    if normalized_cutoff >= 1.0:
        normalized_cutoff = 0.99
    if normalized_cutoff <= 0:
        return signal
    
    # Design Butterworth filter
    b, a = scipy_signal.butter(order, normalized_cutoff, btype='low')
    
    # Apply zero-phase filter
    # Use padlen to avoid edge effects
    padlen = min(3 * max(len(a), len(b)), len(signal) - 1)
    if padlen < 1:
        return signal
    
    filtered = scipy_signal.filtfilt(b, a, signal, padlen=padlen)
    
    return filtered


def denoise_toco(signal: np.ndarray,
                 config: Optional[TocoDenoiseConfig] = None) -> TocoDenoiseResult:
    """
    Main TOCO signal denoising function.
    
    Processing pipeline:
    1. Detect and mark artifacts (spikes, dropouts, plateaus, saturation)
    2. Apply median filter to remove spikes
    3. Interpolate short gaps (dropouts, short plateaus)
    4. Apply gentle low-pass filter
    5. Clip to valid output range
    6. Mark unrecoverable regions in quality mask
    
    Args:
        signal: Raw TOCO signal (numpy array)
        config: Denoising configuration. If None, uses defaults.
        
    Returns:
        TocoDenoiseResult with denoised signal and diagnostics
    """
    if config is None:
        config = TocoDenoiseConfig()
    
    # Convert to float for processing
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    
    # Step 1: Detect artifacts
    spike_mask = detect_spikes(signal, config.spike_threshold)
    dropout_mask = detect_dropouts(signal, config.dropout_threshold)
    saturation_mask = detect_saturation(signal, config.saturation_threshold)
    plateau_mask, plateau_lengths = detect_plateaus(signal, config.plateau_min_length)
    
    # Combine saturation with long plateaus at extreme values
    # (Stuck at 255 or stuck at very low values like 9-11)
    extreme_plateau_mask = plateau_mask & (
        (signal >= config.saturation_threshold) | 
        (signal <= config.saturation_low_threshold)
    )
    saturation_mask = saturation_mask | extreme_plateau_mask
    
    # Step 2: Apply median filter to remove spikes
    working_signal = apply_median_filter(signal, spike_mask, config.median_window)
    
    # Step 3: Interpolate gaps
    # First handle dropouts
    gap_mask = dropout_mask | saturation_mask
    working_signal, remaining_invalid = interpolate_gaps(
        working_signal, gap_mask, config.dropout_max_interpolate
    )
    
    # Handle short plateaus (not at extreme values)
    normal_plateau_mask = plateau_mask & ~extreme_plateau_mask
    short_plateau_mask = normal_plateau_mask & (plateau_lengths <= config.plateau_max_interpolate)
    
    working_signal, plateau_remaining = interpolate_gaps(
        working_signal, short_plateau_mask, config.plateau_max_interpolate
    )
    remaining_invalid = remaining_invalid | plateau_remaining
    
    # Step 4: Apply gentle low-pass filter
    # Only on valid regions to avoid spreading artifacts
    if config.lowpass_cutoff > 0:
        # Create a version with valid data filled in for filtering
        temp_signal = working_signal.copy()
        # Forward fill then backward fill remaining gaps for filtering only
        for i in range(1, n):
            if remaining_invalid[i] and not remaining_invalid[i-1]:
                temp_signal[i] = temp_signal[i-1]
        for i in range(n-2, -1, -1):
            if remaining_invalid[i] and not remaining_invalid[i+1]:
                temp_signal[i] = temp_signal[i+1]
        
        filtered = apply_lowpass_filter(
            temp_signal, config.lowpass_cutoff, config.sample_rate, config.lowpass_order
        )
        # Only use filtered values where we had valid data
        working_signal = np.where(remaining_invalid, working_signal, filtered)
    
    # Step 5: Clip to valid range
    working_signal = np.clip(working_signal, config.output_min, config.output_max)
    
    # Step 6: Build quality mask
    quality_mask = remaining_invalid.astype(np.int8)
    
    # Also mark long plateaus as potentially suspicious (quality=2)
    long_plateau_mask = plateau_mask & (plateau_lengths > config.plateau_max_interpolate) & ~extreme_plateau_mask
    quality_mask = np.where(long_plateau_mask & (quality_mask == 0), 2, quality_mask)
    
    # Statistics
    n_spikes = int(np.sum(spike_mask))
    n_dropouts = int(np.sum(dropout_mask))
    n_plateaus = int(np.sum(plateau_mask))
    n_saturated = int(np.sum(saturation_mask))
    pct_valid = 100.0 * (1.0 - np.mean(quality_mask > 0))
    
    return TocoDenoiseResult(
        signal=working_signal.astype(np.float32),
        quality_mask=quality_mask,
        spike_mask=spike_mask,
        dropout_mask=dropout_mask,
        plateau_mask=plateau_mask,
        saturation_mask=saturation_mask,
        n_spikes=n_spikes,
        n_dropouts=n_dropouts,
        n_plateaus=n_plateaus,
        n_saturated=n_saturated,
        pct_valid=pct_valid,
    )


def denoise_toco_simple(signal: np.ndarray,
                        sample_rate: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple wrapper for TOCO denoising.
    
    Args:
        signal: Raw TOCO signal
        sample_rate: Sample rate in Hz (default 4)
        
    Returns:
        Tuple of (denoised_signal, quality_mask)
        quality_mask: 0=valid, 1=invalid, 2=suspicious
    """
    config = TocoDenoiseConfig(sample_rate=sample_rate)
    result = denoise_toco(signal, config)
    return result.signal, result.quality_mask
