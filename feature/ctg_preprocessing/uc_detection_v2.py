"""
UC (Uterine Contraction) Detection v2 - Robust Detection for Denoised TOCO

This module implements a robust uterine contraction detection algorithm designed
for denoised TOCO signals with smooth v2 baselines.

Key improvements over v1:
1. MAD-based adaptive threshold: noise = 1.4826 * MAD, threshold = max(A_min, k * noise)
2. Prominence-based peak detection (scipy)
3. Better handling of quality mask (invalid regions)
4. Threshold-crossing boundary detection with hold time for noise robustness
5. Smoothed signal for boundary detection to avoid noise artifacts
6. Configurable parameters with sensible defaults

Detection Criteria (Strategy):
---------------------------------
1. PEAK DETECTION:
   - Find local maxima using scipy.signal.find_peaks
   - Minimum prominence: 10 units
   - Minimum distance between peaks: 30 seconds (configurable)

2. AMPLITUDE THRESHOLD (Adaptive MAD-based):
   - Compute noise level: noise = 1.4826 * MAD(above_baseline)
   - Threshold = max(A_min, k * noise), where A_min=20, k=3.0
   - Peak must exceed baseline by threshold

3. CONTRACTION BOUNDARIES (Threshold Crossing Method):
   
   Start point (algorithmic):
     First time point where TOCO(t) - baseline(t) >= θ
     for at least D seconds (to avoid triggering on noise)
     
   End point (algorithmic):
     First time point after the peak where TOCO(t) - baseline(t) < θ
     and stays below for D seconds
   
   Where:
     θ (theta): boundary_threshold (default 8 TOCO units)
     D: boundary_hold_sec (default 15 seconds)
   
   Maximum search window: 90 seconds each direction

4. DURATION VALIDATION:
   - Minimum duration: 30 seconds (physiological minimum)
   - Maximum duration: 180 seconds (physiological maximum)

5. STRENGTH CALCULATION:
   - Strength = peak_value - baseline_value at peak
   - Minimum strength: 20 units (A_min)

6. QUALITY MASK INTEGRATION:
   - Skip peaks in invalid regions (quality_mask > 0)
   - Adjust boundaries to avoid invalid regions

7. ADDITIONAL VALIDATION:
   - Check contraction shape (rise and fall slopes)
   - Ensure peak is near center of contraction (not at edges)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


@dataclass
class UcModelV2:
    """
    Uterine Contraction model v2.
    
    Attributes:
        peak_index: Index of contraction peak
        start_index: Index of contraction start
        end_index: Index of contraction end
        strength: Contraction strength (peak - baseline)
        duration_sec: Duration in seconds
        peak_value: TOCO value at peak
        baseline_value: Baseline value at peak
        rise_time_sec: Time from start to peak (seconds)
        fall_time_sec: Time from peak to end (seconds)
        area: Area under contraction curve (above baseline)
    """
    peak_index: int
    start_index: int
    end_index: int
    strength: float
    duration_sec: float
    peak_value: float
    baseline_value: float
    rise_time_sec: float = 0.0
    fall_time_sec: float = 0.0
    area: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "peakIndex": self.peak_index,
            "startIndex": self.start_index,
            "endIndex": self.end_index,
            "strength": self.strength,
            "durationSec": self.duration_sec,
            "peakValue": self.peak_value,
            "baselineValue": self.baseline_value,
            "riseTimeSec": self.rise_time_sec,
            "fallTimeSec": self.fall_time_sec,
            "area": self.area,
        }


@dataclass
class UcDetectionConfigV2:
    """Configuration for UC detection v2."""
    
    # Sample rate
    sample_rate: int = 4  # Hz
    
    # Peak detection
    min_peak_distance_sec: float = 30.0  # Initial minimum seconds between peaks (for find_peaks)
    min_prominence: float = 10.0  # Minimum prominence for peak detection
    
    # Peak merging - close peaks are merged into one contraction
    merge_peak_distance_sec: float = 40.0  # Peaks closer than this are merged
    min_separate_interval_sec: float = 60.0  # Minimum interval for separate contractions
    
    # Amplitude thresholds (MAD-based)
    a_min: float = 20.0  # Minimum amplitude threshold (A_min)
    k_factor: float = 3.0  # Multiplier for noise: threshold = max(a_min, k * noise)
    mad_scale: float = 1.4826  # Scale factor for MAD to estimate std (normal distribution)
    
    # Boundary detection - threshold crossing method
    # Start point: First time where TOCO(t) - baseline(t) >= boundary_threshold for boundary_hold_sec
    # End point: First time after peak where TOCO(t) - baseline(t) < boundary_threshold for boundary_hold_sec
    boundary_threshold: float = 8.0  # θ: threshold in TOCO units (5-10 typical)
    boundary_hold_sec: float = 15.0  # D: duration to confirm crossing (10-20 sec typical)
    max_search_window_sec: float = 90.0  # Max search window for boundaries (each direction)
    boundary_smoothing_sec: float = 3.0  # Smoothing window for boundary detection
    
    # Legacy parameter (kept for compatibility, not used in new algorithm)
    boundary_threshold_pct: float = 0.10  # Drop to 10% of peak amplitude for boundaries
    
    # Duration validation
    min_duration_sec: float = 30.0  # Minimum contraction duration (physiological)
    max_duration_sec: float = 180.0  # Maximum contraction duration (physiological)
    
    # Strength validation
    min_strength: float = 20.0  # Minimum strength (peak - baseline), same as a_min
    
    # Shape validation - moderately relaxed to catch edge cases
    min_rise_time_sec: float = 10.0  # Minimum rise time
    max_rise_time_sec: float = 80.0  # Maximum rise time (relaxed from 60s for slow contractions)
    max_peak_asymmetry: float = 0.85  # Peak position (0.15-0.85), slightly relaxed from 0.8
    
    # Quality mask
    use_quality_mask: bool = True
    max_invalid_pct: float = 0.3  # Max percentage of invalid samples in contraction


@dataclass
class UcDetectionResultV2:
    """Result of UC detection v2."""
    
    # Detected contractions
    contractions: List[UcModelV2]
    
    # Statistics
    n_contractions: int = 0
    mean_strength: float = 0.0
    mean_duration_sec: float = 0.0
    
    # Adaptive threshold used
    adaptive_threshold: float = 0.0


def compute_adaptive_threshold(toco: np.ndarray,
                                baseline: np.ndarray,
                                config: UcDetectionConfigV2) -> float:
    """
    Compute adaptive threshold based on MAD (Median Absolute Deviation).
    
    Uses: noise = 1.4826 * MAD(above_baseline)
          threshold = max(A_min, k * noise)
    
    The MAD is computed on samples near the baseline (< 50th percentile)
    to avoid inflating noise estimate with actual contractions.
    
    Args:
        toco: TOCO signal
        baseline: Baseline signal
        config: Detection configuration
        
    Returns:
        Adaptive threshold value
    """
    diff = toco - baseline
    
    # Use valid samples only (where diff is reasonable)
    valid_diff = diff[(diff > -50) & (diff < 100)]
    
    if len(valid_diff) < 100:
        return config.a_min
    
    # Only use samples near baseline (below median) for noise estimation
    # This avoids inflating noise estimate with actual contractions
    median_diff = np.median(valid_diff)
    near_baseline = valid_diff[valid_diff <= median_diff]
    
    if len(near_baseline) < 50:
        near_baseline = valid_diff
    
    # Compute MAD (Median Absolute Deviation) on near-baseline samples
    median_val = np.median(near_baseline)
    mad = np.median(np.abs(near_baseline - median_val))
    
    # Estimate noise level: noise = 1.4826 * MAD (for normal distribution)
    noise = config.mad_scale * mad
    
    # Adaptive threshold: max(A_min, k * noise)
    # Also cap at reasonable maximum (e.g., 40 units) to avoid missing real contractions
    threshold = max(config.a_min, config.k_factor * noise)
    threshold = min(threshold, 40.0)  # Cap at 40 units maximum
    
    return threshold


def merge_close_peaks(peaks: np.ndarray,
                      above_baseline: np.ndarray,
                      config: UcDetectionConfigV2) -> np.ndarray:
    """
    Merge peaks that are too close together.
    
    When multiple peaks occur within merge_peak_distance_sec, keep only the
    highest peak. This handles cases where a single contraction has multiple
    local maxima.
    
    Args:
        peaks: Array of peak indices
        above_baseline: Signal above baseline (for determining which peak to keep)
        config: Detection configuration
        
    Returns:
        Array of merged peak indices
    """
    if len(peaks) <= 1:
        return peaks
    
    merge_distance = int(config.merge_peak_distance_sec * config.sample_rate)
    
    merged_peaks = []
    i = 0
    
    while i < len(peaks):
        # Start a group with current peak
        group_start = i
        group_end = i
        
        # Find all peaks within merge distance
        while group_end + 1 < len(peaks):
            if peaks[group_end + 1] - peaks[group_start] <= merge_distance:
                group_end += 1
            else:
                break
        
        # Keep the peak with highest amplitude in this group
        group_peaks = peaks[group_start:group_end + 1]
        best_idx = np.argmax(above_baseline[group_peaks])
        merged_peaks.append(group_peaks[best_idx])
        
        i = group_end + 1
    
    return np.array(merged_peaks)


def filter_by_minimum_interval(contractions: List[UcModelV2],
                                config: UcDetectionConfigV2) -> List[UcModelV2]:
    """
    Ensure minimum interval between separate contractions.
    
    If two contractions are closer than min_separate_interval_sec,
    keep only the stronger one.
    
    Args:
        contractions: List of detected contractions
        config: Detection configuration
        
    Returns:
        Filtered list with minimum intervals enforced
    """
    if len(contractions) <= 1:
        return contractions
    
    min_interval_samples = int(config.min_separate_interval_sec * config.sample_rate)
    
    # Sort by peak index
    sorted_ucs = sorted(contractions, key=lambda uc: uc.peak_index)
    
    filtered = [sorted_ucs[0]]
    
    for uc in sorted_ucs[1:]:
        last_uc = filtered[-1]
        interval = uc.peak_index - last_uc.peak_index
        
        if interval >= min_interval_samples:
            # Sufficient interval, add as separate contraction
            filtered.append(uc)
        else:
            # Too close - keep the stronger one
            if uc.strength > last_uc.strength:
                filtered[-1] = uc
    
    return filtered


def find_contraction_boundaries(toco: np.ndarray,
                                 baseline: np.ndarray,
                                 peak_idx: int,
                                 config: UcDetectionConfigV2) -> Tuple[int, int]:
    """
    Find start and end boundaries of a contraction using threshold crossing method.
    
    Algorithm based on standard definitions:
    
    Start point (algorithmic):
        First time point where TOCO(t) - baseline(t) >= θ
        for at least D seconds (to avoid triggering on noise)
        
    End point (algorithmic):
        First time point after the peak where TOCO(t) - baseline(t) < θ
        and stays below for D seconds
    
    Where:
        θ (theta): boundary_threshold (default 8 TOCO units)
        D: boundary_hold_sec (default 15 seconds)
    
    Args:
        toco: TOCO signal
        baseline: Baseline signal
        peak_idx: Index of contraction peak
        config: Detection configuration
        
    Returns:
        Tuple of (start_index, end_index)
    """
    n = len(toco)
    max_search = int(config.max_search_window_sec * config.sample_rate)
    smooth_window = int(config.boundary_smoothing_sec * config.sample_rate)
    hold_samples = int(config.boundary_hold_sec * config.sample_rate)
    threshold = config.boundary_threshold
    
    # Smooth signal for boundary detection to reduce noise impact
    if smooth_window > 1:
        toco_smooth = uniform_filter1d(toco.astype(np.float64), size=smooth_window)
        baseline_smooth = uniform_filter1d(baseline.astype(np.float64), size=smooth_window)
    else:
        toco_smooth = toco.astype(np.float64)
        baseline_smooth = baseline.astype(np.float64)
    
    # Compute signal above baseline
    above_baseline = toco_smooth - baseline_smooth
    
    # Check peak amplitude
    peak_amplitude = above_baseline[peak_idx]
    if peak_amplitude <= threshold:
        return peak_idx, peak_idx
    
    # =========================================================================
    # Find START point: Search backward from peak
    # First time point where above_baseline >= threshold for at least hold_samples
    # =========================================================================
    search_start = max(0, peak_idx - max_search)
    start_idx = peak_idx
    
    # Create binary mask: 1 where above threshold, 0 otherwise
    above_thresh = (above_baseline >= threshold).astype(np.int8)
    
    # Search backward from peak to find where the signal first crosses threshold
    # and stays above for hold_samples
    for i in range(peak_idx - 1, search_start - 1, -1):
        if above_thresh[i] == 0:
            # Found a point below threshold
            # Check if the next segment stays above threshold for hold_samples
            candidate_start = i + 1
            
            # Verify it stays above threshold for hold_samples after this point
            if candidate_start + hold_samples <= peak_idx:
                segment = above_thresh[candidate_start:candidate_start + hold_samples]
                if np.all(segment == 1):
                    # Valid start point found
                    start_idx = candidate_start
                    break
            else:
                # Not enough samples to verify, but this is still a crossing
                start_idx = candidate_start
                break
    else:
        # Reached search boundary without finding crossing
        # Use the first point that is above threshold
        for i in range(search_start, peak_idx):
            if above_thresh[i] == 1:
                start_idx = i
                break
    
    # =========================================================================
    # Find END point: Search forward from peak
    # First time point after peak where above_baseline < threshold
    # and stays below for hold_samples
    # =========================================================================
    search_end = min(n, peak_idx + max_search)
    end_idx = peak_idx
    
    # Search forward from peak to find where signal drops below threshold
    # and stays below for hold_samples
    i = peak_idx + 1
    while i < search_end:
        if above_thresh[i] == 0:
            # Found a point below threshold
            # Check if it stays below for hold_samples
            candidate_end = i
            
            if candidate_end + hold_samples <= n:
                segment = above_thresh[candidate_end:candidate_end + hold_samples]
                if np.all(segment == 0):
                    # Valid end point found - stays below threshold for D seconds
                    end_idx = candidate_end
                    break
                else:
                    # Temporary dip - find next time it goes below threshold
                    # Skip to where it goes above again
                    next_above = np.where(segment == 1)[0]
                    if len(next_above) > 0:
                        i = candidate_end + next_above[0] + 1
                        continue
            else:
                # Near end of signal, accept this as end point
                end_idx = candidate_end
                break
        i += 1
    else:
        # Reached search boundary without finding valid end
        # Find the last point above threshold
        for i in range(search_end - 1, peak_idx, -1):
            if above_thresh[i] == 1:
                end_idx = i + 1
                break
    
    # Ensure valid bounds
    start_idx = max(0, min(start_idx, peak_idx - 1))
    end_idx = min(n - 1, max(end_idx, peak_idx + 1))
    
    return start_idx, end_idx


def compute_contraction_area(toco: np.ndarray,
                              baseline: np.ndarray,
                              start_idx: int,
                              end_idx: int) -> float:
    """
    Compute area under contraction curve (above baseline).
    
    Args:
        toco: TOCO signal
        baseline: Baseline signal  
        start_idx: Start index
        end_idx: End index
        
    Returns:
        Area in (units * samples)
    """
    if end_idx <= start_idx:
        return 0.0
    
    above = toco[start_idx:end_idx] - baseline[start_idx:end_idx]
    above = np.maximum(above, 0)  # Only count positive area
    return float(np.sum(above))


def detect_uc_v2(toco: np.ndarray,
                  baseline: np.ndarray,
                  quality_mask: Optional[np.ndarray] = None,
                  config: Optional[UcDetectionConfigV2] = None) -> UcDetectionResultV2:
    """
    Detect uterine contractions using robust v2 algorithm.
    
    Algorithm:
    1. Compute adaptive MAD-based threshold
    2. Find peaks using prominence-based detection
    3. Filter peaks by amplitude above baseline
    4. Find contraction boundaries using derivative + threshold methods
    5. Validate duration, strength, and shape
    6. Handle quality mask for invalid regions
    
    Args:
        toco: Denoised TOCO signal (numpy array)
        baseline: Smooth baseline from toco_baseline_v2
        quality_mask: Optional quality mask (0=valid, 1=invalid, 2=suspicious)
        config: Detection configuration. If None, uses defaults.
        
    Returns:
        UcDetectionResultV2 with detected contractions and diagnostics
    """
    if config is None:
        config = UcDetectionConfigV2()
    
    # Convert to float
    toco = np.asarray(toco, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    n = len(toco)
    
    # Handle short signals
    min_samples = int(60 * config.sample_rate)  # At least 1 minute
    if n < min_samples:
        return UcDetectionResultV2(contractions=[], n_contractions=0)
    
    # Ensure baseline matches toco length
    if len(baseline) != n:
        baseline = np.interp(np.arange(n), 
                            np.linspace(0, n-1, len(baseline)), 
                            baseline)
    
    # Create quality mask if not provided
    if quality_mask is None:
        quality_mask = np.zeros(n, dtype=np.int8)
    
    # Compute adaptive threshold using MAD
    adaptive_threshold = compute_adaptive_threshold(toco, baseline, config)
    
    # Compute signal above baseline
    above_baseline = toco - baseline
    
    # Find peaks using scipy (with smaller initial distance for merging later)
    min_distance = int(config.min_peak_distance_sec * config.sample_rate)
    
    peaks, properties = find_peaks(
        above_baseline,
        distance=min_distance,
        prominence=config.min_prominence,
        height=adaptive_threshold,
    )
    
    # Merge close peaks - keep only the highest in each group
    peaks = merge_close_peaks(peaks, above_baseline, config)
    
    # Process each peak
    contractions = []
    
    for peak_idx in peaks:
        # Skip if in definitely invalid region (quality_mask=1)
        # Note: quality_mask=2 means "suspicious" (e.g., plateau at high value during contraction)
        # which is often still valid for detection, so we don't skip those
        if config.use_quality_mask and quality_mask[peak_idx] == 1:
            continue
        
        peak_value = toco[peak_idx]
        baseline_value = baseline[peak_idx]
        strength = peak_value - baseline_value
        
        # Check strength threshold (must be >= A_min)
        if strength < config.min_strength:
            continue
        
        # Find boundaries using improved method
        start_idx, end_idx = find_contraction_boundaries(
            toco, baseline, peak_idx, config
        )
        
        # Calculate duration
        duration_samples = end_idx - start_idx
        duration_sec = duration_samples / config.sample_rate
        
        # Validate duration (30-120 seconds)
        if duration_sec < config.min_duration_sec:
            continue
        if duration_sec > config.max_duration_sec:
            continue
        
        # Calculate rise time and fall time
        rise_time_samples = peak_idx - start_idx
        fall_time_samples = end_idx - peak_idx
        rise_time_sec = rise_time_samples / config.sample_rate
        fall_time_sec = fall_time_samples / config.sample_rate
        
        # Validate rise time (should be reasonable)
        if rise_time_sec < config.min_rise_time_sec:
            continue
        if rise_time_sec > config.max_rise_time_sec:
            continue
        
        # Check peak position (should not be at edges)
        # Peak should be within 20%-80% of contraction duration
        peak_position_ratio = rise_time_samples / duration_samples if duration_samples > 0 else 0.5
        if peak_position_ratio < (1 - config.max_peak_asymmetry) or peak_position_ratio > config.max_peak_asymmetry:
            continue
        
        # Check quality within contraction
        # Only count quality_mask==1 as truly invalid; quality_mask==2 is "suspicious" but often valid
        if config.use_quality_mask:
            contraction_quality = quality_mask[start_idx:end_idx]
            invalid_pct = np.mean(contraction_quality == 1)  # Only count definitely invalid
            if invalid_pct > config.max_invalid_pct:
                continue
        
        # Compute area under contraction
        area = compute_contraction_area(toco, baseline, start_idx, end_idx)
        
        # Create contraction model with all fields
        uc = UcModelV2(
            peak_index=int(peak_idx),
            start_index=int(start_idx),
            end_index=int(end_idx),
            strength=float(strength),
            duration_sec=float(duration_sec),
            peak_value=float(peak_value),
            baseline_value=float(baseline_value),
            rise_time_sec=float(rise_time_sec),
            fall_time_sec=float(fall_time_sec),
            area=float(area),
        )
        contractions.append(uc)
    
    # Filter by minimum interval - ensure ≥60 sec between separate contractions
    contractions = filter_by_minimum_interval(contractions, config)
    
    # Calculate statistics
    n_contractions = len(contractions)
    mean_strength = 0.0
    mean_duration = 0.0
    
    if n_contractions > 0:
        mean_strength = np.mean([uc.strength for uc in contractions])
        mean_duration = np.mean([uc.duration_sec for uc in contractions])
    
    return UcDetectionResultV2(
        contractions=contractions,
        n_contractions=n_contractions,
        mean_strength=mean_strength,
        mean_duration_sec=mean_duration,
        adaptive_threshold=adaptive_threshold,
    )


def detect_uc_simple(toco: np.ndarray,
                     baseline: np.ndarray,
                     sample_rate: int = 4) -> List[UcModelV2]:
    """
    Simple wrapper for UC detection.
    
    Args:
        toco: Denoised TOCO signal
        baseline: Baseline signal
        sample_rate: Sample rate in Hz (default 4)
        
    Returns:
        List of detected UcModelV2 objects
    """
    config = UcDetectionConfigV2(sample_rate=sample_rate)
    result = detect_uc_v2(toco, baseline, config=config)
    return result.contractions


def contractions_to_binary(contractions: List[UcModelV2],
                           signal_length: int) -> np.ndarray:
    """
    Convert list of contractions to a binary signal.
    
    Creates a binary array where 1 indicates a contraction is occurring
    and 0 indicates no contraction.
    
    Args:
        contractions: List of detected UcModelV2 objects
        signal_length: Length of the output binary signal
        
    Returns:
        Binary numpy array of shape (signal_length,) with dtype np.int8
        where 1 = contraction, 0 = no contraction
    """
    binary = np.zeros(signal_length, dtype=np.int8)
    
    for uc in contractions:
        start_idx = max(0, uc.start_index)
        end_idx = min(signal_length, uc.end_index + 1)
        binary[start_idx:end_idx] = 1
    
    return binary


