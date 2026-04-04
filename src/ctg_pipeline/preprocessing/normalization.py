"""
Normalization Module for CTG Signals

Provides various normalization methods for FHR signals and derived features
(baseline, STV) for deep learning applications.

Key Features:
- Multiple normalization strategies
- Channel-specific normalization
- Support for raw FHR (0-255 range) and clean FHR (50-220 bpm range)
- Support for inverse transformation
- Handles NaN values gracefully

Usage:
    from ctg_analysis.preprocessing.normalization import (
        normalize_raw_fhr,
        normalize_fhr,
        normalize_baseline,
        normalize_stv,
        create_normalized_multichannel
    )
"""

import numpy as np
from typing import Optional, Tuple, Dict, Union, List
from dataclasses import dataclass
from enum import Enum


class NormalizationMethod(Enum):
    """Normalization methods for CTG signals."""
    STANDARDIZE = "standardize"           # (x - mean) / std -> ~[-3, 3]
    MINMAX = "minmax"                     # (x - min) / (max - min) -> [0, 1]
    ROBUST = "robust"                     # (x - median) / IQR
    PHYSIOLOGICAL = "physiological"       # (x - 50) / 170 -> [0, 1] for FHR
    PHYSIOLOGICAL_CENTERED = "phys_centered"  # (x - 135) / 85 -> [-1, 1] for FHR
    RAW_FHR = "raw_fhr"                   # (x - 0) / 255 -> [0, 1] for raw FHR (0-255)
    RAW_FHR_CENTERED = "raw_fhr_centered" # (x - 127.5) / 127.5 -> [-1, 1] for raw FHR
    PERCENTILE = "percentile"             # Map 5th-95th percentile to [0, 1]
    LOG_STANDARDIZE = "log_standardize"   # log(x + 1) then standardize
    VARIABILITY_PHYSIOLOGICAL = "var_physiological"  # Map variability (0-25 bpm) to [0, 1]
    VARIABILITY_PHYSIOLOGICAL_CENTERED = "var_phys_centered"  # Map variability (0-25 bpm) to [-1, 1]
    DEVIATION_PHYSIOLOGICAL = "dev_physiological"    # Map deviation (-45 to +45 bpm) to [0, 1]
    DEVIATION_PHYSIOLOGICAL_CENTERED = "dev_phys_centered"  # Map deviation (-45 to +45 bpm) to [-1, 1]
    # TOCO normalization methods (normalize toco - baseline)
    TOCO_PHYSIOLOGICAL = "toco_physiological"         # Map (toco - baseline) 0-100 to [0, 1]
    TOCO_PHYSIOLOGICAL_CENTERED = "toco_phys_centered"  # Map (toco - baseline) 0-100 to [-1, 1], centered at 50


@dataclass
class NormalizationParams:
    """Parameters for normalization (for inverse transform)."""
    method: NormalizationMethod
    offset: float = 0.0
    scale: float = 1.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            'method': self.method.value,
            'offset': self.offset,
            'scale': self.scale,
            'clip_min': self.clip_min,
            'clip_max': self.clip_max
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'NormalizationParams':
        return cls(
            method=NormalizationMethod(d['method']),
            offset=d.get('offset', 0.0),
            scale=d.get('scale', 1.0),
            clip_min=d.get('clip_min'),
            clip_max=d.get('clip_max')
        )


# =============================================================================
# Core Normalization Functions
# =============================================================================
def normalize_signal(
    signal: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.STANDARDIZE,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, NormalizationParams]]:
    """
    Normalize a signal using the specified method.
    
    Args:
        signal: Input signal (can contain NaN)
        method: Normalization method to use
        return_params: If True, return parameters for inverse transform
        
    Returns:
        Normalized signal, optionally with NormalizationParams
    """
    result = signal.copy().astype(np.float32)
    valid_mask = ~np.isnan(result)
    valid_values = result[valid_mask]
    
    if len(valid_values) == 0:
        params = NormalizationParams(method=method)
        if return_params:
            return np.zeros_like(result), params
        return np.zeros_like(result)
    
    if method == NormalizationMethod.PHYSIOLOGICAL:
        # Map FHR range (50-220 bpm) to [0, 1]
        offset, scale = 50.0, 170.0
        result[valid_mask] = (valid_values - offset) / scale
        params = NormalizationParams(method, offset, scale, 0.0, 1.0)
        
    elif method == NormalizationMethod.PHYSIOLOGICAL_CENTERED:
        # Map FHR range (50-220 bpm) to [-1, 1]
        offset, scale = 135.0, 85.0
        result[valid_mask] = (valid_values - offset) / scale
        params = NormalizationParams(method, offset, scale, -1.0, 1.0)
    
    elif method == NormalizationMethod.RAW_FHR:
        # Map raw FHR range (0-255) to [0, 1]
        offset, scale = 0.0, 255.0
        result[valid_mask] = (valid_values - offset) / scale
        params = NormalizationParams(method, offset, scale, 0.0, 1.0)
    
    elif method == NormalizationMethod.RAW_FHR_CENTERED:
        # Map raw FHR range (0-255) to [-1, 1]
        offset, scale = 127.5, 127.5
        result[valid_mask] = (valid_values - offset) / scale
        params = NormalizationParams(method, offset, scale, -1.0, 1.0)
        
    elif method == NormalizationMethod.VARIABILITY_PHYSIOLOGICAL:
        # Map STV range (0-25 bpm) to [0, 1]
        # Based on variability.py clinical thresholds:
        #   < 0.08 (2 bpm): Absent variability
        #   0.08-0.24 (2-6 bpm): Reduced variability
        #   0.24-1.0 (6-25 bpm): Normal variability
        #   > 1.0 (>25 bpm): Marked/Saltatory (clipped)
        offset, scale = 0.0, 25.0
        result[valid_mask] = (valid_values - offset) / scale
        result = np.clip(result, 0.0, 1.0)
        params = NormalizationParams(method, offset, scale, 0.0, 1.0)
    
    elif method == NormalizationMethod.VARIABILITY_PHYSIOLOGICAL_CENTERED:
        # Map STV range (0-25 bpm) to [-1, 1]
        # Center at 12.5 bpm (middle of range)
        # Clinical thresholds after normalization:
        #   < -0.84 (2 bpm): Absent variability
        #   -0.84 to -0.52 (2-6 bpm): Reduced variability
        #   -0.52 to +1.0 (6-25 bpm): Normal variability
        #   > +1.0 (>25 bpm): Marked/Saltatory (clipped)
        offset, scale = 12.5, 12.5
        result[valid_mask] = (valid_values - offset) / scale
        result = np.clip(result, -1.0, 1.0)
        params = NormalizationParams(method, offset, scale, -1.0, 1.0)
        
    elif method == NormalizationMethod.DEVIATION_PHYSIOLOGICAL:
        # Map deviation (FHR - baseline) to [0, 1]
        # Deviation range: approximately -45 to +45 bpm
        offset, scale = -45.0, 90.0
        result[valid_mask] = (valid_values - offset) / scale
        result = np.clip(result, 0.0, 1.0)
        params = NormalizationParams(method, offset, scale, 0.0, 1.0)
    
    elif method == NormalizationMethod.DEVIATION_PHYSIOLOGICAL_CENTERED:
        # Map deviation (FHR - baseline) to [-1, 1]
        # Center at 0 bpm (on baseline), scale by 45 bpm
        offset, scale = 0.0, 45.0
        result[valid_mask] = (valid_values - offset) / scale
        result = np.clip(result, -1.0, 1.0)
        params = NormalizationParams(method, offset, scale, -1.0, 1.0)
        
    elif method == NormalizationMethod.STANDARDIZE:
        offset = float(np.mean(valid_values))
        scale = float(np.std(valid_values))
        if scale < 1e-6:
            scale = 1.0
        result[valid_mask] = (valid_values - offset) / scale
        params = NormalizationParams(method, offset, scale)
        
    elif method == NormalizationMethod.MINMAX:
        min_val = float(np.min(valid_values))
        max_val = float(np.max(valid_values))
        scale = max_val - min_val
        if scale < 1e-6:
            scale = 1.0
        result[valid_mask] = (valid_values - min_val) / scale
        params = NormalizationParams(method, min_val, scale, 0.0, 1.0)
        
    elif method == NormalizationMethod.ROBUST:
        offset = float(np.median(valid_values))
        q75, q25 = np.percentile(valid_values, [75, 25])
        scale = float(q75 - q25)
        if scale < 1e-6:
            scale = 1.0
        result[valid_mask] = (valid_values - offset) / scale
        params = NormalizationParams(method, offset, scale)
        
    elif method == NormalizationMethod.PERCENTILE:
        p5 = float(np.percentile(valid_values, 5))
        p95 = float(np.percentile(valid_values, 95))
        scale = p95 - p5
        if scale < 1e-6:
            scale = 1.0
        result[valid_mask] = (valid_values - p5) / scale
        result = np.clip(result, 0.0, 1.0)
        params = NormalizationParams(method, p5, scale, 0.0, 1.0)
        
    elif method == NormalizationMethod.LOG_STANDARDIZE:
        # Log transform first (for right-skewed data like STV/LTV)
        log_values = np.log1p(np.maximum(valid_values, 0))
        offset = float(np.mean(log_values))
        scale = float(np.std(log_values))
        if scale < 1e-6:
            scale = 1.0
        result[valid_mask] = (log_values - offset) / scale
        params = NormalizationParams(method, offset, scale)
    
    elif method == NormalizationMethod.TOCO_PHYSIOLOGICAL:
        # Map (toco - baseline) range (0-100) to [0, 1]
        # Input should be above-baseline signal
        # 0 = at baseline, 60 = 0.6, 100 = 1.0
        offset, scale = 0.0, 100.0
        result[valid_mask] = (valid_values - offset) / scale
        result = np.clip(result, -0.1, 1.5)  # Allow slight below baseline and strong contractions
        params = NormalizationParams(method, offset, scale, -0.1, 1.5)
    
    elif method == NormalizationMethod.TOCO_PHYSIOLOGICAL_CENTERED:
        # Map (toco - baseline) range (0-100) to [-1, 1], centered at 50
        # Input should be above-baseline signal
        # 0 = -1, 50 = 0, 100 = 1
        offset, scale = 50.0, 50.0
        result[valid_mask] = (valid_values - offset) / scale
        result = np.clip(result, -1.2, 1.5)  # Allow some overflow
        params = NormalizationParams(method, offset, scale, -1.2, 1.5)
    
    else:
        params = NormalizationParams(method)
    
    if return_params:
        return result, params
    return result


def denormalize_signal(
    normalized: np.ndarray,
    params: NormalizationParams
) -> np.ndarray:
    """
    Inverse normalization to recover original scale.
    
    Args:
        normalized: Normalized signal
        params: Parameters from normalize_signal(..., return_params=True)
        
    Returns:
        Signal in original scale
    """
    result = normalized.copy()
    valid_mask = ~np.isnan(result)
    
    if params.method == NormalizationMethod.LOG_STANDARDIZE:
        # Inverse: exp(x * scale + offset) - 1
        result[valid_mask] = np.expm1(result[valid_mask] * params.scale + params.offset)
    else:
        # Standard inverse: x * scale + offset
        result[valid_mask] = result[valid_mask] * params.scale + params.offset
    
    return result


# =============================================================================
# Channel-Specific Normalization
# =============================================================================

def normalize_fhr(
    fhr: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.PHYSIOLOGICAL_CENTERED,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, NormalizationParams]]:
    """
    Normalize clean FHR signal (50-220 bpm range).
    
    Args:
        fhr: Clean FHR signal in bpm (50-220 range)
        method: Normalization method (default: physiological centered)
        return_params: If True, return parameters for inverse transform
        
    Returns:
        Normalized FHR, optionally with parameters
    """
    return normalize_signal(fhr, method, return_params)


def normalize_raw_fhr(
    raw_fhr: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.RAW_FHR_CENTERED,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, NormalizationParams]]:
    """
    Normalize raw FHR signal (0-255 range).
    
    Args:
        raw_fhr: Raw FHR signal (0-255 range)
        method: Normalization method (default: RAW_FHR_CENTERED)
        return_params: If True, return parameters for inverse transform
        
    Returns:
        Normalized raw FHR, optionally with parameters
    """
    return normalize_signal(raw_fhr, method, return_params)


def normalize_baseline(
    baseline: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.PHYSIOLOGICAL_CENTERED,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, NormalizationParams]]:
    """
    Normalize baseline signal.
    
    Args:
        baseline: Baseline signal in bpm
        method: Normalization method (default: same as FHR)
        return_params: If True, return parameters for inverse transform
        
    Returns:
        Normalized baseline, optionally with parameters
    """
    return normalize_signal(baseline, method, return_params)


def normalize_deviation(
    deviation: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.DEVIATION_PHYSIOLOGICAL_CENTERED,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, NormalizationParams]]:
    """
    Normalize deviation (FHR - baseline) signal.
    
    Range: -45 to +45 bpm -> [-1, 1]
    
    Args:
        deviation: Deviation signal (FHR - baseline)
        method: Normalization method (default: DEVIATION_PHYSIOLOGICAL_CENTERED)
        return_params: If True, return parameters for inverse transform
        
    Returns:
        Normalized deviation, optionally with parameters
    """
    return normalize_signal(deviation, method, return_params)


def normalize_stv(
    stv: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.VARIABILITY_PHYSIOLOGICAL,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, NormalizationParams]]:
    """
    Normalize Short-Term Variability (STV).
    
    STV is computed as P95-P5 amplitude in bpm (from variability.py).
    Range: 0-25 bpm -> [0, 1]
    
    Clinical thresholds (aligned with variability.py):
    - < 2 bpm: Absent (< 0.08 after normalization)
    - 2-6 bpm: Reduced (0.08-0.24)
    - 6-25 bpm: Normal (0.24-1.0)
    - > 25 bpm: Marked (clipped to 1.0)
    
    Args:
        stv: STV signal in bpm
        method: Normalization method (default: VARIABILITY_PHYSIOLOGICAL)
        return_params: If True, return parameters for inverse transform
        
    Returns:
        Normalized STV, optionally with parameters
    """
    return normalize_signal(stv, method, return_params)


# =============================================================================
# TOCO Normalization
# =============================================================================

def normalize_toco(
    toco: np.ndarray,
    baseline: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.TOCO_PHYSIOLOGICAL,
    smooth_output: bool = True,
    smooth_window: int = 10,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, NormalizationParams]]:
    """
    Normalize TOCO signal relative to baseline.
    
    This function computes (toco - baseline) first, then normalizes:
    - TOCO_PHYSIOLOGICAL: Maps 0-100 above baseline to [0, 1]
    - TOCO_PHYSIOLOGICAL_CENTERED: Maps 0-100 above baseline to [-1, 1]
    
    Output interpretation (TOCO_PHYSIOLOGICAL, default):
    - 0: At baseline (no contraction)
    - 0.5: Moderate contraction (50 units above baseline)
    - 1.0: Strong contraction (100 units above baseline)
    
    Output interpretation (TOCO_PHYSIOLOGICAL_CENTERED):
    - -1: At baseline (no contraction)
    - 0: Moderate contraction (50 units above baseline)
    - 1: Strong contraction (100 units above baseline)
    
    Args:
        toco: TOCO signal (denoised)
        baseline: Pre-computed TOCO baseline (e.g., from toco_baseline_v2)
        method: Normalization method (default: TOCO_PHYSIOLOGICAL)
        smooth_output: Whether to apply smoothing to reduce noise (default True)
        smooth_window: Smoothing window size in samples (default 10, ~2.5s at 4Hz)
        return_params: If True, return parameters for inverse transform
        
    Returns:
        Normalized TOCO, optionally with parameters
    """
    # Compute deviation from baseline
    above_baseline = (toco - baseline).astype(np.float32)
    
    # Optionally smooth to reduce noise
    if smooth_output and smooth_window > 1:
        from scipy.ndimage import uniform_filter1d
        above_baseline = uniform_filter1d(above_baseline, size=smooth_window, mode='nearest')
    
    return normalize_signal(above_baseline, method, return_params)


# =============================================================================
# Multi-Channel Normalization
# =============================================================================

@dataclass
class MultiChannelNormConfig:
    """Configuration for multi-channel normalization.
    
    Default: All channels use centered normalization to [-1, 1].
    """
    raw_fhr_method: NormalizationMethod = NormalizationMethod.PHYSIOLOGICAL_CENTERED
    clean_fhr_method: NormalizationMethod = NormalizationMethod.PHYSIOLOGICAL_CENTERED
    baseline_method: NormalizationMethod = NormalizationMethod.PHYSIOLOGICAL_CENTERED
    deviation_method: NormalizationMethod = NormalizationMethod.DEVIATION_PHYSIOLOGICAL_CENTERED
    stv_method: NormalizationMethod = NormalizationMethod.VARIABILITY_PHYSIOLOGICAL_CENTERED
    toco_method: NormalizationMethod = NormalizationMethod.TOCO_PHYSIOLOGICAL_CENTERED
    toco_smooth_output: bool = True
    toco_smooth_window: int = 10
    nan_fill_value: float = 0.0


def create_normalized_multichannel(
    raw_fhr: Optional[np.ndarray] = None,
    clean_fhr: Optional[np.ndarray] = None,
    baseline: Optional[np.ndarray] = None,
    deviation: Optional[np.ndarray] = None,
    stv: Optional[np.ndarray] = None,
    toco: Optional[np.ndarray] = None,
    toco_baseline: Optional[np.ndarray] = None,
    config: Optional[MultiChannelNormConfig] = None,
    return_params: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, NormalizationParams]]]:
    """
    Create a normalized multi-channel input tensor for deep learning.
    
    All channels are optional. At least one channel must be provided.
    All channels are normalized to [-1, 1] by default (centered).
    
    Channels (in order, if provided):
    1. Raw FHR (normalized to [-1, 1], optional)
    2. Clean FHR (normalized to [-1, 1], optional)
    3. Baseline (normalized to [-1, 1], optional)
    4. Deviation (normalized to [-1, 1], optional)
    5. STV (normalized to [-1, 1], optional)
    6. TOCO (normalized to [-1, 1], optional, requires toco_baseline)
    
    Args:
        raw_fhr: Raw FHR signal (0-255 range) (optional)
        clean_fhr: Clean FHR signal in bpm (optional)
        baseline: Baseline signal in bpm (optional)
        deviation: Deviation signal (FHR - baseline) in bpm (optional)
        stv: Short-term variability (optional)
        toco: TOCO signal (optional, requires toco_baseline)
        toco_baseline: TOCO baseline signal (required if toco is provided)
        config: Normalization configuration
        return_params: If True, return normalization parameters
        
    Returns:
        Multi-channel array of shape (length, num_channels)
        Optionally with dict of NormalizationParams per channel
        
    Raises:
        ValueError: If no channels are provided or if toco is provided without toco_baseline
    """
    if config is None:
        config = MultiChannelNormConfig()
    
    channels = []
    params = {}
    
    # Channel 1: Normalized Raw FHR (optional)
    if raw_fhr is not None:
        raw_fhr_norm, raw_fhr_params = normalize_raw_fhr(raw_fhr, config.raw_fhr_method, return_params=True)
        channels.append(raw_fhr_norm)
        params['raw_fhr'] = raw_fhr_params
    
    # Channel 2: Normalized Clean FHR (optional)
    if clean_fhr is not None:
        clean_fhr_norm, clean_fhr_params = normalize_fhr(clean_fhr, config.clean_fhr_method, return_params=True)
        channels.append(clean_fhr_norm)
        params['clean_fhr'] = clean_fhr_params
    
    # Channel 3: Normalized Baseline (optional)
    if baseline is not None:
        baseline_norm, baseline_params = normalize_baseline(baseline, config.baseline_method, return_params=True)
        channels.append(baseline_norm)
        params['baseline'] = baseline_params
    
    # Channel 4: Normalized Deviation (optional)
    if deviation is not None:
        deviation_norm, deviation_params = normalize_deviation(deviation, config.deviation_method, return_params=True)
        channels.append(deviation_norm)
        params['deviation'] = deviation_params
    
    # Channel 5: Normalized STV (optional)
    if stv is not None:
        stv_norm, stv_params = normalize_stv(stv, config.stv_method, return_params=True)
        channels.append(stv_norm)
        params['stv'] = stv_params
    
    # Channel 6: Normalized TOCO (optional, requires toco_baseline)
    if toco is not None:
        if toco_baseline is None:
            raise ValueError("toco_baseline is required when toco is provided")
        toco_norm, toco_params = normalize_toco(
            toco, toco_baseline, 
            method=config.toco_method,
            smooth_output=config.toco_smooth_output,
            smooth_window=config.toco_smooth_window,
            return_params=True
        )
        channels.append(toco_norm)
        params['toco'] = toco_params

    # Validate at least one channel is provided
    if len(channels) == 0:
        raise ValueError("At least one channel must be provided")
    
    # Stack channels: shape (length, num_channels)
    if len(channels) > 1:
        result = np.stack(channels, axis=-1).astype(np.float32)
    else:
        result = channels[0][:, np.newaxis].astype(np.float32)
    
    # Fill NaN values
    result = np.nan_to_num(result, nan=config.nan_fill_value)
    
    if return_params:
        return result, params
    return result


def get_channel_names(
    include_raw_fhr: bool = True,
    include_clean_fhr: bool = False,
    include_baseline: bool = False,
    include_deviation: bool = False,
    include_stv: bool = False,
    include_toco: bool = False
) -> List[str]:
    """
    Get ordered list of channel names.
    
    Args:
        include_raw_fhr: Whether raw FHR channel is included (default True)
        include_clean_fhr: Whether clean FHR channel is included
        include_baseline: Whether baseline channel is included
        include_deviation: Whether deviation channel is included
        include_stv: Whether STV channel is included
        include_toco: Whether TOCO channel is included
        
    Returns:
        List of channel names in order
    """
    names = []
    if include_raw_fhr:
        names.append('raw_fhr')
    if include_clean_fhr:
        names.append('clean_fhr')
    if include_baseline:
        names.append('baseline')
    if include_deviation:
        names.append('deviation')
    if include_stv:
        names.append('stv')
    if include_toco:
        names.append('toco')
    return names


# =============================================================================
# Summary Table
# =============================================================================

NORMALIZATION_RECOMMENDATIONS = """
╔═══════════════╦════════════════════════════════════╦════════════════╦═══════════════════════════════════╗
║ Channel       ║ Default Method                     ║ Output Range   ║ Reason                            ║
╠═══════════════╬════════════════════════════════════╬════════════════╬═══════════════════════════════════╣
║ Raw FHR       ║ PHYSIOLOGICAL_CENTERED             ║ [-1, 1]        ║ Centered at 135 bpm               ║
║ FHR (clean)   ║ PHYSIOLOGICAL_CENTERED             ║ [-1, 1]        ║ Centered at 135 bpm               ║
║ Baseline      ║ PHYSIOLOGICAL_CENTERED             ║ [-1, 1]        ║ Centered at 135 bpm               ║
║ Deviation     ║ DEVIATION_PHYSIOLOGICAL_CENTERED   ║ [-1, 1]        ║ Centered at 0 (on baseline)       ║
║ STV           ║ VARIABILITY_PHYSIOLOGICAL_CENTERED ║ [-1, 1]        ║ Centered at 12.5 bpm              ║
║ TOCO          ║ TOCO_PHYSIOLOGICAL                 ║ [0, 1]         ║ Fixed scale, baseline-relative    ║
║ TOCO (alt)    ║ TOCO_PHYSIOLOGICAL_CENTERED        ║ [-1, 1]        ║ Centered at 50 above baseline     ║
╚═══════════════╩════════════════════════════════════╩════════════════╩═══════════════════════════════════╝

PHYSIOLOGICAL_CENTERED (FHR, Baseline): Maps 50-220 bpm → [-1, 1]
  - 50 bpm → -1, 135 bpm → 0, 220 bpm → +1

DEVIATION_PHYSIOLOGICAL_CENTERED: Maps -45 to +45 bpm → [-1, 1]
  - 0: On baseline (normal)
  - +0.33 to +0.56: Acceleration (+15 to +25 bpm)
  - -0.33: Mild deceleration (-15 bpm)
  - -1.0: Severe deceleration (-45 bpm)

VARIABILITY_PHYSIOLOGICAL_CENTERED (STV): Maps 0-25 bpm → [-1, 1]
  - 0 bpm → -1, 12.5 bpm → 0, 25 bpm → +1
  - Clinical thresholds (aligned with variability.py):
    < -0.84 (2 bpm): Absent variability
    -0.84 to -0.52 (2-6 bpm): Reduced variability
    -0.52 to +1.0 (6-25 bpm): Normal variability

Alternative methods (for [0, 1] output):
- RAW_FHR: Maps 0-255 → [0, 1]
- PHYSIOLOGICAL: Maps 50-220 bpm → [0, 1]
- DEVIATION_PHYSIOLOGICAL: Maps -45 to +45 bpm → [0, 1] (0.5 = on baseline)
- VARIABILITY_PHYSIOLOGICAL: Maps 0-25 bpm → [0, 1]

TOCO Normalization Methods (both require baseline parameter):
- TOCO_PHYSIOLOGICAL (default): Maps (toco - baseline) 0-100 → [0, 1]
  - Fixed physiological scale for consistent interpretation
  - 0 = at baseline, 0.6 = 60 units above baseline (typical contraction)
  - Clips to [-0.1, 1.5] to allow some below-baseline and strong contractions
- TOCO_PHYSIOLOGICAL_CENTERED: Maps (toco - baseline) 0-100 → [-1, 1], centered at 50
  - -1 = at baseline, 0 = 50 above baseline, 1 = 100 above baseline
  - Clips to [-1.2, 1.5]

Usage:
  normalize_toco(toco, baseline, method=NormalizationMethod.TOCO_PHYSIOLOGICAL)
  - baseline parameter is required (use estimate_toco_baseline() or similar)
  - Optional smoothing: smooth_output=True, smooth_window=10 (samples at 4Hz = 2.5s)
"""


if __name__ == "__main__":
    print("Normalization Recommendations:")
    print(NORMALIZATION_RECOMMENDATIONS)
    
    # Demo
    np.random.seed(42)
    n = 1000
    
    # Simulate signals
    raw_fhr = np.clip(140 + 15 * np.random.randn(n), 0, 255)
    clean_fhr = np.clip(140 + 10 * np.random.randn(n), 50, 220)
    baseline = np.clip(140 + 5 * np.random.randn(n), 50, 220)
    deviation = clean_fhr - baseline
    stv = np.abs(8 + 4 * np.random.randn(n))  # P95-P5 amplitude in bpm
    toco = np.clip(20 + 30 * np.abs(np.sin(np.linspace(0, 4*np.pi, n))), 0, 100)
    toco_baseline = np.full(n, 20.0)
    
    # Create normalized multi-channel input (with TOCO)
    multichannel, params = create_normalized_multichannel(
        raw_fhr=raw_fhr,
        clean_fhr=clean_fhr,
        baseline=baseline,
        deviation=deviation,
        stv=stv,
        toco=toco,
        toco_baseline=toco_baseline,
        return_params=True
    )
    
    channel_names = get_channel_names(
        include_raw_fhr=True,
        include_clean_fhr=True,
        include_baseline=True,
        include_deviation=True,
        include_stv=True,
        include_toco=True
    )
    
    print(f"\nMulti-channel shape: {multichannel.shape}")
    print(f"Channels: {channel_names}")
    
    for ch_name, ch_params in params.items():
        print(f"  {ch_name}: method={ch_params.method.value}, "
              f"offset={ch_params.offset:.2f}, scale={ch_params.scale:.2f}")
    
    print(f"\nChannel statistics after normalization (all [-1, 1]):")
    for i, name in enumerate(channel_names):
        ch = multichannel[:, i]
        print(f"  {name}: min={ch.min():.3f}, max={ch.max():.3f}, "
              f"mean={ch.mean():.3f}, std={ch.std():.3f}")