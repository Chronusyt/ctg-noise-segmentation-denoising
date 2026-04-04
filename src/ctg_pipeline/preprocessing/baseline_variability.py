"""
CTG Variability (Amplitude) Module for FHR Signals

This module computes *baseline variability amplitude* from FHR signals.
Output array has the same length as input signal.

Key change vs your previous code:
- We compute variability as a *robust amplitude* within each 1-minute window:
    variability = P95(window) - P5(window)   (unit: bpm)
  This aligns with clinical variability categories (minimal ≤5, moderate 6–25, marked >25 bpm)
  much better than using rolling standard deviation.

We still exclude:
- Accelerations (ACC)
- Decelerations (DEC)
- Invalid/artifact regions

Notes:
- If baseline is provided and use_baseline_deviation=True, we compute variability on (FHR - baseline).
  This does NOT change units; it's still bpm. Using deviation can help remove slow drift.
- Unlike the previous "fast std" approach, we DO NOT interpolate invalid points *before* computing
  amplitude, because interpolation can artificially shrink amplitude.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaselineVariabilityConfig:
    """
    Configuration for variability amplitude computation.

    Attributes:
        window_sec: Window size in seconds (default: 60 = 1 min)
        sampling_rate: Sampling rate in Hz (default: 4 Hz)
        min_valid_ratio: Minimum valid samples required in window (default: 0.5)
        use_baseline_deviation: If True, compute on (FHR - baseline) instead of raw FHR
        smooth_output: Apply smoothing to reduce noise in output
        smooth_window_sec: Smoothing window size in seconds (default: 15 sec)
        interpolate_invalid: If True, interpolate NaN regions in the final output (default: True)
        amp_low_pct: lower percentile for robust amplitude (default: 5)
        amp_high_pct: upper percentile for robust amplitude (default: 95)
    """
    window_sec: float = 60.0
    sampling_rate: float = 4.0
    min_valid_ratio: float = 0.5
    use_baseline_deviation: bool = True
    smooth_output: bool = True
    smooth_window_sec: float = 15.0
    interpolate_invalid: bool = True

    amp_low_pct: float = 5.0
    amp_high_pct: float = 95.0

    @property
    def window_samples(self) -> int:
        return int(self.window_sec * self.sampling_rate)

    @property
    def smooth_window_samples(self) -> int:
        return int(self.smooth_window_sec * self.sampling_rate)


def _compute_rolling_amp_percentile(
    signal: np.ndarray,
    window_samples: int,
    valid_mask: np.ndarray,
    min_valid_ratio: float = 0.5,
    low_pct: float = 5.0,
    high_pct: float = 95.0
) -> np.ndarray:
    """
    Compute rolling robust amplitude: P(high_pct) - P(low_pct) within each centered window.

    Args:
        signal: 1D array in bpm (raw FHR or FHR-baseline deviation)
        window_samples: window size in samples (e.g., 60s * fs)
        valid_mask: boolean mask (True=valid sample)
        min_valid_ratio: require at least this fraction of valid samples in the window
        low_pct/high_pct: percentiles for robust amplitude

    Returns:
        amp array, same length as input, unit: bpm
    """
    n = len(signal)
    out = np.full(n, np.nan, dtype=np.float64)
    half = window_samples // 2
    min_valid = int(np.ceil(window_samples * min_valid_ratio))

    is_valid = valid_mask & np.isfinite(signal)

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)

        m = is_valid[start:end]
        if m.sum() < min_valid:
            continue

        w = signal[start:end][m]
        # robust amplitude in bpm
        lo = np.nanpercentile(w, low_pct)
        hi = np.nanpercentile(w, high_pct)
        out[i] = hi - lo

    return out


def compute_baseline_variability(
    fhr: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    acc_mask: Optional[np.ndarray] = None,
    dec_mask: Optional[np.ndarray] = None,
    quality_mask: Optional[np.ndarray] = None,
    config: Optional[BaselineVariabilityConfig] = None
) -> np.ndarray:
    """
    Compute baseline variability amplitude (robust) from FHR signal.

    Returns:
        bv array, same length as input, unit: bpm
        Interpretable with clinical variability categories:
          - absent: <2 bpm
          - minimal/reduced: 2–5 bpm (or <6 bpm)
          - moderate (normal): 6–25 bpm
          - marked: >25 bpm
    """
    if config is None:
        config = BaselineVariabilityConfig()

    fhr = np.asarray(fhr, dtype=np.float64)
    n = len(fhr)

    if baseline is not None and config.use_baseline_deviation:
        baseline = np.asarray(baseline, dtype=np.float64)
        if len(baseline) != n:
            raise ValueError("baseline must have the same length as fhr")
        signal = fhr - baseline  # bpm
    else:
        signal = fhr  # bpm

    # Combined valid mask (True = usable for variability)
    valid = np.ones(n, dtype=bool)

    if acc_mask is not None:
        acc_mask = np.asarray(acc_mask)
        if len(acc_mask) != n:
            raise ValueError("acc_mask must have the same length as fhr")
        valid &= (acc_mask == 0)

    if dec_mask is not None:
        dec_mask = np.asarray(dec_mask)
        if len(dec_mask) != n:
            raise ValueError("dec_mask must have the same length as fhr")
        valid &= (dec_mask == 0)

    if quality_mask is not None:
        quality_mask = np.asarray(quality_mask)
        if len(quality_mask) != n:
            raise ValueError("quality_mask must have the same length as fhr")
        valid &= (quality_mask == 0)

    valid &= np.isfinite(signal)

    # Robust amplitude per centered 1-min window
    bv = _compute_rolling_amp_percentile(
        signal=signal,
        window_samples=config.window_samples,
        valid_mask=valid,
        min_valid_ratio=config.min_valid_ratio,
        low_pct=config.amp_low_pct,
        high_pct=config.amp_high_pct,
    )

    # Optional smoothing - use NaN-aware smoothing to avoid bias from excluded regions
    if config.smooth_output and config.smooth_window_samples > 1:
        bv = _nan_aware_smooth(bv, config.smooth_window_samples)

    # Optional interpolation to fill remaining NaNs (for downstream models)
    if config.interpolate_invalid:
        bv = _interpolate_invalid_regions(bv)

    return bv


def _nan_aware_smooth(x: np.ndarray, window_size: int) -> np.ndarray:
    """
    Smooth array while properly handling NaNs (ignore them, don't treat as 0).
    This prevents excluded regions (ACC/DEC/invalid) from biasing nearby valid values.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    half = window_size // 2

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        window = x[start:end]
        valid_vals = window[np.isfinite(window)]
        if len(valid_vals) > 0:
            out[i] = np.mean(valid_vals)

    return out


def _interpolate_invalid_regions(x: np.ndarray) -> np.ndarray:
    """
    Linear interpolate NaNs in a 1D array.
    """
    x = np.asarray(x, dtype=np.float64)
    out = x.copy()
    valid = np.isfinite(out)

    if not np.any(valid):
        return np.zeros_like(out)

    if np.all(valid):
        return out

    idx = np.arange(len(out))
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


def classify_baseline_variability(bv_value: float) -> str:
    """
    Classify variability amplitude (bpm) using common CTG thresholds.
    """
    if not np.isfinite(bv_value):
        return "unknown"
    if bv_value < 2:
        return "absent"
    elif bv_value < 6:
        return "reduced"   # minimal/reduced
    elif bv_value <= 25:
        return "normal"    # moderate
    else:
        return "marked"    # increased/marked


def get_baseline_variability_statistics(bv: np.ndarray) -> dict:
    bv = np.asarray(bv, dtype=np.float64)
    v = bv[np.isfinite(bv)]
    if len(v) == 0:
        return {
            "bv_mean": np.nan,
            "bv_std": np.nan,
            "bv_min": np.nan,
            "bv_max": np.nan,
            "bv_median": np.nan,
            "bv_classification": "unknown",
            "valid_samples": 0,
            "valid_ratio": 0.0,
        }

    mean_v = float(np.mean(v))
    return {
        "bv_mean": mean_v,
        "bv_std": float(np.std(v)),
        "bv_min": float(np.min(v)),
        "bv_max": float(np.max(v)),
        "bv_median": float(np.median(v)),
        "bv_classification": classify_baseline_variability(mean_v),
        "valid_samples": int(len(v)),
        "valid_ratio": float(len(v) / len(bv)),
    }


def get_baseline_variability_summary(stats: dict) -> str:
    lines = [
        "CTG Variability (Robust Amplitude) Summary",
        "=" * 44,
        f"  Mean:   {stats['bv_mean']:.2f} bpm",
        f"  Std:    {stats['bv_std']:.2f} bpm",
        f"  Range:  {stats['bv_min']:.2f} - {stats['bv_max']:.2f} bpm",
        f"  Median: {stats['bv_median']:.2f} bpm",
        f"  Class:  {stats['bv_classification']}",
        "-" * 44,
        f"  Valid samples: {stats['valid_samples']} ({stats['valid_ratio']*100:.1f}%)",
    ]
    return "\n".join(lines)
