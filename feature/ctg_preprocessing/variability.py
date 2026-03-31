"""
CTG Variability Module — STV (Short-Term Variability) & LTV (Long-Term Variability)

Standard cardiotocographic variability measures based on pulse intervals.

Definitions
-----------
Pulse Interval (PI):
    PI = 60000 / FHR  (ms),  where FHR is in beats per minute (bpm).

Epoch:
    A 3.75-second segment (1/16 of a minute).
    At 4 Hz sampling → 15 samples per epoch.
    At 2 Hz sampling → 7.5 → rounded to 8 samples per epoch.

STV (Short-Term Variability) overall:
    Mean absolute difference of consecutive epoch mean pulse intervals.
        STV = mean( |PI_epoch[i+1] − PI_epoch[i]| )
    Unit: ms

LTV (Long-Term Variability) overall:
    Mean minute-by-minute range of epoch mean pulse intervals.
    For each 1-minute window (16 epochs at 3.75 s):
        LTV_minute = max(PI_epoch) − min(PI_epoch)
    LTV overall = mean(LTV_minute)
    Unit: ms

Reference:
    At FHR = 140 bpm, PI = 428.6 ms.
    A change of 1 bpm → ΔPI ≈ 3.06 ms.
    Hence STV of 3.0 ms ≈ 1.0 bpm at 140 bpm.

Exclusions (optional):
    Accelerations, decelerations, and invalid/artifact regions can be
    excluded from epoch pulse-interval computation via boolean masks.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class STVConfig:
    """
    Configuration for STV (short-term variability) computation.

    Attributes:
        sampling_rate:      Sampling rate in Hz (default 4).
        epoch_sec:          Epoch duration in seconds (default 3.75 = 1/16 min).
        min_valid_ratio:    Minimum fraction of valid samples required per epoch.
        interpolate_output: If True, linearly interpolate NaN gaps in the
                            sample-level output array.
    """
    sampling_rate: float = 4.0
    epoch_sec: float = 3.75
    min_valid_ratio: float = 0.5
    interpolate_output: bool = True

    @property
    def epoch_samples(self) -> int:
        return int(round(self.epoch_sec * self.sampling_rate))


@dataclass
class LTVConfig:
    """
    Configuration for LTV (long-term variability) computation.

    Attributes:
        sampling_rate:          Sampling rate in Hz (default 4).
        epoch_sec:              Epoch duration in seconds (default 3.75).
        minute_sec:             Minute window duration in seconds (default 60).
        min_valid_ratio:        Minimum fraction of valid samples per epoch.
        min_valid_epochs_ratio: Minimum fraction of valid epochs per minute.
        interpolate_output:     If True, linearly interpolate NaN gaps in the
                                sample-level output array.
    """
    sampling_rate: float = 4.0
    epoch_sec: float = 3.75
    minute_sec: float = 60.0
    min_valid_ratio: float = 0.5
    min_valid_epochs_ratio: float = 0.5
    interpolate_output: bool = True

    @property
    def epoch_samples(self) -> int:
        return int(round(self.epoch_sec * self.sampling_rate))

    @property
    def epochs_per_minute(self) -> int:
        return int(round(self.minute_sec / self.epoch_sec))

    @property
    def minute_samples(self) -> int:
        return int(round(self.minute_sec * self.sampling_rate))


# =============================================================================
# Unit conversion helpers
# =============================================================================

def fhr_to_pulse_interval(fhr: np.ndarray) -> np.ndarray:
    """Convert FHR (bpm) → pulse interval (ms):  PI = 60 000 / FHR."""
    fhr = np.asarray(fhr, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = np.where(fhr > 0, 60000.0 / fhr, np.nan)
    return pi


def pulse_interval_to_fhr(pi: np.ndarray) -> np.ndarray:
    """Convert pulse interval (ms) → FHR (bpm):  FHR = 60 000 / PI."""
    pi = np.asarray(pi, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        fhr = np.where(pi > 0, 60000.0 / pi, np.nan)
    return fhr


def ms_to_bpm(ms_value: float, reference_fhr: float = 140.0) -> float:
    """
    Approximate conversion: ms variability → bpm at a reference FHR.

    At reference_fhr, ΔPI ≈ (60 000 / FHR²) · ΔFHR
    ⇒  ΔFHR ≈ ΔPI · FHR² / 60 000
    """
    return ms_value * (reference_fhr ** 2) / 60000.0


def bpm_to_ms(bpm_value: float, reference_fhr: float = 140.0) -> float:
    """
    Approximate conversion: bpm variability → ms at a reference FHR.

    ΔPI ≈ ΔFHR · 60 000 / FHR²
    """
    return bpm_value * 60000.0 / (reference_fhr ** 2)


# =============================================================================
# Internal helpers
# =============================================================================

def _build_valid_mask(
    n: int,
    fhr: np.ndarray,
    acc_mask: Optional[np.ndarray] = None,
    dec_mask: Optional[np.ndarray] = None,
    quality_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return boolean mask where True = sample is usable."""
    valid = np.isfinite(fhr) & (fhr > 0)
    if acc_mask is not None:
        acc_mask = np.asarray(acc_mask)
        if len(acc_mask) != n:
            raise ValueError("acc_mask length must match fhr length")
        valid &= (acc_mask == 0)
    if dec_mask is not None:
        dec_mask = np.asarray(dec_mask)
        if len(dec_mask) != n:
            raise ValueError("dec_mask length must match fhr length")
        valid &= (dec_mask == 0)
    if quality_mask is not None:
        quality_mask = np.asarray(quality_mask)
        if len(quality_mask) != n:
            raise ValueError("quality_mask length must match fhr length")
        valid &= (quality_mask == 0)
    return valid


def _compute_epoch_pulse_intervals(
    fhr: np.ndarray,
    valid_mask: np.ndarray,
    epoch_samples: int,
    min_valid_ratio: float,
) -> np.ndarray:
    """
    Compute mean pulse interval (ms) per epoch.

    Args:
        fhr:             FHR signal in bpm (1-D).
        valid_mask:      Boolean mask (True = valid sample).
        epoch_samples:   Number of samples per epoch.
        min_valid_ratio: Minimum fraction of valid samples in an epoch.

    Returns:
        1-D array of length n_epochs.  NaN for epochs with too few valid
        samples or non-positive mean FHR.
    """
    n = len(fhr)
    n_epochs = n // epoch_samples
    epoch_pi = np.full(n_epochs, np.nan, dtype=np.float64)
    min_valid = max(1, int(np.ceil(epoch_samples * min_valid_ratio)))

    for i in range(n_epochs):
        start = i * epoch_samples
        end = start + epoch_samples
        m = valid_mask[start:end]
        if m.sum() < min_valid:
            continue
        valid_fhr = fhr[start:end][m]
        if np.all(valid_fhr > 0):
            # Convert each sample to PI first, then average.
            # This is more faithful than 60000/mean(FHR), which
            # underestimates mean PI due to Jensen's inequality.
            epoch_pi[i] = np.mean(60000.0 / valid_fhr)

    return epoch_pi


def _expand_to_samples(
    block_values: np.ndarray,
    block_samples: int,
    total_samples: int,
) -> np.ndarray:
    """Repeat each block-level value across its sample range."""
    out = np.full(total_samples, np.nan, dtype=np.float64)
    for i, v in enumerate(block_values):
        start = i * block_samples
        end = min(start + block_samples, total_samples)
        out[start:end] = v
    return out


def _interpolate_nans(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN regions in a 1-D array."""
    out = x.copy()
    valid = np.isfinite(out)
    if not np.any(valid):
        return np.zeros_like(out)
    if np.all(valid):
        return out
    idx = np.arange(len(out))
    out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    return out


# =============================================================================
# STV — Short-Term Variability
# =============================================================================

def compute_stv_epochs(
    fhr: np.ndarray,
    acc_mask: Optional[np.ndarray] = None,
    dec_mask: Optional[np.ndarray] = None,
    quality_mask: Optional[np.ndarray] = None,
    config: Optional[STVConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute epoch-level STV.

    Returns:
        epoch_pi:   Mean pulse interval per epoch (ms).
        stv_epochs: |PI[i] − PI[i−1]| per epoch (ms).
                    stv_epochs[0] is always NaN (no predecessor).
    """
    if config is None:
        config = STVConfig()

    fhr = np.asarray(fhr, dtype=np.float64)
    n = len(fhr)
    valid = _build_valid_mask(n, fhr, acc_mask, dec_mask, quality_mask)

    epoch_pi = _compute_epoch_pulse_intervals(
        fhr, valid, config.epoch_samples, config.min_valid_ratio,
    )

    n_epochs = len(epoch_pi)
    stv_epochs = np.full(n_epochs, np.nan, dtype=np.float64)
    for i in range(1, n_epochs):
        if np.isfinite(epoch_pi[i]) and np.isfinite(epoch_pi[i - 1]):
            stv_epochs[i] = abs(epoch_pi[i] - epoch_pi[i - 1])

    return epoch_pi, stv_epochs


def compute_stv(
    fhr: np.ndarray,
    acc_mask: Optional[np.ndarray] = None,
    dec_mask: Optional[np.ndarray] = None,
    quality_mask: Optional[np.ndarray] = None,
    config: Optional[STVConfig] = None,
) -> np.ndarray:
    """
    Compute STV and return a **sample-level** array (ms).

    Each sample inherits the STV value of its epoch (the epoch-to-epoch
    difference ending at that epoch).

    Returns:
        1-D array, same length as *fhr*, unit: ms.
    """
    if config is None:
        config = STVConfig()

    fhr = np.asarray(fhr, dtype=np.float64)
    _, stv_epochs = compute_stv_epochs(
        fhr, acc_mask, dec_mask, quality_mask, config,
    )

    stv = _expand_to_samples(stv_epochs, config.epoch_samples, len(fhr))

    if config.interpolate_output:
        stv = _interpolate_nans(stv)

    return stv


def compute_stv_overall(
    fhr: np.ndarray,
    acc_mask: Optional[np.ndarray] = None,
    dec_mask: Optional[np.ndarray] = None,
    quality_mask: Optional[np.ndarray] = None,
    config: Optional[STVConfig] = None,
) -> float:
    """
    Compute **STV overall** — a single scalar (ms).

    STV overall = mean( |PI_epoch[i+1] − PI_epoch[i]| )

    Returns:
        STV overall in ms, or NaN if insufficient data.
    """
    _, stv_epochs = compute_stv_epochs(
        fhr, acc_mask, dec_mask, quality_mask, config,
    )
    valid = stv_epochs[np.isfinite(stv_epochs)]
    if len(valid) == 0:
        return float(np.nan)
    return float(np.mean(valid))


# =============================================================================
# LTV — Long-Term Variability
# =============================================================================

def compute_ltv_minutes(
    fhr: np.ndarray,
    acc_mask: Optional[np.ndarray] = None,
    dec_mask: Optional[np.ndarray] = None,
    quality_mask: Optional[np.ndarray] = None,
    config: Optional[LTVConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute minute-level LTV.

    Returns:
        epoch_pi:    Mean pulse interval per epoch (ms).
        ltv_minutes: Range of epoch PI within each 1-minute window (ms).
                     NaN for minutes with too few valid epochs.
    """
    if config is None:
        config = LTVConfig()

    fhr = np.asarray(fhr, dtype=np.float64)
    n = len(fhr)
    valid = _build_valid_mask(n, fhr, acc_mask, dec_mask, quality_mask)

    epoch_pi = _compute_epoch_pulse_intervals(
        fhr, valid, config.epoch_samples, config.min_valid_ratio,
    )

    epm = config.epochs_per_minute          # 16 for 3.75-s epochs
    n_epochs = len(epoch_pi)
    n_minutes = n_epochs // epm
    min_valid_epochs = max(1, int(np.ceil(epm * config.min_valid_epochs_ratio)))

    ltv_min = np.full(n_minutes, np.nan, dtype=np.float64)
    for i in range(n_minutes):
        start = i * epm
        end = start + epm
        minute_pi = epoch_pi[start:end]
        valid_pi = minute_pi[np.isfinite(minute_pi)]
        if len(valid_pi) >= min_valid_epochs:
            ltv_min[i] = float(np.max(valid_pi) - np.min(valid_pi))

    return epoch_pi, ltv_min


def compute_ltv(
    fhr: np.ndarray,
    acc_mask: Optional[np.ndarray] = None,
    dec_mask: Optional[np.ndarray] = None,
    quality_mask: Optional[np.ndarray] = None,
    config: Optional[LTVConfig] = None,
) -> np.ndarray:
    """
    Compute LTV and return a **sample-level** array (ms).

    Each sample inherits the LTV value of its 1-minute window.

    Returns:
        1-D array, same length as *fhr*, unit: ms.
    """
    if config is None:
        config = LTVConfig()

    fhr = np.asarray(fhr, dtype=np.float64)
    _, ltv_min = compute_ltv_minutes(
        fhr, acc_mask, dec_mask, quality_mask, config,
    )

    ltv = _expand_to_samples(ltv_min, config.minute_samples, len(fhr))

    if config.interpolate_output:
        ltv = _interpolate_nans(ltv)

    return ltv


def compute_ltv_overall(
    fhr: np.ndarray,
    acc_mask: Optional[np.ndarray] = None,
    dec_mask: Optional[np.ndarray] = None,
    quality_mask: Optional[np.ndarray] = None,
    config: Optional[LTVConfig] = None,
) -> float:
    """
    Compute **LTV overall** — a single scalar (ms).

    LTV overall = mean of per-minute pulse-interval ranges.

    Returns:
        LTV overall in ms, or NaN if insufficient data.
    """
    _, ltv_min = compute_ltv_minutes(
        fhr, acc_mask, dec_mask, quality_mask, config,
    )
    valid = ltv_min[np.isfinite(ltv_min)]
    if len(valid) == 0:
        return float(np.nan)
    return float(np.mean(valid))


# =============================================================================
# Statistics & classification
# =============================================================================

def classify_stv(stv_ms: float) -> str:
    """
    Classify STV overall (ms) using approximate Dawes–Redman thresholds
    (term gestation).

    Categories:
        abnormal:   < 3.0 ms  (≈ < 1.0 bpm @ 140 bpm)
        borderline: 3.0 – 4.0 ms
        normal:     ≥ 4.0 ms

    Note: exact thresholds vary with gestational age and system.
    """
    if not np.isfinite(stv_ms):
        return "unknown"
    if stv_ms < 3.0:
        return "abnormal"
    elif stv_ms < 4.0:
        return "borderline"
    else:
        return "normal"


def get_stv_statistics(stv: np.ndarray) -> dict:
    """
    Summary statistics from a sample-level (or epoch-level) STV array (ms).
    """
    stv = np.asarray(stv, dtype=np.float64)
    v = stv[np.isfinite(stv)]
    if len(v) == 0:
        return {
            "stv_mean_ms": np.nan,
            "stv_std_ms": np.nan,
            "stv_min_ms": np.nan,
            "stv_max_ms": np.nan,
            "stv_median_ms": np.nan,
            "stv_classification": "unknown",
            "valid_samples": 0,
            "valid_ratio": 0.0,
        }

    mean_v = float(np.mean(v))
    return {
        "stv_mean_ms": mean_v,
        "stv_std_ms": float(np.std(v)),
        "stv_min_ms": float(np.min(v)),
        "stv_max_ms": float(np.max(v)),
        "stv_median_ms": float(np.median(v)),
        "stv_classification": classify_stv(mean_v),
        "valid_samples": int(len(v)),
        "valid_ratio": float(len(v) / len(stv)),
    }


def get_ltv_statistics(ltv: np.ndarray) -> dict:
    """
    Summary statistics from a sample-level (or minute-level) LTV array (ms).
    """
    ltv = np.asarray(ltv, dtype=np.float64)
    v = ltv[np.isfinite(ltv)]
    if len(v) == 0:
        return {
            "ltv_mean_ms": np.nan,
            "ltv_std_ms": np.nan,
            "ltv_min_ms": np.nan,
            "ltv_max_ms": np.nan,
            "ltv_median_ms": np.nan,
            "valid_samples": 0,
            "valid_ratio": 0.0,
        }

    return {
        "ltv_mean_ms": float(np.mean(v)),
        "ltv_std_ms": float(np.std(v)),
        "ltv_min_ms": float(np.min(v)),
        "ltv_max_ms": float(np.max(v)),
        "ltv_median_ms": float(np.median(v)),
        "valid_samples": int(len(v)),
        "valid_ratio": float(len(v) / len(ltv)),
    }


def get_stv_summary(stats: dict) -> str:
    """Human-readable STV summary."""
    mean_ms = stats.get("stv_mean_ms", np.nan)
    lines = [
        "STV (Short-Term Variability) Summary",
        "=" * 44,
        f"  Mean:   {stats['stv_mean_ms']:.2f} ms"
        f"  (~{ms_to_bpm(mean_ms):.2f} bpm @ 140 bpm)",
        f"  Std:    {stats['stv_std_ms']:.2f} ms",
        f"  Range:  {stats['stv_min_ms']:.2f} – {stats['stv_max_ms']:.2f} ms",
        f"  Median: {stats['stv_median_ms']:.2f} ms",
        f"  Class:  {stats['stv_classification']}",
        "-" * 44,
        f"  Valid samples: {stats['valid_samples']}"
        f"  ({stats['valid_ratio']*100:.1f}%)",
    ]
    return "\n".join(lines)


def get_ltv_summary(stats: dict) -> str:
    """Human-readable LTV summary."""
    mean_ms = stats.get("ltv_mean_ms", np.nan)
    lines = [
        "LTV (Long-Term Variability) Summary",
        "=" * 44,
        f"  Mean:   {stats['ltv_mean_ms']:.2f} ms"
        f"  (~{ms_to_bpm(mean_ms):.2f} bpm @ 140 bpm)",
        f"  Std:    {stats['ltv_std_ms']:.2f} ms",
        f"  Range:  {stats['ltv_min_ms']:.2f} – {stats['ltv_max_ms']:.2f} ms",
        f"  Median: {stats['ltv_median_ms']:.2f} ms",
        "-" * 44,
        f"  Valid samples: {stats['valid_samples']}"
        f"  ({stats['valid_ratio']*100:.1f}%)",
    ]
    return "\n".join(lines)
