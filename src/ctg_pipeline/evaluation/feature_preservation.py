"""Physiological feature preservation metrics for FHR reconstruction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np

from ctg_pipeline.preprocessing.fhr_baseline_optimized import BaselineConfig, analyse_baseline_optimized
from ctg_pipeline.preprocessing.variability import LTVConfig, STVConfig, compute_ltv_overall, compute_stv_overall


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for 1-minute FHR feature computation."""

    sample_rate: float = 4.0
    valid_min: float = 50.0
    valid_max: float = 220.0


def _as_2d(signals: np.ndarray) -> np.ndarray:
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected [N,L] or [L] signals, got shape {arr.shape}")
    return arr


def _sanitize_fhr(signal: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Prepare reconstructed FHR for traditional feature extraction."""
    out = np.asarray(signal, dtype=np.float64).copy()
    bad = ~np.isfinite(out)
    valid = (~bad) & (out >= cfg.valid_min) & (out <= cfg.valid_max)
    if np.any(valid):
        idx = np.arange(len(out))
        out[~valid] = np.interp(idx[~valid], idx[valid], out[valid])
    else:
        out[:] = 140.0
    return np.clip(out, cfg.valid_min, cfg.valid_max)


def compute_signal_features(signals: np.ndarray, config: FeatureConfig | None = None) -> Dict[str, np.ndarray]:
    """
    Compute scalar baseline / STV / LTV for each FHR segment.

    Returns arrays with shape [N]. Baseline is the mean of the optimized
    baseline trace; STV/LTV follow the existing pulse-interval implementations.
    """
    cfg = config or FeatureConfig()
    arr = _as_2d(signals)
    n, length = arr.shape
    baseline = np.full(n, np.nan, dtype=np.float64)
    stv = np.full(n, np.nan, dtype=np.float64)
    ltv = np.full(n, np.nan, dtype=np.float64)

    baseline_cfg = BaselineConfig(
        window_size=max(4, min(length, int(round(60.0 * cfg.sample_rate)))),
        window_step=max(1, min(length, int(round(15.0 * cfg.sample_rate)))),
        smoothing_window=max(1, min(length, int(round(15.0 * cfg.sample_rate)))),
        min_valid_ratio=0.4,
    )
    stv_cfg = STVConfig(sampling_rate=cfg.sample_rate)
    ltv_cfg = LTVConfig(sampling_rate=cfg.sample_rate)

    for i in range(n):
        signal = _sanitize_fhr(arr[i], cfg)
        zero_mask = np.zeros(length, dtype=np.uint8)
        try:
            baseline_trace = analyse_baseline_optimized(
                signal,
                config=baseline_cfg,
                mask=zero_mask,
                sample_rate=cfg.sample_rate,
            )
            baseline[i] = float(np.nanmean(baseline_trace))
        except Exception:
            baseline[i] = float(np.nanmedian(signal))
        stv[i] = compute_stv_overall(signal, quality_mask=zero_mask, config=stv_cfg)
        ltv[i] = compute_ltv_overall(signal, quality_mask=zero_mask, config=ltv_cfg)

    return {"baseline": baseline, "stv": stv, "ltv": ltv}


def summarize_feature_preservation(
    reconstructed: np.ndarray,
    clean: np.ndarray,
    config: FeatureConfig | None = None,
) -> Dict[str, float]:
    """Summarize reconstructed-vs-clean feature deviations."""
    pred_features = compute_signal_features(reconstructed, config=config)
    clean_features = compute_signal_features(clean, config=config)
    out: Dict[str, float] = {}

    for name in ("baseline", "stv", "ltv"):
        diff = pred_features[name] - clean_features[name]
        out[f"{name}_mae"] = float(np.nanmean(np.abs(diff)))
        out[f"{name}_bias_mean"] = float(np.nanmean(diff))
        out[f"{name}_bias_median"] = float(np.nanmedian(diff))
        out[f"{name}_clean_mean"] = float(np.nanmean(clean_features[name]))
        out[f"{name}_reconstructed_mean"] = float(np.nanmean(pred_features[name]))

    return out


def feature_title(features: Dict[str, np.ndarray], index: int) -> str:
    """Compact feature string for figure titles."""
    return (
        f"B={features['baseline'][index]:.2f}, "
        f"STV={features['stv'][index]:.2f}, "
        f"LTV={features['ltv'][index]:.2f}"
    )


def metric_subset(metrics: Dict[str, float], keys: Iterable[str]) -> Dict[str, float]:
    """Return a stable subset while tolerating older metric JSON files."""
    return {key: float(metrics.get(key, np.nan)) for key in keys}
