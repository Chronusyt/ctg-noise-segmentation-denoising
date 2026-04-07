"""Unified physiological feature and event-label computation for FHR segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from ctg_pipeline.preprocessing.acc_detection_figo_v2 import (
    AccelerationConfig,
    AccelerationCriterion,
    detect_accelerations_figo,
)
from ctg_pipeline.preprocessing.baseline_variability import (
    BaselineVariabilityConfig,
    classify_baseline_variability,
    compute_baseline_variability,
)
from ctg_pipeline.preprocessing.dec_detection_figo_v2 import (
    DecelerationConfig,
    DecelerationCriterion,
    detect_decelerations_figo,
)
from ctg_pipeline.preprocessing.fhr_baseline_optimized import BaselineConfig, analyse_baseline_optimized
from ctg_pipeline.preprocessing.variability import LTVConfig, STVConfig, compute_ltv_overall, compute_stv_overall


BV_CLASS_TO_ID = {
    "unknown": -1,
    "absent": 0,
    "reduced": 1,
    "minimal": 1,
    "normal": 2,
    "moderate": 2,
    "marked": 3,
}


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for 1-minute clean-FHR physiological labels."""

    sample_rate: float = 4.0
    valid_min: float = 50.0
    valid_max: float = 220.0
    min_valid_ratio: float = 0.4

    @property
    def sample_rate_int(self) -> int:
        return int(round(self.sample_rate))


def _as_2d(signals: np.ndarray) -> np.ndarray:
    arr = np.asarray(signals, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected [N,L] or [L] signals, got shape {arr.shape}")
    return arr


def _sanitize_fhr(signal: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    """Prepare an FHR segment for traditional feature extraction."""
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
    Compute scalar baseline / STV / LTV for FHR segment(s).

    This legacy-compatible helper is used by the current feature-preservation
    evaluation and figures. The multitask dataset builder uses
    `compute_multitask_physiology_labels`, which also computes BV and acc/dec
    event labels. Returned arrays all have shape [N].
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


def _baseline_config(length: int, cfg: FeatureConfig) -> BaselineConfig:
    return BaselineConfig(
        window_size=max(4, min(length, int(round(60.0 * cfg.sample_rate)))),
        window_step=max(1, min(length, int(round(15.0 * cfg.sample_rate)))),
        smoothing_window=max(1, min(length, int(round(15.0 * cfg.sample_rate)))),
        min_valid_ratio=cfg.min_valid_ratio,
    )


def _compute_baseline_trace(signal: np.ndarray, zero_mask: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    try:
        baseline_trace = analyse_baseline_optimized(
            signal,
            config=_baseline_config(len(signal), cfg),
            mask=zero_mask,
            sample_rate=cfg.sample_rate,
        )
    except Exception:
        baseline_trace = np.full(len(signal), float(np.nanmedian(signal)), dtype=np.float64)
    if baseline_trace.shape[0] != signal.shape[0]:
        baseline_trace = np.resize(baseline_trace, signal.shape[0]).astype(np.float64)
    return np.asarray(baseline_trace, dtype=np.float64)


def _bv_class_id(value: float) -> int:
    label = classify_baseline_variability(value)
    return int(BV_CLASS_TO_ID.get(label, -1))


def compute_multitask_physiology_labels(
    signals: np.ndarray,
    config: FeatureConfig | None = None,
) -> Dict[str, np.ndarray]:
    """
    Compute v1 physiological multitask labels from clean FHR segment(s).

    Returned fields:
    - baseline / stv / ltv / baseline_variability: float arrays, shape [N]
    - baseline_variability_class: int array, shape [N]
    - acc_labels / dec_labels: uint8 arrays, shape [N, L]
    - acc_counts / dec_counts: int arrays, shape [N]

    The supervised v1 feature labels are intentionally derived from clean FHR
    only. Acceleration/deceleration point labels use the existing FIGO 15x15
    detectors, and STV/LTV/BV are computed after excluding detected acc/dec
    regions following the historical `extract_features.py` feature flow.
    """
    cfg = config or FeatureConfig()
    arr = _as_2d(signals)
    n, length = arr.shape

    baseline = np.full(n, np.nan, dtype=np.float64)
    stv = np.full(n, np.nan, dtype=np.float64)
    ltv = np.full(n, np.nan, dtype=np.float64)
    baseline_variability = np.full(n, np.nan, dtype=np.float64)
    baseline_variability_class = np.full(n, -1, dtype=np.int32)
    acc_labels = np.zeros((n, length), dtype=np.uint8)
    dec_labels = np.zeros((n, length), dtype=np.uint8)
    acc_counts = np.zeros(n, dtype=np.int32)
    dec_counts = np.zeros(n, dtype=np.int32)

    acc_cfg = AccelerationConfig(
        sample_rate=cfg.sample_rate_int,
        criterion=AccelerationCriterion.RULE_15_15,
    )
    dec_cfg = DecelerationConfig(
        sample_rate=cfg.sample_rate_int,
        criterion=DecelerationCriterion.RULE_15_15,
    )
    stv_cfg = STVConfig(sampling_rate=cfg.sample_rate)
    ltv_cfg = LTVConfig(sampling_rate=cfg.sample_rate)
    bv_cfg = BaselineVariabilityConfig(
        window_sec=60.0,
        sampling_rate=cfg.sample_rate,
        min_valid_ratio=0.5,
        use_baseline_deviation=True,
        smooth_output=True,
        smooth_window_sec=15.0,
        interpolate_invalid=True,
    )

    for i in range(n):
        signal = _sanitize_fhr(arr[i], cfg)
        zero_mask = np.zeros(length, dtype=np.uint8)
        baseline_trace = _compute_baseline_trace(signal, zero_mask, cfg)
        baseline[i] = float(np.nanmean(baseline_trace))

        try:
            accelerations, acc_binary = detect_accelerations_figo(
                signal,
                baseline_trace,
                zero_mask,
                acc_cfg,
            )
        except Exception:
            accelerations, acc_binary = [], np.zeros(length, dtype=np.uint8)

        try:
            decelerations, dec_binary = detect_decelerations_figo(
                signal,
                baseline_trace,
                zero_mask,
                uc_models=None,
                config=dec_cfg,
            )
        except Exception:
            decelerations, dec_binary = [], np.zeros(length, dtype=np.uint8)

        acc_binary = np.asarray(acc_binary, dtype=np.uint8)
        dec_binary = np.asarray(dec_binary, dtype=np.uint8)
        acc_labels[i] = acc_binary
        dec_labels[i] = dec_binary
        acc_counts[i] = len(accelerations)
        dec_counts[i] = len(decelerations)

        stv[i] = compute_stv_overall(
            signal,
            acc_mask=acc_binary,
            dec_mask=dec_binary,
            quality_mask=zero_mask,
            config=stv_cfg,
        )
        if not np.isfinite(stv[i]):
            stv[i] = compute_stv_overall(signal, quality_mask=zero_mask, config=stv_cfg)
        if not np.isfinite(stv[i]):
            stv[i] = 0.0
        ltv[i] = compute_ltv_overall(
            signal,
            acc_mask=acc_binary,
            dec_mask=dec_binary,
            quality_mask=zero_mask,
            config=ltv_cfg,
        )
        if not np.isfinite(ltv[i]):
            ltv[i] = compute_ltv_overall(signal, quality_mask=zero_mask, config=ltv_cfg)
        if not np.isfinite(ltv[i]):
            ltv[i] = 0.0
        bv_trace = compute_baseline_variability(
            fhr=signal,
            baseline=baseline_trace,
            acc_mask=acc_binary,
            dec_mask=dec_binary,
            quality_mask=zero_mask,
            config=bv_cfg,
        )
        baseline_variability[i] = float(np.nanmean(bv_trace))
        if not np.isfinite(baseline_variability[i]):
            bv_trace = compute_baseline_variability(
                fhr=signal,
                baseline=baseline_trace,
                quality_mask=zero_mask,
                config=bv_cfg,
            )
            baseline_variability[i] = float(np.nanmean(bv_trace))
        if not np.isfinite(baseline_variability[i]):
            baseline_variability[i] = 0.0
        baseline_variability_class[i] = _bv_class_id(baseline_variability[i])

    return {
        "baseline": baseline,
        "stv": stv,
        "ltv": ltv,
        "baseline_variability": baseline_variability,
        "baseline_variability_class": baseline_variability_class,
        "acc_labels": acc_labels,
        "dec_labels": dec_labels,
        "acc_counts": acc_counts,
        "dec_counts": dec_counts,
    }


def feature_title(features: Dict[str, np.ndarray], index: int) -> str:
    """Compact feature string for figure titles."""
    return (
        f"B={features['baseline'][index]:.2f}, "
        f"STV={features['stv'][index]:.2f}, "
        f"LTV={features['ltv'][index]:.2f}"
    )
