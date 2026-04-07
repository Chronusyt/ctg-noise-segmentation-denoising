"""Physiological feature preservation metrics for FHR reconstruction."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from ctg_pipeline.features.physiology import FeatureConfig, compute_signal_features, feature_title


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


def metric_subset(metrics: Dict[str, float], keys: Iterable[str]) -> Dict[str, float]:
    """Return a stable subset while tolerating older metric JSON files."""
    return {key: float(metrics.get(key, np.nan)) for key in keys}
