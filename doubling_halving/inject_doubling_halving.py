from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
FEATURE_ROOT = PROJECT_ROOT / "feature"

if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from ctg_preprocessing.fhr_baseline_optimized import BaselineConfig, analyse_baseline_optimized

try:
    from .config_injection import InjectionConfig, get_default_config
except ImportError:
    from config_injection import InjectionConfig, get_default_config


@dataclass
class SegmentSpec:
    region_type: str
    double_subtype: Optional[str]
    start_idx: int
    end_idx: int
    duration_sec: float
    ratio_used: float
    sigma_used: float
    transition_len_points: int
    ceiling_used: Optional[float]
    mean_baseline: float
    mean_noisy_value: float
    actual_mean_ratio: float


def _interpolate_or_fill(signal: np.ndarray, fill_value: float = 140.0) -> np.ndarray:
    out = np.asarray(signal, dtype=np.float64).copy()
    nan_mask = ~np.isfinite(out)
    if not np.any(nan_mask):
        return out

    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        out[:] = fill_value
        return out
    if len(valid_idx) == 1:
        out[nan_mask] = out[valid_idx[0]]
        return out

    out[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, out[valid_idx])
    return out


def compute_local_baseline(clean_signal: np.ndarray, cfg: InjectionConfig) -> Tuple[np.ndarray, str]:
    signal = np.asarray(clean_signal, dtype=np.float64)
    zero_mask = np.zeros(len(signal), dtype=np.uint8)

    if cfg.baseline_strategy == "feature":
        try:
            baseline_cfg = BaselineConfig(
                window_size=cfg.feature_baseline_window_size,
                window_step=cfg.feature_baseline_window_step,
                smoothing_window=cfg.feature_baseline_smoothing_window,
                variability_threshold=cfg.feature_baseline_variability_threshold,
                min_valid_ratio=cfg.feature_baseline_min_valid_ratio,
            )
            baseline = analyse_baseline_optimized(
                signal,
                config=baseline_cfg,
                mask=zero_mask,
                sample_rate=cfg.fs,
            ).astype(np.float64)
            if len(baseline) == len(signal) and np.all(np.isfinite(baseline)):
                return baseline, "feature_optimized"
        except Exception:
            pass

    fill_ready = _interpolate_or_fill(signal)
    win = int(cfg.baseline_window_sec * cfg.fs)
    if win % 2 == 0:
        win += 1
    baseline = median_filter(fill_ready, size=max(3, win), mode="reflect").astype(np.float64)
    return baseline, "median_fallback"


def sample_duration(region_type: str, rng: np.random.Generator, cfg: InjectionConfig) -> int:
    if region_type == "half":
        main_range = cfg.halving_duration_main
        tail_range = cfg.halving_duration_tail
        main_weight = cfg.halving_main_weight
    else:
        main_range = cfg.doubling_duration_main
        tail_range = cfg.doubling_duration_tail
        main_weight = cfg.doubling_main_weight

    sec_range = main_range if rng.random() < main_weight else tail_range
    duration_sec = float(rng.uniform(sec_range[0], sec_range[1]))
    duration_sec = min(duration_sec, cfg.max_duration_sec)
    return max(1, int(round(duration_sec * cfg.fs)))


def sample_transition_len(rng: np.random.Generator, cfg: InjectionConfig) -> int:
    low, high = cfg.transition_len_points
    return int(rng.integers(low, high + 1))


def _sample_ratio(region_type: str, rng: np.random.Generator, cfg: InjectionConfig) -> float:
    low, high = cfg.halving_ratio_range if region_type == "half" else cfg.doubling_ratio_range
    return float(rng.uniform(low, high))


def _sample_sigma(region_type: str, rng: np.random.Generator, cfg: InjectionConfig) -> float:
    low, high = cfg.halving_sigma_range if region_type == "half" else cfg.doubling_sigma_range
    return float(rng.uniform(low, high))


def _smooth_noise(noise: np.ndarray) -> np.ndarray:
    if len(noise) <= 2:
        return noise
    kernel_len = min(5, len(noise))
    kernel = np.ones(kernel_len, dtype=np.float64) / kernel_len
    return np.convolve(noise, kernel, mode="same")


def apply_short_transition(segment_values: np.ndarray, original_signal: np.ndarray, start_idx: int, end_idx: int, transition_len: int) -> np.ndarray:
    blended = np.asarray(segment_values, dtype=np.float64).copy()
    seg_len = len(blended)
    if seg_len == 0:
        return blended

    transition_len = max(0, min(transition_len, seg_len // 2))
    if transition_len == 0:
        return blended

    left_ref = original_signal[start_idx - 1] if start_idx > 0 else blended[0]
    right_ref = original_signal[end_idx] if end_idx < len(original_signal) else blended[-1]

    for i in range(transition_len):
        alpha = (i + 1) / float(transition_len + 1)
        blended[i] = (1.0 - alpha) * left_ref + alpha * blended[i]

    for i in range(transition_len):
        alpha = (transition_len - i) / float(transition_len + 1)
        pos = seg_len - transition_len + i
        blended[pos] = alpha * blended[pos] + (1.0 - alpha) * right_ref

    return blended


def _find_injection_window(
    baseline: np.ndarray,
    clean_signal: np.ndarray,
    occupied_mask: np.ndarray,
    duration_points: int,
    transition_len: int,
    rng: np.random.Generator,
    cfg: InjectionConfig,
    region_type: str,
    subtype: Optional[str] = None,
    baseline_min: Optional[float] = None,
    baseline_max: Optional[float] = None,
) -> Optional[Tuple[int, int]]:
    n = len(clean_signal)
    if duration_points >= n:
        return None

    valid_baseline = (
        np.isfinite(baseline)
        & (baseline >= 80.0)
        & (baseline <= 200.0)
        & np.isfinite(clean_signal)
        & (clean_signal >= cfg.valid_fhr_min)
        & (clean_signal <= cfg.valid_fhr_max)
    )

    padding = max(transition_len, 1)
    max_start = n - duration_points
    attempts = 0

    while attempts < cfg.max_placement_attempts:
        start = int(rng.integers(0, max_start + 1))
        end = start + duration_points
        win_start = max(0, start - padding)
        win_end = min(n, end + padding)

        if np.any(occupied_mask[win_start:win_end]):
            attempts += 1
            continue
        if not np.all(valid_baseline[start:end]):
            attempts += 1
            continue
        window_mean_baseline = float(np.mean(baseline[start:end]))
        if baseline_min is not None and window_mean_baseline < baseline_min:
            attempts += 1
            continue
        if baseline_max is not None and window_mean_baseline > baseline_max:
            attempts += 1
            continue
        return start, end

    return None


def _core_slice(length: int, transition_len: int) -> slice:
    if transition_len <= 0:
        return slice(0, length)
    if length <= 2 * transition_len + 2:
        return slice(0, length)
    return slice(transition_len, length - transition_len)


def _build_halving_values(
    baseline_segment: np.ndarray,
    ratio_used: float,
    sigma_used: float,
    clean_signal: np.ndarray,
    start_idx: int,
    end_idx: int,
    transition_len: int,
    rng: np.random.Generator,
    cfg: InjectionConfig,
) -> np.ndarray:
    target = baseline_segment * ratio_used
    noise = _smooth_noise(rng.normal(0.0, sigma_used, size=len(target)))
    plateau = target + noise
    plateau = apply_short_transition(plateau, clean_signal, start_idx, end_idx, transition_len)
    return np.clip(plateau, cfg.valid_fhr_min, cfg.valid_fhr_max)


def _build_doubling_values(
    baseline_segment: np.ndarray,
    ratio_used: float,
    sigma_used: float,
    ceiling_used: float,
    clean_signal: np.ndarray,
    start_idx: int,
    end_idx: int,
    transition_len: int,
    rng: np.random.Generator,
    cfg: InjectionConfig,
) -> np.ndarray:
    target = baseline_segment * ratio_used
    noise = _smooth_noise(rng.normal(0.0, sigma_used, size=len(target)))
    plateau = target + noise

    over_ceiling = plateau >= ceiling_used
    if np.any(over_ceiling):
        # Do not hard-flatten the whole segment to a single ceiling value.
        # Instead, compress clipped points into a narrow band just below the
        # ceiling and keep a small subset exactly at the ceiling to mimic the
        # high-platform / clipped appearance seen in real doubling artifacts.
        ceiling_band = float(rng.uniform(max(1.5, sigma_used * 0.7), max(3.5, sigma_used * 1.6)))
        downward_jitter = np.abs(
            _smooth_noise(
                rng.normal(loc=ceiling_band * 0.45, scale=max(0.6, ceiling_band * 0.35), size=len(target))
            )
        )
        plateau = plateau.copy()
        plateau[over_ceiling] = ceiling_used - np.clip(downward_jitter[over_ceiling], 0.0, ceiling_band)

        if np.mean(over_ceiling) > 0.4:
            exact_hit_prob = float(rng.uniform(0.12, 0.30))
            exact_hits = over_ceiling & (rng.random(len(target)) < exact_hit_prob)
            plateau[exact_hits] = ceiling_used

    plateau = apply_short_transition(plateau, clean_signal, start_idx, end_idx, transition_len)
    return np.clip(plateau, cfg.valid_fhr_min, cfg.valid_fhr_max)


def inject_halving_segment(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    baseline: np.ndarray,
    artifact_mask: np.ndarray,
    occupied_mask: np.ndarray,
    rng: np.random.Generator,
    cfg: InjectionConfig,
) -> Optional[SegmentSpec]:
    duration_points = sample_duration("half", rng, cfg)
    transition_len = sample_transition_len(rng, cfg)
    window = _find_injection_window(
        baseline,
        clean_signal,
        occupied_mask,
        duration_points,
        transition_len,
        rng,
        cfg,
        region_type="half",
    )
    if window is None:
        return None

    start_idx, end_idx = window
    ratio_used = _sample_ratio("half", rng, cfg)
    sigma_used = _sample_sigma("half", rng, cfg)
    baseline_segment = baseline[start_idx:end_idx]

    segment_values = _build_halving_values(
        baseline_segment,
        ratio_used,
        sigma_used,
        clean_signal,
        start_idx,
        end_idx,
        transition_len,
        rng,
        cfg,
    )

    noisy_signal[start_idx:end_idx] = segment_values
    artifact_mask[start_idx:end_idx] = 1
    occupied_mask[start_idx:end_idx] = True

    core = _core_slice(len(segment_values), transition_len)
    actual_ratio = float(np.mean(segment_values[core] / baseline_segment[core]))
    return SegmentSpec(
        region_type="half",
        double_subtype=None,
        start_idx=start_idx,
        end_idx=end_idx,
        duration_sec=(end_idx - start_idx) / cfg.fs,
        ratio_used=ratio_used,
        sigma_used=sigma_used,
        transition_len_points=transition_len,
        ceiling_used=None,
        mean_baseline=float(np.mean(baseline_segment)),
        mean_noisy_value=float(np.mean(segment_values)),
        actual_mean_ratio=actual_ratio,
    )


def inject_unclipped_doubling_segment(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    baseline: np.ndarray,
    artifact_mask: np.ndarray,
    occupied_mask: np.ndarray,
    rng: np.random.Generator,
    cfg: InjectionConfig,
) -> Optional[SegmentSpec]:
    baseline_tiers = (105.0, 110.0, cfg.unclipped_double_baseline_max)
    window = None
    baseline_segment = None
    ratio_used = None
    sigma_used = None
    start_idx = end_idx = 0
    transition_len = 0

    for baseline_max in baseline_tiers:
        duration_points = sample_duration("double", rng, cfg)
        transition_len = sample_transition_len(rng, cfg)
        candidate_window = _find_injection_window(
            baseline,
            clean_signal,
            occupied_mask,
            duration_points,
            transition_len,
            rng,
            cfg,
            region_type="double",
            subtype="unclipped",
            baseline_max=baseline_max,
        )
        if candidate_window is None:
            continue

        start_idx, end_idx = candidate_window
        candidate_baseline = baseline[start_idx:end_idx]
        local_baseline = float(np.mean(candidate_baseline))
        max_ratio_allowed = cfg.valid_fhr_max / max(local_baseline, 1.0)
        if max_ratio_allowed < cfg.doubling_ratio_range[0]:
            continue

        ratio_high = min(cfg.doubling_ratio_range[1], max_ratio_allowed)
        if ratio_high < cfg.doubling_ratio_range[0]:
            continue

        ratio_used = float(rng.uniform(cfg.doubling_ratio_range[0], ratio_high))
        sigma_used = _sample_sigma("double", rng, cfg)
        baseline_segment = candidate_baseline
        window = candidate_window
        break

    if window is None or baseline_segment is None or ratio_used is None or sigma_used is None:
        return None

    target = baseline_segment * ratio_used
    noise = _smooth_noise(rng.normal(0.0, sigma_used, size=len(target)))
    segment_values = target + noise
    segment_values = apply_short_transition(segment_values, clean_signal, start_idx, end_idx, transition_len)
    segment_values = np.clip(segment_values, cfg.valid_fhr_min, cfg.valid_fhr_max)

    noisy_signal[start_idx:end_idx] = segment_values
    artifact_mask[start_idx:end_idx] = 2
    occupied_mask[start_idx:end_idx] = True

    core = _core_slice(len(segment_values), transition_len)
    actual_ratio = float(np.mean(segment_values[core] / baseline_segment[core]))
    return SegmentSpec(
        region_type="double",
        double_subtype="unclipped",
        start_idx=start_idx,
        end_idx=end_idx,
        duration_sec=(end_idx - start_idx) / cfg.fs,
        ratio_used=ratio_used,
        sigma_used=sigma_used,
        transition_len_points=transition_len,
        ceiling_used=None,
        mean_baseline=float(np.mean(baseline_segment)),
        mean_noisy_value=float(np.mean(segment_values)),
        actual_mean_ratio=actual_ratio,
    )


def inject_clipped_doubling_segment(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    baseline: np.ndarray,
    artifact_mask: np.ndarray,
    occupied_mask: np.ndarray,
    rng: np.random.Generator,
    cfg: InjectionConfig,
) -> Optional[SegmentSpec]:
    window = None
    baseline_segment = None
    ratio_used = None
    sigma_used = None
    ceiling_used = None
    start_idx = end_idx = 0
    transition_len = 0

    for _ in range(4):
        duration_points = sample_duration("double", rng, cfg)
        transition_len = sample_transition_len(rng, cfg)
        candidate_window = _find_injection_window(
            baseline,
            clean_signal,
            occupied_mask,
            duration_points,
            transition_len,
            rng,
            cfg,
            region_type="double",
            subtype="clipped",
            baseline_min=cfg.clipped_double_baseline_min,
            baseline_max=cfg.clipped_double_baseline_max,
        )
        if candidate_window is None:
            continue

        start_idx, end_idx = candidate_window
        candidate_baseline = baseline[start_idx:end_idx]
        candidate_ratio = _sample_ratio("double", rng, cfg)
        candidate_sigma = _sample_sigma("double", rng, cfg)
        candidate_target = candidate_baseline * candidate_ratio
        target_peak = float(np.percentile(candidate_target, 80))
        if target_peak < cfg.double_ceiling_range[0]:
            continue

        ceiling_upper = min(cfg.double_ceiling_range[1], max(cfg.double_ceiling_range[0], target_peak))
        candidate_ceiling = float(rng.uniform(cfg.double_ceiling_range[0], ceiling_upper))

        window = candidate_window
        baseline_segment = candidate_baseline
        ratio_used = candidate_ratio
        sigma_used = candidate_sigma
        ceiling_used = candidate_ceiling
        break

    if window is None or baseline_segment is None or ratio_used is None or sigma_used is None or ceiling_used is None:
        return None

    segment_values = _build_doubling_values(
        baseline_segment,
        ratio_used,
        sigma_used,
        ceiling_used,
        clean_signal,
        start_idx,
        end_idx,
        transition_len,
        rng,
        cfg,
    )

    noisy_signal[start_idx:end_idx] = segment_values
    artifact_mask[start_idx:end_idx] = 2
    occupied_mask[start_idx:end_idx] = True

    core = _core_slice(len(segment_values), transition_len)
    actual_ratio = float(np.mean(segment_values[core] / baseline_segment[core]))
    return SegmentSpec(
        region_type="double",
        double_subtype="clipped",
        start_idx=start_idx,
        end_idx=end_idx,
        duration_sec=(end_idx - start_idx) / cfg.fs,
        ratio_used=ratio_used,
        sigma_used=sigma_used,
        transition_len_points=transition_len,
        ceiling_used=ceiling_used,
        mean_baseline=float(np.mean(baseline_segment)),
        mean_noisy_value=float(np.mean(segment_values)),
        actual_mean_ratio=actual_ratio,
    )


def inject_doubling_segment(
    noisy_signal: np.ndarray,
    clean_signal: np.ndarray,
    baseline: np.ndarray,
    artifact_mask: np.ndarray,
    occupied_mask: np.ndarray,
    rng: np.random.Generator,
    cfg: InjectionConfig,
) -> Optional[SegmentSpec]:
    candidates = []
    if cfg.enable_unclipped_doubling:
        candidates.append(("unclipped", cfg.double_unclipped_weight))
    if cfg.enable_clipped_doubling:
        candidates.append(("clipped", cfg.double_clipped_weight))
    if not candidates:
        return None

    names = [name for name, _ in candidates]
    weights = np.asarray([weight for _, weight in candidates], dtype=np.float64)
    if np.sum(weights) <= 0:
        weights = np.ones(len(weights), dtype=np.float64)
    weights = weights / np.sum(weights)

    preferred = names[int(rng.choice(len(names), p=weights))]
    ordered = [preferred] + [name for name in names if name != preferred]

    for subtype in ordered:
        if subtype == "unclipped":
            segment = inject_unclipped_doubling_segment(
                noisy_signal, clean_signal, baseline, artifact_mask, occupied_mask, rng, cfg
            )
        else:
            segment = inject_clipped_doubling_segment(
                noisy_signal, clean_signal, baseline, artifact_mask, occupied_mask, rng, cfg
            )
        if segment is not None:
            return segment
    return None


def _segment_to_dict(segment: SegmentSpec) -> Dict[str, float]:
    return {
        "region_type": segment.region_type,
        "double_subtype": segment.double_subtype,
        "start_idx": segment.start_idx,
        "end_idx": segment.end_idx,
        "duration_sec": round(segment.duration_sec, 3),
        "ratio_used": round(segment.ratio_used, 4),
        "sigma_used": round(segment.sigma_used, 4),
        "transition_len_points": int(segment.transition_len_points),
        "ceiling_used": None if segment.ceiling_used is None else round(segment.ceiling_used, 3),
        "mean_baseline": round(segment.mean_baseline, 3),
        "mean_noisy_value": round(segment.mean_noisy_value, 3),
        "actual_mean_ratio": round(segment.actual_mean_ratio, 4),
    }


def inject_one_signal(
    clean_signal: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    cfg: Optional[InjectionConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    cfg = cfg if cfg is not None else get_default_config()
    rng = rng if rng is not None else np.random.default_rng(cfg.random_seed)

    clean_signal = np.asarray(clean_signal, dtype=np.float64)
    noisy_signal = clean_signal.copy()
    artifact_mask = np.zeros(len(clean_signal), dtype=np.uint8)
    occupied_mask = np.zeros(len(clean_signal), dtype=bool)

    if baseline is None:
        baseline, baseline_strategy = compute_local_baseline(clean_signal, cfg)
    else:
        baseline = np.asarray(baseline, dtype=np.float64)
        baseline_strategy = "provided"

    halving_segments: List[Dict[str, float]] = []
    doubling_segments: List[Dict[str, float]] = []

    if cfg.enable_halving and rng.random() < cfg.halving_prob:
        halving_segment = inject_halving_segment(
            noisy_signal, clean_signal, baseline, artifact_mask, occupied_mask, rng, cfg
        )
        if halving_segment is not None:
            halving_segments.append(_segment_to_dict(halving_segment))

    if cfg.enable_doubling and rng.random() < cfg.doubling_prob:
        doubling_segment = inject_doubling_segment(
            noisy_signal, clean_signal, baseline, artifact_mask, occupied_mask, rng, cfg
        )
        if doubling_segment is not None:
            doubling_segments.append(_segment_to_dict(doubling_segment))

    noisy_signal = np.clip(noisy_signal, cfg.valid_fhr_min, cfg.valid_fhr_max)

    metadata: Dict[str, object] = {
        "has_halving": bool(len(halving_segments) > 0),
        "has_doubling": bool(len(doubling_segments) > 0),
        "halving_segments": halving_segments,
        "doubling_segments": doubling_segments,
        "double_subtype": [seg["double_subtype"] for seg in doubling_segments if seg["double_subtype"] is not None],
        "duration_sec": [seg["duration_sec"] for seg in halving_segments + doubling_segments],
        "ratio_used": [seg["ratio_used"] for seg in halving_segments + doubling_segments],
        "actual_mean_ratio": [seg["actual_mean_ratio"] for seg in halving_segments + doubling_segments],
        "ceiling_used": [seg["ceiling_used"] for seg in doubling_segments if seg["ceiling_used"] is not None],
        "baseline_strategy": baseline_strategy,
        "baseline": baseline.astype(np.float32),
    }

    return noisy_signal.astype(np.float32), artifact_mask, metadata
