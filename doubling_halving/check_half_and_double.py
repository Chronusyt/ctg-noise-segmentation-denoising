"""
Real doubling / halving candidate mining for batch1 fetal data.

This script keeps the original pipeline idea:
1. Read raw .fetal data
2. Split into 20-minute segments
3. Apply segment-level quality gating
4. Use artifact-corrected FHR only for baseline estimation
5. Detect half / double candidates from raw_fhr / baseline
6. Score contiguous candidate regions by ratio closeness, boundary jump,
   within-region stability, and flatness
7. Export summary.csv, candidate_segments.npz, plots, and overall stats
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

try:
    from .config import DetectionConfig, FEATURE_ROOT, RuntimeConfig, config_to_dict, resolve_runtime_config
except ImportError:
    from config import DetectionConfig, FEATURE_ROOT, RuntimeConfig, config_to_dict, resolve_runtime_config

if str(FEATURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FEATURE_ROOT))

from ctg_io.fetal_reader import read_fetal
from ctg_preprocessing.artifact_correction import correct_artifacts
from ctg_preprocessing.fhr_baseline_optimized import BaselineConfig, analyse_baseline_optimized
from ctg_preprocessing.signal_quality import assess_signal_quality


SUMMARY_FIELDS = [
    "candidate_id",
    "file_id",
    "file_path",
    "segment_idx",
    "segment_start_sec",
    "segment_end_sec",
    "region_type",
    "start_idx",
    "end_idx",
    "start_sec",
    "end_sec",
    "duration_sec",
    "mean_ratio",
    "ratio_std",
    "mean_fhr",
    "mean_baseline",
    "quality_percent",
    "quality_bad_percent",
    "boundary_jump",
    "fhr_range",
    "confidence",
    "baseline_strategy",
    "plot_path",
]


def _contiguous_regions(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e - s >= min_len]


def _boundary_jump(fhr: np.ndarray, idx: int, look: int = 8) -> float:
    n = len(fhr)
    if idx < 0 or idx >= n:
        return 0.0
    val = float(fhr[idx])
    if not np.isfinite(val) or val <= 50 or val >= 220:
        return 0.0

    best = 0.0
    for sl in (slice(max(0, idx - look), idx), slice(idx + 1, min(n, idx + 1 + look))):
        neighbors = np.asarray(fhr[sl], dtype=np.float64)
        neighbors = neighbors[np.isfinite(neighbors) & (neighbors > 50) & (neighbors < 220)]
        if len(neighbors) > 0:
            best = max(best, abs(float(np.median(neighbors)) - val))
    return best


def _region_stats(fhr_raw: np.ndarray, ratio: np.ndarray, start: int, end: int) -> Dict[str, float]:
    seg = np.asarray(fhr_raw[start:end], dtype=np.float64)
    seg = seg[np.isfinite(seg) & (seg > 50) & (seg < 220)]

    ratio_vals = np.asarray(ratio[start:end], dtype=np.float64)
    ratio_vals = ratio_vals[np.isfinite(ratio_vals)]

    if len(seg) > 2:
        cv = float(np.std(seg) / max(np.mean(seg), 1.0))
        fhr_range = float(np.max(seg) - np.min(seg))
    else:
        cv = 0.08
        fhr_range = 25.0

    return {
        "cv": cv,
        "fhr_range": fhr_range,
        "ratio_std": float(np.std(ratio_vals)) if len(ratio_vals) > 1 else 0.0,
    }


def _confidence(mean_ratio: float, target: float, fhr_raw: np.ndarray, ratio: np.ndarray, start: int, end: int) -> float:
    max_dev = 0.12 if target < 1.0 else 0.35
    ratio_score = max(0.0, 1.0 - abs(mean_ratio - target) / max_dev)

    jump = max(_boundary_jump(fhr_raw, start), _boundary_jump(fhr_raw, end - 1))
    jump_score = min(1.0, jump / 40.0)

    stats = _region_stats(fhr_raw, ratio, start, end)
    stability_score = max(0.0, 1.0 - stats["cv"] / 0.08)
    flatness_score = max(0.0, 1.0 - stats["fhr_range"] / 25.0)

    return float(
        0.30 * ratio_score
        + 0.20 * jump_score
        + 0.25 * stability_score
        + 0.25 * flatness_score
    )


def _interpolate_or_fill(signal: np.ndarray, fill_value: float = 140.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    out = signal.copy()
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


def compute_baseline(
    fhr_clean: np.ndarray,
    quality_mask: np.ndarray,
    detect_cfg: DetectionConfig,
) -> Tuple[np.ndarray, str]:
    clean = np.asarray(fhr_clean, dtype=np.float64)
    quality_mask = np.asarray(quality_mask, dtype=np.uint8)

    if detect_cfg.baseline_strategy == "feature":
        try:
            baseline_cfg = BaselineConfig(
                window_size=detect_cfg.feature_baseline_window_size,
                window_step=detect_cfg.feature_baseline_window_step,
                smoothing_window=detect_cfg.feature_baseline_smoothing_window,
                variability_threshold=detect_cfg.feature_baseline_variability_threshold,
                min_valid_ratio=detect_cfg.feature_baseline_min_valid_ratio,
            )
            baseline = analyse_baseline_optimized(
                clean,
                config=baseline_cfg,
                mask=quality_mask,
                sample_rate=detect_cfg.sample_rate,
            ).astype(np.float64)
            if len(baseline) == len(clean) and np.all(np.isfinite(baseline)):
                return baseline, "feature_optimized"
        except Exception:
            pass

    fill_ready = _interpolate_or_fill(clean)
    win = int(detect_cfg.baseline_window_sec * detect_cfg.sample_rate)
    if win % 2 == 0:
        win += 1
    baseline = median_filter(fill_ready, size=max(3, win), mode="reflect").astype(np.float64)
    return baseline, "median_fallback"


def detect_half_double(
    fhr_raw: np.ndarray,
    baseline: np.ndarray,
    detect_cfg: DetectionConfig,
) -> Tuple[List[dict], np.ndarray]:
    n = len(fhr_raw)
    raw = np.asarray(fhr_raw, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)

    raw_ok = (raw >= detect_cfg.valid_fhr_min) & (raw <= detect_cfg.valid_fhr_max)
    baseline_ok = (baseline >= detect_cfg.baseline_min) & (baseline <= detect_cfg.baseline_max)
    valid = raw_ok & baseline_ok

    ratio = np.full(n, np.nan, dtype=np.float64)
    ratio[valid] = raw[valid] / baseline[valid]

    min_samples = max(1, int(detect_cfg.min_duration_sec * detect_cfg.sample_rate))
    max_samples = max(min_samples, int(detect_cfg.max_duration_sec * detect_cfg.sample_rate))

    detections: List[dict] = []

    regions_to_scan = [
        ("half", 0.5, detect_cfg.half_ratio_lo, detect_cfg.half_ratio_hi),
        ("double", 2.0, detect_cfg.double_ratio_lo, detect_cfg.double_ratio_hi),
    ]

    for region_type, target_ratio, ratio_lo, ratio_hi in regions_to_scan:
        mask = np.zeros(n, dtype=bool)
        mask[valid] = (ratio[valid] >= ratio_lo) & (ratio[valid] <= ratio_hi)

        if region_type == "double":
            clipped = (raw >= 210) & baseline_ok & (baseline < 150)
            mask |= clipped

        for start, end in _contiguous_regions(mask, min_samples):
            if end - start > max_samples:
                continue

            ratio_vals = ratio[start:end]
            ratio_vals = ratio_vals[np.isfinite(ratio_vals)]
            if region_type == "half" and len(ratio_vals) < max(1, min_samples // 2):
                continue

            mean_ratio = float(np.nanmean(ratio_vals)) if len(ratio_vals) > 0 else target_ratio
            confidence = _confidence(mean_ratio, target_ratio, raw, ratio, start, end)
            if confidence < detect_cfg.min_confidence:
                continue

            region_stats = _region_stats(raw, ratio, start, end)
            jump = max(_boundary_jump(raw, start), _boundary_jump(raw, end - 1))

            detections.append(
                {
                    "region_type": region_type,
                    "start_idx": start,
                    "end_idx": end,
                    "duration_sec": round((end - start) / detect_cfg.sample_rate, 3),
                    "mean_ratio": round(mean_ratio, 4),
                    "ratio_std": round(region_stats["ratio_std"], 4),
                    "mean_fhr": round(float(np.nanmean(raw[start:end])), 2),
                    "mean_baseline": round(float(np.nanmean(baseline[start:end])), 2),
                    "boundary_jump": round(jump, 2),
                    "fhr_range": round(region_stats["fhr_range"], 2),
                    "confidence": round(confidence, 3),
                }
            )

    detections.sort(key=lambda item: (item["start_idx"], item["region_type"]))
    return detections, ratio


def plot_segment(
    fhr_raw: np.ndarray,
    fhr_clean: np.ndarray,
    baseline: np.ndarray,
    ratio: np.ndarray,
    quality_mask: np.ndarray,
    detections: List[dict],
    file_id: str,
    segment_idx: int,
    segment_offset_min: float,
    save_path: Path,
    detect_cfg: DetectionConfig,
) -> None:
    n = len(fhr_raw)
    t_min = np.arange(n, dtype=np.float64) / detect_cfg.sample_rate / 60.0 + segment_offset_min

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [4.0, 1.5, 1.0]},
    )
    ax_sig, ax_ratio, ax_q = axes

    colors = {"half": "#E65100", "double": "#1B5E20"}
    legend_seen = set()

    ax_sig.plot(t_min, fhr_raw, color="#1976D2", lw=0.9, alpha=0.9, label="Raw FHR")
    ax_sig.plot(t_min, fhr_clean, color="#90A4AE", lw=0.8, alpha=0.8, label="Clean FHR")
    ax_sig.plot(t_min, baseline, color="#222222", lw=1.2, ls="--", label="Baseline")

    for det in detections:
        start = det["start_idx"]
        end = min(det["end_idx"] - 1, n - 1)
        region_type = det["region_type"]
        label = None
        if region_type not in legend_seen:
            label = f"{region_type} candidate"
            legend_seen.add(region_type)

        ax_sig.axvspan(t_min[start], t_min[end], color=colors[region_type], alpha=0.22, label=label)
        ax_ratio.axvspan(t_min[start], t_min[end], color=colors[region_type], alpha=0.22)

        x_mid = (t_min[start] + t_min[end]) / 2
        y_top = min(245.0, float(np.nanmax(fhr_raw[max(0, start - 4):min(n, end + 4)])) + 6.0)
        ax_sig.text(
            x_mid,
            y_top,
            f"{region_type} | conf {det['confidence']:.2f}",
            fontsize=8,
            ha="center",
            va="bottom",
            color=colors[region_type],
            bbox={"facecolor": "white", "edgecolor": colors[region_type], "alpha": 0.75, "pad": 2},
        )

    ax_sig.set_title(f"{file_id} | segment {segment_idx + 1}")
    ax_sig.set_ylabel("FHR (bpm)")
    ax_sig.set_ylim(40, 250)
    ax_sig.grid(True, alpha=0.25)
    ax_sig.legend(loc="upper right", fontsize=8, ncol=2)

    ax_ratio.plot(t_min, ratio, color="#6A1B9A", lw=0.9, label="raw / baseline")
    ax_ratio.axhline(0.5, color=colors["half"], ls="--", lw=0.9, alpha=0.8, label="0.5x")
    ax_ratio.axhline(2.0, color=colors["double"], ls="--", lw=0.9, alpha=0.8, label="2.0x")
    ax_ratio.set_ylabel("Ratio")
    ax_ratio.set_ylim(0.0, 2.5)
    ax_ratio.grid(True, alpha=0.25)
    ax_ratio.legend(loc="upper right", fontsize=8, ncol=3)

    ax_q.fill_between(t_min, 0, quality_mask, color="#D32F2F", alpha=0.35, step="mid", label="Unreliable mask")
    ax_q.set_ylabel("Mask")
    ax_q.set_xlabel("Time (min)")
    ax_q.set_ylim(-0.1, 1.2)
    ax_q.set_yticks([0, 1])
    ax_q.grid(True, alpha=0.25)
    ax_q.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=detect_cfg.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def _candidate_payload_arrays(candidates: List[dict]) -> Dict[str, np.ndarray]:
    if not candidates:
        return {
            "candidate_id": np.array([], dtype=object),
            "file_id": np.array([], dtype=object),
            "file_path": np.array([], dtype=object),
            "segment_idx": np.array([], dtype=np.int32),
            "region_type": np.array([], dtype=object),
            "segment_start_sec": np.array([], dtype=np.float32),
            "segment_end_sec": np.array([], dtype=np.float32),
            "start_idx": np.array([], dtype=np.int32),
            "end_idx": np.array([], dtype=np.int32),
            "start_sec": np.array([], dtype=np.float32),
            "end_sec": np.array([], dtype=np.float32),
            "duration_sec": np.array([], dtype=np.float32),
            "mean_ratio": np.array([], dtype=np.float32),
            "ratio_std": np.array([], dtype=np.float32),
            "mean_fhr": np.array([], dtype=np.float32),
            "mean_baseline": np.array([], dtype=np.float32),
            "quality_percent": np.array([], dtype=np.float32),
            "quality_bad_percent": np.array([], dtype=np.float32),
            "boundary_jump": np.array([], dtype=np.float32),
            "fhr_range": np.array([], dtype=np.float32),
            "confidence": np.array([], dtype=np.float32),
            "plot_path": np.array([], dtype=object),
            "baseline_strategy": np.array([], dtype=object),
            "raw_segment": np.array([], dtype=object),
            "clean_segment": np.array([], dtype=object),
            "baseline_segment": np.array([], dtype=object),
            "ratio_segment": np.array([], dtype=object),
            "quality_mask_segment": np.array([], dtype=object),
            "detection_mask_segment": np.array([], dtype=object),
        }

    def obj_array(key: str) -> np.ndarray:
        return np.array([np.asarray(item[key]) for item in candidates], dtype=object)

    def value_array(key: str, dtype=None) -> np.ndarray:
        values = [item[key] for item in candidates]
        return np.array(values, dtype=dtype) if dtype is not None else np.array(values)

    return {
        "candidate_id": value_array("candidate_id", dtype=object),
        "file_id": value_array("file_id", dtype=object),
        "file_path": value_array("file_path", dtype=object),
        "segment_idx": value_array("segment_idx", dtype=np.int32),
        "region_type": value_array("region_type", dtype=object),
        "segment_start_sec": value_array("segment_start_sec", dtype=np.float32),
        "segment_end_sec": value_array("segment_end_sec", dtype=np.float32),
        "start_idx": value_array("start_idx", dtype=np.int32),
        "end_idx": value_array("end_idx", dtype=np.int32),
        "start_sec": value_array("start_sec", dtype=np.float32),
        "end_sec": value_array("end_sec", dtype=np.float32),
        "duration_sec": value_array("duration_sec", dtype=np.float32),
        "mean_ratio": value_array("mean_ratio", dtype=np.float32),
        "ratio_std": value_array("ratio_std", dtype=np.float32),
        "mean_fhr": value_array("mean_fhr", dtype=np.float32),
        "mean_baseline": value_array("mean_baseline", dtype=np.float32),
        "quality_percent": value_array("quality_percent", dtype=np.float32),
        "quality_bad_percent": value_array("quality_bad_percent", dtype=np.float32),
        "boundary_jump": value_array("boundary_jump", dtype=np.float32),
        "fhr_range": value_array("fhr_range", dtype=np.float32),
        "confidence": value_array("confidence", dtype=np.float32),
        "plot_path": value_array("plot_path", dtype=object),
        "baseline_strategy": value_array("baseline_strategy", dtype=object),
        "raw_segment": obj_array("raw_segment"),
        "clean_segment": obj_array("clean_segment"),
        "baseline_segment": obj_array("baseline_segment"),
        "ratio_segment": obj_array("ratio_segment"),
        "quality_mask_segment": obj_array("quality_mask_segment"),
        "detection_mask_segment": obj_array("detection_mask_segment"),
    }


def _write_summary_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_candidate_npz(candidates: List[dict], npz_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **_candidate_payload_arrays(candidates))


def _confidence_histogram(detections: List[dict], bins: Iterable[float]) -> Dict[str, int]:
    edges = list(bins)
    if len(edges) < 2:
        return {}
    values = np.array([det["confidence"] for det in detections], dtype=np.float64)
    counts: Dict[str, int] = {}
    for left, right in zip(edges[:-1], edges[1:]):
        if right >= 1.0:
            label = f"[{left:.2f}, 1.00]"
            mask = (values >= left) & (values <= right)
        else:
            label = f"[{left:.2f}, {right:.2f})"
            mask = (values >= left) & (values < right)
        counts[label] = int(np.sum(mask))
    return counts


def build_batch_stats(
    detections: List[dict],
    aggregate: Dict[str, int],
    errors: List[dict],
    runtime_cfg: RuntimeConfig,
    detect_cfg: DetectionConfig,
) -> Dict[str, object]:
    durations = np.array([det["duration_sec"] for det in detections], dtype=np.float64)
    half_durations = np.array([det["duration_sec"] for det in detections if det["region_type"] == "half"], dtype=np.float64)
    double_durations = np.array([det["duration_sec"] for det in detections if det["region_type"] == "double"], dtype=np.float64)

    def duration_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
        if len(values) == 0:
            return {"mean": None, "median": None, "max": None}
        return {
            "mean": round(float(np.mean(values)), 3),
            "median": round(float(np.median(values)), 3),
            "max": round(float(np.max(values)), 3),
        }

    stats = {
        "total_files": aggregate["files_total"],
        "processed_files": aggregate["files_processed"],
        "files_with_detections": aggregate["files_with_detections"],
        "files_with_errors": len(errors),
        "segments_total": aggregate["segments_total"],
        "segments_short_skipped": aggregate["segments_short_skipped"],
        "segments_low_quality_skipped": aggregate["segments_low_quality_skipped"],
        "segments_analyzed": aggregate["segments_analyzed"],
        "segments_with_detections": aggregate["segments_with_detections"],
        "total_regions": len(detections),
        "half_regions": sum(1 for det in detections if det["region_type"] == "half"),
        "double_regions": sum(1 for det in detections if det["region_type"] == "double"),
        "duration_stats_all_sec": duration_stats(durations),
        "duration_stats_half_sec": duration_stats(half_durations),
        "duration_stats_double_sec": duration_stats(double_durations),
        "confidence_histogram": _confidence_histogram(detections, detect_cfg.confidence_bins),
        "runtime": {
            "data_dir": str(runtime_cfg.data_dir) if runtime_cfg.data_dir else None,
            "output_dir": str(runtime_cfg.output_dir),
            "num_workers": runtime_cfg.num_workers,
        },
    }
    if errors:
        stats["errors"] = errors
    return stats


def _plot_filename(file_id: str, segment_idx: int, detections: List[dict]) -> str:
    half_count = sum(1 for det in detections if det["region_type"] == "half")
    double_count = sum(1 for det in detections if det["region_type"] == "double")
    return f"{file_id}__seg{segment_idx + 1:04d}__half{half_count}_double{double_count}.png"


def process_one_file(filepath: str, runtime_cfg: RuntimeConfig, detect_cfg: DetectionConfig) -> Tuple[str, List[dict], List[dict], Dict[str, int], Optional[dict]]:
    file_path = Path(filepath)
    file_id = file_path.stem

    stats = {
        "files_processed": 1,
        "files_with_detections": 0,
        "segments_total": 0,
        "segments_short_skipped": 0,
        "segments_low_quality_skipped": 0,
        "segments_analyzed": 0,
        "segments_with_detections": 0,
    }

    detections_out: List[dict] = []
    candidates_out: List[dict] = []

    try:
        fetal = read_fetal(str(file_path))
        if fetal.fhr is None:
            return file_id, detections_out, candidates_out, stats, None

        fhr_raw = np.asarray(fetal.fhr, dtype=np.float64)

        if len(fhr_raw) < detect_cfg.min_segment_samples:
            return file_id, detections_out, candidates_out, stats, None

        total_len = len(fhr_raw)
        segment_starts = list(range(0, total_len, detect_cfg.segment_samples))

        for segment_idx, start_global in enumerate(segment_starts):
            end_global = min(start_global + detect_cfg.segment_samples, total_len)
            stats["segments_total"] += 1

            if end_global - start_global < detect_cfg.min_segment_samples:
                stats["segments_short_skipped"] += 1
                continue

            raw_segment = fhr_raw[start_global:end_global].astype(np.float64)
            quality_mask, quality_stats = assess_signal_quality(raw_segment, sample_rate=detect_cfg.sample_rate)
            quality_pct = float(quality_stats["reliability_percent"])
            if quality_pct < detect_cfg.min_quality_percent:
                stats["segments_low_quality_skipped"] += 1
                continue

            stats["segments_analyzed"] += 1
            clean_segment, _ = correct_artifacts(raw_segment, quality_mask)
            baseline, baseline_strategy = compute_baseline(clean_segment, quality_mask, detect_cfg)
            detections, ratio = detect_half_double(raw_segment, baseline, detect_cfg)

            if not detections:
                continue

            stats["segments_with_detections"] += 1
            if stats["files_with_detections"] == 0:
                stats["files_with_detections"] = 1

            segment_offset_sec = start_global / detect_cfg.sample_rate
            segment_end_sec = end_global / detect_cfg.sample_rate
            quality_bad_percent = round(float(np.mean(quality_mask > 0) * 100.0), 3)

            plot_rel = ""
            if runtime_cfg.save_plots:
                plot_rel = str(Path("plots") / file_id / _plot_filename(file_id, segment_idx, detections))
                plot_abs = runtime_cfg.output_dir / plot_rel
                plot_segment(
                    raw_segment,
                    clean_segment,
                    baseline,
                    ratio,
                    quality_mask,
                    detections,
                    file_id=file_id,
                    segment_idx=segment_idx,
                    segment_offset_min=segment_offset_sec / 60.0,
                    save_path=plot_abs,
                    detect_cfg=detect_cfg,
                )

            for det_idx, det in enumerate(detections):
                candidate_id = f"{file_id}__seg{segment_idx + 1:04d}__{det['region_type']}__{det_idx + 1:02d}"
                detection_mask = np.zeros(len(raw_segment), dtype=np.uint8)
                detection_mask[det["start_idx"]:det["end_idx"]] = 1

                record = dict(det)
                record.update(
                    {
                        "candidate_id": candidate_id,
                        "file_id": file_id,
                        "file_path": str(file_path),
                        "segment_idx": segment_idx,
                        "segment_start_sec": round(segment_offset_sec, 3),
                        "segment_end_sec": round(segment_end_sec, 3),
                        "start_sec": round(segment_offset_sec + det["start_idx"] / detect_cfg.sample_rate, 3),
                        "end_sec": round(segment_offset_sec + det["end_idx"] / detect_cfg.sample_rate, 3),
                        "quality_percent": round(quality_pct, 3),
                        "quality_bad_percent": quality_bad_percent,
                        "baseline_strategy": baseline_strategy,
                        "plot_path": plot_rel,
                    }
                )
                detections_out.append(record)

                candidates_out.append(
                    {
                        **record,
                        "raw_segment": raw_segment.astype(np.float32),
                        "clean_segment": np.asarray(clean_segment, dtype=np.float32),
                        "baseline_segment": np.asarray(baseline, dtype=np.float32),
                        "ratio_segment": np.asarray(ratio, dtype=np.float32),
                        "quality_mask_segment": np.asarray(quality_mask, dtype=np.uint8),
                        "detection_mask_segment": detection_mask,
                    }
                )

    except Exception:
        return file_id, detections_out, candidates_out, stats, {
            "file_id": file_id,
            "file_path": str(file_path),
            "traceback": traceback.format_exc(),
        }

    return file_id, detections_out, candidates_out, stats, None


def _empty_aggregate(total_files: int = 0) -> Dict[str, int]:
    return {
        "files_total": total_files,
        "files_processed": 0,
        "files_with_detections": 0,
        "segments_total": 0,
        "segments_short_skipped": 0,
        "segments_low_quality_skipped": 0,
        "segments_analyzed": 0,
        "segments_with_detections": 0,
    }


def _merge_stats(dst: Dict[str, int], src: Dict[str, int]) -> None:
    for key, value in src.items():
        if key == "files_total":
            continue
        dst[key] = dst.get(key, 0) + int(value)


def _write_outputs(
    detections: List[dict],
    candidates: List[dict],
    errors: List[dict],
    aggregate: Dict[str, int],
    runtime_cfg: RuntimeConfig,
    detect_cfg: DetectionConfig,
) -> Dict[str, object]:
    runtime_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = runtime_cfg.output_dir / "summary.csv"
    candidate_npz = runtime_cfg.output_dir / "candidate_segments.npz"
    stats_json = runtime_cfg.output_dir / "batch_stats.json"
    config_json = runtime_cfg.output_dir / "run_config.json"
    errors_json = runtime_cfg.output_dir / "errors.json"

    _write_summary_csv(detections, summary_csv)
    if runtime_cfg.export_candidates:
        _write_candidate_npz(candidates, candidate_npz)

    batch_stats = build_batch_stats(detections, aggregate, errors, runtime_cfg, detect_cfg)
    with open(stats_json, "w", encoding="utf-8") as handle:
        json.dump(batch_stats, handle, ensure_ascii=False, indent=2)
    with open(config_json, "w", encoding="utf-8") as handle:
        json.dump(config_to_dict(runtime_cfg, detect_cfg), handle, ensure_ascii=False, indent=2)
    with open(errors_json, "w", encoding="utf-8") as handle:
        json.dump(errors, handle, ensure_ascii=False, indent=2)

    return {
        "summary_csv": summary_csv,
        "candidate_npz": candidate_npz,
        "stats_json": stats_json,
        "config_json": config_json,
        "errors_json": errors_json,
        "batch_stats": batch_stats,
    }


def run_batch(runtime_cfg: RuntimeConfig, detect_cfg: DetectionConfig) -> Dict[str, object]:
    if runtime_cfg.data_dir is None:
        raise FileNotFoundError(
            "No default batch1 fetal directory was found. "
            "Please pass --data-dir or set CTG_BATCH1_FETAL_DIR."
        )
    if not runtime_cfg.data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {runtime_cfg.data_dir}")

    fetal_files = sorted(runtime_cfg.data_dir.glob("*.fetal"))
    aggregate = _empty_aggregate(total_files=len(fetal_files))
    all_detections: List[dict] = []
    all_candidates: List[dict] = []
    errors: List[dict] = []

    print(f"Found {len(fetal_files):,} .fetal files in {runtime_cfg.data_dir}")
    if not fetal_files:
        return _write_outputs(all_detections, all_candidates, errors, aggregate, runtime_cfg, detect_cfg)

    started_at = time.time()
    print(f"Processing with {runtime_cfg.num_workers} worker(s)")

    if runtime_cfg.num_workers == 1:
        iterator = ((None, file_path) for file_path in fetal_files)
        for _, file_path in iterator:
            _, detections, candidates, stats, err = process_one_file(str(file_path), runtime_cfg, detect_cfg)
            _merge_stats(aggregate, stats)
            all_detections.extend(detections)
            all_candidates.extend(candidates)
            if err:
                errors.append(err)
    else:
        with ProcessPoolExecutor(max_workers=runtime_cfg.num_workers) as pool:
            futures = {
                pool.submit(process_one_file, str(file_path), runtime_cfg, detect_cfg): file_path
                for file_path in fetal_files
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                _, detections, candidates, stats, err = future.result()
                _merge_stats(aggregate, stats)
                all_detections.extend(detections)
                all_candidates.extend(candidates)
                if err:
                    errors.append(err)

                if idx % 200 == 0 or idx == len(fetal_files):
                    elapsed = max(time.time() - started_at, 1e-6)
                    speed = idx / elapsed
                    print(
                        f"  [{idx:>6,}/{len(fetal_files):,}] "
                        f"{speed:7.2f} files/s | "
                        f"detections={len(all_detections):,} | "
                        f"errors={len(errors):,}"
                    )

    outputs = _write_outputs(all_detections, all_candidates, errors, aggregate, runtime_cfg, detect_cfg)
    elapsed = time.time() - started_at
    stats = outputs["batch_stats"]

    print("=" * 68)
    print(f"Completed in {elapsed:.1f}s")
    print(f"Processed files          : {stats['processed_files']:,}")
    print(f"Files with detections    : {stats['files_with_detections']:,}")
    print(f"Half regions             : {stats['half_regions']:,}")
    print(f"Double regions           : {stats['double_regions']:,}")
    print(f"Total regions            : {stats['total_regions']:,}")
    print(f"Low-quality segments     : {stats['segments_low_quality_skipped']:,}")
    print(f"Confidence histogram     : {stats['confidence_histogram']}")
    print(f"Summary CSV              : {outputs['summary_csv']}")
    if runtime_cfg.export_candidates:
        print(f"Candidate NPZ            : {outputs['candidate_npz']}")
    print(f"Stats JSON               : {outputs['stats_json']}")
    print("=" * 68)
    return outputs


def run_single(filepath: str, runtime_cfg: RuntimeConfig, detect_cfg: DetectionConfig) -> Dict[str, object]:
    file_path = Path(filepath).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Input .fetal file does not exist: {file_path}")

    aggregate = _empty_aggregate(total_files=1)
    _, detections, candidates, stats, err = process_one_file(str(file_path), runtime_cfg, detect_cfg)
    _merge_stats(aggregate, stats)
    errors = [err] if err else []

    outputs = _write_outputs(detections, candidates, errors, aggregate, runtime_cfg, detect_cfg)

    if err:
        print(err["traceback"])
        return outputs

    if not detections:
        print("No half / double counting candidates detected.")
        print(f"Summary CSV: {outputs['summary_csv']}")
        return outputs

    print(f"Detected {len(detections)} candidate region(s) in {file_path.stem}:")
    for idx, det in enumerate(detections, start=1):
        print(
            f"  [{idx}] {det['region_type']:6s} "
            f"idx {det['start_idx']:>5d}-{det['end_idx']:>5d} "
            f"dur={det['duration_sec']:.1f}s "
            f"ratio={det['mean_ratio']:.3f} "
            f"fhr={det['mean_fhr']:.1f} "
            f"baseline={det['mean_baseline']:.1f} "
            f"conf={det['confidence']:.3f}"
        )
    print(f"Summary CSV: {outputs['summary_csv']}")
    if runtime_cfg.export_candidates:
        print(f"Candidate NPZ: {outputs['candidate_npz']}")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mine real doubling / halving candidates from raw batch1 fetal files."
    )
    parser.add_argument("fetal_path", nargs="?", help="Optional single .fetal file to process.")
    parser.add_argument("--file", dest="single_file", help="Single .fetal file to process.")
    parser.add_argument("--data-dir", help="Directory containing batch1 .fetal files.")
    parser.add_argument("--output-dir", help="Output directory for summary / plots / npz exports.")
    parser.add_argument("--workers", type=int, help="Number of batch workers.")
    parser.add_argument("--min-quality", type=float, help="Override minimum segment reliability percent.")
    parser.add_argument("--min-confidence", type=float, help="Override minimum candidate confidence threshold.")
    parser.add_argument(
        "--baseline-strategy",
        choices=["feature", "median"],
        help="Baseline estimation strategy. 'feature' uses feature/ctg_preprocessing baseline with median fallback.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Do not export detection plots.")
    parser.add_argument("--no-candidate-export", action="store_true", help="Do not export candidate_segments.npz.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    runtime_cfg = resolve_runtime_config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_workers=args.workers,
        save_plots=not args.no_plots,
        export_candidates=not args.no_candidate_export,
    )

    detect_cfg = DetectionConfig()
    if args.min_quality is not None:
        detect_cfg = replace(detect_cfg, min_quality_percent=float(args.min_quality))
    if args.min_confidence is not None:
        detect_cfg = replace(detect_cfg, min_confidence=float(args.min_confidence))
    if args.baseline_strategy is not None:
        detect_cfg = replace(detect_cfg, baseline_strategy=args.baseline_strategy)

    single_file = args.single_file or args.fetal_path
    if single_file:
        run_single(single_file, runtime_cfg, detect_cfg)
    else:
        run_batch(runtime_cfg, detect_cfg)


if __name__ == "__main__":
    main()
