from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path("/home/yt/CTG_test")
OUTPUT_DIR = PROJECT_ROOT / "doubling_halving" / "output"
REVIEW_DIR = PROJECT_ROOT / "doubling_halving" / "review_samples"

SUMMARY_CSV = OUTPUT_DIR / "summary.csv"
CANDIDATE_NPZ = OUTPUT_DIR / "candidate_segments.npz"
PLOTS_DIR = OUTPUT_DIR / "plots"

SELECTED_SUMMARY_CSV = REVIEW_DIR / "selected_summary.csv"
SELECTED_SEGMENTS_NPZ = REVIEW_DIR / "selected_segments.npz"
REVIEW_TABLE_CSV = REVIEW_DIR / "review_table.csv"
REVIEW_PLOTS_DIR = REVIEW_DIR / "plots"

CONFIDENCE_THRESHOLD = 0.85
MIN_DURATION_SEC = 3.0
TARGET_HALF = 50
TARGET_DOUBLE = 30
RANDOM_SEED = 20260331

SUMMARY_FIELDS = [
    "candidate_id",
    "file_id",
    "segment_idx",
    "region_type",
    "duration_sec",
    "mean_ratio",
    "mean_baseline",
    "mean_fhr",
    "confidence",
    "plot_path",
]

REVIEW_TABLE_FIELDS = [
    "candidate_id",
    "file_id",
    "region_type",
    "duration_sec",
    "mean_ratio",
    "confidence",
    "plot_path",
    "human_label",
]

SEGMENT_EXPORT_FIELDS = [
    "raw_segment",
    "clean_segment",
    "baseline_segment",
    "ratio_segment",
    "quality_mask_segment",
    "detection_mask_segment",
    "candidate_id",
    "region_type",
    "duration_sec",
    "mean_ratio",
    "confidence",
]


def load_summary_rows() -> List[dict]:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing summary.csv: {SUMMARY_CSV}")

    with open(SUMMARY_CSV, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["segment_idx"] = int(row["segment_idx"])
        row["duration_sec"] = float(row["duration_sec"])
        row["mean_ratio"] = float(row["mean_ratio"])
        row["mean_baseline"] = float(row["mean_baseline"])
        row["mean_fhr"] = float(row["mean_fhr"])
        row["confidence"] = float(row["confidence"])
    return rows


def first_stage_filter(rows: List[dict]) -> List[dict]:
    return [
        row for row in rows
        if row["confidence"] >= CONFIDENCE_THRESHOLD and row["duration_sec"] >= MIN_DURATION_SEC
    ]


def sample_by_region(rows: List[dict], region_type: str, target_count: int, rng: random.Random) -> List[dict]:
    candidates = [row for row in rows if row["region_type"] == region_type]
    if len(candidates) <= target_count:
        return list(candidates)
    return rng.sample(candidates, target_count)


def ensure_output_dirs() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_selected_summary(rows: List[dict]) -> List[dict]:
    return [
        {field: row[field] for field in SUMMARY_FIELDS}
        for row in rows
    ]


def build_review_table(rows: List[dict]) -> List[dict]:
    review_rows = []
    for row in rows:
        review_row = {field: row[field] for field in REVIEW_TABLE_FIELDS if field != "human_label"}
        review_row["human_label"] = ""
        review_rows.append(review_row)
    return review_rows


def copy_selected_plots(rows: List[dict]) -> int:
    copied = 0
    seen = set()
    for row in rows:
        rel_plot = Path(row["plot_path"])
        if not row["plot_path"] or rel_plot in seen:
            continue
        src = OUTPUT_DIR / rel_plot
        dst = REVIEW_DIR / rel_plot
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        seen.add(rel_plot)
    return copied


def export_selected_segments(selected_rows: List[dict]) -> None:
    if not CANDIDATE_NPZ.exists():
        raise FileNotFoundError(f"Missing candidate_segments.npz: {CANDIDATE_NPZ}")

    selected_ids = [row["candidate_id"] for row in selected_rows]
    id_to_pos = {candidate_id: idx for idx, candidate_id in enumerate(selected_ids)}

    npz = np.load(CANDIDATE_NPZ, allow_pickle=True)
    all_ids = npz["candidate_id"]
    selected_positions = [
        idx for idx, candidate_id in enumerate(all_ids)
        if candidate_id in id_to_pos
    ]

    selected_positions.sort(key=lambda idx: id_to_pos[all_ids[idx]])

    export_data: Dict[str, np.ndarray] = {}
    for field in SEGMENT_EXPORT_FIELDS:
        export_data[field] = npz[field][selected_positions]

    np.savez_compressed(SELECTED_SEGMENTS_NPZ, **export_data)


def describe_distribution(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def print_stats(all_rows: List[dict], filtered_rows: List[dict], selected_rows: List[dict]) -> None:
    total_half = sum(row["region_type"] == "half" for row in all_rows)
    total_double = sum(row["region_type"] == "double" for row in all_rows)

    filtered_half = [row for row in filtered_rows if row["region_type"] == "half"]
    filtered_double = [row for row in filtered_rows if row["region_type"] == "double"]
    selected_half = [row for row in selected_rows if row["region_type"] == "half"]
    selected_double = [row for row in selected_rows if row["region_type"] == "double"]

    duration_stats = describe_distribution([row["duration_sec"] for row in selected_rows])
    ratio_stats = describe_distribution([row["mean_ratio"] for row in selected_rows])
    conf_stats = describe_distribution([row["confidence"] for row in selected_rows])

    print("=" * 68)
    print(f"Input summary          : {SUMMARY_CSV}")
    print(f"Input candidate npz    : {CANDIDATE_NPZ}")
    print("-" * 68)
    print(f"Original candidates    : half={total_half}, double={total_double}")
    print(
        f"After threshold        : half={len(filtered_half)}, double={len(filtered_double)} "
        f"(confidence>={CONFIDENCE_THRESHOLD}, duration>={MIN_DURATION_SEC}s)"
    )
    print(f"Selected samples       : half={len(selected_half)}, double={len(selected_double)}")
    print("-" * 68)
    print(
        f"Duration distribution  : mean={duration_stats['mean']:.3f}s, "
        f"median={duration_stats['median']:.3f}s"
    )
    print(
        f"Ratio distribution     : mean={ratio_stats['mean']:.4f}, "
        f"std={ratio_stats['std']:.4f}"
    )
    print(
        f"Confidence distribution: mean={conf_stats['mean']:.4f}, "
        f"median={conf_stats['median']:.4f}, min={conf_stats['min']:.4f}, max={conf_stats['max']:.4f}"
    )
    print("-" * 68)
    print(f"selected_summary.csv   : {SELECTED_SUMMARY_CSV}")
    print(f"selected_segments.npz  : {SELECTED_SEGMENTS_NPZ}")
    print(f"review_table.csv       : {REVIEW_TABLE_CSV}")
    print(f"review plots dir       : {REVIEW_PLOTS_DIR}")
    print("=" * 68)


def main() -> None:
    ensure_output_dirs()

    all_rows = load_summary_rows()
    filtered_rows = first_stage_filter(all_rows)

    rng = random.Random(RANDOM_SEED)
    selected_half = sample_by_region(filtered_rows, "half", TARGET_HALF, rng)
    selected_double = sample_by_region(filtered_rows, "double", TARGET_DOUBLE, rng)
    selected_rows = selected_half + selected_double

    selected_summary = build_selected_summary(selected_rows)
    review_table = build_review_table(selected_rows)

    write_csv(SELECTED_SUMMARY_CSV, selected_summary, SUMMARY_FIELDS)
    write_csv(REVIEW_TABLE_CSV, review_table, REVIEW_TABLE_FIELDS)
    export_selected_segments(selected_rows)
    copy_selected_plots(selected_rows)
    print_stats(all_rows, filtered_rows, selected_rows)


if __name__ == "__main__":
    main()
