"""Validate clinical physiological multitask dataset alignment and labels."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.features.physiology import FeatureConfig, compute_multitask_physiology_labels
from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_DATASETS_ROOT, resolve_repo_path


def load_multitask_split(data_dir: str, split: str) -> np.lib.npyio.NpzFile:
    return np.load(os.path.join(data_dir, f"{split}_dataset_multitask.npz"))


def check_split(data_dir: str, split: str, sample_count: int, rng: np.random.Generator) -> tuple[list[str], bool]:
    data = load_multitask_split(data_dir, split)
    n = int(data["noisy_signals"].shape[0])
    ok = True
    lines = [f"[{split}]", f"n_samples: {n}"]

    expected = {
        "clean_signals": data["noisy_signals"].shape,
        "masks": (n, data["noisy_signals"].shape[1], 5),
        "artifact_labels": (n, data["noisy_signals"].shape[1], 5),
        "baseline_labels": (n,),
        "stv_labels": (n,),
        "ltv_labels": (n,),
        "baseline_variability_labels": (n,),
        "baseline_variability_class_labels": (n,),
        "acc_labels": data["noisy_signals"].shape,
        "dec_labels": data["noisy_signals"].shape,
        "acc_counts": (n,),
        "dec_counts": (n,),
        "parent_index": (n,),
        "chunk_index": (n,),
    }
    for key, shape in expected.items():
        present = key in data.files
        shape_ok = present and data[key].shape == shape
        lines.append(f"{key}: present={present}, shape={data[key].shape if present else 'missing'}, expected={shape}, ok={shape_ok}")
        ok = ok and shape_ok

    if "pred_masks" in data.files:
        pred_shape_ok = data["pred_masks"].shape == expected["masks"]
        lines.append(f"pred_masks: present=True, shape={data['pred_masks'].shape}, ok={pred_shape_ok}")
        ok = ok and pred_shape_ok

    for key in (
        "noisy_signals",
        "clean_signals",
        "masks",
        "artifact_labels",
        "baseline_labels",
        "stv_labels",
        "ltv_labels",
        "baseline_variability_labels",
    ):
        finite_ok = bool(np.all(np.isfinite(data[key])))
        lines.append(f"{key}: finite_ok={finite_ok}")
        ok = ok and finite_ok

    for key in ("acc_labels", "dec_labels"):
        values = np.unique(data[key])
        binary_ok = bool(np.all(np.isin(values, [0, 1])))
        length_ok = data[key].shape[1] == data["clean_signals"].shape[1]
        positive_ratio = float(np.mean(data[key] > 0))
        lines.append(
            f"{key}: binary_ok={binary_ok}, length_ok={length_ok}, "
            f"positive_ratio={positive_ratio:.6f}, unique={values[:10].tolist()}"
        )
        ok = ok and binary_ok and length_ok

    aligned_shape_ok = (
        data["noisy_signals"].shape == data["clean_signals"].shape
        and data["masks"].shape == data["artifact_labels"].shape
        and data["masks"].shape[:2] == data["clean_signals"].shape
    )
    lines.append(f"core_shape_alignment_ok: {aligned_shape_ok}")
    ok = ok and aligned_shape_ok

    sample_count = min(sample_count, n)
    indices = np.sort(rng.choice(n, size=sample_count, replace=False)) if sample_count > 0 else np.array([], dtype=int)
    recomputed = compute_multitask_physiology_labels(data["clean_signals"][indices], config=FeatureConfig(sample_rate=4.0))
    tolerances = {
        "baseline": 1e-4,
        "stv": 1e-4,
        "ltv": 1e-4,
        "baseline_variability": 1e-4,
    }
    for feature, label_key in (
        ("baseline", "baseline_labels"),
        ("stv", "stv_labels"),
        ("ltv", "ltv_labels"),
        ("baseline_variability", "baseline_variability_labels"),
    ):
        diff = np.abs(recomputed[feature] - data[label_key][indices])
        max_diff = float(np.nanmax(diff)) if diff.size else 0.0
        feature_ok = bool(max_diff <= tolerances[feature])
        lines.append(f"{feature} recompute_check: sample_count={sample_count}, max_abs_diff={max_diff:.8f}, ok={feature_ok}")
        ok = ok and feature_ok
    for feature, label_key in (("acc_labels", "acc_labels"), ("dec_labels", "dec_labels")):
        exact_ok = bool(np.array_equal(recomputed[feature], data[label_key][indices]))
        mismatch_count = int(np.sum(recomputed[feature] != data[label_key][indices]))
        lines.append(f"{feature} recompute_check: sample_count={sample_count}, exact_ok={exact_ok}, mismatch_count={mismatch_count}")
        ok = ok and exact_ok

    bv_class_ok = bool(
        np.array_equal(
            recomputed["baseline_variability_class"],
            data["baseline_variability_class_labels"][indices],
        )
    )
    lines.append(f"baseline_variability_class recompute_check: sample_count={sample_count}, ok={bv_class_ok}")
    ok = ok and bv_class_ok
    lines.append(f"split_ok: {ok}")
    return lines, ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Check clinical multitask dataset")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "clinical_multitask_physiology_v1"),
    )
    parser.add_argument(
        "--report_path",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "summary" / "clinical_main" / "multitask_data_check_report.txt"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--sample_count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = str(resolve_repo_path(args.data_dir))
    report_path = str(resolve_repo_path(args.report_path))
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_ok = True
    lines = ["Clinical multitask dataset check report", f"data_dir: {data_dir}", ""]
    for split in args.splits:
        split_lines, split_ok = check_split(data_dir, split, args.sample_count, rng)
        lines.extend(split_lines)
        lines.append("")
        all_ok = all_ok and split_ok
    lines.append(f"overall_ok: {all_ok}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Check report saved to {report_path}")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
