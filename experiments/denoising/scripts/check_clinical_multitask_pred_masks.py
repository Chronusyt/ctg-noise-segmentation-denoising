"""Check cached predicted masks for the clinical multitask dataset."""
from __future__ import annotations

import argparse
import json
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

from ctg_pipeline.utils.pathing import DENOISING_DATASETS_ROOT, resolve_repo_path


CLASS_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]


def load_base_split(data_dir: str, split: str) -> dict:
    path = os.path.join(data_dir, f"{split}_dataset_multitask.npz")
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def load_cache_split(cache_dir: str, split: str) -> dict:
    path = os.path.join(cache_dir, f"{split}_pred_masks.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def binary_overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    return {
        "iou": tp / (tp + fp + fn + 1e-8),
        "dice": (2.0 * tp) / (pred.sum() + gt.sum() + 1e-8),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check cached predicted masks for clinical multitask data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "clinical_multitask_physiology_v1"),
    )
    parser.add_argument(
        "--pred_mask_cache_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "clinical_multitask_physiology_v1_pred_masks"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument(
        "--report_path",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "clinical_multitask_physiology_v1_pred_masks" / "pred_mask_check_report.txt"),
    )
    args = parser.parse_args()

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.pred_mask_cache_dir = str(resolve_repo_path(args.pred_mask_cache_dir))
    args.report_path = str(resolve_repo_path(args.report_path))
    os.makedirs(os.path.dirname(args.report_path), exist_ok=True)

    lines = [
        "Clinical multitask predicted-mask cache check",
        f"data_dir: {args.data_dir}",
        f"pred_mask_cache_dir: {args.pred_mask_cache_dir}",
        "",
    ]
    report = {"overall_ok": True, "splits": {}}

    for split in args.splits:
        base = load_base_split(args.data_dir, split)
        cache = load_cache_split(args.pred_mask_cache_dir, split)
        gt_masks = np.asarray(base["masks"], dtype=np.float32)

        if "sample_indices" in cache:
            indices = np.asarray(cache["sample_indices"], dtype=np.int64)
            base_gt = gt_masks[indices]
            base_parent = np.asarray(base["parent_index"], dtype=np.int32)[indices] if "parent_index" in base else None
            base_chunk = np.asarray(base["chunk_index"], dtype=np.int32)[indices] if "chunk_index" in base else None
        else:
            indices = None
            base_gt = gt_masks
            base_parent = np.asarray(base["parent_index"], dtype=np.int32) if "parent_index" in base else None
            base_chunk = np.asarray(base["chunk_index"], dtype=np.int32) if "chunk_index" in base else None

        soft = np.asarray(cache["pred_masks_soft"])
        hard = np.asarray(cache["pred_masks_hard"])
        expected_shape = base_gt.shape

        shape_ok = soft.shape == expected_shape and hard.shape == expected_shape
        soft_range_ok = bool(np.isfinite(soft).all() and soft.min() >= 0.0 and soft.max() <= 1.0)
        hard_binary_ok = bool(np.isin(hard, [0, 1]).all())
        parent_ok = True if base_parent is None or "parent_index" not in cache else np.array_equal(base_parent, cache["parent_index"])
        chunk_ok = True if base_chunk is None or "chunk_index" not in cache else np.array_equal(base_chunk, cache["chunk_index"])

        union_stats = binary_overlap_metrics(hard.max(axis=-1), base_gt.max(axis=-1))
        split_ok = bool(shape_ok and soft_range_ok and hard_binary_ok and parent_ok and chunk_ok)
        report["overall_ok"] = bool(report["overall_ok"] and split_ok)
        report["splits"][split] = {
            "shape_ok": shape_ok,
            "soft_range_ok": soft_range_ok,
            "hard_binary_ok": hard_binary_ok,
            "parent_index_ok": parent_ok,
            "chunk_index_ok": chunk_ok,
            "n_samples": int(soft.shape[0]),
            "is_subset_cache": indices is not None,
            "union_iou_hard_vs_gt": float(union_stats["iou"]),
            "union_dice_hard_vs_gt": float(union_stats["dice"]),
            "soft_mean_probability": float(np.mean(soft)),
            "hard_positive_ratio": float(np.mean(hard > 0)),
        }

        lines.extend(
            [
                f"[{split}]",
                f"shape_ok: {shape_ok}",
                f"soft_range_ok: {soft_range_ok}",
                f"hard_binary_ok: {hard_binary_ok}",
                f"parent_index_ok: {parent_ok}",
                f"chunk_index_ok: {chunk_ok}",
                f"n_samples: {soft.shape[0]}",
                f"is_subset_cache: {indices is not None}",
                f"soft_mean_probability: {np.mean(soft):.6f}",
                f"hard_positive_ratio: {np.mean(hard > 0):.6f}",
                f"union_iou_hard_vs_gt: {union_stats['iou']:.6f}",
                f"union_dice_hard_vs_gt: {union_stats['dice']:.6f}",
                "",
            ]
        )

    lines.append(f"overall_ok: {report['overall_ok']}")
    with open(args.report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    json_path = os.path.splitext(args.report_path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Pred-mask cache check report saved to {args.report_path}", flush=True)


if __name__ == "__main__":
    main()
