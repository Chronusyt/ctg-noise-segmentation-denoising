"""Build cached predicted multilabel masks for the clinical multitask dataset."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.models.unet1d_multilabel_segmentation import UNet1DMultilabelSegmentation
from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_DATASETS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path


CLASS_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]


def load_split(data_dir: str, split: str) -> Dict[str, np.ndarray]:
    path = os.path.join(data_dir, f"{split}_dataset_multitask.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def maybe_subset(arrays: Dict[str, np.ndarray], max_samples: int, seed: int) -> tuple[Dict[str, np.ndarray], np.ndarray | None, int]:
    n_total = int(arrays["noisy_signals"].shape[0])
    if max_samples <= 0 or max_samples >= n_total:
        return arrays, None, n_total
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_total, size=max_samples, replace=False))
    out = {}
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n_total,):
            out[key] = value[indices]
        else:
            out[key] = value
    return out, indices.astype(np.int32), n_total


def load_model(model_path: str, device: str) -> UNet1DMultilabelSegmentation:
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = UNet1DMultilabelSegmentation(
        in_channels=int(cfg.get("in_channels", 1)),
        out_channels=int(cfg.get("out_channels", 5)),
        base_channels=int(cfg.get("base_channels", 32)),
        depth=int(cfg.get("depth", 3)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    return model


def predict_masks(model: torch.nn.Module, noisy_signals: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    probs = []
    n = noisy_signals.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            print(f"  预测进度：{start}-{end} / {n} ({end / n * 100:.1f}%)", flush=True)
            x = torch.from_numpy(noisy_signals[start:end, np.newaxis, :].astype(np.float32)).to(device)
            logits = model(x)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs, axis=0)  # [N, 5, L]
    return np.transpose(probs, (0, 2, 1))  # [N, L, 5]


def binary_overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    iou = tp / (tp + fp + fn + 1e-8)
    dice = (2.0 * tp) / (pred.sum() + gt.sum() + 1e-8)
    return {"iou": float(iou), "dice": float(dice), "tp": tp, "fp": fp, "fn": fn}


def summarize_split(
    split: str,
    soft_masks: np.ndarray,
    hard_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float,
    n_total: int,
    sample_indices: np.ndarray | None,
) -> dict:
    summary = {
        "split": split,
        "n_samples": int(soft_masks.shape[0]),
        "n_total_source_samples": int(n_total),
        "is_subset_cache": bool(sample_indices is not None),
        "soft_mean_probability": float(np.mean(soft_masks)),
        "hard_positive_ratio": float(np.mean(hard_masks > 0)),
        "hard_threshold": float(threshold),
        "per_class": {},
    }

    union_soft = soft_masks.max(axis=-1)
    union_hard = hard_masks.max(axis=-1)
    union_gt = gt_masks.max(axis=-1)
    summary["union"] = {
        "soft_mean_probability": float(np.mean(union_soft)),
        "hard_positive_ratio": float(np.mean(union_hard > 0)),
        "gt_positive_ratio": float(np.mean(union_gt > 0)),
        **binary_overlap_metrics(union_hard, union_gt),
    }

    for class_idx, name in enumerate(CLASS_NAMES):
        summary["per_class"][name] = {
            "soft_mean_probability": float(np.mean(soft_masks[..., class_idx])),
            "hard_positive_ratio": float(np.mean(hard_masks[..., class_idx] > 0)),
            "gt_positive_ratio": float(np.mean(gt_masks[..., class_idx] > 0)),
            **binary_overlap_metrics(hard_masks[..., class_idx], gt_masks[..., class_idx]),
        }
    return summary


def write_outputs(output_dir: str, summaries: list[dict], soft_dtype: str, threshold: float) -> None:
    json_path = os.path.join(output_dir, "build_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    report_path = os.path.join(output_dir, "pred_mask_build_report.txt")
    lines = [
        "Clinical multitask predicted-mask cache report",
        "",
        f"soft_dtype: {soft_dtype}",
        f"hard_threshold: {threshold}",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"[{summary['split']}]",
                f"n_samples: {summary['n_samples']} / source_total={summary['n_total_source_samples']}",
                f"is_subset_cache: {summary['is_subset_cache']}",
                f"soft_mean_probability: {summary['soft_mean_probability']:.6f}",
                f"hard_positive_ratio: {summary['hard_positive_ratio']:.6f}",
                f"union_iou_hard_vs_gt: {summary['union']['iou']:.6f}",
                f"union_dice_hard_vs_gt: {summary['union']['dice']:.6f}",
                "",
            ]
        )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    md_path = os.path.join(output_dir, "DATASET.md")
    md_lines = [
        "# Clinical Multitask Predicted Mask Cache",
        "",
        "This cache stores predicted multilabel artifact masks generated by the frozen clinical multilabel segmentation model.",
        "It does not overwrite the original multitask dataset.",
        "",
        "Files:",
        "- `train_pred_masks.npz` / `val_pred_masks.npz` / `test_pred_masks.npz`",
        "- `build_summary.json`",
        "- `pred_mask_build_report.txt`",
        "",
        "Fields per split file:",
        "- `pred_masks_soft`: shape `[N, 240, 5]`, soft probabilities from sigmoid logits",
        "- `pred_masks_hard`: shape `[N, 240, 5]`, thresholded binary masks",
        "- `parent_index`: shape `[N]`, copied from multitask dataset when available",
        "- `chunk_index`: shape `[N]`, copied from multitask dataset when available",
        "- `sample_indices`: shape `[N]`, only present for subset smoke caches",
        "- `class_names`: shape `[5]`",
        "- `hard_threshold`: scalar threshold used to build hard masks",
        "",
        f"soft_dtype: `{soft_dtype}`",
        f"hard_threshold: `{threshold}`",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cached predicted masks for clinical multitask data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "clinical_multitask_physiology_v1"),
    )
    parser.add_argument(
        "--segmentation_model",
        type=str,
        default=str(DENOISING_RESULTS_ROOT / "multilabel_segmentation_clinical" / "best_model.pt"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "clinical_multitask_physiology_v1_pred_masks"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--hard_threshold", type=float, default=0.5)
    parser.add_argument("--soft_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max_samples_per_split", type=int, default=0, help="Smoke only; 0 means full split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.segmentation_model = str(resolve_repo_path(args.segmentation_model))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("Clinical multitask predicted mask cache 构建启动", flush=True)
    print(f"使用设备: {device}", flush=True)
    print(f"数据目录: {args.data_dir}", flush=True)
    print(f"segmentation 模型: {args.segmentation_model}", flush=True)
    print(f"输出目录: {args.output_dir}", flush=True)
    print(f"soft dtype: {args.soft_dtype}", flush=True)
    print(f"hard threshold: {args.hard_threshold}", flush=True)
    print(f"max_samples_per_split: {args.max_samples_per_split}", flush=True)

    model = load_model(args.segmentation_model, device)
    soft_dtype = np.float16 if args.soft_dtype == "float16" else np.float32
    summaries = []

    for split_idx, split in enumerate(args.splits):
        print(f"[{split}] 加载 multitask 数据...", flush=True)
        arrays = load_split(args.data_dir, split)
        arrays, sample_indices, n_total = maybe_subset(arrays, args.max_samples_per_split, args.seed + split_idx)
        noisy = np.asarray(arrays["noisy_signals"], dtype=np.float32)
        gt_masks = np.asarray(arrays["masks"], dtype=np.float32)
        print(f"[{split}] noisy shape={noisy.shape}, gt mask shape={gt_masks.shape}", flush=True)

        soft_masks = predict_masks(model, noisy, device, args.batch_size)
        hard_masks = (soft_masks >= float(args.hard_threshold)).astype(np.uint8)
        summary = summarize_split(
            split,
            soft_masks=soft_masks,
            hard_masks=hard_masks,
            gt_masks=gt_masks,
            threshold=args.hard_threshold,
            n_total=n_total,
            sample_indices=sample_indices,
        )
        summaries.append(summary)

        out_path = os.path.join(args.output_dir, f"{split}_pred_masks.npz")
        payload = {
            "pred_masks_soft": soft_masks.astype(soft_dtype),
            "pred_masks_hard": hard_masks.astype(np.uint8),
            "class_names": np.asarray(CLASS_NAMES),
            "hard_threshold": np.asarray([args.hard_threshold], dtype=np.float32),
        }
        for key in ("parent_index", "chunk_index"):
            if key in arrays:
                payload[key] = np.asarray(arrays[key], dtype=np.int32)
        if sample_indices is not None:
            payload["sample_indices"] = sample_indices
            payload["n_total_source_samples"] = np.asarray([n_total], dtype=np.int32)
        np.savez_compressed(out_path, **payload)
        print(f"[{split}] pred mask cache 已保存到: {out_path}", flush=True)

    write_outputs(args.output_dir, summaries, soft_dtype=args.soft_dtype, threshold=args.hard_threshold)
    print(f"Pred-mask cache 构建完成: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
