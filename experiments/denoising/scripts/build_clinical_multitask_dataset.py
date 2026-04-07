"""Build clinical physiological multitask data from clean clinical FHR targets."""
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

from ctg_pipeline.features.physiology import FeatureConfig, compute_multitask_physiology_labels
from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_DATASETS_ROOT, resolve_repo_path


def load_direct_split(direct_dir: str, split: str) -> np.lib.npyio.NpzFile:
    return np.load(os.path.join(direct_dir, f"{split}_dataset_denoising.npz"))


def load_mask_split(mask_dir: str, split: str) -> np.lib.npyio.NpzFile:
    return np.load(os.path.join(mask_dir, f"{split}_dataset_mask_guided.npz"))


def validate_alignment(direct: np.lib.npyio.NpzFile, gt: np.lib.npyio.NpzFile, pred: np.lib.npyio.NpzFile | None) -> list[str]:
    lines: list[str] = []
    for key in ("noisy_signals", "clean_signals", "artifact_labels", "parent_index", "chunk_index"):
        direct_gt = np.array_equal(direct[key], gt[key])
        pred_ok = True if pred is None else np.array_equal(direct[key], pred[key])
        lines.append(f"{key}: direct==gt {direct_gt}, direct==pred {pred_ok}")
        if not direct_gt or not pred_ok:
            raise ValueError(f"Alignment check failed for {key}")
    return lines


def compute_labels_with_progress(
    clean: np.ndarray,
    split: str,
    cfg: FeatureConfig,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Compute clean-signal physiological labels in chunks with readable progress logs."""
    n = clean.shape[0]
    if chunk_size <= 0 or chunk_size >= n:
        print(f"[{split}] 标签计算：一次性处理 {n} 条 clean signal", flush=True)
        return compute_multitask_physiology_labels(clean, config=cfg)

    chunks: list[dict[str, np.ndarray]] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        print(
            f"[{split}] 标签计算进度：{start}-{end} / {n} "
            f"({end / n * 100:.1f}%)",
            flush=True,
        )
        chunks.append(compute_multitask_physiology_labels(clean[start:end], config=cfg))

    keys = chunks[0].keys()
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def build_split(
    split: str,
    direct_dir: str,
    gt_mask_dir: str,
    pred_mask_dir: str | None,
    output_dir: str,
    cfg: FeatureConfig,
    chunk_size: int,
) -> dict:
    direct = load_direct_split(direct_dir, split)
    gt = load_mask_split(gt_mask_dir, split)
    pred = load_mask_split(pred_mask_dir, split) if pred_mask_dir else None
    alignment_lines = validate_alignment(direct, gt, pred)

    noisy = np.asarray(direct["noisy_signals"], dtype=np.float32)
    clean = np.asarray(direct["clean_signals"], dtype=np.float32)
    artifact_labels = np.asarray(direct["artifact_labels"], dtype=np.float32)
    gt_masks = np.asarray(gt["masks"], dtype=np.float32)
    pred_masks = np.asarray(pred["masks"], dtype=np.float32) if pred is not None else None

    print(f"[{split}] 开始基于 clean_signals 生成 physiological labels: shape={clean.shape}", flush=True)
    features = compute_labels_with_progress(clean, split=split, cfg=cfg, chunk_size=chunk_size)

    out_path = os.path.join(output_dir, f"{split}_dataset_multitask.npz")
    payload = {
        "noisy_signals": noisy,
        "clean_signals": clean,
        "masks": gt_masks,
        "artifact_labels": artifact_labels,
        "baseline_labels": features["baseline"].astype(np.float32),
        "stv_labels": features["stv"].astype(np.float32),
        "ltv_labels": features["ltv"].astype(np.float32),
        "baseline_variability_labels": features["baseline_variability"].astype(np.float32),
        "baseline_variability_class_labels": features["baseline_variability_class"].astype(np.int32),
        "acc_labels": features["acc_labels"].astype(np.uint8),
        "dec_labels": features["dec_labels"].astype(np.uint8),
        "acc_counts": features["acc_counts"].astype(np.int32),
        "dec_counts": features["dec_counts"].astype(np.int32),
        "parent_index": np.asarray(direct["parent_index"], dtype=np.int32),
        "chunk_index": np.asarray(direct["chunk_index"], dtype=np.int32),
        "feature_label_names": np.asarray(["baseline", "stv", "ltv", "baseline_variability"]),
        "event_label_names": np.asarray(["acc_label", "dec_label"]),
        "feature_label_source": np.asarray(["clean_signals"]),
        "acc_dec_rule": np.asarray(["FIGO_15x15_clean_signal"]),
        "baseline_variability_v1": np.asarray(["continuous_regression_mean_bv_bpm"]),
        "sample_rate_hz": np.asarray([cfg.sample_rate], dtype=np.float32),
    }
    if pred_masks is not None:
        payload["pred_masks"] = pred_masks

    print(f"[{split}] 保存 multitask 数据到: {out_path}", flush=True)
    np.savez_compressed(out_path, **payload)
    return {
        "split": split,
        "path": out_path,
        "n_samples": int(noisy.shape[0]),
        "signal_shape": list(noisy.shape),
        "mask_shape": list(gt_masks.shape),
        "baseline_mean": float(np.nanmean(features["baseline"])),
        "stv_mean": float(np.nanmean(features["stv"])),
        "ltv_mean": float(np.nanmean(features["ltv"])),
        "baseline_variability_mean": float(np.nanmean(features["baseline_variability"])),
        "acc_positive_ratio": float(np.mean(features["acc_labels"] > 0)),
        "dec_positive_ratio": float(np.mean(features["dec_labels"] > 0)),
        "acc_count_mean": float(np.mean(features["acc_counts"])),
        "dec_count_mean": float(np.mean(features["dec_counts"])),
        "alignment": alignment_lines,
    }


def write_description(output_dir: str, summaries: list[dict], cfg: FeatureConfig) -> None:
    md_path = os.path.join(output_dir, "DATASET.md")
    json_path = os.path.join(output_dir, "build_summary.json")
    lines = [
        "# Clinical Physiological Multitask Dataset v1",
        "",
        "All physiological feature and event labels are computed only from `clean_signals`.",
        "This dataset does not overwrite the original clinical denoising or mask-guided datasets.",
        "",
        "Fields:",
        "- `noisy_signals`: float32, shape [N, 240], noisy FHR input",
        "- `clean_signals`: float32, shape [N, 240], clean FHR reconstruction target",
        "- `masks`: float32, shape [N, 240, 5], GT multilabel artifact masks",
        "- `pred_masks`: float32, shape [N, 240, 5], predicted masks when available",
        "- `artifact_labels`: float32, shape [N, 240, 5], GT artifact labels",
        "- `baseline_labels`: float32, shape [N], mean optimized baseline label from clean FHR",
        "- `stv_labels`: float32, shape [N], event-excluded STV label from clean FHR",
        "- `ltv_labels`: float32, shape [N], event-excluded LTV label from clean FHR",
        "- `baseline_variability_labels`: float32, shape [N], continuous mean baseline-variability amplitude in bpm",
        "- `baseline_variability_class_labels`: int32, shape [N], auxiliary class id: -1 unknown, 0 absent, 1 minimal/reduced, 2 moderate/normal, 3 marked",
        "- `acc_labels`: uint8, shape [N, 240], pointwise acceleration labels from clean FHR",
        "- `dec_labels`: uint8, shape [N, 240], pointwise deceleration labels from clean FHR",
        "- `acc_counts`: int32, shape [N], number of detected acceleration events",
        "- `dec_counts`: int32, shape [N], number of detected deceleration events",
        "- `parent_index`: int32, shape [N], parent 20-min segment index",
        "- `chunk_index`: int32, shape [N], 1-min chunk index within parent",
        "",
        "Label definitions:",
        "- reconstruction target: `clean_signals`",
        "- baseline: mean of optimized FHR baseline trace from clean FHR",
        "- STV/LTV: existing pulse-interval STV/LTV implementation, excluding detected acceleration/deceleration regions",
        "- baseline variability: v1 continuous regression target, mean robust 1-min baseline-deviation amplitude in bpm",
        "- acceleration/deceleration: pointwise binary labels using existing FIGO 15x15 detectors on clean FHR",
        "",
        f"sample_rate_hz: {cfg.sample_rate}",
        "",
        "Splits:",
    ]
    for summary in summaries:
        lines.append(
            f"- {summary['split']}: n={summary['n_samples']}, "
            f"signal_shape={summary['signal_shape']}, mask_shape={summary['mask_shape']}"
        )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)


def write_label_source_report(summary_dir: str, output_dir: str) -> None:
    path = os.path.join(summary_dir, "multitask_label_source_report.txt")
    lines = [
        "Clinical physiological multitask label source report",
        "",
        "Reusable code found:",
        "- baseline: src/ctg_pipeline/preprocessing/fhr_baseline_optimized.py :: analyse_baseline_optimized, BaselineConfig",
        "- STV/LTV: src/ctg_pipeline/preprocessing/variability.py :: compute_stv_overall, compute_ltv_overall, STVConfig, LTVConfig",
        "- baseline variability: src/ctg_pipeline/preprocessing/baseline_variability.py :: compute_baseline_variability, classify_baseline_variability, BaselineVariabilityConfig",
        "- acceleration: src/ctg_pipeline/preprocessing/acc_detection_figo_v2.py :: detect_accelerations_figo, AccelerationConfig, AccelerationCriterion",
        "- deceleration: src/ctg_pipeline/preprocessing/dec_detection_figo_v2.py :: detect_decelerations_figo, DecelerationConfig, DecelerationCriterion",
        "- historical integrated feature flow: experiments/denoising/scripts/extract_features.py",
        "- unified v1 wrapper used here: src/ctg_pipeline/features/physiology.py :: compute_multitask_physiology_labels",
        "",
        "Reuse decision:",
        "- baseline/STV/LTV/BV/acc/dec are all reused from existing project modules.",
        "- No new clinical rule system is introduced in this round.",
        "- A thin wrapper was added only to make label generation consistent and reusable.",
        "",
        f"multitask_output_dir: {output_dir}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_label_definition(summary_dir: str) -> None:
    path = os.path.join(summary_dir, "multitask_label_definition.txt")
    lines = [
        "Clinical physiological multitask label definition v1",
        "",
        "Scope:",
        "- clinical data line only",
        "- labels are computed from clean target FHR only",
        "- no multitask model training is introduced in this round",
        "",
        "Labels:",
        "- clean_signal: reconstruction target, shape [L], copied from existing clinical clean target",
        "- baseline: scalar float32, shape [N], mean optimized baseline trace in bpm",
        "- STV: scalar float32, shape [N], pulse-interval STV in ms, excluding detected acc/dec regions",
        "- LTV: scalar float32, shape [N], pulse-interval LTV in ms, excluding detected acc/dec regions",
        "- baseline_variability: scalar float32, shape [N], continuous regression target, mean robust 1-min baseline-deviation amplitude in bpm",
        "- baseline_variability_class: auxiliary int32, shape [N], -1 unknown, 0 absent, 1 minimal/reduced, 2 moderate/normal, 3 marked",
        "- acc_label: uint8, shape [N,L], pointwise acceleration label from clean FHR and FIGO 15x15 detector",
        "- dec_label: uint8, shape [N,L], pointwise deceleration label from clean FHR and FIGO 15x15 detector",
        "",
        "Baseline variability decision:",
        "- v1 main label uses continuous regression rather than a primary class label.",
        "- Reason: the existing module computes a continuous variability trace in bpm, and regression is consistent with baseline/STV/LTV scalar heads.",
        "- The clinical class id is still saved as auxiliary metadata for later classification experiments.",
        "",
        "Acceleration/deceleration decision:",
        "- v1 uses pointwise binary labels rather than only sample-level counts.",
        "- Reason: the project already has FIGO v2 detectors that return binary event signals, and point labels are better aligned with future multitask temporal heads.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clinical multitask dataset labels")
    parser.add_argument(
        "--direct_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "denoising_baseline_clinical"),
    )
    parser.add_argument(
        "--gt_mask_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "multilabel_guided_denoising_clinical_gt"),
    )
    parser.add_argument(
        "--pred_mask_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "multilabel_guided_denoising_clinical_pred"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "clinical_multitask_physiology_v1"),
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "summary" / "clinical_main"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--sample_rate", type=float, default=4.0)
    parser.add_argument("--chunk_size", type=int, default=4096, help="Number of samples per label-computation chunk; <=0 disables chunking")
    parser.add_argument("--write_reports_only", action="store_true", help="Write label source/definition reports without building npz files")
    args = parser.parse_args()

    direct_dir = str(resolve_repo_path(args.direct_dir))
    gt_mask_dir = str(resolve_repo_path(args.gt_mask_dir))
    pred_mask_dir = str(resolve_repo_path(args.pred_mask_dir)) if args.pred_mask_dir else None
    output_dir = str(resolve_repo_path(args.output_dir))
    summary_dir = str(resolve_repo_path(args.summary_dir))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    cfg = FeatureConfig(sample_rate=args.sample_rate)
    write_label_source_report(summary_dir, output_dir)
    write_label_definition(summary_dir)
    if args.write_reports_only:
        print(f"Label source/definition reports saved to {summary_dir}")
        return
    summaries = [
        build_split(split, direct_dir, gt_mask_dir, pred_mask_dir, output_dir, cfg, args.chunk_size)
        for split in args.splits
    ]
    write_description(output_dir, summaries, cfg)

    report_path = os.path.join(summary_dir, "multitask_dataset_build_report.txt")
    lines = [
        "Clinical multitask dataset build report",
        f"output_dir: {output_dir}",
        f"direct_dir: {direct_dir}",
        f"gt_mask_dir: {gt_mask_dir}",
        f"pred_mask_dir: {pred_mask_dir}",
        "",
    ]
    for summary in summaries:
        lines.append(
            f"{summary['split']}: n={summary['n_samples']}, "
            f"baseline_mean={summary['baseline_mean']:.4f}, "
            f"stv_mean={summary['stv_mean']:.4f}, "
            f"ltv_mean={summary['ltv_mean']:.4f}, "
            f"baseline_variability_mean={summary['baseline_variability_mean']:.4f}, "
            f"acc_positive_ratio={summary['acc_positive_ratio']:.6f}, "
            f"dec_positive_ratio={summary['dec_positive_ratio']:.6f}, "
            f"acc_count_mean={summary['acc_count_mean']:.4f}, "
            f"dec_count_mean={summary['dec_count_mean']:.4f}"
        )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Multitask dataset written to {output_dir}")
    print(f"Build report saved to {report_path}")


if __name__ == "__main__":
    main()
