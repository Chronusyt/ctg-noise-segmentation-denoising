"""Analyze the performance gap from GT-mask constrained editing to pred-mask constrained editing."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, resolve_repo_path


LOWER_IS_BETTER = {
    "overall_mse",
    "corrupted_region_mse",
    "clean_region_mse",
    "boundary_near_clean_mse",
    "far_from_mask_clean_mse",
    "overall_mae",
    "corrupted_region_mae",
    "clean_region_mae",
    "boundary_near_clean_mae",
    "far_from_mask_clean_mae",
    "baseline_mae",
    "stv_mae",
    "ltv_mae",
    "bv_mae",
}


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_baseline_pred_guided(summary_dir: str) -> dict:
    path = os.path.join(summary_dir, "pred_mask_test_metrics.json")
    data = load_json(path)
    overall = data.get("overall", {})
    features = data.get("feature_preservation", {})
    return {
        "overall_mse": overall.get("overall_mse"),
        "corrupted_region_mse": overall.get("corrupted_region_mse"),
        "clean_region_mse": overall.get("clean_region_mse"),
        "overall_mae": overall.get("overall_mae"),
        "corrupted_region_mae": overall.get("corrupted_region_mae"),
        "clean_region_mae": overall.get("clean_region_mae"),
        "baseline_mae": features.get("baseline_mae"),
        "stv_mae": features.get("stv_mae"),
        "ltv_mae": features.get("ltv_mae"),
        "bv_mae": None,
    }


def extract_multitask_metrics(path: str) -> dict:
    data = load_json(path)
    recon = data.get("reconstruction", {})
    events = data.get("event_prediction", {})
    scalar = data.get("scalar_head", {})
    derived = data.get("reconstructed_signal_derived_features", {})
    return {
        "path": path,
        "metadata": data.get("metadata", {}),
        "reconstruction": recon,
        "acc": events.get("acceleration", {}),
        "dec": events.get("deceleration", {}),
        "scalar_head": scalar,
        "derived": derived,
    }


def delta(pred_value: float | None, gt_value: float | None) -> float | None:
    if pred_value is None or gt_value is None:
        return None
    return float(pred_value - gt_value)


def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:.4f}"


def choose_worst_overlap_class(test_summary: dict) -> tuple[str, dict]:
    per_class = test_summary.get("per_class", {})
    return min(per_class.items(), key=lambda item: item[1].get("dice", 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze GT-mask vs pred-mask constrained editing gap")
    parser.add_argument(
        "--baseline_summary_dir",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "summary" / "clinical_parallel_20260407_140308"),
    )
    parser.add_argument(
        "--gt_metrics",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "physiological_multitask" / "clinical_v2_2_gt_mask_constrained_A" / "eval" / "test_metrics.json"),
    )
    parser.add_argument(
        "--v1_metrics",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "physiological_multitask" / "clinical_v1_no_mask" / "eval" / "test_metrics.json"),
    )
    parser.add_argument("--pred_metrics", type=str, required=True)
    parser.add_argument("--pred_mask_build_summary", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    args.baseline_summary_dir = str(resolve_repo_path(args.baseline_summary_dir))
    args.gt_metrics = str(resolve_repo_path(args.gt_metrics))
    args.v1_metrics = str(resolve_repo_path(args.v1_metrics))
    args.pred_metrics = str(resolve_repo_path(args.pred_metrics))
    args.pred_mask_build_summary = str(resolve_repo_path(args.pred_mask_build_summary))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    gt = extract_multitask_metrics(args.gt_metrics)
    pred = extract_multitask_metrics(args.pred_metrics)
    v1 = extract_multitask_metrics(args.v1_metrics)
    pred_guided = load_baseline_pred_guided(args.baseline_summary_dir)
    build_summary = load_json(args.pred_mask_build_summary)
    test_summary = next(item for item in build_summary if item.get("split") == "test")
    worst_class_name, worst_class = choose_worst_overlap_class(test_summary)

    recon_keys = [
        "overall_mse",
        "corrupted_region_mse",
        "clean_region_mse",
        "boundary_near_clean_mse",
        "far_from_mask_clean_mse",
        "overall_mae",
        "corrupted_region_mae",
        "clean_region_mae",
        "boundary_near_clean_mae",
        "far_from_mask_clean_mae",
    ]
    event_keys = ["precision", "recall", "f1", "positive_rate_pred", "positive_rate_gt"]
    derived_keys = ["baseline_mae", "stv_mae", "ltv_mae", "bv_mae"]

    gap = {
        "gt_to_pred_reconstruction": {key: delta(pred["reconstruction"].get(key), gt["reconstruction"].get(key)) for key in recon_keys},
        "pred_vs_v1_reconstruction": {key: delta(pred["reconstruction"].get(key), v1["reconstruction"].get(key)) for key in recon_keys},
        "pred_vs_pred_guided_reconstruction": {
            key: delta(pred["reconstruction"].get(key), pred_guided.get(key))
            for key in ("overall_mse", "corrupted_region_mse", "clean_region_mse", "overall_mae", "corrupted_region_mae", "clean_region_mae")
        },
        "gt_to_pred_events": {
            "acceleration": {key: delta(pred["acc"].get(key), gt["acc"].get(key)) for key in event_keys},
            "deceleration": {key: delta(pred["dec"].get(key), gt["dec"].get(key)) for key in event_keys},
        },
        "gt_to_pred_derived_features": {key: delta(pred["derived"].get(key), gt["derived"].get(key)) for key in derived_keys},
        "pred_mask_quality_test": test_summary,
    }

    worst_recon_metric = max(
        ("corrupted_region_mse", "clean_region_mse", "boundary_near_clean_mse", "far_from_mask_clean_mse"),
        key=lambda key: abs(gap["gt_to_pred_reconstruction"].get(key) or 0.0),
    )
    worst_derived_metric = max(derived_keys, key=lambda key: abs(gap["gt_to_pred_derived_features"].get(key) or 0.0))

    lines = [
        "GT -> pred constrained editing gap report",
        "=====================================",
        "",
        f"gt_metrics: {args.gt_metrics}",
        f"pred_metrics: {args.pred_metrics}",
        f"v1_metrics: {args.v1_metrics}",
        f"pred_guided_summary_dir: {args.baseline_summary_dir}",
        f"pred_mask_build_summary: {args.pred_mask_build_summary}",
        "",
        "[核心结论]",
        f"- GT -> pred 最大的 reconstruction 变化项: {worst_recon_metric}, delta={fmt(gap['gt_to_pred_reconstruction'][worst_recon_metric])}",
        f"- GT -> pred 最大的 derived feature 变化项: {worst_derived_metric}, delta={fmt(gap['gt_to_pred_derived_features'][worst_derived_metric])}",
        f"- pred-mask test union IoU/Dice: {fmt(test_summary['union']['iou'])} / {fmt(test_summary['union']['dice'])}",
        f"- pred-mask 最弱类别: {worst_class_name}, IoU={fmt(worst_class['iou'])}, Dice={fmt(worst_class['dice'])}",
        "",
        "[GT -> pred reconstruction delta]  (pred - gt; 正值表示 pred 更差，负值表示 pred 更好)",
    ]
    for key in recon_keys:
        lines.append(f"- {key}: {fmt(gap['gt_to_pred_reconstruction'][key])}")

    lines.extend(
        [
            "",
            "[pred vs v1 reconstruction delta]  (pred - v1; 正值表示 pred 更差，负值表示 pred 更好)",
        ]
    )
    for key in recon_keys:
        lines.append(f"- {key}: {fmt(gap['pred_vs_v1_reconstruction'][key])}")

    lines.extend(
        [
            "",
            "[pred vs pred-mask guided reconstruction delta]  (pred - pred_guided; 正值表示 pred 更差，负值表示 pred 更好)",
        ]
    )
    for key, value in gap["pred_vs_pred_guided_reconstruction"].items():
        lines.append(f"- {key}: {fmt(value)}")

    lines.extend(["", "[GT -> pred event delta]  (pred - gt)"])
    for event_name in ("acceleration", "deceleration"):
        lines.append(f"- {event_name}:")
        for key in event_keys:
            lines.append(f"  - {key}: {fmt(gap['gt_to_pred_events'][event_name][key])}")

    lines.extend(["", "[GT -> pred derived physiology delta]  (pred - gt; 正值表示 pred 更差)"])
    for key in derived_keys:
        lines.append(f"- {key}: {fmt(gap['gt_to_pred_derived_features'][key])}")

    lines.extend(
        [
            "",
            "[pred-mask 质量诊断 | test split]",
            f"- union soft_mean_probability: {fmt(test_summary['union']['soft_mean_probability'])}",
            f"- union hard_positive_ratio: {fmt(test_summary['union']['hard_positive_ratio'])}",
            f"- union gt_positive_ratio: {fmt(test_summary['union']['gt_positive_ratio'])}",
            f"- union IoU: {fmt(test_summary['union']['iou'])}",
            f"- union Dice: {fmt(test_summary['union']['dice'])}",
            f"- worst per-class overlap: {worst_class_name} | IoU={fmt(worst_class['iou'])}, Dice={fmt(worst_class['dice'])}",
        ]
    )

    txt_path = os.path.join(args.output_dir, "gt_vs_pred_gap_report.txt")
    json_path = os.path.join(args.output_dir, "gt_vs_pred_gap_report.json")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(gap, f, indent=2, ensure_ascii=False)
    print(f"GT -> pred gap report saved to {txt_path}", flush=True)


if __name__ == "__main__":
    main()
