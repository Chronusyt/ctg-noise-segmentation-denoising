"""Compare physiological multitask models with fixed clinical denoising baselines."""
from __future__ import annotations

import argparse
import csv
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


METRICS = [
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
    "derived_baseline_mae",
    "derived_stv_mae",
    "derived_ltv_mae",
    "derived_bv_mae",
    "head_baseline_mae",
    "head_stv_mae",
    "head_ltv_mae",
    "head_bv_mae",
    "acc_f1",
    "dec_f1",
]
METADATA_FIELDS = [
    "input_mode",
    "experiment_variant",
    "model_variant",
    "backbone_type",
    "loss_balance_mode",
]


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if is_nan(value):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def baseline_row(name: str, path: str, direct: bool = False) -> dict:
    data = load_json(path)
    if direct:
        overall = data["learned_denoiser"].get("overall", {})
        features = data["learned_denoiser"].get("feature_preservation", {})
    else:
        overall = data.get("overall", {})
        features = data.get("feature_preservation", {})
    return {
        "method": name,
        "source": path,
        "input_mode": None,
        "experiment_variant": None,
        "model_variant": None,
        "backbone_type": None,
        "loss_balance_mode": None,
        "overall_mse": overall.get("overall_mse"),
        "corrupted_region_mse": overall.get("corrupted_region_mse"),
        "clean_region_mse": overall.get("clean_region_mse"),
        "boundary_near_clean_mse": None,
        "far_from_mask_clean_mse": None,
        "overall_mae": overall.get("overall_mae"),
        "corrupted_region_mae": overall.get("corrupted_region_mae"),
        "clean_region_mae": overall.get("clean_region_mae"),
        "boundary_near_clean_mae": None,
        "far_from_mask_clean_mae": None,
        "derived_baseline_mae": features.get("baseline_mae"),
        "derived_stv_mae": features.get("stv_mae"),
        "derived_ltv_mae": features.get("ltv_mae"),
        "derived_bv_mae": None,
        "head_baseline_mae": None,
        "head_stv_mae": None,
        "head_ltv_mae": None,
        "head_bv_mae": None,
        "acc_f1": None,
        "dec_f1": None,
    }


def infer_multitask_label(data: dict) -> str:
    metadata = data.get("metadata", {})
    architecture_variant = metadata.get("model_variant", "")
    backbone_type = metadata.get("backbone_type", "unet")
    loss_balance_mode = metadata.get("loss_balance_mode", "static")
    experiment_variant = metadata.get("experiment_variant", "")
    model_variant = metadata.get("model_variant", "")
    input_mode = metadata.get("input_mode", "")
    gate_mode = metadata.get("gate_mode", "none")
    if architecture_variant in {"legacy_single_residual", "expert_residual"}:
        experiment_map = {
            "physiological_multitask_v1_no_mask": "Physiological multitask v1 no-mask",
            "physiological_multitask_v2_gt_mask_aux": "Physiological multitask v2 gt-mask auxiliary",
            "physiological_multitask_v2_2_gt_mask_constrained_editing": "Physiological multitask v2.2 gt-mask constrained",
            "physiological_multitask_v3_pred_mask_constrained_editing": "Physiological multitask v3 pred-mask constrained",
        }
        base = (
            experiment_map.get(experiment_variant, experiment_variant)
            or ("pred-mask" if input_mode == "pred_mask" else "gt-mask" if input_mode == "gt_mask" else "no-mask")
        )
        return f"{base} | {architecture_variant} | {backbone_type} | {loss_balance_mode}"
    if model_variant == "physiological_multitask_v3_pred_mask_constrained_editing":
        return "Physiological multitask v3 (pred-mask constrained editing)"
    if model_variant == "physiological_multitask_v2_2_gt_mask_constrained_editing":
        return "Physiological multitask v2.2 (gt-mask constrained editing)"
    if model_variant == "physiological_multitask_v2_gt_mask_aux" or (input_mode == "gt_mask" and gate_mode == "none"):
        return "Physiological multitask v2 (gt-mask auxiliary)"
    if model_variant == "physiological_multitask_v1_no_mask" or input_mode == "no_mask":
        return "Physiological multitask v1 no-mask"
    return "Physiological multitask"


def multitask_row(path: str, label: str | None = None) -> dict:
    data = load_json(path)
    recon = data.get("reconstruction", {})
    scalar = data.get("scalar_head", {})
    derived = data.get("reconstructed_signal_derived_features", {})
    events = data.get("event_prediction", {})
    acc = events.get("acceleration", {})
    dec = events.get("deceleration", {})
    return {
        "method": label or infer_multitask_label(data),
        "source": path,
        "input_mode": data.get("metadata", {}).get("input_mode"),
        "experiment_variant": data.get("metadata", {}).get("experiment_variant"),
        "model_variant": data.get("metadata", {}).get("model_variant"),
        "backbone_type": data.get("metadata", {}).get("backbone_type"),
        "loss_balance_mode": data.get("metadata", {}).get("loss_balance_mode"),
        "overall_mse": recon.get("overall_mse"),
        "corrupted_region_mse": recon.get("corrupted_region_mse"),
        "clean_region_mse": recon.get("clean_region_mse"),
        "boundary_near_clean_mse": recon.get("boundary_near_clean_mse"),
        "far_from_mask_clean_mse": recon.get("far_from_mask_clean_mse"),
        "overall_mae": recon.get("overall_mae"),
        "corrupted_region_mae": recon.get("corrupted_region_mae"),
        "clean_region_mae": recon.get("clean_region_mae"),
        "boundary_near_clean_mae": recon.get("boundary_near_clean_mae"),
        "far_from_mask_clean_mae": recon.get("far_from_mask_clean_mae"),
        "derived_baseline_mae": derived.get("baseline_mae"),
        "derived_stv_mae": derived.get("stv_mae"),
        "derived_ltv_mae": derived.get("ltv_mae"),
        "derived_bv_mae": derived.get("bv_mae"),
        "head_baseline_mae": scalar.get("baseline_mae"),
        "head_stv_mae": scalar.get("stv_mae"),
        "head_ltv_mae": scalar.get("ltv_mae"),
        "head_bv_mae": scalar.get("bv_mae"),
        "acc_f1": acc.get("f1"),
        "dec_f1": dec.get("f1"),
    }


def maybe_add_multitask(rows: list[dict], path: str, label: str | None = None) -> None:
    if not path:
        return
    resolved = str(resolve_repo_path(path))
    if not os.path.isfile(resolved):
        print(f"Skip missing multitask metrics: {resolved}")
        return
    rows.append(multitask_row(resolved, label=label))


def write_markdown(path: str, rows: list[dict]) -> None:
    headers = ["method", *METADATA_FIELDS, *METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [row["method"], *[fmt(row.get(key)) for key in METADATA_FIELDS], *[fmt(row.get(key)) for key in METRICS]]
            )
            + " |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare physiological multitask models with fixed clinical baselines")
    parser.add_argument(
        "--baseline_summary_dir",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "summary" / "clinical_parallel_20260407_140308"),
    )
    parser.add_argument(
        "--v1_metrics",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "physiological_multitask" / "clinical_v1_no_mask" / "eval" / "test_metrics.json"),
    )
    parser.add_argument(
        "--v2_metrics",
        type=str,
        default="",
        help="Optional legacy v2 metrics path; leave empty to omit v2 from the comparison table.",
    )
    parser.add_argument(
        "--v2_2_metrics",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "physiological_multitask" / "clinical_v2_2_gt_mask_constrained_A" / "eval" / "test_metrics.json"),
    )
    parser.add_argument(
        "--v3_metrics",
        type=str,
        default="",
    )
    parser.add_argument(
        "--multitask_metrics",
        nargs="*",
        default=[],
        help="Additional multitask metrics json files to include in the comparison table.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "physiological_multitask" / "comparison"),
    )
    parser.add_argument("--include_gt_oracle", action="store_true")
    args = parser.parse_args()

    baseline_dir = str(resolve_repo_path(args.baseline_summary_dir))
    output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    rows = [
        baseline_row("Direct denoising", os.path.join(baseline_dir, "direct_test_metrics.json"), direct=True),
        baseline_row("Pred-mask guided denoising", os.path.join(baseline_dir, "pred_mask_test_metrics.json"), direct=False),
    ]
    if args.include_gt_oracle:
        rows.append(baseline_row("GT-mask oracle denoising", os.path.join(baseline_dir, "gt_mask_test_metrics.json"), direct=False))

    maybe_add_multitask(rows, args.v1_metrics, label="Physiological multitask v1 no-mask")
    maybe_add_multitask(rows, args.v2_2_metrics, label="Physiological multitask v2.2 (gt-mask constrained editing)")
    maybe_add_multitask(rows, args.v3_metrics, label="Physiological multitask v3 (pred-mask constrained editing)")
    maybe_add_multitask(rows, args.v2_metrics, label="Physiological multitask v2 (gt-mask auxiliary)")
    for path in args.multitask_metrics:
        maybe_add_multitask(rows, path)

    csv_path = os.path.join(output_dir, "physiological_multitask_comparison.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "source", *METADATA_FIELDS, *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(output_dir, "physiological_multitask_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(output_dir, "physiological_multitask_comparison.md")
    write_markdown(md_path, rows)

    print(f"Comparison saved to {csv_path}")
    print(f"Markdown saved to {md_path}")
    for row in rows:
        print(row["method"], {key: fmt(row.get(key)) for key in METRICS})


if __name__ == "__main__":
    main()
