"""Diagnose clinical feature-preservation error distributions."""
from __future__ import annotations

import argparse
import csv
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

from ctg_pipeline.features.physiology import FeatureConfig, compute_signal_features
from ctg_pipeline.models.unet1d_denoiser import UNet1DDenoiser
from ctg_pipeline.models.unet1d_mask_guided_denoiser import UNet1DMaskGuidedDenoiser
from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_DATASETS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path


METHODS = {
    "direct": ("Direct denoising", "denoising_baseline_clinical"),
    "pred": ("Pred-mask guided", "multilabel_guided_denoising_clinical_pred"),
    "gt": ("GT-mask oracle", "multilabel_guided_denoising_clinical_gt"),
}


def load_metrics(result_dir: str) -> Dict[str, float]:
    path = os.path.join(result_dir, "eval", "test_metrics.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "learned_denoiser" in data:
        overall = data["learned_denoiser"].get("overall", {})
        features = data["learned_denoiser"].get("feature_preservation", {})
    else:
        overall = data.get("overall", {})
        features = data.get("feature_preservation", {})
    merged = dict(overall)
    merged.update(features)
    return {k: float(v) for k, v in merged.items()}


def load_direct_test(datasets_root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = os.path.join(datasets_root, "denoising_baseline_clinical", "test_dataset_denoising.npz")
    data = np.load(path)
    return (
        np.asarray(data["noisy_signals"], dtype=np.float32),
        np.asarray(data["clean_signals"], dtype=np.float32),
        np.asarray(data["artifact_labels"], dtype=np.float32),
    )


def load_masks(datasets_root: str, source: str) -> np.ndarray:
    path = os.path.join(datasets_root, f"multilabel_guided_denoising_clinical_{source}", "test_dataset_mask_guided.npz")
    return np.asarray(np.load(path)["masks"], dtype=np.float32)


def predict_direct(model_path: str, noisy: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model = UNet1DDenoiser(in_channels=1, out_channels=1, base_channels=32, depth=3)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        for start in range(0, noisy.shape[0], batch_size):
            x = torch.from_numpy(noisy[start : start + batch_size, np.newaxis, :]).to(device)
            preds.append(model(x).cpu().numpy()[:, 0, :])
    return np.concatenate(preds, axis=0)


def predict_mask_guided(model_path: str, noisy: np.ndarray, masks: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model = UNet1DMaskGuidedDenoiser(in_channels=6, out_channels=1, base_channels=32, depth=3)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model = model.to(device).eval()
    preds = []
    model_input = np.concatenate([noisy[:, np.newaxis, :], np.transpose(masks, (0, 2, 1))], axis=1).astype(np.float32)
    with torch.no_grad():
        for start in range(0, model_input.shape[0], batch_size):
            x = torch.from_numpy(model_input[start : start + batch_size]).to(device)
            preds.append(model(x).cpu().numpy()[:, 0, :])
    return np.concatenate(preds, axis=0)


def summarize_abs_errors(abs_err: np.ndarray) -> Dict[str, float]:
    finite = abs_err[np.isfinite(abs_err)]
    if finite.size == 0:
        return {k: float("nan") for k in ("mean", "median", "p90", "p95", "p99", "max", "top1pct_share")}
    n_top = max(1, int(np.ceil(finite.size * 0.01)))
    top = np.partition(finite, -n_top)[-n_top:]
    total = float(np.sum(finite))
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
        "top1pct_share": float(np.sum(top) / total) if total > 0 else 0.0,
    }


def summarize_bias(diff: np.ndarray) -> Dict[str, float]:
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return {k: float("nan") for k in ("mean", "median", "p05", "p25", "p75", "p95")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "p95": float(np.percentile(finite, 95)),
    }


def format_delta(method: str, metric: str, metrics: dict) -> str:
    return f"{metrics[method][metric] - metrics['direct'][metric]:+.4f}"


def write_report(report_path: str, metrics: dict, summaries: dict, bias_summaries: dict, top_rows: list[dict], n_samples: int) -> None:
    lines = [
        "Clinical feature-preservation error diagnosis",
        f"n_samples: {n_samples}",
        "",
        "Current aggregate metrics:",
    ]
    for key, (name, _subdir) in METHODS.items():
        lines.append(
            f"- {name}: overall_mse={metrics[key]['overall_mse']:.4f}, "
            f"overall_mae={metrics[key]['overall_mae']:.4f}, "
            f"baseline_mae={metrics[key]['baseline_mae']:.4f}, "
            f"stv_mae={metrics[key]['stv_mae']:.4f}, "
            f"ltv_mae={metrics[key]['ltv_mae']:.4f}"
        )
    lines.extend(["", "Interpretation checks:"])
    for method in ("pred", "gt"):
        lines.append(
            f"- {METHODS[method][0]} vs direct: "
            f"overall_mse {format_delta(method, 'overall_mse', metrics)}, "
            f"overall_mae {format_delta(method, 'overall_mae', metrics)}, "
            f"baseline_mae {format_delta(method, 'baseline_mae', metrics)}, "
            f"stv_mae {format_delta(method, 'stv_mae', metrics)}, "
            f"ltv_mae {format_delta(method, 'ltv_mae', metrics)}"
        )
    lines.extend(["", "Feature absolute-error distribution:"])
    for method in ("direct", "pred", "gt"):
        lines.append(f"[{METHODS[method][0]}]")
        for feature in ("baseline", "stv", "ltv"):
            s = summaries[method][feature]
            lines.append(
                f"  {feature}: mean={s['mean']:.4f}, median={s['median']:.4f}, "
                f"p95={s['p95']:.4f}, p99={s['p99']:.4f}, max={s['max']:.4f}, "
                f"top1pct_share={s['top1pct_share']:.3f}"
            )
    lines.extend(["", "Feature bias distribution (reconstructed - clean):"])
    for method in ("direct", "pred", "gt"):
        lines.append(f"[{METHODS[method][0]}]")
        for feature in ("baseline", "stv", "ltv"):
            s = bias_summaries[method][feature]
            lines.append(
                f"  {feature}: mean={s['mean']:.4f}, median={s['median']:.4f}, "
                f"p05={s['p05']:.4f}, p95={s['p95']:.4f}"
            )
    lines.extend([
        "",
        "Outlier readout:",
        "- top1pct_share > 0.20 suggests a small tail materially contributes to feature error.",
        "- Inspect feature_error_outliers.csv for the largest per-sample errors.",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose clinical feature-preservation errors")
    parser.add_argument("--datasets_root", type=str, default=str(DENOISING_DATASETS_ROOT))
    parser.add_argument("--results_root", type=str, default=str(DENOISING_RESULTS_ROOT))
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "summary" / "clinical_main"),
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means use all test samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets_root = str(resolve_repo_path(args.datasets_root))
    results_root = str(resolve_repo_path(args.results_root))
    summary_dir = str(resolve_repo_path(args.summary_dir))
    os.makedirs(summary_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    noisy, clean, _labels = load_direct_test(datasets_root)
    if args.max_samples and args.max_samples < noisy.shape[0]:
        rng = np.random.default_rng(args.seed)
        indices = np.sort(rng.choice(noisy.shape[0], size=args.max_samples, replace=False))
        noisy = noisy[indices]
        clean = clean[indices]
    else:
        indices = np.arange(noisy.shape[0])

    pred_masks = load_masks(datasets_root, "pred")[indices]
    gt_masks = load_masks(datasets_root, "gt")[indices]

    metrics = {key: load_metrics(os.path.join(results_root, subdir)) for key, (_name, subdir) in METHODS.items()}

    print("Predicting direct...", flush=True)
    direct = predict_direct(os.path.join(results_root, "denoising_baseline_clinical", "best_model.pt"), noisy, device, args.batch_size)
    print("Predicting pred-mask guided...", flush=True)
    pred = predict_mask_guided(
        os.path.join(results_root, "multilabel_guided_denoising_clinical_pred", "best_model.pt"),
        noisy,
        pred_masks,
        device,
        args.batch_size,
    )
    print("Predicting GT-mask oracle...", flush=True)
    gt = predict_mask_guided(
        os.path.join(results_root, "multilabel_guided_denoising_clinical_gt", "best_model.pt"),
        noisy,
        gt_masks,
        device,
        args.batch_size,
    )

    feature_cfg = FeatureConfig(sample_rate=4.0)
    clean_features = compute_signal_features(clean, config=feature_cfg)
    recons = {"direct": direct, "pred": pred, "gt": gt}
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    bias_summaries: dict[str, dict[str, dict[str, float]]] = {}
    top_rows: list[dict] = []

    for method, recon in recons.items():
        recon_features = compute_signal_features(recon, config=feature_cfg)
        summaries[method] = {}
        bias_summaries[method] = {}
        for feature in ("baseline", "stv", "ltv"):
            diff = recon_features[feature] - clean_features[feature]
            abs_err = np.abs(diff)
            summaries[method][feature] = summarize_abs_errors(abs_err)
            bias_summaries[method][feature] = summarize_bias(diff)
            top_idx = np.argsort(-abs_err)[:10]
            for rank, local_idx in enumerate(top_idx, start=1):
                top_rows.append(
                    {
                        "method": method,
                        "feature": feature,
                        "rank": rank,
                        "sample_index": int(indices[local_idx]),
                        "abs_error": float(abs_err[local_idx]),
                        "bias": float(diff[local_idx]),
                        "clean_feature": float(clean_features[feature][local_idx]),
                        "reconstructed_feature": float(recon_features[feature][local_idx]),
                    }
                )

    report_path = os.path.join(summary_dir, "feature_error_diagnosis.txt")
    json_path = os.path.join(summary_dir, "feature_error_diagnosis.json")
    csv_path = os.path.join(summary_dir, "feature_error_outliers.csv")
    write_report(report_path, metrics, summaries, bias_summaries, top_rows, clean.shape[0])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metrics": metrics, "feature_error_summary": summaries, "feature_bias_summary": bias_summaries},
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(top_rows[0].keys()))
        writer.writeheader()
        writer.writerows(top_rows)
    print(f"Diagnosis report saved to {report_path}")
    print(f"Outlier table saved to {csv_path}")


if __name__ == "__main__":
    main()
