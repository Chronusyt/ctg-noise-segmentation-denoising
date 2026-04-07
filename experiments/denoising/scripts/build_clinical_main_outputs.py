"""
Build paper-ready clinical main tables and one comparison figure.

This script does not train models. It reads existing clinical evaluation
results and checkpoints for:
1. Direct denoising
2. Pred-mask guided denoising
3. GT-mask oracle denoising
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.evaluation.feature_preservation import FeatureConfig, compute_signal_features, feature_title
from ctg_pipeline.models.unet1d_denoiser import UNet1DDenoiser
from ctg_pipeline.models.unet1d_mask_guided_denoiser import UNet1DMaskGuidedDenoiser
from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_DATASETS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path

METHODS = [
    ("direct", "Direct denoising", "denoising_baseline_clinical"),
    ("pred", "Pred-mask guided", "multilabel_guided_denoising_clinical_pred"),
    ("gt", "GT-mask oracle", "multilabel_guided_denoising_clinical_gt"),
]

TABLE_METRICS = [
    "overall_mse",
    "overall_mae",
    "corrupted_region_mse",
    "corrupted_region_mae",
    "clean_region_mse",
    "clean_region_mae",
    "baseline_mae",
    "stv_mae",
    "ltv_mae",
]


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and value != value


def load_metrics(result_dir: str) -> dict | None:
    for path in (
        os.path.join(result_dir, "test_metrics.json"),
        os.path.join(result_dir, "eval", "test_metrics.json"),
    ):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def resolve_result_dir(results_roots: list[str], subdir: str) -> Tuple[str, dict | None]:
    for root in results_roots:
        result_dir = os.path.join(root, subdir)
        data = load_metrics(result_dir)
        if data is not None:
            return result_dir, data
    return os.path.join(results_roots[0], subdir), None


def extract_metrics(data: dict | None) -> Dict[str, float]:
    if data is None:
        return {key: float("nan") for key in TABLE_METRICS}
    if "learned_denoiser" in data:
        overall = data["learned_denoiser"].get("overall", {})
        features = data["learned_denoiser"].get("feature_preservation", {})
    else:
        overall = data.get("overall", {})
        features = data.get("feature_preservation", {})
    merged = dict(overall)
    merged.update(features)
    return {key: float(merged.get(key, np.nan)) for key in TABLE_METRICS}


def write_tables(rows: list[dict], table_dir: str) -> None:
    os.makedirs(table_dir, exist_ok=True)
    csv_path = os.path.join(table_dir, "clinical_main_results.csv")
    md_path = os.path.join(table_dir, "clinical_main_results.md")
    json_path = os.path.join(table_dir, "clinical_main_results.json")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "source_dir", *TABLE_METRICS])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    headers = ["method", *TABLE_METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [row["method"]]
        values.extend("N/A" if _is_nan(row[key]) else f"{row[key]:.4f}" for key in TABLE_METRICS)
        lines.append("| " + " | ".join(values) + " |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Tables saved to {table_dir}")


def load_direct_dataset(datasets_root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = os.path.join(datasets_root, "denoising_baseline_clinical", "test_dataset_denoising.npz")
    data = np.load(path)
    return (
        np.asarray(data["noisy_signals"], dtype=np.float32),
        np.asarray(data["clean_signals"], dtype=np.float32),
        np.asarray(data["artifact_labels"], dtype=np.float32),
    )


def load_mask_dataset(datasets_root: str, source: str) -> np.ndarray:
    path = os.path.join(
        datasets_root,
        f"multilabel_guided_denoising_clinical_{source}",
        "test_dataset_mask_guided.npz",
    )
    data = np.load(path)
    return np.asarray(data["masks"], dtype=np.float32)


def load_checkpoint_path(results_roots: list[str], subdir: str) -> str:
    for root in results_roots:
        path = os.path.join(root, subdir, "best_model.pt")
        if os.path.isfile(path):
            return path
    return os.path.join(results_roots[0], subdir, "best_model.pt")


def predict_direct(model_path: str, noisy: np.ndarray, device: str) -> np.ndarray:
    model = UNet1DDenoiser(in_channels=1, out_channels=1, base_channels=32, depth=3)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model = model.to(device).eval()
    x = torch.from_numpy(noisy[np.newaxis, np.newaxis, :].astype(np.float32)).to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()[0, 0]


def predict_mask_guided(model_path: str, noisy: np.ndarray, masks: np.ndarray, device: str) -> np.ndarray:
    model = UNet1DMaskGuidedDenoiser(in_channels=6, out_channels=1, base_channels=32, depth=3)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model = model.to(device).eval()
    model_input = np.concatenate(
        [noisy[np.newaxis, np.newaxis, :], np.transpose(masks[np.newaxis, :, :], (0, 2, 1))],
        axis=1,
    ).astype(np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(model_input).to(device)).cpu().numpy()[0, 0]


def plot_clinical_comparison(
    noisy: np.ndarray,
    clean: np.ndarray,
    labels: np.ndarray,
    direct: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    sample_index: int,
    save_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figure")
        return

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    series = [
        ("Noisy input", noisy, None),
        ("Clean target", clean, None),
        ("Direct denoising", direct, clean),
        ("Pred-mask guided", pred, clean),
        ("GT-mask oracle", gt, clean),
    ]
    feature_cfg = FeatureConfig(sample_rate=4.0)
    all_features = compute_signal_features(np.stack([s for _name, s, _ref in series]), config=feature_cfg)
    clean_features = compute_signal_features(clean, config=feature_cfg)
    noise_mask = (labels > 0.5).any(axis=-1)
    x = np.arange(len(clean))

    fig, axes = plt.subplots(len(series), 1, figsize=(15, 12), sharex=True)
    for i, (name, signal, reference) in enumerate(series):
        ax = axes[i]
        ax.plot(x, signal, color="black" if i < 2 else "#1f77b4", linewidth=0.8)
        ax.fill_between(x, 45, 225, where=noise_mask, color="red", alpha=0.15)
        ax.axhline(all_features["baseline"][i], color="orange", linestyle="--", linewidth=1.0, alpha=0.9)
        title = f"{name} | {feature_title(all_features, i)}"
        if reference is not None:
            mae = float(np.mean(np.abs(signal - reference)))
            title += (
                f" | signal MAE={mae:.3f}"
                f" | dB={all_features['baseline'][i] - clean_features['baseline'][0]:+.3f}"
                f" dSTV={all_features['stv'][i] - clean_features['stv'][0]:+.3f}"
                f" dLTV={all_features['ltv'][i] - clean_features['ltv'][0]:+.3f}"
            )
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("FHR")
        ax.set_ylim(45, 225)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (samples @4Hz)")
    fig.suptitle(f"Clinical main reconstruction comparison | test sample index={sample_index}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clinical main paper table and figure")
    parser.add_argument("--datasets_root", type=str, default=str(DENOISING_DATASETS_ROOT))
    parser.add_argument("--results_root", type=str, default=str(DENOISING_RESULTS_ROOT))
    parser.add_argument(
        "--fallback_results_root",
        action="append",
        default=None,
        help="Extra result roots, e.g. artifacts/runs/results/denoising",
    )
    parser.add_argument(
        "--table_dir",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "paper_tables" / "clinical_main"),
    )
    parser.add_argument(
        "--figure_dir",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "paper_figures" / "clinical_main"),
    )
    parser.add_argument("--sample_index", type=int, default=-1, help="-1 selects the noisiest clinical test segment")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    datasets_root = str(resolve_repo_path(args.datasets_root))
    results_root = str(resolve_repo_path(args.results_root))
    table_dir = str(resolve_repo_path(args.table_dir))
    figure_dir = str(resolve_repo_path(args.figure_dir))
    fallback_roots = args.fallback_results_root or [str(ARTIFACTS_ROOT / "runs" / "results" / "denoising")]
    results_roots = [results_root]
    for root in fallback_roots:
        resolved = str(resolve_repo_path(root))
        if resolved not in results_roots:
            results_roots.append(resolved)

    rows = []
    for _key, method_name, subdir in METHODS:
        source_dir, data = resolve_result_dir(results_roots, subdir)
        rows.append({"method": method_name, "source_dir": source_dir, **extract_metrics(data)})
    write_tables(rows, table_dir)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    noisy, clean, labels = load_direct_dataset(datasets_root)
    noise_coverage = (labels > 0.5).any(axis=-1).mean(axis=1)
    sample_index = int(np.argmax(noise_coverage)) if args.sample_index < 0 else int(args.sample_index)
    sample_index = max(0, min(sample_index, noisy.shape[0] - 1))

    pred_masks = load_mask_dataset(datasets_root, "pred")
    gt_masks = load_mask_dataset(datasets_root, "gt")
    direct_model = load_checkpoint_path(results_roots, "denoising_baseline_clinical")
    pred_model = load_checkpoint_path(results_roots, "multilabel_guided_denoising_clinical_pred")
    gt_model = load_checkpoint_path(results_roots, "multilabel_guided_denoising_clinical_gt")

    direct_recon = predict_direct(direct_model, noisy[sample_index], device)
    pred_recon = predict_mask_guided(pred_model, noisy[sample_index], pred_masks[sample_index], device)
    gt_recon = predict_mask_guided(gt_model, noisy[sample_index], gt_masks[sample_index], device)

    fig_path = os.path.join(figure_dir, f"clinical_main_comparison_sample_{sample_index}.png")
    plot_clinical_comparison(
        noisy=noisy[sample_index],
        clean=clean[sample_index],
        labels=labels[sample_index],
        direct=direct_recon,
        pred=pred_recon,
        gt=gt_recon,
        sample_index=sample_index,
        save_path=fig_path,
    )


if __name__ == "__main__":
    main()
