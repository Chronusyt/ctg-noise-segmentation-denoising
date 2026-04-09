"""Evaluate physiological multitask models on the clinical multitask dataset."""
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

from ctg_pipeline.features.physiology import FeatureConfig, compute_multitask_physiology_labels
from ctg_pipeline.models.unet1d_physiological_multitask import UNet1DPhysiologicalMultitask
from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_DATASETS_ROOT, resolve_repo_path


SCALAR_KEYS = ("baseline", "stv", "ltv", "baseline_variability")


def load_multitask_npz(data_dir: str, split: str) -> dict:
    path = os.path.join(data_dir, f"{split}_dataset_multitask.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def subset_arrays(data: dict, max_samples: int, seed: int) -> tuple[dict, np.ndarray]:
    n = data["noisy_signals"].shape[0]
    if max_samples <= 0 or max_samples >= n:
        return data, np.arange(n)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n, size=max_samples, replace=False))
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n,):
            out[key] = value[indices]
        else:
            out[key] = value
    return out, indices


def infer_input_mode(args: argparse.Namespace, ckpt: dict) -> str:
    if args.input_mode:
        return args.input_mode
    return ckpt.get("config", {}).get("input_mode", "no_mask")


def make_model_from_checkpoint(ckpt: dict, input_mode: str) -> UNet1DPhysiologicalMultitask:
    config = ckpt.get("config", {})
    in_channels = int(config.get("in_channels", 6 if input_mode == "gt_mask" else 1))
    return UNet1DPhysiologicalMultitask(
        in_channels=in_channels,
        base_channels=int(config.get("base_channels", 32)),
        depth=int(config.get("depth", 3)),
        scalar_hidden_channels=int(config.get("scalar_hidden_channels", 128)),
        dropout=float(config.get("dropout", 0.0)),
        residual_reconstruction=bool(config.get("residual_reconstruction", True)),
    )


def denormalize(values: np.ndarray, label_stats: dict, key: str) -> np.ndarray:
    stats = label_stats[key]
    return values * float(stats["std"]) + float(stats["mean"])


def build_input_chunk(data: dict, start: int, end: int, input_mode: str) -> np.ndarray:
    noisy = data["noisy_signals"][start:end, np.newaxis, :].astype(np.float32)
    if input_mode == "no_mask":
        return noisy
    if input_mode == "gt_mask":
        masks = np.transpose(data["masks"][start:end], (0, 2, 1)).astype(np.float32)
        return np.concatenate([noisy, masks], axis=1)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def predict(
    model: torch.nn.Module,
    data: dict,
    label_stats: dict,
    device: str,
    batch_size: int,
    input_mode: str,
) -> dict:
    model.eval()
    preds = {
        "reconstruction": [],
        "acc_logits": [],
        "dec_logits": [],
        "baseline": [],
        "stv": [],
        "ltv": [],
        "baseline_variability": [],
    }
    with torch.no_grad():
        n = data["noisy_signals"].shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(build_input_chunk(data, start, end, input_mode)).to(device)
            out = model(x)
            preds["reconstruction"].append(out["reconstruction"].cpu().numpy()[:, 0, :])
            preds["acc_logits"].append(out["acc_logits"].cpu().numpy()[:, 0, :])
            preds["dec_logits"].append(out["dec_logits"].cpu().numpy()[:, 0, :])
            for key in SCALAR_KEYS:
                raw = out[key].cpu().numpy()
                preds[key].append(denormalize(raw, label_stats, key))
    return {key: np.concatenate(value, axis=0) for key, value in preds.items()}


def reconstruction_metrics(pred: np.ndarray, clean: np.ndarray, noise_mask: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.float64).ravel()
    clean = clean.astype(np.float64).ravel()
    noise_mask = noise_mask.astype(bool).ravel()
    clean_mask = ~noise_mask
    return {
        "overall_mse": float(np.mean((pred - clean) ** 2)),
        "overall_mae": float(np.mean(np.abs(pred - clean))),
        "corrupted_region_mse": float(np.mean((pred[noise_mask] - clean[noise_mask]) ** 2)) if np.any(noise_mask) else float("nan"),
        "corrupted_region_mae": float(np.mean(np.abs(pred[noise_mask] - clean[noise_mask]))) if np.any(noise_mask) else float("nan"),
        "clean_region_mse": float(np.mean((pred[clean_mask] - clean[clean_mask]) ** 2)) if np.any(clean_mask) else float("nan"),
        "clean_region_mae": float(np.mean(np.abs(pred[clean_mask] - clean[clean_mask]))) if np.any(clean_mask) else float("nan"),
    }


def binary_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    pred = probs >= threshold
    gt = labels.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate_pred": float(pred.mean()),
        "positive_rate_gt": float(gt.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def scalar_metrics(preds: dict, data: dict) -> Dict[str, float]:
    mapping = {
        "baseline": "baseline_labels",
        "stv": "stv_labels",
        "ltv": "ltv_labels",
        "baseline_variability": "baseline_variability_labels",
    }
    out: Dict[str, float] = {}
    for key, label_key in mapping.items():
        diff = preds[key].astype(np.float64) - data[label_key].astype(np.float64)
        prefix = "bv" if key == "baseline_variability" else key
        out[f"{prefix}_mae"] = float(np.mean(np.abs(diff)))
        out[f"{prefix}_mse"] = float(np.mean(diff**2))
        out[f"{prefix}_bias_mean"] = float(np.mean(diff))
        out[f"{prefix}_bias_median"] = float(np.median(diff))
    return out


def compute_derived_features(reconstructed: np.ndarray, chunk_size: int) -> dict:
    chunks = []
    n = reconstructed.shape[0]
    cfg = FeatureConfig(sample_rate=4.0)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        print(f"Derived feature progress: {start}-{end} / {n} ({end / n * 100:.1f}%)", flush=True)
        chunks.append(compute_multitask_physiology_labels(reconstructed[start:end], config=cfg))
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0].keys()}


def derived_feature_metrics(derived: dict, data: dict) -> Dict[str, float]:
    mapping = {
        "baseline": "baseline_labels",
        "stv": "stv_labels",
        "ltv": "ltv_labels",
        "baseline_variability": "baseline_variability_labels",
    }
    out: Dict[str, float] = {}
    for key, label_key in mapping.items():
        diff = derived[key].astype(np.float64) - data[label_key].astype(np.float64)
        prefix = "bv" if key == "baseline_variability" else key
        out[f"{prefix}_mae"] = float(np.mean(np.abs(diff)))
        out[f"{prefix}_mse"] = float(np.mean(diff**2))
        out[f"{prefix}_bias_mean"] = float(np.mean(diff))
        out[f"{prefix}_bias_median"] = float(np.median(diff))
    return out


def choose_visual_indices(data: dict, n_vis: int, seed: int) -> np.ndarray:
    n = data["noisy_signals"].shape[0]
    if n_vis <= 0:
        return np.array([], dtype=int)
    noise_mask = (data["artifact_labels"] > 0.5).any(axis=-1)
    coverage = noise_mask.mean(axis=1)
    if np.any(coverage > 0):
        return np.argsort(-coverage)[: min(n_vis, n)]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(n_vis, n), replace=False))


def plot_visualizations(
    data: dict,
    preds: dict,
    derived: dict | None,
    save_dir: str,
    indices: np.ndarray,
    threshold: float,
) -> None:
    if indices.size == 0:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip visualization")
        return
    os.makedirs(save_dir, exist_ok=True)
    x = np.arange(data["noisy_signals"].shape[1])
    acc_prob = 1.0 / (1.0 + np.exp(-preds["acc_logits"]))
    dec_prob = 1.0 / (1.0 + np.exp(-preds["dec_logits"]))

    for rank, idx in enumerate(indices):
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        noisy = data["noisy_signals"][idx]
        clean = data["clean_signals"][idx]
        recon = preds["reconstruction"][idx]
        noise_mask = (data["artifact_labels"][idx] > 0.5).any(axis=-1)

        axes[0].plot(x, noisy, "k-", linewidth=0.8, label="noisy")
        axes[0].plot(x, clean, "g-", linewidth=0.9, label="clean target")
        axes[0].plot(x, recon, "b-", linewidth=0.9, label="multitask recon")
        axes[0].fill_between(
            x,
            min(noisy.min(), clean.min()) - 5,
            max(noisy.max(), clean.max()) + 5,
            where=noise_mask,
            color="red",
            alpha=0.15,
            label="noise region",
        )
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_ylabel("FHR")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, data["acc_labels"][idx], "g-", linewidth=1.0, label="acc GT")
        axes[1].plot(x, acc_prob[idx], "b-", linewidth=0.9, label="acc pred prob")
        axes[1].axhline(threshold, color="gray", linestyle="--", linewidth=0.8)
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].set_ylabel("ACC")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x, data["dec_labels"][idx], "g-", linewidth=1.0, label="dec GT")
        axes[2].plot(x, dec_prob[idx], "b-", linewidth=0.9, label="dec pred prob")
        axes[2].axhline(threshold, color="gray", linestyle="--", linewidth=0.8)
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].set_ylabel("DEC")
        axes[2].grid(True, alpha=0.3)

        derived_text = ""
        if derived is not None:
            derived_text = (
                f"\nrecon-derived baseline/STV/LTV/BV: "
                f"{derived['baseline'][idx]:.2f} / {derived['stv'][idx]:.2f} / "
                f"{derived['ltv'][idx]:.2f} / {derived['baseline_variability'][idx]:.2f}"
            )
        scalar_text = (
            f"head baseline GT/pred: {data['baseline_labels'][idx]:.2f} / {preds['baseline'][idx]:.2f}\n"
            f"head STV GT/pred: {data['stv_labels'][idx]:.2f} / {preds['stv'][idx]:.2f}\n"
            f"head LTV GT/pred: {data['ltv_labels'][idx]:.2f} / {preds['ltv'][idx]:.2f}\n"
            f"head BV GT/pred: {data['baseline_variability_labels'][idx]:.2f} / {preds['baseline_variability'][idx]:.2f}"
            f"{derived_text}"
        )
        axes[3].axis("off")
        axes[3].text(0.02, 0.85, scalar_text, transform=axes[3].transAxes, fontsize=10, va="top")
        axes[3].set_title(f"sample_index={idx}")

        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"multitask_sample_{rank}_idx_{int(idx)}.png"), dpi=150)
        plt.close(fig)
    print(f"Visualizations saved to {save_dir}")


def flatten_metrics(results: dict) -> dict:
    out = {}
    for section, values in results.items():
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        out[f"{section}_{key}_{subkey}"] = subvalue
                else:
                    out[f"{section}_{key}"] = value
    return out


def write_outputs(output_dir: str, results: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(output_dir, "test_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for section, values in results.items():
            f.write(f"[{section}]\n")
            if isinstance(values, dict):
                for key, value in values.items():
                    if isinstance(value, dict):
                        f.write(f"  {key}:\n")
                        for subkey, subvalue in value.items():
                            f.write(f"    {subkey}: {subvalue}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  {values}\n")
            f.write("\n")

    flat = flatten_metrics(results)
    csv_path = os.path.join(output_dir, "test_metrics_flat.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in flat.items():
            writer.writerow([key, value])
    print(f"Metrics saved to {json_path}, {txt_path}, {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate physiological multitask model (v1 no-mask / v2 gt-mask auxiliary)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "clinical_multitask_physiology_v1"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "physiological_multitask" / "clinical_v1_no_mask" / "best_model.pt"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "physiological_multitask" / "clinical_v1_no_mask" / "eval"),
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_samples", type=int, default=0, help="Debug only; 0 means full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_vis", type=int, default=5)
    parser.add_argument("--input_mode", choices=["no_mask", "gt_mask"], default="")
    parser.add_argument("--no_derived_features", action="store_true")
    parser.add_argument("--derived_chunk_size", type=int, default=4096)
    args = parser.parse_args()

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.model_path = str(resolve_repo_path(args.model_path))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)

    data, _indices = subset_arrays(load_multitask_npz(args.data_dir, args.split), args.max_samples, args.seed)
    print(f"Loaded split={args.split}, samples={data['noisy_signals'].shape[0]}", flush=True)

    ckpt = torch.load(args.model_path, map_location=device)
    label_stats = ckpt.get("label_stats") or ckpt.get("config", {}).get("label_stats")
    if label_stats is None:
        raise KeyError("Checkpoint missing label_stats")
    input_mode = infer_input_mode(args, ckpt)
    print(f"输入模式: {input_mode}", flush=True)

    model = make_model_from_checkpoint(ckpt, input_mode=input_mode)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model = model.to(device)

    preds = predict(model, data, label_stats, device, args.batch_size, input_mode=input_mode)
    noise_mask = (data["artifact_labels"] > 0.5).any(axis=-1)
    results = {
        "metadata": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "split": args.split,
            "n_samples": int(data["noisy_signals"].shape[0]),
            "input_mode": input_mode,
            "model_variant": ckpt.get("config", {}).get("model_variant", ""),
        },
        "reconstruction": reconstruction_metrics(preds["reconstruction"], data["clean_signals"], noise_mask),
        "event_prediction": {
            "acceleration": binary_metrics(preds["acc_logits"], data["acc_labels"], args.threshold),
            "deceleration": binary_metrics(preds["dec_logits"], data["dec_labels"], args.threshold),
        },
        "scalar_head": scalar_metrics(preds, data),
    }

    derived = None
    if not args.no_derived_features:
        print("Computing reconstructed-signal-derived physiological features...", flush=True)
        derived = compute_derived_features(preds["reconstruction"], args.derived_chunk_size)
        results["reconstructed_signal_derived_features"] = derived_feature_metrics(derived, data)
    else:
        results["reconstructed_signal_derived_features"] = {}

    write_outputs(args.output_dir, results)
    vis_indices = choose_visual_indices(data, args.n_vis, args.seed)
    plot_visualizations(data, preds, derived, os.path.join(args.output_dir, "figures"), vis_indices, args.threshold)


if __name__ == "__main__":
    main()
