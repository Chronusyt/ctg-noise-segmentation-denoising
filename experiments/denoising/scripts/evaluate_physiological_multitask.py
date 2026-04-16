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
from ctg_pipeline.data.multitask_dataset import (
    build_parent_chunk_index,
    context_offsets,
    fetch_same_parent_neighbor_context,
    load_multitask_arrays,
)
from ctg_pipeline.models.unet1d_physiological_multitask import ARTIFACT_CLASS_ORDER, UNet1DPhysiologicalMultitask
from ctg_pipeline.utils.editing import build_edit_gate_torch, compute_region_masks_torch
from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_DATASETS_ROOT, resolve_repo_path


SCALAR_KEYS = ("baseline", "stv", "ltv", "baseline_variability")


def pred_mask_cache_path_for_split(pred_mask_cache_dir: str | None, split: str) -> str | None:
    if not pred_mask_cache_dir:
        return None
    return os.path.join(pred_mask_cache_dir, f"{split}_pred_masks.npz")


def load_multitask_npz(
    data_dir: str,
    split: str,
    pred_mask_cache_dir: str | None = None,
    pred_mask_variant: str = "soft",
) -> dict:
    path = os.path.join(data_dir, f"{split}_dataset_multitask.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return load_multitask_arrays(
        path,
        pred_mask_cache_path=pred_mask_cache_path_for_split(pred_mask_cache_dir, split),
        pred_mask_variant=pred_mask_variant,
    )


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


def infer_gate_mode(args: argparse.Namespace, ckpt: dict) -> str:
    if args.gate_mode:
        return args.gate_mode
    return ckpt.get("config", {}).get("gate_mode", "none")


def infer_model_variant(ckpt: dict) -> str:
    model_variant = ckpt.get("config", {}).get("model_variant", "legacy_single_residual")
    if model_variant in {"legacy_single_residual", "expert_residual", "typed_scale_residual"}:
        return model_variant
    return "legacy_single_residual"


def infer_backbone_type(ckpt: dict) -> str:
    backbone_type = ckpt.get("config", {}).get("backbone_type", "unet")
    if backbone_type in {
        "unet",
        "modern_tcn",
        "multiscale_unet",
        "tcn_unet",
        "multiscale_tcn_unet",
        "convnext1d_unet",
        "multiscale_convnext1d_unet",
        "modern_tcn_unet",
        "multiscale_modern_tcn_unet",
    }:
        return backbone_type
    return "unet"


def infer_loss_balance_mode(ckpt: dict) -> str:
    loss_balance_mode = ckpt.get("config", {}).get("loss_balance_mode", "static")
    if loss_balance_mode in {"static", "gradnorm"}:
        return loss_balance_mode
    return "static"


def infer_bottleneck_type(ckpt: dict) -> str:
    bottleneck_type = ckpt.get("config", {}).get("bottleneck_type", "none")
    if bottleneck_type in {"none", "mhsa"}:
        return bottleneck_type
    return "none"


def infer_use_context_chunks(args: argparse.Namespace, ckpt: dict) -> bool:
    return bool(args.use_context_chunks or ckpt.get("config", {}).get("use_context_chunks", False))


def infer_context_mode(args: argparse.Namespace, ckpt: dict) -> str:
    if args.context_mode:
        return args.context_mode
    return ckpt.get("config", {}).get("context_mode", "same_parent_neighbors")


def infer_context_radius(args: argparse.Namespace, ckpt: dict) -> int:
    if args.context_radius >= 0:
        return int(args.context_radius)
    return int(ckpt.get("config", {}).get("context_radius", 5))


def infer_context_include_center(args: argparse.Namespace, ckpt: dict) -> bool:
    if args.context_include_center:
        return True
    return bool(ckpt.get("config", {}).get("context_include_center", False))


def build_context_chunk(
    data: dict,
    parent_chunk_to_row: dict[int, dict[int, int]] | None,
    start: int,
    end: int,
    *,
    context_radius: int,
    context_include_center: bool,
) -> dict[str, np.ndarray]:
    if parent_chunk_to_row is None:
        raise ValueError("Context evaluation requires parent/chunk lookup")
    offsets = context_offsets(context_radius, context_include_center)
    context_rows = []
    for row_index in range(start, end):
        context_rows.append(
            fetch_same_parent_neighbor_context(
                row_index=row_index,
                noisy_signals=np.asarray(data["noisy_signals"], dtype=np.float32),
                pred_masks=np.asarray(data["pred_masks"], dtype=np.float32) if "pred_masks" in data else None,
                parent_index=np.asarray(data["parent_index"], dtype=np.int64),
                chunk_index=np.asarray(data["chunk_index"], dtype=np.int64),
                parent_chunk_to_row=parent_chunk_to_row,
                offsets=offsets,
                context_use_pred_mask=True,
            )
        )
    return {
        "context_noisy_signal": np.stack([row["context_noisy_signal"] for row in context_rows], axis=0),
        "context_pred_mask": np.stack([row["context_pred_mask"] for row in context_rows], axis=0),
        "context_valid": np.stack([row["context_valid"] for row in context_rows], axis=0),
        "context_chunk_index": np.stack([row["context_chunk_index"] for row in context_rows], axis=0),
    }


def make_model_from_checkpoint(ckpt: dict, input_mode: str) -> UNet1DPhysiologicalMultitask:
    config = ckpt.get("config", {})
    in_channels = int(config.get("in_channels", 6 if input_mode in {"gt_mask", "pred_mask"} else 1))
    return UNet1DPhysiologicalMultitask(
        in_channels=in_channels,
        base_channels=int(config.get("base_channels", 32)),
        depth=int(config.get("depth", 3)),
        scalar_hidden_channels=int(config.get("scalar_hidden_channels", 128)),
        dropout=float(config.get("dropout", 0.0)),
        residual_reconstruction=bool(config.get("residual_reconstruction", True)),
        model_variant=infer_model_variant(ckpt),
        backbone_type=infer_backbone_type(ckpt),
        bottleneck_type=infer_bottleneck_type(ckpt),
        modern_tcn_blocks_per_stage=int(config.get("modern_tcn_blocks_per_stage", 2)),
        modern_tcn_kernel_size=int(config.get("modern_tcn_kernel_size", 5)),
        modern_tcn_expansion=int(config.get("modern_tcn_expansion", 2)),
        use_context_chunks=bool(config.get("use_context_chunks", False)),
        context_fusion=str(config.get("context_fusion", "film")),
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
    if input_mode == "pred_mask":
        if "pred_masks" not in data:
            raise KeyError("input_mode=pred_mask but data is missing pred_masks. Provide pred-mask cache or embedded pred_masks.")
        masks = np.transpose(data["pred_masks"][start:end], (0, 2, 1)).astype(np.float32)
        return np.concatenate([noisy, masks], axis=1)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def build_gate_chunk(data: dict, start: int, end: int, input_mode: str, gate_mode: str, ckpt_config: dict, device: str) -> torch.Tensor | None:
    if gate_mode == "none":
        return None

    if input_mode == "gt_mask":
        mask_arr = data["masks"][start:end]
    elif input_mode == "pred_mask":
        if "pred_masks" not in data:
            raise KeyError("gate_mode requires pred_masks, but evaluation data is missing pred_masks.")
        mask_arr = data["pred_masks"][start:end]
    else:
        return None

    mask = torch.from_numpy(np.transpose(mask_arr, (0, 2, 1)).astype(np.float32)).to(device)
    return build_edit_gate_torch(
        mask,
        gate_mode=gate_mode,
        clean_gate_value=float(ckpt_config.get("clean_gate_value", 0.1)),
        dilation_radius=int(ckpt_config.get("gate_dilation_radius", 5)),
        smooth_kernel_size=int(ckpt_config.get("gate_smooth_kernel", 5)),
    )


def predict(
    model: torch.nn.Module,
    data: dict,
    label_stats: dict,
    device: str,
    batch_size: int,
    input_mode: str,
    gate_mode: str,
    ckpt_config: dict,
    use_context_chunks: bool = False,
    context_mode: str = "same_parent_neighbors",
    context_radius: int = 5,
    context_include_center: bool = False,
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
    parent_chunk_to_row = None
    if use_context_chunks:
        if context_mode != "same_parent_neighbors":
            raise ValueError(f"Unsupported context_mode: {context_mode}")
        if "parent_index" not in data or "chunk_index" not in data:
            raise KeyError("Context evaluation requires parent_index and chunk_index")
        parent_chunk_to_row = build_parent_chunk_index(data["parent_index"], data["chunk_index"])
    with torch.no_grad():
        n = data["noisy_signals"].shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(build_input_chunk(data, start, end, input_mode)).to(device)
            edit_gate = build_gate_chunk(data, start, end, input_mode, gate_mode, ckpt_config, device)
            context_kwargs = {}
            if use_context_chunks:
                context_batch = build_context_chunk(
                    data,
                    parent_chunk_to_row,
                    start,
                    end,
                    context_radius=context_radius,
                    context_include_center=context_include_center,
                )
                context_kwargs = {
                    "context_noisy_signal": torch.from_numpy(context_batch["context_noisy_signal"]).to(device),
                    "context_pred_mask": torch.from_numpy(context_batch["context_pred_mask"]).to(device),
                    "context_valid": torch.from_numpy(context_batch["context_valid"]).to(device),
                }
            out = model(x, edit_gate=edit_gate, **context_kwargs)
            preds["reconstruction"].append(out["reconstruction"].cpu().numpy()[:, 0, :])
            preds["acc_logits"].append(out["acc_logits"].cpu().numpy()[:, 0, :])
            preds["dec_logits"].append(out["dec_logits"].cpu().numpy()[:, 0, :])
            for key in SCALAR_KEYS:
                raw = out[key].cpu().numpy()
                preds[key].append(denormalize(raw, label_stats, key))
    return {key: np.concatenate(value, axis=0) for key, value in preds.items()}


def masked_error_stats(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, prefix: str) -> Dict[str, float]:
    pred = pred.astype(np.float64).ravel()
    target = target.astype(np.float64).ravel()
    mask = mask.astype(bool).ravel()
    if not np.any(mask):
        return {
            f"{prefix}_mse": float("nan"),
            f"{prefix}_mae": float("nan"),
            f"{prefix}_ratio": float(mask.mean()),
        }
    diff = pred[mask] - target[mask]
    return {
        f"{prefix}_mse": float(np.mean(diff**2)),
        f"{prefix}_mae": float(np.mean(np.abs(diff))),
        f"{prefix}_ratio": float(mask.mean()),
    }


def reconstruction_metrics(pred: np.ndarray, clean: np.ndarray, region_masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    out = {
        "overall_mse": float(np.mean((pred.astype(np.float64) - clean.astype(np.float64)) ** 2)),
        "overall_mae": float(np.mean(np.abs(pred.astype(np.float64) - clean.astype(np.float64)))),
    }
    out.update(masked_error_stats(pred, clean, region_masks["corrupted"], "corrupted_region"))
    out.update(masked_error_stats(pred, clean, region_masks["clean"], "clean_region"))
    out.update(masked_error_stats(pred, clean, region_masks["boundary_near_clean"], "boundary_near_clean"))
    out.update(masked_error_stats(pred, clean, region_masks["far_clean"], "far_from_mask_clean"))
    return out


def artifact_reconstruction_metrics(pred: np.ndarray, clean: np.ndarray, masks: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for idx, name in enumerate(ARTIFACT_CLASS_ORDER):
        out.update(masked_error_stats(pred, clean, masks[..., idx], f"{name}_region"))
    scale_mask = np.max(masks[..., :2], axis=-1)
    other_mask = np.max(masks[..., 2:], axis=-1)
    out.update(masked_error_stats(pred, clean, scale_mask, "scale_artifact"))
    out.update(masked_error_stats(pred, clean, other_mask, "other_artifact"))
    return out


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


def mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate(([0], np.asarray(mask, dtype=np.uint8), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def masked_signal(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.asarray(signal, dtype=np.float64).copy()
    out[~np.asarray(mask, dtype=bool)] = np.nan
    return out


def plot_event_panel(
    ax,
    x: np.ndarray,
    clean: np.ndarray,
    recon: np.ndarray,
    noise_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_prob: np.ndarray,
    threshold: float,
    event_name: str,
    gt_color: str,
    pred_color: str,
    y_limits: tuple[float, float],
) -> tuple[int, int]:
    pred_mask = pred_prob >= threshold

    ax.fill_between(
        x,
        y_limits[0],
        y_limits[1],
        where=noise_mask,
        color="red",
        alpha=0.08,
    )
    ax.plot(x, clean, color="0.75", linewidth=0.8, label="clean FHR")
    ax.plot(x, recon, color="#6f9cf5", linewidth=0.8, alpha=0.7, label="recon FHR")

    if np.any(gt_mask):
        ax.plot(x, masked_signal(clean, gt_mask), color=gt_color, linewidth=2.2, label=f"{event_name} GT on clean")
        for start, end in mask_to_segments(gt_mask):
            ax.axvspan(start, end, color=gt_color, alpha=0.10)

    if np.any(pred_mask):
        ax.plot(
            x,
            masked_signal(clean, pred_mask),
            color=pred_color,
            linewidth=2.0,
            linestyle="--",
            label=f"{event_name} pred region on clean",
        )

    ax.set_ylim(*y_limits)
    ax.set_ylabel("FHR")
    ax.grid(True, alpha=0.3)

    ax_prob = ax.twinx()
    ax_prob.plot(x, pred_prob, color=pred_color, linewidth=1.0, alpha=0.95, label=f"{event_name} pred prob")
    ax_prob.axhline(threshold, color="gray", linestyle="--", linewidth=0.8, label=f"{event_name} threshold")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_ylabel("Prob")

    handles, labels = ax.get_legend_handles_labels()
    prob_handles, prob_labels = ax_prob.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles + prob_handles, labels + prob_labels):
        if label not in dedup:
            dedup[label] = handle
    ax.legend(list(dedup.values()), list(dedup.keys()), loc="upper right", fontsize=8)
    ax.set_title(
        f"{event_name} waveform view | GT samples={int(np.sum(gt_mask))} | "
        f"pred samples={int(np.sum(pred_mask))}"
    )
    return int(np.sum(gt_mask)), int(np.sum(pred_mask))


def plot_visualizations(
    data: dict,
    preds: dict,
    derived: dict | None,
    save_dir: str,
    indices: np.ndarray,
    acc_threshold: float,
    dec_threshold: float,
    gate_mode: str,
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
        y_limits = (
            float(min(noisy.min(), clean.min(), recon.min()) - 5),
            float(max(noisy.max(), clean.max(), recon.max()) + 5),
        )

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
        axes[0].set_ylim(*y_limits)

        acc_gt_samples, acc_pred_samples = plot_event_panel(
            axes[1],
            x=x,
            clean=clean,
            recon=recon,
            noise_mask=noise_mask,
            gt_mask=data["acc_labels"][idx].astype(bool),
            pred_prob=acc_prob[idx],
            threshold=acc_threshold,
            event_name="ACC",
            gt_color="#2ca02c",
            pred_color="#1f77b4",
            y_limits=y_limits,
        )

        dec_gt_samples, dec_pred_samples = plot_event_panel(
            axes[2],
            x=x,
            clean=clean,
            recon=recon,
            noise_mask=noise_mask,
            gt_mask=data["dec_labels"][idx].astype(bool),
            pred_prob=dec_prob[idx],
            threshold=dec_threshold,
            event_name="DEC",
            gt_color="#8c564b",
            pred_color="#d62728",
            y_limits=y_limits,
        )

        derived_text = ""
        if derived is not None:
            derived_text = (
                f"\nrecon-derived baseline/STV/LTV/BV: "
                f"{derived['baseline'][idx]:.2f} / {derived['stv'][idx]:.2f} / "
                f"{derived['ltv'][idx]:.2f} / {derived['baseline_variability'][idx]:.2f}"
            )
        parent_text = ""
        if "parent_index" in data and "chunk_index" in data:
            parent_text = f"\nparent/chunk: {int(data['parent_index'][idx])} / {int(data['chunk_index'][idx])}"
        scalar_text = (
            f"gate_mode: {gate_mode}\n"
            f"ACC GT/pred samples: {acc_gt_samples} / {acc_pred_samples}\n"
            f"DEC GT/pred samples: {dec_gt_samples} / {dec_pred_samples}\n"
            f"head baseline GT/pred: {data['baseline_labels'][idx]:.2f} / {preds['baseline'][idx]:.2f}\n"
            f"head STV GT/pred: {data['stv_labels'][idx]:.2f} / {preds['stv'][idx]:.2f}\n"
            f"head LTV GT/pred: {data['ltv_labels'][idx]:.2f} / {preds['ltv'][idx]:.2f}\n"
            f"head BV GT/pred: {data['baseline_variability_labels'][idx]:.2f} / {preds['baseline_variability'][idx]:.2f}"
            f"{derived_text}"
            f"{parent_text}"
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

    md_path = os.path.join(output_dir, "test_metrics.md")
    lines = ["# Physiological Multitask Evaluation", ""]
    for section, values in results.items():
        lines.append(f"## {section}")
        if isinstance(values, dict):
            for key, value in values.items():
                if isinstance(value, dict):
                    lines.append(f"- {key}:")
                    for subkey, subvalue in value.items():
                        lines.append(f"  - {subkey}: {subvalue}")
                else:
                    lines.append(f"- {key}: {value}")
        else:
            lines.append(f"- value: {values}")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Metrics saved to {json_path}, {txt_path}, {csv_path}, {md_path}")


def write_boundary_report(output_dir: str, reconstruction: dict, boundary_k: int) -> None:
    diagnostics_dir = os.path.join(output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    txt_path = os.path.join(diagnostics_dir, "boundary_error_report.txt")
    json_path = os.path.join(diagnostics_dir, "boundary_error_report.json")
    report = {
        "boundary_k": int(boundary_k),
        "clean_region_mse": reconstruction.get("clean_region_mse"),
        "clean_region_mae": reconstruction.get("clean_region_mae"),
        "boundary_near_clean_mse": reconstruction.get("boundary_near_clean_mse"),
        "boundary_near_clean_mae": reconstruction.get("boundary_near_clean_mae"),
        "far_from_mask_clean_mse": reconstruction.get("far_from_mask_clean_mse"),
        "far_from_mask_clean_mae": reconstruction.get("far_from_mask_clean_mae"),
        "clean_region_ratio": reconstruction.get("clean_region_ratio"),
        "boundary_near_clean_ratio": reconstruction.get("boundary_near_clean_ratio"),
        "far_from_mask_clean_ratio": reconstruction.get("far_from_mask_clean_ratio"),
    }
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Boundary Error Report\n")
        f.write("=====================\n")
        for key, value in report.items():
            f.write(f"{key}: {value}\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Boundary diagnostics saved to {txt_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate physiological multitask model")
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
    parser.add_argument("--acc_threshold", type=float, default=-1.0)
    parser.add_argument("--dec_threshold", type=float, default=-1.0)
    parser.add_argument("--max_samples", type=int, default=0, help="Debug only; 0 means full split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_vis", type=int, default=5)
    parser.add_argument("--pred_mask_cache_dir", type=str, default="")
    parser.add_argument("--pred_mask_variant", choices=["soft", "hard"], default="soft")
    parser.add_argument("--input_mode", choices=["no_mask", "gt_mask", "pred_mask"], default="")
    parser.add_argument("--gate_mode", choices=["none", "union_soft", "union_dilated_soft"], default="")
    parser.add_argument("--use_context_chunks", action="store_true")
    parser.add_argument("--context_mode", choices=["same_parent_neighbors"], default="")
    parser.add_argument("--context_radius", type=int, default=-1)
    parser.add_argument("--context_include_center", action="store_true")
    parser.add_argument("--boundary_k", type=int, default=-1)
    parser.add_argument("--no_derived_features", action="store_true")
    parser.add_argument("--derived_chunk_size", type=int, default=4096)
    args = parser.parse_args()

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.model_path = str(resolve_repo_path(args.model_path))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    args.pred_mask_cache_dir = str(resolve_repo_path(args.pred_mask_cache_dir)) if args.pred_mask_cache_dir else ""
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)

    data, _indices = subset_arrays(
        load_multitask_npz(
            args.data_dir,
            args.split,
            pred_mask_cache_dir=args.pred_mask_cache_dir,
            pred_mask_variant=args.pred_mask_variant,
        ),
        args.max_samples,
        args.seed,
    )
    print(f"Loaded split={args.split}, samples={data['noisy_signals'].shape[0]}", flush=True)

    ckpt = torch.load(args.model_path, map_location=device)
    ckpt_config = ckpt.get("config", {})
    label_stats = ckpt.get("label_stats") or ckpt_config.get("label_stats")
    if label_stats is None:
        raise KeyError("Checkpoint missing label_stats")
    input_mode = infer_input_mode(args, ckpt)
    gate_mode = infer_gate_mode(args, ckpt)
    model_variant = infer_model_variant(ckpt)
    backbone_type = infer_backbone_type(ckpt)
    loss_balance_mode = infer_loss_balance_mode(ckpt)
    bottleneck_type = infer_bottleneck_type(ckpt)
    use_context_chunks = infer_use_context_chunks(args, ckpt)
    context_mode = infer_context_mode(args, ckpt)
    context_radius = infer_context_radius(args, ckpt)
    context_include_center = infer_context_include_center(args, ckpt)
    boundary_k = args.boundary_k if args.boundary_k >= 0 else int(ckpt_config.get("boundary_k", 5))
    acc_threshold = args.acc_threshold if args.acc_threshold >= 0 else args.threshold
    dec_threshold = args.dec_threshold if args.dec_threshold >= 0 else args.threshold
    if model_variant == "typed_scale_residual" and input_mode != "pred_mask":
        raise ValueError("typed_scale_residual checkpoint 第一版只支持 pred_mask 评估")
    if use_context_chunks and input_mode != "pred_mask":
        raise ValueError("Stage-1 context conditioning 只支持 pred_mask 评估")
    if use_context_chunks and model_variant != "legacy_single_residual":
        raise ValueError("Stage-1 context conditioning 只支持 legacy_single_residual 评估")
    print(f"输入模式: {input_mode}", flush=True)
    print(f"模型变体: {model_variant}", flush=True)
    print(f"backbone: {backbone_type}", flush=True)
    print(f"bottleneck: {bottleneck_type}", flush=True)
    print(f"loss balance: {loss_balance_mode}", flush=True)
    print(f"pred mask cache: {args.pred_mask_cache_dir or 'embedded_or_none'}", flush=True)
    print(f"pred mask variant: {args.pred_mask_variant}", flush=True)
    print(f"use context chunks: {use_context_chunks}", flush=True)
    if use_context_chunks:
        print(
            f"context mode: {context_mode}, radius={context_radius}, "
            f"include_center={context_include_center}",
            flush=True,
        )
    print(f"gate 模式: {gate_mode}", flush=True)
    print(f"boundary_k: {boundary_k}", flush=True)
    print(f"thresholds: acc={acc_threshold}, dec={dec_threshold}", flush=True)

    model = make_model_from_checkpoint(ckpt, input_mode=input_mode)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model = model.to(device)

    preds = predict(
        model,
        data,
        label_stats,
        device,
        args.batch_size,
        input_mode=input_mode,
        gate_mode=gate_mode,
        ckpt_config=ckpt_config,
        use_context_chunks=use_context_chunks,
        context_mode=context_mode,
        context_radius=context_radius,
        context_include_center=context_include_center,
    )

    region_masks_torch = compute_region_masks_torch(
        torch.from_numpy(np.transpose(data["masks"], (0, 2, 1)).astype(np.float32)),
        boundary_k=boundary_k,
    )
    region_masks = {key: value.numpy()[:, 0, :] for key, value in region_masks_torch.items()}

    results = {
        "metadata": {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "split": args.split,
            "n_samples": int(data["noisy_signals"].shape[0]),
            "input_mode": input_mode,
            "pred_mask_cache_dir": args.pred_mask_cache_dir,
            "pred_mask_variant": args.pred_mask_variant,
            "gate_mode": gate_mode,
            "boundary_k": boundary_k,
            "model_variant": model_variant,
            "backbone_type": backbone_type,
            "bottleneck_type": bottleneck_type,
            "loss_balance_mode": loss_balance_mode,
            "use_context_chunks": use_context_chunks,
            "context_mode": context_mode if use_context_chunks else None,
            "context_radius": context_radius if use_context_chunks else None,
            "context_include_center": context_include_center if use_context_chunks else None,
            "experiment_variant": ckpt_config.get("experiment_variant", ckpt_config.get("model_variant", "")),
            "artifact_class_order": ckpt_config.get("artifact_class_order", list(ARTIFACT_CLASS_ORDER)),
        },
        "reconstruction": reconstruction_metrics(preds["reconstruction"], data["clean_signals"], region_masks),
        "artifact_reconstruction": artifact_reconstruction_metrics(
            preds["reconstruction"],
            data["clean_signals"],
            data["masks"],
        ),
        "event_prediction": {
            "acceleration": binary_metrics(preds["acc_logits"], data["acc_labels"], acc_threshold),
            "deceleration": binary_metrics(preds["dec_logits"], data["dec_labels"], dec_threshold),
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
    write_boundary_report(args.output_dir, results["reconstruction"], boundary_k=boundary_k)
    vis_indices = choose_visual_indices(data, args.n_vis, args.seed)
    plot_visualizations(
        data,
        preds,
        derived,
        os.path.join(args.output_dir, "figures"),
        vis_indices,
        acc_threshold,
        dec_threshold,
        gate_mode=gate_mode,
    )


if __name__ == "__main__":
    main()
