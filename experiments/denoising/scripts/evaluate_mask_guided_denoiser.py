"""
Mask-guided denoiser 评估与可视化（multilabel segmentation → denoising 两阶段）

与 direct denoising baseline 独立，不修改原脚本。
"""
from __future__ import annotations

import argparse
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

from ctg_pipeline.models.unet1d_mask_guided_denoiser import UNet1DMaskGuidedDenoiser
from ctg_pipeline.utils.pathing import DENOISING_DATASETS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path

CLASS_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def _infer_experiment_tag(*paths: str) -> str:
    for path in paths:
        if path and "clinical" in path:
            return "clinical"
    return "hard"


def _infer_mask_source(current: str, *paths: str) -> str:
    if current != "pred":
        return current
    for path in paths:
        if not path:
            continue
        if "_gt" in path or "gt" in os.path.basename(path):
            return "gt"
        if "_pred" in path or "pred" in os.path.basename(path):
            return "pred"
    return current


def compute_metrics(
    pred: np.ndarray,
    clean: np.ndarray,
    noise_mask: np.ndarray,
) -> Dict[str, float]:
    """计算 overall / corrupted_region / clean_region 的 MSE 和 MAE。"""
    pred = pred.ravel().astype(np.float64)
    clean = clean.ravel().astype(np.float64)
    noise_mask = noise_mask.ravel().astype(bool)
    corrupted = noise_mask
    clean_region = ~noise_mask

    overall_mse = np.mean((pred - clean) ** 2)
    overall_mae = np.mean(np.abs(pred - clean))
    corrupted_mse = np.mean((pred[corrupted] - clean[corrupted]) ** 2) if np.any(corrupted) else np.nan
    corrupted_mae = np.mean(np.abs(pred[corrupted] - clean[corrupted])) if np.any(corrupted) else np.nan
    clean_mse = np.mean((pred[clean_region] - clean[clean_region]) ** 2) if np.any(clean_region) else np.nan
    clean_mae = np.mean(np.abs(pred[clean_region] - clean[clean_region])) if np.any(clean_region) else np.nan

    return {
        "overall_mse": float(overall_mse),
        "corrupted_region_mse": float(corrupted_mse),
        "clean_region_mse": float(clean_mse),
        "overall_mae": float(overall_mae),
        "corrupted_region_mae": float(corrupted_mae),
        "clean_region_mae": float(clean_mae),
    }


def compute_per_sample(
    pred: np.ndarray,
    clean: np.ndarray,
    noise_mask: np.ndarray,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Per-sample 指标。"""
    N = pred.shape[0]
    per_mse = np.zeros(N)
    per_corr_mse = np.zeros(N)
    per_clean_mse = np.zeros(N)
    per_mae = np.zeros(N)
    per_corr_mae = np.zeros(N)
    per_clean_mae = np.zeros(N)

    for i in range(N):
        m = compute_metrics(pred[i], clean[i], noise_mask[i])
        per_mse[i] = m["overall_mse"]
        per_corr_mse[i] = m["corrupted_region_mse"] if not np.isnan(m["corrupted_region_mse"]) else np.nan
        per_clean_mse[i] = m["clean_region_mse"] if not np.isnan(m["clean_region_mse"]) else np.nan
        per_mae[i] = m["overall_mae"]
        per_corr_mae[i] = m["corrupted_region_mae"] if not np.isnan(m["corrupted_region_mae"]) else np.nan
        per_clean_mae[i] = m["clean_region_mae"] if not np.isnan(m["clean_region_mae"]) else np.nan

    for i in range(N):
        if not np.any(noise_mask[i]):
            per_corr_mse[i] = np.nan
            per_corr_mae[i] = np.nan
        if not np.any(~noise_mask[i]):
            per_clean_mse[i] = np.nan
            per_clean_mae[i] = np.nan

    mean_metrics = {
        "overall_mse": float(np.nanmean(per_mse)),
        "corrupted_region_mse": float(np.nanmean(per_corr_mse)),
        "clean_region_mse": float(np.nanmean(per_clean_mse)),
        "overall_mae": float(np.nanmean(per_mae)),
        "corrupted_region_mae": float(np.nanmean(per_corr_mae)),
        "clean_region_mae": float(np.nanmean(per_clean_mae)),
    }
    return mean_metrics, {}


def load_mask_guided_dataset(data_dir: str, split: str) -> tuple:
    """加载 mask-guided 数据。"""
    path = os.path.join(data_dir, f"{split}_dataset_mask_guided.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    data = np.load(path)
    noisy = np.asarray(data["noisy_signals"], dtype=np.float32)
    clean = np.asarray(data["clean_signals"], dtype=np.float32)
    masks = np.asarray(data["masks"], dtype=np.float32)
    labels = np.asarray(data["artifact_labels"], dtype=np.float32)
    parent_index = np.asarray(data["parent_index"], dtype=np.int32)
    chunk_index = np.asarray(data["chunk_index"], dtype=np.int32)
    return noisy, clean, masks, labels, parent_index, chunk_index


def predict_batch(
    model: torch.nn.Module,
    noisy: np.ndarray,
    masks: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """批量预测，输入 [N, 6, L]，返回 [N, L]。"""
    model.eval()
    N = noisy.shape[0]
    input_arr = np.concatenate([
        noisy[:, np.newaxis, :],
        np.transpose(masks, (0, 2, 1)),
    ], axis=1)
    preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = torch.from_numpy(input_arr[i : i + batch_size]).to(device)
            y = model(x)
            preds.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds.squeeze()


def plot_samples(
    noisy: np.ndarray,
    masks: np.ndarray,
    reconstructed: np.ndarray,
    clean: np.ndarray,
    noise_mask: np.ndarray,
    parent_index: np.ndarray,
    chunk_index: np.ndarray,
    sample_metrics: list,
    indices: np.ndarray,
    save_dir: str,
    mask_source: str,
) -> None:
    """绘制：noisy, masks(5类), reconstructed, clean + 噪声区域 shading。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    os.makedirs(save_dir, exist_ok=True)
    L = noisy.shape[1]
    x = np.arange(L)

    for k, idx in enumerate(indices):
        n_sig = noisy[idx]
        m = masks[idx]  # [L, 5]
        r_sig = reconstructed[idx]
        c_sig = clean[idx]
        nm = noise_mask[idx]
        pid = parent_index[idx]
        cid = chunk_index[idx]
        metrics = sample_metrics[k] if k < len(sample_metrics) else {}

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        axes[0].plot(x, n_sig, "k-", linewidth=0.8, label="noisy")
        axes[0].fill_between(x, n_sig.min() - 5, n_sig.max() + 5, where=nm, alpha=0.3, color="red", label="noise region")
        axes[0].set_ylabel("FHR")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"Parent {pid}, Chunk {cid} | mask_source={mask_source}")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Masks (5 classes)")
        for c, (name, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            axes[1].fill_between(x, c - 0.4, c + 0.4, where=m[:, c] > 0.5, alpha=0.7, color=col)
        axes[1].set_ylim(-0.5, 4.5)
        axes[1].set_yticks(range(5))
        axes[1].set_yticklabels(CLASS_NAMES, fontsize=8)
        axes[1].set_ylabel("mask")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x, r_sig, "b-", linewidth=0.8, label="reconstructed")
        axes[2].fill_between(x, r_sig.min() - 5, r_sig.max() + 5, where=nm, alpha=0.3, color="red")
        axes[2].set_ylabel("FHR")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].set_title(f"Reconstructed | overall_mse={metrics.get('overall_mse', 0):.4f} corr_mse={metrics.get('corrupted_region_mse', 0):.4f} clean_mse={metrics.get('clean_region_mse', 0):.4f}")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(x, c_sig, "g-", linewidth=0.8, label="clean (GT)")
        axes[3].fill_between(x, c_sig.min() - 5, c_sig.max() + 5, where=nm, alpha=0.3, color="red")
        axes[3].set_ylabel("FHR")
        axes[3].set_xlabel("Time (samples)")
        axes[3].legend(loc="upper right", fontsize=8)
        axes[3].set_title("Clean (ground truth)")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{k}_idx_{idx}_parent_{pid}_chunk_{cid}.png"), dpi=150)
        plt.close()
    print(f"  可视化已保存到 {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="评估 mask-guided denoiser")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="mask-guided 数据集目录",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="mask-guided denoiser 模型路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="评估结果输出目录",
    )
    parser.add_argument(
        "--mask_source",
        type=str,
        choices=["gt", "pred"],
        default="pred",
        help="mask 来源",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--n_vis", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefer_noisy", action="store_true", default=True)
    parser.add_argument("--no_prefer_noisy", action="store_true")
    args = parser.parse_args()

    args.mask_source = _infer_mask_source(args.mask_source, args.data_dir, args.model_path, args.output_dir)
    experiment_tag = _infer_experiment_tag(args.data_dir, args.model_path, args.output_dir)

    if args.data_dir is None:
        args.data_dir = str(DENOISING_DATASETS_ROOT / f"multilabel_guided_denoising_{experiment_tag}_{args.mask_source}")
    if args.model_path is None:
        args.model_path = str(DENOISING_RESULTS_ROOT / f"multilabel_guided_denoising_{experiment_tag}_{args.mask_source}" / "best_model.pt")
    if args.output_dir is None:
        args.output_dir = str(DENOISING_RESULTS_ROOT / f"multilabel_guided_denoising_{experiment_tag}_{args.mask_source}")

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.model_path = str(resolve_repo_path(args.model_path))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"mask_source={args.mask_source}, 使用设备: {device}")

    print("加载 test 数据...")
    noisy, clean, masks, labels, parent_index, chunk_index = load_mask_guided_dataset(args.data_dir, "test")
    noise_mask = (labels > 0.5).any(axis=-1)
    print(f"  noisy: {noisy.shape}, clean: {clean.shape}, masks: {masks.shape}")

    print("加载模型...")
    model = UNet1DMaskGuidedDenoiser(in_channels=6, out_channels=1, base_channels=32, depth=3)
    ckpt = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    print("预测...")
    reconstructed = predict_batch(model, noisy, masks, device)

    overall = compute_metrics(reconstructed, clean, noise_mask)
    per_sample_mean, _ = compute_per_sample(reconstructed, clean, noise_mask)

    print("\n========== Mask-guided denoiser 评估 ==========")
    print("Overall:")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")
    print("Per-sample 平均:")
    for k, v in per_sample_mean.items():
        print(f"  {k}: {v:.4f}")

    results = {
        "mask_source": args.mask_source,
        "overall": overall,
        "per_sample_mean": per_sample_mean,
    }
    txt_path = os.path.join(args.output_dir, "test_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"========== Mask-guided denoiser (mask_source={args.mask_source}) ==========\n")
        f.write("Overall:\n")
        for k, v in overall.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("Per-sample 平均:\n")
        for k, v in per_sample_mean.items():
            f.write(f"  {k}: {v:.4f}\n")
    print(f"\n指标已保存到 {txt_path}")

    json_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存到 {json_path}")

    rng = np.random.default_rng(args.seed)
    n_vis = min(args.n_vis, len(noisy))
    if (args.prefer_noisy and not args.no_prefer_noisy) and np.any(noise_mask):
        has_noise = noise_mask.any(axis=1)
        noisy_idx = np.where(has_noise)[0]
        if len(noisy_idx) >= n_vis:
            vis_indices = rng.choice(noisy_idx, size=n_vis, replace=False)
        else:
            vis_indices = rng.choice(len(noisy), size=n_vis, replace=False)
    else:
        vis_indices = rng.choice(len(noisy), size=n_vis, replace=False)

    sample_metrics_list = [compute_metrics(reconstructed[idx], clean[idx], noise_mask[idx]) for idx in vis_indices]
    vis_dir = os.path.join(args.output_dir, "visualizations")
    plot_samples(
        noisy, masks, reconstructed, clean, noise_mask,
        parent_index, chunk_index,
        sample_metrics_list,
        vis_indices,
        vis_dir,
        args.mask_source,
    )


if __name__ == "__main__":
    main()
