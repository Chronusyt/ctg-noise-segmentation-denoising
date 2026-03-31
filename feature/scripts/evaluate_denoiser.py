"""
直接去噪模型评估与可视化（direct denoising baseline）

与 segmentation 实验独立，不修改原脚本。

功能：
1. 计算 overall / corrupted_region / clean_region 的 MSE 和 MAE
2. identity baseline 对比
3. 可视化：noisy / reconstructed / clean + 噪声区域 shading
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)

from models.unet1d_denoiser import UNet1DDenoiser


def compute_metrics(
    pred: np.ndarray,
    clean: np.ndarray,
    noise_mask: np.ndarray,
) -> Dict[str, float]:
    """
    pred, clean: [N, L] 或 [L]
    noise_mask: [N, L] 或 [L]，True 表示噪声位置
    返回 overall_mse, corrupted_region_mse, clean_region_mse, overall_mae, corrupted_region_mae, clean_region_mae
    """
    pred = pred.ravel().astype(np.float64)
    clean = clean.ravel().astype(np.float64)
    noise_mask = noise_mask.ravel().astype(bool)

    corrupted = noise_mask
    clean_region = ~noise_mask

    overall_mse = np.mean((pred - clean) ** 2)
    overall_mae = np.mean(np.abs(pred - clean))

    if np.any(corrupted):
        corrupted_mse = np.mean((pred[corrupted] - clean[corrupted]) ** 2)
        corrupted_mae = np.mean(np.abs(pred[corrupted] - clean[corrupted]))
    else:
        corrupted_mse = np.nan
        corrupted_mae = np.nan

    if np.any(clean_region):
        clean_mse = np.mean((pred[clean_region] - clean[clean_region]) ** 2)
        clean_mae = np.mean(np.abs(pred[clean_region] - clean[clean_region]))
    else:
        clean_mse = np.nan
        clean_mae = np.nan

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
    """Per-sample 指标，返回 (mean_metrics, per_sample_metrics)。"""
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
        per_corr_mse[i] = m["corrupted_region_mse"] if not np.isnan(m["corrupted_region_mse"]) else 0.0
        per_clean_mse[i] = m["clean_region_mse"] if not np.isnan(m["clean_region_mse"]) else 0.0
        per_mae[i] = m["overall_mae"]
        per_corr_mae[i] = m["corrupted_region_mae"] if not np.isnan(m["corrupted_region_mae"]) else 0.0
        per_clean_mae[i] = m["clean_region_mae"] if not np.isnan(m["clean_region_mae"]) else 0.0

    # 对 corrupted/clean 为空的样本，用 nan 表示
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
    per_sample = {
        "overall_mse": per_mse,
        "corrupted_region_mse": per_corr_mse,
        "clean_region_mse": per_clean_mse,
        "overall_mae": per_mae,
        "corrupted_region_mae": per_corr_mae,
        "clean_region_mae": per_clean_mae,
    }
    return mean_metrics, per_sample


def load_denoising_dataset(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 denoising 数据。"""
    path = os.path.join(data_dir, f"{split}_dataset_denoising.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    data = np.load(path)
    noisy = np.asarray(data["noisy_signals"], dtype=np.float32)
    clean = np.asarray(data["clean_signals"], dtype=np.float32)
    labels = np.asarray(data["artifact_labels"], dtype=np.float32)
    parent_index = np.asarray(data["parent_index"], dtype=np.int32)
    chunk_index = np.asarray(data["chunk_index"], dtype=np.int32)
    return noisy, clean, labels, parent_index, chunk_index


def predict_batch(
    model: torch.nn.Module,
    noisy: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """批量预测，返回 [N, L]。"""
    model.eval()
    N = noisy.shape[0]
    if noisy.ndim == 2:
        noisy = noisy[:, np.newaxis, :]
    preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = torch.from_numpy(noisy[i : i + batch_size]).to(device)
            y = model(x)
            preds.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return preds.squeeze()


def plot_samples(
    noisy: np.ndarray,
    reconstructed: np.ndarray,
    clean: np.ndarray,
    noise_mask: np.ndarray,
    parent_index: np.ndarray,
    chunk_index: np.ndarray,
    sample_metrics: list,
    indices: np.ndarray,
    save_dir: str,
) -> None:
    """绘制样本：noisy / reconstructed / clean + 噪声区域 shading。"""
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
        r_sig = reconstructed[idx]
        c_sig = clean[idx]
        nm = noise_mask[idx]
        pid = parent_index[idx]
        cid = chunk_index[idx]
        m = sample_metrics[k] if k < len(sample_metrics) else {}

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(x, n_sig, "k-", linewidth=0.8, label="noisy")
        axes[0].fill_between(x, n_sig.min() - 5, n_sig.max() + 5, where=nm, alpha=0.3, color="red", label="noise region")
        axes[0].set_ylabel("FHR")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"Parent {pid}, Chunk {cid} | overall_mse={m.get('overall_mse', 0):.4f} corr_mse={m.get('corrupted_region_mse', 0):.4f} clean_mse={m.get('clean_region_mse', 0):.4f}")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, r_sig, "b-", linewidth=0.8, label="reconstructed")
        axes[1].fill_between(x, r_sig.min() - 5, r_sig.max() + 5, where=nm, alpha=0.3, color="red")
        axes[1].set_ylabel("FHR")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].set_title("Reconstructed (learned denoiser)")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x, c_sig, "g-", linewidth=0.8, label="clean (GT)")
        axes[2].fill_between(x, c_sig.min() - 5, c_sig.max() + 5, where=nm, alpha=0.3, color="red")
        axes[2].set_ylabel("FHR")
        axes[2].set_xlabel("Time (samples)")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].set_title("Clean (ground truth)")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{k}_idx_{idx}_parent_{pid}_chunk_{cid}.png"), dpi=150)
        plt.close()
    print(f"  可视化已保存到 {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="评估直接去噪模型")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/denoising_baseline_hard",
        help="denoising 数据集目录",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/denoising_baseline_hard/best_model.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/denoising_baseline_hard",
        help="评估结果输出目录",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--n_vis", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefer_noisy",
        action="store_true",
        default=True,
        help="优先选择含噪声样本可视化",
    )
    parser.add_argument("--no_prefer_noisy", action="store_true")
    args = parser.parse_args()

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(_FEATURE_ROOT, args.data_dir)
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(_FEATURE_ROOT, args.model_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(_FEATURE_ROOT, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载 test 数据...")
    noisy, clean, labels, parent_index, chunk_index = load_denoising_dataset(args.data_dir, "test")
    noise_mask = (labels > 0.5).any(axis=-1)
    print(f"  noisy: {noisy.shape}, clean: {clean.shape}")

    # ========== Identity baseline ==========
    print("\n========== Identity baseline（noisy 直接作为输出）==========")
    identity_pred = noisy.copy()
    identity_overall = compute_metrics(identity_pred, clean, noise_mask)
    identity_per_sample, _ = compute_per_sample(identity_pred, clean, noise_mask)
    print("Overall:")
    for k, v in identity_overall.items():
        print(f"  {k}: {v:.4f}")
    print("Per-sample 平均:")
    for k, v in identity_per_sample.items():
        print(f"  {k}: {v:.4f}")

    # ========== Learned denoiser ==========
    print("\n========== Learned denoiser ==========")
    model = UNet1DDenoiser(in_channels=1, out_channels=1, base_channels=32, depth=3)
    ckpt = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    print("预测...")
    reconstructed = predict_batch(model, noisy, device)

    learned_overall = compute_metrics(reconstructed, clean, noise_mask)
    learned_per_sample, per_sample_dict = compute_per_sample(reconstructed, clean, noise_mask)

    print("Overall:")
    for k, v in learned_overall.items():
        print(f"  {k}: {v:.4f}")
    print("Per-sample 平均:")
    for k, v in learned_per_sample.items():
        print(f"  {k}: {v:.4f}")

    # ========== 保存指标 ==========
    results = {
        "identity_baseline": {
            "overall": identity_overall,
            "per_sample_mean": identity_per_sample,
        },
        "learned_denoiser": {
            "overall": learned_overall,
            "per_sample_mean": learned_per_sample,
        },
    }
    txt_path = os.path.join(args.output_dir, "test_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("========== Identity baseline ==========\n")
        f.write("Overall:\n")
        for k, v in identity_overall.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("Per-sample 平均:\n")
        for k, v in identity_per_sample.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\n========== Learned denoiser ==========\n")
        f.write("Overall:\n")
        for k, v in learned_overall.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("Per-sample 平均:\n")
        for k, v in learned_per_sample.items():
            f.write(f"  {k}: {v:.4f}\n")
    print(f"\n指标已保存到 {txt_path}")

    json_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存到 {json_path}")

    # ========== 可视化 ==========
    rng = np.random.default_rng(args.seed)
    n_vis = min(args.n_vis, len(noisy))
    if (args.prefer_noisy and not args.no_prefer_noisy) and np.any(noise_mask):
        has_noise = noise_mask.any(axis=1)
        noisy_idx = np.where(has_noise)[0]
        if len(noisy_idx) >= n_vis:
            vis_indices = rng.choice(noisy_idx, size=n_vis, replace=False)
            print(f"  可视化：从 {len(noisy_idx)} 个含噪声样本中抽样 {n_vis} 个")
        else:
            vis_indices = rng.choice(len(noisy), size=n_vis, replace=False)
    else:
        vis_indices = rng.choice(len(noisy), size=n_vis, replace=False)

    sample_metrics_list = []
    for idx in vis_indices:
        m = compute_metrics(reconstructed[idx], clean[idx], noise_mask[idx])
        sample_metrics_list.append(m)

    vis_dir = os.path.join(args.output_dir, "visualizations")
    plot_samples(
        noisy, reconstructed, clean, noise_mask,
        parent_index, chunk_index,
        sample_metrics_list,
        vis_indices,
        vis_dir,
    )


if __name__ == "__main__":
    main()
