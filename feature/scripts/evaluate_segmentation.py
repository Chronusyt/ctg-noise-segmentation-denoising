"""
噪声分割模型评估与可视化

功能：
1. 在 test 集上计算 Precision、Recall、F1、IoU、Dice（per-sample 与 overall）
2. 随机选择样本，绘制 noisy_signal、GT mask、predicted mask
3. 噪声区域用 shading 标出
4. 保存到 results/segmentation/
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)

from models.unet1d_segmentation import UNet1DSegmentation


def compute_metrics(
    pred_bin: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-6,
) -> dict[str, float]:
    """
    单样本二分类指标。
    pred_bin, gt: [L] 或 [1,L]，取值 0/1。
    """
    pred_flat = pred_bin.ravel().astype(np.float64)
    gt_flat = gt.ravel().astype(np.float64)
    tp = (pred_flat * gt_flat).sum()
    fp = (pred_flat * (1 - gt_flat)).sum()
    fn = ((1 - pred_flat) * gt_flat).sum()
    tn = ((1 - pred_flat) * (1 - gt_flat)).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
    }


def evaluate_overall(
    pred_bin: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-6,
) -> dict[str, float]:
    """Overall 指标：将所有样本展平后计算。"""
    return compute_metrics(pred_bin, gt, smooth)


def evaluate_per_sample(
    pred_bin: np.ndarray,
    gt: np.ndarray,
) -> Tuple[dict[str, float], dict[str, np.ndarray]]:
    """
    Per-sample 指标。
    pred_bin, gt: [N, L] 或 [N, 1, L]
    返回 (mean_metrics, per_sample_metrics)
    """
    pred_flat = pred_bin.reshape(pred_bin.shape[0], -1)
    gt_flat = gt.reshape(gt.shape[0], -1)
    n = pred_flat.shape[0]
    per_prec = np.zeros(n)
    per_rec = np.zeros(n)
    per_f1 = np.zeros(n)
    per_iou = np.zeros(n)
    per_dice = np.zeros(n)
    for i in range(n):
        m = compute_metrics(pred_flat[i : i + 1], gt_flat[i : i + 1])
        per_prec[i] = m["precision"]
        per_rec[i] = m["recall"]
        per_f1[i] = m["f1"]
        per_iou[i] = m["iou"]
        per_dice[i] = m["dice"]
    mean_metrics = {
        "precision": float(np.mean(per_prec)),
        "recall": float(np.mean(per_rec)),
        "f1": float(np.mean(per_f1)),
        "iou": float(np.mean(per_iou)),
        "dice": float(np.mean(per_dice)),
    }
    per_sample = {
        "precision": per_prec,
        "recall": per_rec,
        "f1": per_f1,
        "iou": per_iou,
        "dice": per_dice,
    }
    return mean_metrics, per_sample


def load_dataset(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载数据集，返回 (signals, masks)。"""
    path = os.path.join(data_dir, f"{split}_dataset.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    data = np.load(path)
    signals = np.asarray(data["signals"], dtype=np.float32)
    masks = np.asarray(data["masks"], dtype=np.float32)
    return signals, masks


def predict_batch(
    model: torch.nn.Module,
    signals: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """批量预测，返回 [N, L] 二值 mask。"""
    model.eval()
    N = signals.shape[0]
    # [N, L] -> [N, 1, L]
    if signals.ndim == 2:
        signals = signals[:, np.newaxis, :]
    preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = torch.from_numpy(signals[i : i + batch_size]).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append(probs)
    preds = np.concatenate(preds, axis=0)
    pred_bin = (preds >= 0.5).astype(np.float32)
    return pred_bin.squeeze()


def plot_samples(
    signals: np.ndarray,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    indices: np.ndarray,
    save_dir: str,
) -> None:
    """绘制若干样本：noisy_signal、GT mask、predicted mask，噪声区域 shading。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    os.makedirs(save_dir, exist_ok=True)
    L = signals.shape[1]
    x = np.arange(L)

    for k, idx in enumerate(indices):
        sig = signals[idx]
        gt = gt_masks[idx]
        pred = pred_masks[idx]

        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
        # Row 0: noisy signal
        axes[0].plot(x, sig, "k-", linewidth=0.8, label="noisy signal")
        axes[0].fill_between(x, sig.min() - 1, sig.max() + 1, where=gt > 0.5, alpha=0.3, color="red", label="GT noise")
        axes[0].set_ylabel("FHR")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"Sample {idx} (index {k})")
        axes[0].grid(True, alpha=0.3)

        # Row 1: GT mask
        axes[1].fill_between(x, 0, 1, where=gt > 0.5, alpha=0.6, color="red", label="GT mask")
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].set_ylabel("GT mask")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        # Row 2: predicted mask
        axes[2].fill_between(x, 0, 1, where=pred > 0.5, alpha=0.6, color="blue", label="Pred mask")
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].set_ylabel("Pred mask")
        axes[2].set_xlabel("Time (samples)")
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{k}_idx_{idx}.png"), dpi=150)
        plt.close()
    print(f"  可视化已保存到 {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="评估噪声分割模型")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/segmentation",
        help="分割数据集目录",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/segmentation/best_model.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/segmentation",
        help="评估结果与可视化输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda/cpu",
    )
    parser.add_argument(
        "--n_vis",
        type=int,
        default=5,
        help="可视化样本数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于选取可视化样本）",
    )
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
    signals, masks = load_dataset(args.data_dir, "test")
    print(f"  signals: {signals.shape}, masks: {masks.shape}")

    print("加载模型...")
    model = UNet1DSegmentation(in_channels=1, out_channels=1, base_channels=32, depth=3)
    ckpt = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    print("预测...")
    pred_bin = predict_batch(model, signals, device)
    if pred_bin.ndim == 3:
        pred_bin = pred_bin.squeeze(1)
    gt = masks

    print("\n========== 评估指标 ==========")
    # Per-sample 平均
    mean_metrics, _ = evaluate_per_sample(pred_bin, gt)
    print("\nPer-sample 平均:")
    for k, v in mean_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Overall
    overall = evaluate_overall(pred_bin, gt)
    print("\nOverall:")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")

    # 保存指标
    metrics_path = os.path.join(args.output_dir, "test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Per-sample 平均:\n")
        for k, v in mean_metrics.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nOverall:\n")
        for k, v in overall.items():
            f.write(f"  {k}: {v:.4f}\n")
    print(f"\n指标已保存到 {metrics_path}")

    # 可视化
    rng = np.random.default_rng(args.seed)
    n_vis = min(args.n_vis, len(signals))
    vis_indices = rng.choice(len(signals), size=n_vis, replace=False)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    plot_samples(signals, masks, pred_bin, vis_indices, vis_dir)


if __name__ == "__main__":
    main()
