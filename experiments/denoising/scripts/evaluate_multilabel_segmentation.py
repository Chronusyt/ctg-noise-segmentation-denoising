"""
五类噪声分割模型评估与可视化（multi-label temporal segmentation）

与 binary segmentation 独立，不修改原 evaluate_segmentation.py。

功能：
1. 在 test 集上对每类输出 precision/recall/f1/iou/dice
2. macro / micro / overall average
3. 可视化：noisy_signal + 5 类 GT/pred mask
4. 保存到 results/multilabel_segmentation_hard/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.models.unet1d_multilabel_segmentation import UNet1DMultilabelSegmentation
from ctg_pipeline.utils.pathing import DENOISING_DATASETS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path

CLASS_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]  # 5 类不同颜色


def compute_metrics_binary(
    pred_bin: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """二分类指标。pred_bin, gt: 展平后 0/1。"""
    pred_flat = pred_bin.ravel().astype(np.float64)
    gt_flat = gt.ravel().astype(np.float64)
    tp = (pred_flat * gt_flat).sum()
    fp = (pred_flat * (1 - gt_flat)).sum()
    fn = ((1 - pred_flat) * gt_flat).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    union = tp + fp + fn
    iou = (tp + smooth) / (union + smooth)
    dice = (2 * tp + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
    }


def evaluate_per_class(
    pred_bin: np.ndarray,
    gt: np.ndarray,
    smooth: float = 1e-6,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, float]]:
    """
    pred_bin, gt: [N, 5, L] 或 [N, L, 5]
    返回 (per_class_metrics, macro_avg, micro_avg)
    """
    if pred_bin.shape[1] == 5:
        pred_bin = pred_bin  # [N, 5, L]
        gt = gt
    else:
        pred_bin = np.transpose(pred_bin, (0, 2, 1))  # [N, L, 5] -> [N, 5, L]
        gt = np.transpose(gt, (0, 2, 1))

    per_class: Dict[str, Dict[str, float]] = {}
    for c, name in enumerate(CLASS_NAMES):
        p = pred_bin[:, c].ravel()
        g = gt[:, c].ravel()
        per_class[name] = compute_metrics_binary(p, g, smooth)

    # macro: 各类指标平均
    macro = {}
    for metric in ["precision", "recall", "f1", "iou", "dice"]:
        macro[metric] = float(np.mean([per_class[n][metric] for n in CLASS_NAMES]))

    # micro: 将所有类展平后整体计算
    pred_all = pred_bin.ravel()
    gt_all = gt.ravel()
    micro = compute_metrics_binary(pred_all, gt_all, smooth)

    return per_class, macro, micro


def load_multilabel_dataset(data_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载 multilabel 数据。"""
    path = os.path.join(data_dir, f"{split}_dataset_multilabel.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    data = np.load(path)
    signals = np.asarray(data["signals"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.float32)
    return signals, labels


def predict_batch(
    model: torch.nn.Module,
    signals: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """批量预测，返回 [N, 5, L] 二值 mask。"""
    model.eval()
    N = signals.shape[0]
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
    return pred_bin


def plot_multilabel_samples(
    signals: np.ndarray,
    gt_labels: np.ndarray,
    pred_labels: np.ndarray,
    indices: np.ndarray,
    save_dir: str,
) -> None:
    """
    绘制 multilabel 样本：noisy_signal + 5 类 GT/pred mask。
    gt_labels, pred_labels: [N, L, 5]
    """
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
        gt = gt_labels[idx]  # [L, 5]
        pred = pred_labels[idx]  # [L, 5]

        # 布局：1 行 signal + 2 行（GT 5 类 + Pred 5 类）或合并
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        # Row 0: noisy signal + 5 类 shading
        ax0 = axes[0]
        ax0.plot(x, sig, "k-", linewidth=0.8, label="noisy signal")
        for c, (name, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            ax0.fill_between(x, sig.min() - 1, sig.max() + 1, where=gt[:, c] > 0.5,
                alpha=0.25, color=col, label=f"GT {name}")
        ax0.set_ylabel("FHR")
        ax0.legend(loc="upper right", fontsize=7, ncol=2)
        ax0.set_title(f"Sample {idx} (index {k}) - Noisy signal + GT masks")
        ax0.grid(True, alpha=0.3)

        # Row 1: GT masks 5 类
        ax1 = axes[1]
        for c, (name, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            ax1.fill_between(x, c - 0.4, c + 0.4, where=gt[:, c] > 0.5, alpha=0.7, color=col)
        ax1.set_ylim(-0.5, 4.5)
        ax1.set_yticks(range(5))
        ax1.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax1.set_ylabel("GT")
        ax1.set_title("GT masks (5 classes)")
        ax1.grid(True, alpha=0.3)

        # Row 2: Pred masks 5 类
        ax2 = axes[2]
        for c, (name, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            ax2.fill_between(x, c - 0.4, c + 0.4, where=pred[:, c] > 0.5, alpha=0.7, color=col)
        ax2.set_ylim(-0.5, 4.5)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(CLASS_NAMES, fontsize=8)
        ax2.set_ylabel("Pred")
        ax2.set_title("Predicted masks (5 classes)")
        ax2.grid(True, alpha=0.3)

        # Row 3: 叠加对比（GT vs Pred，不同透明度）
        ax3 = axes[3]
        ax3.plot(x, sig, "k-", linewidth=0.5, alpha=0.5)
        for c, col in enumerate(COLORS):
            ax3.fill_between(x, sig.min() - 1, sig.max() + 1, where=pred[:, c] > 0.5,
                alpha=0.35, color=col, label=CLASS_NAMES[c])
        ax3.set_ylabel("FHR")
        ax3.set_xlabel("Time (samples)")
        ax3.set_title("Predicted masks overlay")
        ax3.legend(loc="upper right", fontsize=7, ncol=2)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{k}_idx_{idx}.png"), dpi=150)
        plt.close()
    print(f"  可视化已保存到 {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="评估五类噪声分割模型")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "multilabel_segmentation_hard"),
        help="multilabel 数据集目录",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DENOISING_RESULTS_ROOT / "multilabel_segmentation_hard" / "best_model.pt"),
        help="模型权重路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DENOISING_RESULTS_ROOT / "multilabel_segmentation_hard"),
        help="评估结果输出目录",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--n_vis", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prefer_noisy",
        action="store_true",
        default=True,
        help="优先选择含噪声的样本进行可视化（默认开启）",
    )
    parser.add_argument(
        "--no_prefer_noisy",
        action="store_true",
        help="禁用优先含噪声，完全随机抽样",
    )
    args = parser.parse_args()

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.model_path = str(resolve_repo_path(args.model_path))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载 test 数据...")
    signals, labels = load_multilabel_dataset(args.data_dir, "test")
    print(f"  signals: {signals.shape}, labels: {labels.shape}")

    print("加载模型...")
    model = UNet1DMultilabelSegmentation(in_channels=1, out_channels=5, base_channels=32, depth=3)
    ckpt = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    print("预测...")
    pred_bin = predict_batch(model, signals, device)  # [N, 5, L]
    gt = np.transpose(labels, (0, 2, 1))  # [N, L, 5] -> [N, 5, L]

    print("\n========== 评估指标 ==========")
    per_class, macro, micro = evaluate_per_class(pred_bin, gt)

    print("\nPer-class metrics:")
    for name in CLASS_NAMES:
        m = per_class[name]
        print(f"  {name}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} IoU={m['iou']:.4f} Dice={m['dice']:.4f}")

    print("\nMacro average:")
    for k, v in macro.items():
        print(f"  {k}: {v:.4f}")

    print("\nMicro average:")
    for k, v in micro.items():
        print(f"  {k}: {v:.4f}")

    # overall = micro（将所有点展平）
    overall = micro

    # 保存 txt
    txt_path = os.path.join(args.output_dir, "test_metrics.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Per-class metrics:\n")
        for name in CLASS_NAMES:
            m = per_class[name]
            f.write(f"  {name}: precision={m['precision']:.4f} recall={m['recall']:.4f} f1={m['f1']:.4f} iou={m['iou']:.4f} dice={m['dice']:.4f}\n")
        f.write("\nMacro average:\n")
        for k, v in macro.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nMicro average:\n")
        for k, v in micro.items():
            f.write(f"  {k}: {v:.4f}\n")
        f.write("\nOverall (same as micro):\n")
        for k, v in overall.items():
            f.write(f"  {k}: {v:.4f}\n")
    print(f"\n指标已保存到 {txt_path}")

    # 保存 json
    json_data = {
        "per_class": per_class,
        "macro": macro,
        "micro": micro,
        "overall": overall,
    }
    json_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存到 {json_path}")

    # 可视化：优先选择含噪声的样本，避免 GT/Pred 全空白
    rng = np.random.default_rng(args.seed)
    n_vis = min(args.n_vis, len(signals))
    if (args.prefer_noisy and not args.no_prefer_noisy) and labels.size > 0:
        has_noise = (labels > 0.5).any(axis=(1, 2))
        noisy_idx = np.where(has_noise)[0]
        if len(noisy_idx) >= n_vis:
            vis_indices = rng.choice(noisy_idx, size=n_vis, replace=False)
            print(f"  可视化：从 {len(noisy_idx)} 个含噪声样本中抽样 {n_vis} 个")
        else:
            vis_indices = rng.choice(len(signals), size=n_vis, replace=False)
            print(f"  可视化：含噪声样本不足 {n_vis}，随机抽样")
    else:
        vis_indices = rng.choice(len(signals), size=n_vis, replace=False)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    pred_for_vis = np.transpose(pred_bin, (0, 2, 1))  # [N, 5, L] -> [N, L, 5]
    plot_multilabel_samples(signals, labels, pred_for_vis, vis_indices, vis_dir)


if __name__ == "__main__":
    main()
