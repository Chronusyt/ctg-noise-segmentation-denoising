"""
五类噪声分割模型训练脚本（multi-label temporal segmentation）

与 binary segmentation 独立，不修改原 train_segmentation.py。

功能：
1. 加载 multilabel train/val 数据
2. 训练 5 通道 U-Net
3. BCEWithLogitsLoss + pos_weight 处理类别不平衡
4. 可选 Dice loss（按通道平均）
5. 保存到 results/multilabel_segmentation_hard/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.models.unet1d_multilabel_segmentation import UNet1DMultilabelSegmentation
from ctg_pipeline.utils.pathing import DENOISING_DATASETS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path

CLASS_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]


def compute_pos_weight(labels: np.ndarray) -> np.ndarray:
    """
    计算每类 pos_weight = neg_count / pos_count。
    labels: [N, L, 5]
    返回 [5]
    """
    pos_weight = np.ones(5, dtype=np.float32)
    for c in range(5):
        pos = (labels[:, :, c] > 0.5).sum()
        neg = (labels[:, :, c] <= 0.5).sum()
        if pos > 0:
            pos_weight[c] = float(neg) / float(pos)
        else:
            pos_weight[c] = 1.0
    return pos_weight


def dice_loss_per_channel(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """按通道计算 Dice loss 并平均。pred/target: [B, 5, L]"""
    pred = torch.sigmoid(pred)
    losses = []
    for c in range(pred.shape[1]):
        p = pred[:, c].reshape(-1)
        t = target[:, c].reshape(-1)
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice = (2 * inter + smooth) / (union + smooth)
        losses.append(1 - dice)
    return torch.stack(losses).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    bce_weight: float = 1.0,
    dice_weight: float = 0.0,
) -> torch.Tensor:
    """BCE + 可选 Dice。pred/target: [B, 5, L]。pos_weight 按通道加权。"""
    if pos_weight is not None:
        pos_weight = pos_weight.to(pred.device)
        # 按通道计算 BCE 并加权（pos_weight 在通道维）
        bce_per_ch = []
        for c in range(pred.shape[1]):
            bc = F.binary_cross_entropy_with_logits(
                pred[:, c], target[:, c],
                pos_weight=pos_weight[c : c + 1],
                reduction="mean",
            )
            bce_per_ch.append(bc)
        bce = torch.stack(bce_per_ch).mean()
    else:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
    if dice_weight > 0:
        d = dice_loss_per_channel(pred, target)
        return bce_weight * bce + dice_weight * d
    return bce


def load_multilabel_dataset(data_dir: str, split: str) -> tuple:
    """加载 multilabel 数据，返回 (signals, labels)。"""
    fname = f"{split}_dataset_multilabel.npz"
    path = os.path.join(data_dir, fname)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    data = np.load(path)
    signals = np.asarray(data["signals"], dtype=np.float32)
    labels = np.asarray(data["labels"], dtype=np.float32)
    return signals, labels


def main():
    parser = argparse.ArgumentParser(description="训练五类噪声分割模型")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "multilabel_segmentation_hard"),
        help="multilabel 数据集目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DENOISING_RESULTS_ROOT / "multilabel_segmentation_hard"),
        help="输出目录",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--use_pos_weight", action="store_true", default=True,
        help="使用 pos_weight 处理类别不平衡",
    )
    parser.add_argument("--no_pos_weight", action="store_true",
        help="禁用 pos_weight",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    use_pos_weight = args.use_pos_weight and not args.no_pos_weight

    # 立即刷新输出，避免缓冲导致终端无显示
    print("Multilabel 训练启动...", flush=True)
    sys.stdout.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)

    print("加载数据...", flush=True)
    train_sig, train_labels = load_multilabel_dataset(args.data_dir, "train")
    val_sig, val_labels = load_multilabel_dataset(args.data_dir, "val")
    print(f"  train: {train_sig.shape}, labels {train_labels.shape}", flush=True)
    print(f"  val: {val_sig.shape}, labels {val_labels.shape}", flush=True)

    # 统计每类正样本比例与 pos_weight
    pos_ratios = []
    pos_weight_arr = compute_pos_weight(train_labels)
    print("\n每类正样本比例与 pos_weight:", flush=True)
    for c, name in enumerate(CLASS_NAMES):
        r = (train_labels[:, :, c] > 0.5).mean()
        pos_ratios.append(float(r))
        print(f"  {name}: pos_ratio={r:.4f}, pos_weight={pos_weight_arr[c]:.4f}", flush=True)
    pos_weight = torch.from_numpy(pos_weight_arr).float() if use_pos_weight else None
    if use_pos_weight:
        print("  已启用 pos_weight 处理类别不平衡", flush=True)

    # [N, L] -> [N, 1, L], labels [N, L, 5] -> [N, 5, L]
    train_sig = train_sig[:, np.newaxis, :]
    train_labels = np.transpose(train_labels, (0, 2, 1))  # [N, 5, L]
    val_sig = val_sig[:, np.newaxis, :]
    val_labels = np.transpose(val_labels, (0, 2, 1))

    train_ds = TensorDataset(
        torch.from_numpy(train_sig),
        torch.from_numpy(train_labels),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_sig),
        torch.from_numpy(val_labels),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = UNet1DMultilabelSegmentation(in_channels=1, out_channels=5, base_channels=32, depth=3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    config = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dice_weight": args.dice_weight,
        "use_pos_weight": use_pos_weight,
        "pos_weight": pos_weight_arr.tolist() if use_pos_weight else None,
        "pos_ratios": pos_ratios,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = combined_loss(
                logits, y,
                pos_weight=pos_weight,
                bce_weight=1.0,
                dice_weight=args.dice_weight,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = combined_loss(
                    logits, y,
                    pos_weight=pos_weight,
                    bce_weight=1.0,
                    dice_weight=args.dice_weight,
                )
                val_loss += loss.item()
                n_val += 1
        val_loss = val_loss / max(n_val, 1)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                save_path,
            )
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f} [best, saved]", flush=True)
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}", flush=True)

    np.savez(
        os.path.join(args.output_dir, "loss_curves.npz"),
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
    )
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label="Train")
        plt.plot(val_losses, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Multilabel Segmentation Training Loss")
        plt.savefig(os.path.join(args.output_dir, "loss_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  (loss 曲线图保存失败: {e})")
    print(f"\n训练完成。best_model.pt、config.json、loss_curves 已保存到 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
