"""
噪声分割模型训练脚本

功能：
1. 加载 train/val 数据
2. 训练 1D U-Net
3. BCEWithLogitsLoss + 可选 Dice loss
4. GPU 支持
5. 保存 best_model.pt、loss 曲线
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.models.unet1d_segmentation import UNet1DSegmentation
from ctg_pipeline.utils.pathing import DENOISING_DATASETS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice loss for binary segmentation."""
    pred = torch.sigmoid(pred)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 0.0,
) -> torch.Tensor:
    """BCE + 可选 Dice loss。"""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
    if dice_weight > 0:
        d = dice_loss(pred, target)
        return bce_weight * bce + dice_weight * d
    return bce


def load_dataset(data_dir: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """加载 train/val/test 数据，返回 (signals, masks)。"""
    path = os.path.join(data_dir, f"{split}_dataset.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    data = np.load(path)
    signals = np.asarray(data["signals"], dtype=np.float32)
    masks = np.asarray(data["masks"], dtype=np.float32)
    return signals, masks


def main():
    parser = argparse.ArgumentParser(description="训练噪声分割模型")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "segmentation"),
        help="分割数据集目录（含 train/val_dataset.npz）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DENOISING_RESULTS_ROOT / "segmentation"),
        help="模型与曲线输出目录",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="训练轮数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率",
    )
    parser.add_argument(
        "--dice_weight",
        type=float,
        default=0.5,
        help="Dice loss 权重，0 表示仅用 BCE",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda/cpu，空则自动选择",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    print("加载数据...")
    train_sig, train_mask = load_dataset(args.data_dir, "train")
    val_sig, val_mask = load_dataset(args.data_dir, "val")
    print(f"  train: {train_sig.shape}, val: {val_mask.shape}")

    # [N, L] -> [N, 1, L]
    train_sig = train_sig[:, np.newaxis, :]
    train_mask = train_mask[:, np.newaxis, :]
    val_sig = val_sig[:, np.newaxis, :]
    val_mask = val_mask[:, np.newaxis, :]

    train_ds = TensorDataset(
        torch.from_numpy(train_sig),
        torch.from_numpy(train_mask),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_sig),
        torch.from_numpy(val_mask),
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

    model = UNet1DSegmentation(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_loss = float("inf")
    train_losses: list[float] = []
    val_losses: list[float] = []

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
                },
                save_path,
            )
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f} [best, saved]")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # 保存 loss 曲线数据与图像
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
        plt.title("Segmentation Training Loss")
        plt.savefig(os.path.join(args.output_dir, "loss_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  (loss 曲线图保存失败: {e})")
    print(f"\n训练完成。best_model.pt 与 loss_curves.npz 已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
