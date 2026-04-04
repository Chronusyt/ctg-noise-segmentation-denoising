"""
直接去噪模型训练脚本（direct denoising baseline）

与 segmentation 实验独立，不修改原脚本。

功能：
1. 加载 denoising train/val 数据
2. 训练 1D U-Net 去噪模型
3. MSE + 可选 L1 组合 loss
4. 保存到 results/denoising_baseline_hard/
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

from ctg_pipeline.models.unet1d_denoiser import UNet1DDenoiser
from ctg_pipeline.utils.pathing import DENOISING_DATASETS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    l1_weight: float = 0.0,
) -> torch.Tensor:
    """MSE + 可选 L1。"""
    mse = F.mse_loss(pred, target)
    if l1_weight > 0:
        l1 = F.l1_loss(pred, target)
        return mse_weight * mse + l1_weight * l1
    return mse


def load_denoising_dataset(data_dir: str, split: str) -> tuple:
    """加载 denoising 数据，返回 (noisy_signals, clean_signals)。"""
    path = os.path.join(data_dir, f"{split}_dataset_denoising.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    data = np.load(path)
    noisy = np.asarray(data["noisy_signals"], dtype=np.float32)
    clean = np.asarray(data["clean_signals"], dtype=np.float32)
    return noisy, clean


def main():
    parser = argparse.ArgumentParser(description="训练直接去噪模型")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "denoising_baseline_hard"),
        help="denoising 数据集目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DENOISING_RESULTS_ROOT / "denoising_baseline_hard"),
        help="输出目录",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    print("Direct denoising 训练启动...", flush=True)
    sys.stdout.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)

    print("加载数据...", flush=True)
    train_noisy, train_clean = load_denoising_dataset(args.data_dir, "train")
    val_noisy, val_clean = load_denoising_dataset(args.data_dir, "val")
    print(f"  train: {train_noisy.shape}", flush=True)
    print(f"  val: {val_noisy.shape}", flush=True)

    train_noisy = train_noisy[:, np.newaxis, :]  # [N, 1, L]
    train_clean = train_clean[:, np.newaxis, :]
    val_noisy = val_noisy[:, np.newaxis, :]
    val_clean = val_clean[:, np.newaxis, :]

    train_ds = TensorDataset(
        torch.from_numpy(train_noisy),
        torch.from_numpy(train_clean),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_noisy),
        torch.from_numpy(val_clean),
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

    model = UNet1DDenoiser(in_channels=1, out_channels=1, base_channels=32, depth=3)
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
        "mse_weight": args.mse_weight,
        "l1_weight": args.l1_weight,
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
            pred = model(x)
            loss = combined_loss(
                pred, y,
                mse_weight=args.mse_weight,
                l1_weight=args.l1_weight,
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
                pred = model(x)
                loss = combined_loss(
                    pred, y,
                    mse_weight=args.mse_weight,
                    l1_weight=args.l1_weight,
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

    # 保存 last_model
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_losses[-1],
            "config": config,
        },
        os.path.join(args.output_dir, "last_model.pt"),
    )

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
        plt.title("Denoiser Training Loss")
        plt.savefig(os.path.join(args.output_dir, "loss_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  (loss 曲线图保存失败: {e})")
    print(f"\n训练完成。best_model.pt、last_model.pt、config.json 已保存到 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
