"""
Mask-guided denoiser 训练脚本（multilabel segmentation → denoising 两阶段）

与 direct denoising baseline 独立，不修改原脚本。

输入: concat(noisy, masks) [B, 6, 240]
输出: clean [B, 1, 240]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)

from models.unet1d_mask_guided_denoiser import UNet1DMaskGuidedDenoiser


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


def load_mask_guided_dataset(data_dir: str, split: str) -> tuple:
    """加载 mask-guided 数据，返回 (noisy, masks, clean)。输入将构造为 concat(noisy, masks) [N, 6, L]。"""
    path = os.path.join(data_dir, f"{split}_dataset_mask_guided.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    data = np.load(path)
    noisy = np.asarray(data["noisy_signals"], dtype=np.float32)
    clean = np.asarray(data["clean_signals"], dtype=np.float32)
    masks = np.asarray(data["masks"], dtype=np.float32)
    return noisy, masks, clean


def main():
    parser = argparse.ArgumentParser(description="训练 mask-guided denoiser")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="mask-guided 数据集目录，默认根据 mask_source 自动选择",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认根据 mask_source 自动选择",
    )
    parser.add_argument(
        "--mask_source",
        type=str,
        choices=["gt", "pred"],
        default="pred",
        help="mask 来源，需与 data_dir 一致",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mse_weight", type=float, default=1.0)
    parser.add_argument("--l1_weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.mask_source = _infer_mask_source(args.mask_source, args.data_dir, args.output_dir)
    experiment_tag = _infer_experiment_tag(args.data_dir, args.output_dir)

    if args.data_dir is None:
        args.data_dir = os.path.join(_FEATURE_ROOT, f"datasets/multilabel_guided_denoising_{experiment_tag}_{args.mask_source}")
    if args.output_dir is None:
        args.output_dir = os.path.join(_FEATURE_ROOT, f"results/multilabel_guided_denoising_{experiment_tag}_{args.mask_source}")

    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(_FEATURE_ROOT, args.data_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(_FEATURE_ROOT, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Mask-guided denoiser 训练启动 (mask_source={args.mask_source})...", flush=True)
    sys.stdout.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)

    print("加载数据...", flush=True)
    train_noisy, train_masks, train_clean = load_mask_guided_dataset(args.data_dir, "train")
    val_noisy, val_masks, val_clean = load_mask_guided_dataset(args.data_dir, "val")
    print(f"  train: noisy {train_noisy.shape}, masks {train_masks.shape}", flush=True)
    print(f"  val: noisy {val_noisy.shape}, masks {val_masks.shape}", flush=True)

    # 构造输入 [N, 6, L] = [noisy, m0, m1, m2, m3, m4]
    train_input = np.concatenate([
        train_noisy[:, np.newaxis, :],
        np.transpose(train_masks, (0, 2, 1)),
    ], axis=1)
    val_input = np.concatenate([
        val_noisy[:, np.newaxis, :],
        np.transpose(val_masks, (0, 2, 1)),
    ], axis=1)
    train_clean = train_clean[:, np.newaxis, :]
    val_clean = val_clean[:, np.newaxis, :]

    train_ds = TensorDataset(
        torch.from_numpy(train_input),
        torch.from_numpy(train_clean),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_input),
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

    model = UNet1DMaskGuidedDenoiser(in_channels=6, out_channels=1, base_channels=32, depth=3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    config = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "mask_source": args.mask_source,
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
        plt.title(f"Mask-Guided Denoiser Training Loss ({args.mask_source})")
        plt.savefig(os.path.join(args.output_dir, "loss_curves.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  (loss 曲线图保存失败: {e})")
    print(f"\n训练完成。best_model.pt、last_model.pt、config.json 已保存到 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
