"""Train physiological multitask models on the clinical multitask dataset."""
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.data.multitask_dataset import ClinicalMultitaskDataset
from ctg_pipeline.models.unet1d_physiological_multitask import UNet1DPhysiologicalMultitask
from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_DATASETS_ROOT, resolve_repo_path


SCALAR_KEYS = ("baseline", "stv", "ltv", "baseline_variability")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_subset(dataset: ClinicalMultitaskDataset, max_samples: int, seed: int) -> ClinicalMultitaskDataset | Subset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_samples, replace=False))
    return Subset(dataset, indices.tolist())


def compute_label_stats(dataset: ClinicalMultitaskDataset) -> Dict[str, Dict[str, float]]:
    values = {
        "baseline": dataset.baseline,
        "stv": dataset.stv,
        "ltv": dataset.ltv,
        "baseline_variability": dataset.baseline_variability,
    }
    stats: Dict[str, Dict[str, float]] = {}
    for key, arr in values.items():
        arr = np.asarray(arr, dtype=np.float64)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr))
        stats[key] = {"mean": mean, "std": std if std > 1e-6 else 1.0}
    return stats


def compute_pos_weight(arr: np.ndarray, max_pos_weight: float) -> float:
    arr = np.asarray(arr) > 0.5
    pos = float(arr.sum())
    neg = float(arr.size - arr.sum())
    if pos <= 0:
        return 1.0
    return float(min(max(neg / pos, 1.0), max_pos_weight))


def positive_ratio(arr: np.ndarray) -> float:
    return float(np.asarray(arr, dtype=np.float32).mean())


def normalize_scalar(batch: dict, key: str, label_stats: Dict[str, Dict[str, float]], device: str) -> torch.Tensor:
    value = batch[f"{key}_label"].to(device)
    mean = label_stats[key]["mean"]
    std = label_stats[key]["std"]
    return (value - mean) / std


def build_model_input(batch: dict, device: str, input_mode: str) -> torch.Tensor:
    noisy = batch["noisy_signal"].to(device)
    if input_mode == "no_mask":
        return noisy
    if input_mode == "gt_mask":
        mask = batch["mask"].to(device)
        return torch.cat([noisy, mask], dim=1)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def build_corrupted_region_mask(batch: dict, device: str) -> torch.Tensor:
    mask = batch["mask"].to(device)
    if mask.ndim != 3:
        raise ValueError(f"Expected mask shape [B, 5, L], got {mask.shape}")
    return mask.amax(dim=1, keepdim=True).clamp(0.0, 1.0)


def pointwise_reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return (pred - target) ** 2
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(f"Unknown reconstruction loss: {kind}")


def scalar_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "smooth_l1":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unknown scalar loss: {kind}")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if float(denom.detach().cpu()) <= 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return (values * mask).sum() / denom


def selective_reconstruction_terms(
    pred: torch.Tensor,
    target: torch.Tensor,
    corrupted_mask: torch.Tensor,
    kind: str,
    mode: str,
    lambda_corrupt: float,
    lambda_clean: float,
) -> Dict[str, torch.Tensor]:
    pointwise = pointwise_reconstruction_loss(pred, target, kind)
    clean_mask = 1.0 - corrupted_mask
    recon_corrupt = masked_mean(pointwise, corrupted_mask)
    recon_clean = masked_mean(pointwise, clean_mask)

    if mode == "overall":
        recon_total = pointwise.mean()
    elif mode == "selective":
        recon_total = lambda_corrupt * recon_corrupt + lambda_clean * recon_clean
    else:
        raise ValueError(f"Unknown reconstruction_mode: {mode}")

    return {
        "reconstruction": recon_total,
        "recon_corrupt": recon_corrupt,
        "recon_clean": recon_clean,
    }


class BinaryEventLoss(nn.Module):
    """Binary event loss with BCE or sigmoid focal loss."""

    def __init__(
        self,
        loss_type: str,
        pos_weight: float,
        focal_gamma: float,
        focal_alpha: float,
        device: str,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.focal_gamma = float(focal_gamma)
        self.focal_alpha = float(focal_alpha)
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)], dtype=torch.float32, device=device))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

        if self.loss_type != "focal":
            raise ValueError(f"Unsupported event loss type: {self.loss_type}")

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = targets * self.focal_alpha + (1.0 - targets) * (1.0 - self.focal_alpha)
        modulating = (1.0 - p_t).clamp(min=0.0) ** self.focal_gamma
        return (alpha_t * modulating * bce).mean()


def compute_losses(
    outputs: dict,
    batch: dict,
    device: str,
    label_stats: Dict[str, Dict[str, float]],
    acc_loss_fn: nn.Module,
    dec_loss_fn: nn.Module,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    clean = batch["clean_signal"].to(device)
    acc_label = batch["acc_label"].to(device)
    dec_label = batch["dec_label"].to(device)
    corrupted_mask = build_corrupted_region_mask(batch, device)

    losses = selective_reconstruction_terms(
        outputs["reconstruction"],
        clean,
        corrupted_mask,
        kind=args.reconstruction_loss,
        mode=args.reconstruction_mode,
        lambda_corrupt=args.lambda_corrupt,
        lambda_clean=args.lambda_clean,
    )
    losses.update(
        {
            "acc": acc_loss_fn(outputs["acc_logits"], acc_label),
            "dec": dec_loss_fn(outputs["dec_logits"], dec_label),
            "baseline": scalar_loss(
                outputs["baseline"],
                normalize_scalar(batch, "baseline", label_stats, device),
                args.scalar_loss,
            ),
            "stv": scalar_loss(outputs["stv"], normalize_scalar(batch, "stv", label_stats, device), args.scalar_loss),
            "ltv": scalar_loss(outputs["ltv"], normalize_scalar(batch, "ltv", label_stats, device), args.scalar_loss),
            "baseline_variability": scalar_loss(
                outputs["baseline_variability"],
                normalize_scalar(batch, "baseline_variability", label_stats, device),
                args.scalar_loss,
            ),
        }
    )
    losses["total"] = (
        args.reconstruction_weight * losses["reconstruction"]
        + args.acc_weight * losses["acc"]
        + args.dec_weight * losses["dec"]
        + args.baseline_weight * losses["baseline"]
        + args.stv_weight * losses["stv"]
        + args.ltv_weight * losses["ltv"]
        + args.bv_weight * losses["baseline_variability"]
    )
    return losses


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    label_stats: Dict[str, Dict[str, float]],
    acc_loss_fn: nn.Module,
    dec_loss_fn: nn.Module,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {
        "total": 0.0,
        "reconstruction": 0.0,
        "recon_corrupt": 0.0,
        "recon_clean": 0.0,
        "acc": 0.0,
        "dec": 0.0,
        "baseline": 0.0,
        "stv": 0.0,
        "ltv": 0.0,
        "baseline_variability": 0.0,
    }
    n_batches = 0

    for batch in loader:
        x = build_model_input(batch, device, args.input_mode)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            outputs = model(x)
            losses = compute_losses(outputs, batch, device, label_stats, acc_loss_fn, dec_loss_fn, args)
            if is_train:
                losses["total"].backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def write_train_log_header(path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "lr",
                "train_total",
                "train_recon_total",
                "train_recon_corrupt",
                "train_recon_clean",
                "train_acc",
                "train_dec",
                "train_baseline",
                "train_stv",
                "train_ltv",
                "train_bv",
                "val_total",
                "val_recon_total",
                "val_recon_corrupt",
                "val_recon_clean",
                "val_acc",
                "val_dec",
                "val_baseline",
                "val_stv",
                "val_ltv",
                "val_bv",
            ]
        )


def append_train_log(path: str, epoch: int, lr: float, train: dict, val: dict) -> None:
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                lr,
                train["total"],
                train["reconstruction"],
                train["recon_corrupt"],
                train["recon_clean"],
                train["acc"],
                train["dec"],
                train["baseline"],
                train["stv"],
                train["ltv"],
                train["baseline_variability"],
                val["total"],
                val["reconstruction"],
                val["recon_corrupt"],
                val["recon_clean"],
                val["acc"],
                val["dec"],
                val["baseline"],
                val["stv"],
                val["ltv"],
                val["baseline_variability"],
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train physiological multitask model (v1 no-mask / v2 gt-mask auxiliary)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "clinical_multitask_physiology_v1"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "physiological_multitask" / "clinical_v1_no_mask"),
    )
    parser.add_argument("--input_mode", choices=["no_mask", "gt_mask"], default="no_mask")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--scalar_hidden_channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--no_residual_reconstruction",
        action="store_true",
        help="Disable reconstruction = noisy + residual output",
    )
    parser.add_argument("--reconstruction_loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--reconstruction_mode", choices=["overall", "selective"], default="overall")
    parser.add_argument("--lambda_corrupt", type=float, default=3.0)
    parser.add_argument("--lambda_clean", type=float, default=1.0)
    parser.add_argument("--scalar_loss", choices=["mse", "smooth_l1"], default="smooth_l1")
    parser.add_argument("--acc_event_loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--dec_event_loss", choices=["bce", "focal"], default="bce")
    parser.add_argument("--acc_focal_gamma", type=float, default=2.0)
    parser.add_argument("--dec_focal_gamma", type=float, default=2.0)
    parser.add_argument("--acc_focal_alpha", type=float, default=0.25)
    parser.add_argument("--dec_focal_alpha", type=float, default=0.75)
    parser.add_argument("--reconstruction_weight", type=float, default=1.0)
    parser.add_argument("--acc_weight", type=float, default=0.5)
    parser.add_argument("--dec_weight", type=float, default=0.5)
    parser.add_argument("--baseline_weight", type=float, default=0.2)
    parser.add_argument("--stv_weight", type=float, default=0.2)
    parser.add_argument("--ltv_weight", type=float, default=0.2)
    parser.add_argument("--bv_weight", type=float, default=0.2)
    parser.add_argument("--acc_pos_weight", type=float, default=0.0, help="0 means compute from train labels")
    parser.add_argument("--dec_pos_weight", type=float, default=0.0, help="0 means compute from train labels")
    parser.add_argument("--max_pos_weight", type=float, default=50.0)
    parser.add_argument("--clip_grad_norm", type=float, default=5.0)
    parser.add_argument("--max_train_samples", type=int, default=0, help="Debug only; 0 means full train split")
    parser.add_argument("--max_val_samples", type=int, default=0, help="Debug only; 0 means full val split")
    args = parser.parse_args()

    args.data_dir = str(resolve_repo_path(args.data_dir))
    args.output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.startswith("cuda")
    print("Physiological multitask 训练启动", flush=True)
    print(f"使用设备: {device}", flush=True)
    print(f"输入模式: {args.input_mode}", flush=True)
    print(f"重建模式: {args.reconstruction_mode}", flush=True)
    print(f"事件损失: acc={args.acc_event_loss}, dec={args.dec_event_loss}", flush=True)
    print(f"数据目录: {args.data_dir}", flush=True)

    train_path = os.path.join(args.data_dir, "train_dataset_multitask.npz")
    val_path = os.path.join(args.data_dir, "val_dataset_multitask.npz")
    train_full = ClinicalMultitaskDataset(train_path)
    val_full = ClinicalMultitaskDataset(val_path)
    train_ds = make_subset(train_full, args.max_train_samples, args.seed)
    val_ds = make_subset(val_full, args.max_val_samples, args.seed + 1)
    print(f"train samples: {len(train_ds)} (full={len(train_full)})", flush=True)
    print(f"val samples: {len(val_ds)} (full={len(val_full)})", flush=True)

    label_stats = compute_label_stats(train_full)
    acc_positive_ratio = positive_ratio(train_full.acc_labels)
    dec_positive_ratio = positive_ratio(train_full.dec_labels)
    acc_pos_weight = args.acc_pos_weight if args.acc_pos_weight > 0 else compute_pos_weight(train_full.acc_labels, args.max_pos_weight)
    dec_pos_weight = args.dec_pos_weight if args.dec_pos_weight > 0 else compute_pos_weight(train_full.dec_labels, args.max_pos_weight)
    print(f"scalar label stats: {label_stats}", flush=True)
    print(
        f"event 正样本比例: acc={acc_positive_ratio:.6f}, dec={dec_positive_ratio:.6f}",
        flush=True,
    )
    print(
        f"event pos_weight: acc={acc_pos_weight:.4f}, dec={dec_pos_weight:.4f}",
        flush=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    in_channels = 1 if args.input_mode == "no_mask" else 6
    model = UNet1DPhysiologicalMultitask(
        in_channels=in_channels,
        base_channels=args.base_channels,
        depth=args.depth,
        scalar_hidden_channels=args.scalar_hidden_channels,
        dropout=args.dropout,
        residual_reconstruction=not args.no_residual_reconstruction,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    acc_loss_fn = BinaryEventLoss(
        args.acc_event_loss,
        pos_weight=acc_pos_weight,
        focal_gamma=args.acc_focal_gamma,
        focal_alpha=args.acc_focal_alpha,
        device=device,
    )
    dec_loss_fn = BinaryEventLoss(
        args.dec_event_loss,
        pos_weight=dec_pos_weight,
        focal_gamma=args.dec_focal_gamma,
        focal_alpha=args.dec_focal_alpha,
        device=device,
    )

    model_variant = "physiological_multitask_v1_no_mask" if args.input_mode == "no_mask" else "physiological_multitask_v2_gt_mask_aux"
    config = vars(args).copy()
    config.update(
        {
            "device": device,
            "label_stats": label_stats,
            "acc_pos_weight_used": acc_pos_weight,
            "dec_pos_weight_used": dec_pos_weight,
            "acc_positive_ratio": acc_positive_ratio,
            "dec_positive_ratio": dec_positive_ratio,
            "model_class": "UNet1DPhysiologicalMultitask",
            "model_variant": model_variant,
            "in_channels": in_channels,
            "residual_reconstruction": not args.no_residual_reconstruction,
        }
    )
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    train_log_path = os.path.join(args.output_dir, "train_log.csv")
    write_train_log_header(train_log_path)
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, label_stats, acc_loss_fn, dec_loss_fn, args, optimizer=optimizer
        )
        val_metrics = run_epoch(
            model, val_loader, device, label_stats, acc_loss_fn, dec_loss_fn, args, optimizer=None
        )
        scheduler.step(val_metrics["total"])
        lr = float(optimizer.param_groups[0]["lr"])
        append_train_log(train_log_path, epoch, lr, train_metrics, val_metrics)
        history.append({"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics})

        message = (
            f"Epoch {epoch}: "
            f"train_total={train_metrics['total']:.4f}, "
            f"train_recon_total={train_metrics['reconstruction']:.4f}, "
            f"train_recon_corrupt={train_metrics['recon_corrupt']:.4f}, "
            f"train_recon_clean={train_metrics['recon_clean']:.4f}, "
            f"train_acc={train_metrics['acc']:.4f}, train_dec={train_metrics['dec']:.4f}, "
            f"train_baseline={train_metrics['baseline']:.4f}, train_stv={train_metrics['stv']:.4f}, "
            f"train_ltv={train_metrics['ltv']:.4f}, train_bv={train_metrics['baseline_variability']:.4f}, "
            f"val_total={val_metrics['total']:.4f}, "
            f"val_recon_total={val_metrics['reconstruction']:.4f}, "
            f"val_recon_corrupt={val_metrics['recon_corrupt']:.4f}, "
            f"val_recon_clean={val_metrics['recon_clean']:.4f}, "
            f"val_acc={val_metrics['acc']:.4f}, val_dec={val_metrics['dec']:.4f}, "
            f"val_baseline={val_metrics['baseline']:.4f}, val_stv={val_metrics['stv']:.4f}, "
            f"val_ltv={val_metrics['ltv']:.4f}, val_bv={val_metrics['baseline_variability']:.4f}"
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                    "label_stats": label_stats,
                },
                os.path.join(args.output_dir, "best_model.pt"),
            )
            message += " [best, saved]"
        print(message, flush=True)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": config,
                "label_stats": label_stats,
            },
            os.path.join(args.output_dir, "last_model.pt"),
        )
        with open(os.path.join(args.output_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"best_val_total": best_val, "history": history}, f, indent=2, ensure_ascii=False)

    np.savez(
        os.path.join(args.output_dir, "loss_curves.npz"),
        train_total=np.asarray([h["train"]["total"] for h in history], dtype=np.float32),
        val_total=np.asarray([h["val"]["total"] for h in history], dtype=np.float32),
        train_recon_total=np.asarray([h["train"]["reconstruction"] for h in history], dtype=np.float32),
        val_recon_total=np.asarray([h["val"]["reconstruction"] for h in history], dtype=np.float32),
        train_recon_corrupt=np.asarray([h["train"]["recon_corrupt"] for h in history], dtype=np.float32),
        val_recon_corrupt=np.asarray([h["val"]["recon_corrupt"] for h in history], dtype=np.float32),
        train_recon_clean=np.asarray([h["train"]["recon_clean"] for h in history], dtype=np.float32),
        val_recon_clean=np.asarray([h["val"]["recon_clean"] for h in history], dtype=np.float32),
    )
    print(f"训练完成，输出目录: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
