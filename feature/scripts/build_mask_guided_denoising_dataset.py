"""
构建 mask-guided denoising 数据集（multilabel segmentation → denoising 两阶段实验）

与 direct denoising baseline 独立，不修改原脚本。

支持两种 mask 来源：
- gt: 使用真实 artifact_labels
- pred: 使用 multilabel segmentation 模型预测
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

from models.unet1d_multilabel_segmentation import UNet1DMultilabelSegmentation

SEG_LEN_1MIN = 240


def _infer_experiment_tag(*paths: str) -> str:
    for path in paths:
        if path and "clinical" in path:
            return "clinical"
    return "hard"


def _infer_default_segmentation_model(experiment_tag: str) -> str:
    return f"results/multilabel_segmentation_{experiment_tag}/best_model.pt"


def predict_masks(
    model: torch.nn.Module,
    noisy: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """对 noisy [N, L] 预测 5 类 mask，返回 [N, L, 5]。"""
    model.eval()
    N = noisy.shape[0]
    if noisy.ndim == 2:
        noisy = noisy[:, np.newaxis, :]  # [N, 1, L]
    preds = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = torch.from_numpy(noisy[i : i + batch_size]).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, 5, L]
            preds.append(probs)
    preds = np.concatenate(preds, axis=0)  # [N, 5, L]
    return np.transpose(preds, (0, 2, 1))  # [N, L, 5]


def main():
    parser = argparse.ArgumentParser(description="构建 mask-guided denoising 数据集")
    parser.add_argument(
        "--source_dir",
        type=str,
        default="datasets/denoising_baseline_hard",
        help="源数据集目录（denoising dataset，含 train/val/test_dataset_denoising.npz）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认根据 mask_source 自动设为 multilabel_guided_denoising_hard_pred 或 _gt",
    )
    parser.add_argument(
        "--mask_source",
        type=str,
        choices=["gt", "pred"],
        default="pred",
        help="mask 来源：gt=真实标签，pred=segmentation 模型预测",
    )
    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="results/multilabel_segmentation_hard/best_model.pt",
        help="multilabel segmentation 模型路径（mask_source=pred 时使用）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
    )
    args = parser.parse_args()

    source_dir_arg = args.source_dir
    experiment_tag = _infer_experiment_tag(source_dir_arg, args.output_dir, args.segmentation_model)

    if args.segmentation_model == "results/multilabel_segmentation_hard/best_model.pt" and experiment_tag != "hard":
        args.segmentation_model = _infer_default_segmentation_model(experiment_tag)

    if not os.path.isabs(args.source_dir):
        args.source_dir = os.path.join(_FEATURE_ROOT, args.source_dir)
    if not os.path.isabs(args.segmentation_model):
        args.segmentation_model = os.path.join(_FEATURE_ROOT, args.segmentation_model)
    if args.output_dir is None:
        suffix = "pred" if args.mask_source == "pred" else "gt"
        args.output_dir = os.path.join(_FEATURE_ROOT, f"datasets/multilabel_guided_denoising_{experiment_tag}_{suffix}")
    elif not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(_FEATURE_ROOT, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"mask_source={args.mask_source}, 使用设备: {device}", flush=True)

    for split in ["train", "val", "test"]:
        path = os.path.join(args.source_dir, f"{split}_dataset_denoising.npz")
        if not os.path.isfile(path):
            print(f"错误：{path} 不存在")
            return
        data = np.load(path)
        noisy = np.asarray(data["noisy_signals"], dtype=np.float32)
        clean = np.asarray(data["clean_signals"], dtype=np.float32)
        artifact_labels = np.asarray(data["artifact_labels"], dtype=np.float32)
        parent_index = np.asarray(data["parent_index"], dtype=np.int32)
        chunk_index = np.asarray(data["chunk_index"], dtype=np.int32)

        if args.mask_source == "gt":
            masks = artifact_labels.copy()
            print(f"  {split}: 使用 GT mask, shape={masks.shape}")
        else:
            print(f"  {split}: 使用 segmentation 模型预测 mask...", flush=True)
            model = UNet1DMultilabelSegmentation(in_channels=1, out_channels=5, base_channels=32, depth=3)
            ckpt = torch.load(args.segmentation_model, map_location=device)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            model = model.to(device)
            masks = predict_masks(model, noisy, device)
            print(f"  {split}: 预测完成, shape={masks.shape}")

        out_path = os.path.join(args.output_dir, f"{split}_dataset_mask_guided.npz")
        np.savez_compressed(
            out_path,
            noisy_signals=noisy,
            clean_signals=clean,
            masks=masks,
            artifact_labels=artifact_labels,
            parent_index=parent_index,
            chunk_index=chunk_index,
        )
        print(f"  已保存 {out_path}")

    print(f"\n完成。输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
