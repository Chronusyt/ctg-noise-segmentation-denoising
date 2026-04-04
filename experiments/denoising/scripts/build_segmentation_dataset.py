"""
从 paired_dataset 构建噪声分割数据集

流程（先 split parent 再切，避免数据泄漏）：
1. 加载 paired_dataset.npz（noisy_signals, artifact_labels）
2. 构建二分类 mask：noise_mask = artifact_labels.any(axis=-1)
3. 按 parent sample 划分 train/val/test（80/10/10）
4. 对每个 split 的 parent 切分为 1 min（240 点）非重叠子段
5. 保存 segmentation_dataset.npz、train/val/test_dataset.npz
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.utils.dataset_split import resolve_parent_and_chunk_indices, split_parent_groups
from ctg_pipeline.utils.pathing import DENOISING_DATASETS_ROOT, resolve_repo_path

SEG_LEN_1MIN = 240  # 1 min @ 4 Hz


def load_paired_dataset(
    paired_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 paired_dataset，返回 noisy/labels 及真实 parent/chunk 元数据。"""
    data = np.load(paired_path)
    signals = np.asarray(data["noisy_signals"], dtype=np.float32)
    artifact_labels = np.asarray(data["artifact_labels"], dtype=np.float32)
    parent_index, chunk_index = resolve_parent_and_chunk_indices(data, signals.shape[0])
    # 处理 NaN：用 0 填充，避免训练时 loss 为 nan
    if np.isnan(signals).any():
        nan_count = np.isnan(signals).sum()
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  警告：signals 含 {nan_count} 个 NaN，已填充为 0")
    return signals, artifact_labels, parent_index, chunk_index


def build_noise_mask(artifact_labels: np.ndarray) -> np.ndarray:
    """从 artifact_labels [N,L,5] 构建二分类 mask [N,L]。"""
    return (artifact_labels.any(axis=-1)).astype(np.float32)


def slice_parents_into_1min(
    signals: np.ndarray,
    masks: np.ndarray,
    source_parent_index: np.ndarray,
    source_chunk_index: np.ndarray,
    selected_parents: np.ndarray,
    seg_len: int = SEG_LEN_1MIN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对指定 parent 切分为 1 min 子段。
    返回 (signals [M, seg_len], masks [M, seg_len], parent_indices [M], chunk_index [M])
    """
    L = signals.shape[1]
    n_segs_per_parent = L // seg_len
    if n_segs_per_parent == 0:
        raise ValueError(f"Signal length {L} < seg_len {seg_len}")

    segs_sig = []
    segs_mask = []
    out_parent_idx = []
    out_chunk_idx = []

    for parent_id in selected_parents:
        sample_rows = np.where(source_parent_index == parent_id)[0]
        if sample_rows.size == 0:
            continue
        order = np.argsort(source_chunk_index[sample_rows], kind="stable")
        for row in sample_rows[order]:
            base_chunk = int(source_chunk_index[row]) * n_segs_per_parent
            for j in range(n_segs_per_parent):
                start = j * seg_len
                end = start + seg_len
                segs_sig.append(signals[row, start:end])
                segs_mask.append(masks[row, start:end])
                out_parent_idx.append(int(parent_id))
                out_chunk_idx.append(base_chunk + j)

    return (
        np.array(segs_sig, dtype=np.float32),
        np.array(segs_mask, dtype=np.float32),
        np.array(out_parent_idx, dtype=np.int32),
        np.array(out_chunk_idx, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description="构建噪声分割数据集")
    parser.add_argument(
        "--paired",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "denoising_20min" / "paired_dataset.npz"),
        help="paired_dataset.npz 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "segmentation"),
        help="输出目录",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="train 比例",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="val 比例",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="test 比例",
    )
    parser.add_argument(
        "--seg_len",
        type=int,
        default=SEG_LEN_1MIN,
        help="1 min 子段长度（240 @ 4Hz）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()

    paired_path = str(resolve_repo_path(args.paired))
    if not os.path.isfile(paired_path):
        print(f"错误：paired_dataset 不存在 {paired_path}")
        return

    output_dir = str(resolve_repo_path(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1: 加载 paired_dataset...")
    signals, artifact_labels, parent_index, source_chunk_index = load_paired_dataset(paired_path)
    print(f"  noisy_signals: {signals.shape}, artifact_labels: {artifact_labels.shape}")

    print("\nStep 2: 构建 noise_mask...")
    masks = build_noise_mask(artifact_labels)
    noise_ratio = masks.mean() * 100
    print(f"  noise_mask: {masks.shape}, 噪声占比: {noise_ratio:.2f}%")

    n_unique_parents = len(np.unique(parent_index))
    print("\nStep 3: 按 parent 划分 train/val/test（先 split 再切，避免泄漏）...")
    train_parents, val_parents, test_parents = split_parent_groups(
        parent_index,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"  train parents: {len(train_parents)}, val parents: {len(val_parents)}, test parents: {len(test_parents)}")
    print(f"  unique parents: {n_unique_parents}")

    print("\nStep 4: 对每个 split 切分为 1 min 子段...")
    train_sig, train_mask, train_parent_idx, train_chunk_idx = slice_parents_into_1min(
        signals, masks, parent_index, source_chunk_index, train_parents, seg_len=args.seg_len
    )
    val_sig, val_mask, val_parent_idx, val_chunk_idx = slice_parents_into_1min(
        signals, masks, parent_index, source_chunk_index, val_parents, seg_len=args.seg_len
    )
    test_sig, test_mask, test_parent_idx, test_chunk_idx = slice_parents_into_1min(
        signals, masks, parent_index, source_chunk_index, test_parents, seg_len=args.seg_len
    )
    print(f"  train segments: {train_sig.shape[0]}, val: {val_sig.shape[0]}, test: {test_sig.shape[0]}")

    # 完整数据集（用于兼容性，按 train+val+test 顺序拼接）
    seg_signals = np.concatenate([train_sig, val_sig, test_sig], axis=0)
    seg_masks = np.concatenate([train_mask, val_mask, test_mask], axis=0)
    parent_indices = np.concatenate([train_parent_idx, val_parent_idx, test_parent_idx], axis=0)
    chunk_indices = np.concatenate([train_chunk_idx, val_chunk_idx, test_chunk_idx], axis=0)

    print("\nStep 5: 保存...")
    np.savez_compressed(
        os.path.join(output_dir, "segmentation_dataset.npz"),
        signals=seg_signals,
        masks=seg_masks,
        parent_indices=parent_indices,
        chunk_index=chunk_indices,
    )
    np.savez_compressed(
        os.path.join(output_dir, "train_dataset.npz"),
        signals=train_sig,
        masks=train_mask,
        parent_indices=train_parent_idx,
        chunk_index=train_chunk_idx,
    )
    np.savez_compressed(
        os.path.join(output_dir, "val_dataset.npz"),
        signals=val_sig,
        masks=val_mask,
        parent_indices=val_parent_idx,
        chunk_index=val_chunk_idx,
    )
    np.savez_compressed(
        os.path.join(output_dir, "test_dataset.npz"),
        signals=test_sig,
        masks=test_mask,
        parent_indices=test_parent_idx,
        chunk_index=test_chunk_idx,
    )
    print(f"  已保存到 {output_dir}")
    print(f"  - segmentation_dataset.npz (完整)")
    print(f"  - train_dataset.npz, val_dataset.npz, test_dataset.npz")


if __name__ == "__main__":
    main()
