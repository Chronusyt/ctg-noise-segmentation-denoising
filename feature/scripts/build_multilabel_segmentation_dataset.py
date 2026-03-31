"""
从 paired_dataset 构建五类噪声分割数据集（multi-label temporal segmentation）

与 binary segmentation 独立，不修改原 build_segmentation_dataset.py。

流程（先 split parent 再切，避免数据泄漏）：
1. 加载 paired_dataset（noisy_signals, artifact_labels [N,L,5]）
2. 保留五类标签，不合并成 binary mask
3. 按 parent 划分 train/val/test（80/10/10）
4. 对每个 split 切分为 1 min（240 点）非重叠子段
5. 输出到 datasets/multilabel_segmentation_hard/
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)

SEG_LEN_1MIN = 240  # 1 min @ 4 Hz
CLASS_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]


def load_paired_dataset(paired_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载 paired_dataset，返回 (noisy_signals, artifact_labels)。"""
    data = np.load(paired_path)
    signals = np.asarray(data["noisy_signals"], dtype=np.float32)
    artifact_labels = np.asarray(data["artifact_labels"], dtype=np.float32)
    if np.isnan(signals).any():
        nan_count = np.isnan(signals).sum()
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  警告：signals 含 {nan_count} 个 NaN，已填充为 0")
    return signals, artifact_labels


def split_parents(
    n_parents: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 parent 索引划分 train/val/test。"""
    np.random.seed(seed)
    perm = np.random.permutation(n_parents)
    n_train = int(n_parents * train_ratio)
    n_val = int(n_parents * val_ratio)
    n_test = n_parents - n_train - n_val
    train_parents = perm[:n_train]
    val_parents = perm[n_train : n_train + n_val]
    test_parents = perm[n_train + n_val :]
    return train_parents, val_parents, test_parents


def slice_parents_into_1min(
    signals: np.ndarray,
    artifact_labels: np.ndarray,
    parent_indices: np.ndarray,
    seg_len: int = SEG_LEN_1MIN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对指定 parent 切分为 1 min 子段，保留五类标签。
    返回 (signals [M, seg_len], labels [M, seg_len, 5], parent_index [M], chunk_index [M])
    """
    L = signals.shape[1]
    n_segs_per_parent = L // seg_len
    if n_segs_per_parent == 0:
        raise ValueError(f"Signal length {L} < seg_len {seg_len}")

    segs_sig = []
    segs_labels = []
    out_parent_idx = []
    out_chunk_idx = []

    for p in parent_indices:
        for j in range(n_segs_per_parent):
            start = j * seg_len
            end = start + seg_len
            segs_sig.append(signals[p, start:end])
            segs_labels.append(artifact_labels[p, start:end, :])
            out_parent_idx.append(p)
            out_chunk_idx.append(j)

    return (
        np.array(segs_sig, dtype=np.float32),
        np.array(segs_labels, dtype=np.float32),
        np.array(out_parent_idx, dtype=np.int32),
        np.array(out_chunk_idx, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description="构建五类噪声分割数据集（multilabel）")
    parser.add_argument(
        "--paired",
        type=str,
        default="datasets/denoising_20min_hard/paired_dataset_hard.npz",
        help="paired_dataset 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/multilabel_segmentation_hard",
        help="输出目录（与 binary 独立）",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
    )
    parser.add_argument(
        "--seg_len", type=int, default=SEG_LEN_1MIN,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    paired_path = args.paired
    if not os.path.isabs(paired_path):
        paired_path = os.path.join(_FEATURE_ROOT, paired_path)
    if not os.path.isfile(paired_path):
        print(f"错误：paired_dataset 不存在 {paired_path}")
        return

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_FEATURE_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1: 加载 paired_dataset...")
    signals, artifact_labels = load_paired_dataset(paired_path)
    print(f"  noisy_signals: {signals.shape}, artifact_labels: {artifact_labels.shape}")

    N = signals.shape[0]
    print("\nStep 2: 按 parent 划分 train/val/test（先 split 再切，避免泄漏）...")
    train_parents, val_parents, test_parents = split_parents(
        N,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"  train parents: {len(train_parents)}, val: {len(val_parents)}, test: {len(test_parents)}")

    print("\nStep 3: 对每个 split 切分为 1 min 子段（保留五类标签）...")
    train_sig, train_labels, train_parent_idx, train_chunk_idx = slice_parents_into_1min(
        signals, artifact_labels, train_parents, seg_len=args.seg_len
    )
    val_sig, val_labels, val_parent_idx, val_chunk_idx = slice_parents_into_1min(
        signals, artifact_labels, val_parents, seg_len=args.seg_len
    )
    test_sig, test_labels, test_parent_idx, test_chunk_idx = slice_parents_into_1min(
        signals, artifact_labels, test_parents, seg_len=args.seg_len
    )
    print(f"  train segments: {train_sig.shape[0]}, val: {val_sig.shape[0]}, test: {test_sig.shape[0]}")

    print("\nStep 4: 保存...")
    for name, sig, labels, pidx, cidx in [
        ("train_dataset_multilabel", train_sig, train_labels, train_parent_idx, train_chunk_idx),
        ("val_dataset_multilabel", val_sig, val_labels, val_parent_idx, val_chunk_idx),
        ("test_dataset_multilabel", test_sig, test_labels, test_parent_idx, test_chunk_idx),
    ]:
        np.savez_compressed(
            os.path.join(output_dir, f"{name}.npz"),
            signals=sig,
            labels=labels,
            parent_index=pidx,
            chunk_index=cidx,
        )
    print(f"  已保存到 {output_dir}")
    print(f"  - train_dataset_multilabel.npz")
    print(f"  - val_dataset_multilabel.npz")
    print(f"  - test_dataset_multilabel.npz")


if __name__ == "__main__":
    main()
