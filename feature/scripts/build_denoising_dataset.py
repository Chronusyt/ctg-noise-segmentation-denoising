"""
从 paired_dataset 构建直接去噪数据集（direct denoising baseline）

与 segmentation 实验独立，不修改原脚本。

流程（先 split parent 再切，避免数据泄漏）：
1. 加载 paired_dataset（clean_signals, noisy_signals, artifact_labels）
2. 按 parent 划分 train/val/test（80/10/10）
3. 对每个 split 切分为 1 min（240 点）非重叠子段
4. 输出到 datasets/denoising_baseline_hard/
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


def load_paired_dataset(paired_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载 paired_dataset，返回 (clean_signals, noisy_signals, artifact_labels)。"""
    data = np.load(paired_path)
    clean = np.asarray(data["clean_signals"], dtype=np.float32)
    noisy = np.asarray(data["noisy_signals"], dtype=np.float32)
    labels = np.asarray(data["artifact_labels"], dtype=np.float32)
    if np.isnan(noisy).any():
        nan_count = np.isnan(noisy).sum()
        noisy = np.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  警告：noisy 含 {nan_count} 个 NaN，已填充为 0")
    if np.isnan(clean).any():
        clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    return clean, noisy, labels


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
    clean: np.ndarray,
    noisy: np.ndarray,
    artifact_labels: np.ndarray,
    parent_indices: np.ndarray,
    seg_len: int = SEG_LEN_1MIN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    对指定 parent 切分为 1 min 子段。
    返回 (clean [M,L], noisy [M,L], artifact_labels [M,L,5], parent_index [M], chunk_index [M])
    """
    L = noisy.shape[1]
    n_segs_per_parent = L // seg_len
    if n_segs_per_parent == 0:
        raise ValueError(f"Signal length {L} < seg_len {seg_len}")

    segs_clean = []
    segs_noisy = []
    segs_labels = []
    out_parent_idx = []
    out_chunk_idx = []

    for p in parent_indices:
        for j in range(n_segs_per_parent):
            start = j * seg_len
            end = start + seg_len
            segs_clean.append(clean[p, start:end])
            segs_noisy.append(noisy[p, start:end])
            segs_labels.append(artifact_labels[p, start:end, :])
            out_parent_idx.append(p)
            out_chunk_idx.append(j)

    return (
        np.array(segs_clean, dtype=np.float32),
        np.array(segs_noisy, dtype=np.float32),
        np.array(segs_labels, dtype=np.float32),
        np.array(out_parent_idx, dtype=np.int32),
        np.array(out_chunk_idx, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser(description="构建直接去噪数据集")
    parser.add_argument(
        "--paired",
        type=str,
        default="datasets/denoising_20min_hard/paired_dataset_hard.npz",
        help="paired_dataset 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/denoising_baseline_hard",
        help="输出目录",
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
    clean, noisy, artifact_labels = load_paired_dataset(paired_path)
    print(f"  clean: {clean.shape}, noisy: {noisy.shape}, artifact_labels: {artifact_labels.shape}")

    N = clean.shape[0]
    print("\nStep 2: 按 parent 划分 train/val/test（先 split 再切，避免泄漏）...")
    train_parents, val_parents, test_parents = split_parents(
        N,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"  train parents: {len(train_parents)}, val: {len(val_parents)}, test: {len(test_parents)}")

    print("\nStep 3: 对每个 split 切分为 1 min 子段...")
    train_clean, train_noisy, train_labels, train_pidx, train_cidx = slice_parents_into_1min(
        clean, noisy, artifact_labels, train_parents, seg_len=args.seg_len
    )
    val_clean, val_noisy, val_labels, val_pidx, val_cidx = slice_parents_into_1min(
        clean, noisy, artifact_labels, val_parents, seg_len=args.seg_len
    )
    test_clean, test_noisy, test_labels, test_pidx, test_cidx = slice_parents_into_1min(
        clean, noisy, artifact_labels, test_parents, seg_len=args.seg_len
    )
    print(f"  train segments: {train_noisy.shape[0]}, val: {val_noisy.shape[0]}, test: {test_noisy.shape[0]}")

    def stats(noisy_arr, clean_arr, labels_arr, name):
        noise_mask = (labels_arr > 0.5).any(axis=-1)
        cov = noise_mask.mean() * 100
        per_class = [(labels_arr[:, :, c] > 0.5).any(axis=1).mean() * 100 for c in range(5)]
        mse = ((noisy_arr - clean_arr) ** 2).mean()
        return cov, per_class, mse

    print("\nStep 4: 统计信息...")
    for name, n, c, l in [
        ("train", train_noisy, train_clean, train_labels),
        ("val", val_noisy, val_clean, val_labels),
        ("test", test_noisy, test_clean, test_labels),
    ]:
        cov, per_class, mse = stats(n, c, l, name)
        print(f"  {name}: 样本数={len(n)}, 噪声覆盖率={cov:.2f}%, noisy-clean MSE={mse:.4f}")
        print(f"    各类 sample-level 出现比例: ", end="")
        for i, cn in enumerate(CLASS_NAMES):
            print(f"{cn}={per_class[i]:.1f}%", end="  ")
        print()

    print("\nStep 5: 保存...")
    for name, nc, nn, nl, pi, ci in [
        ("train_dataset_denoising", train_clean, train_noisy, train_labels, train_pidx, train_cidx),
        ("val_dataset_denoising", val_clean, val_noisy, val_labels, val_pidx, val_cidx),
        ("test_dataset_denoising", test_clean, test_noisy, test_labels, test_pidx, test_cidx),
    ]:
        np.savez_compressed(
            os.path.join(output_dir, f"{name}.npz"),
            clean_signals=nc,
            noisy_signals=nn,
            artifact_labels=nl,
            parent_index=pi,
            chunk_index=ci,
        )
    print(f"  已保存到 {output_dir}")
    print(f"  - train_dataset_denoising.npz")
    print(f"  - val_dataset_denoising.npz")
    print(f"  - test_dataset_denoising.npz")


if __name__ == "__main__":
    main()
