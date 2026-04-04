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

from scripts.dataset_split_utils import resolve_parent_and_chunk_indices, split_parent_groups

SEG_LEN_1MIN = 240  # 1 min @ 4 Hz
CLASS_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]


def load_paired_dataset(
    paired_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """加载 paired_dataset，返回 noisy/labels 及真实 parent/chunk 元数据。"""
    data = np.load(paired_path)
    signals = np.asarray(data["noisy_signals"], dtype=np.float32)
    artifact_labels = np.asarray(data["artifact_labels"], dtype=np.float32)
    parent_index, chunk_index = resolve_parent_and_chunk_indices(data, signals.shape[0])
    if np.isnan(signals).any():
        nan_count = np.isnan(signals).sum()
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  警告：signals 含 {nan_count} 个 NaN，已填充为 0")
    return signals, artifact_labels, parent_index, chunk_index


def slice_parents_into_1min(
    signals: np.ndarray,
    artifact_labels: np.ndarray,
    source_parent_index: np.ndarray,
    source_chunk_index: np.ndarray,
    selected_parents: np.ndarray,
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
                segs_labels.append(artifact_labels[row, start:end, :])
                out_parent_idx.append(int(parent_id))
                out_chunk_idx.append(base_chunk + j)

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
    signals, artifact_labels, parent_index, source_chunk_index = load_paired_dataset(paired_path)
    print(f"  noisy_signals: {signals.shape}, artifact_labels: {artifact_labels.shape}")

    n_unique_parents = len(np.unique(parent_index))
    print("\nStep 2: 按 parent 划分 train/val/test（先 split 再切，避免泄漏）...")
    train_parents, val_parents, test_parents = split_parent_groups(
        parent_index,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"  train parents: {len(train_parents)}, val: {len(val_parents)}, test: {len(test_parents)}")
    print(f"  unique parents: {n_unique_parents}")

    print("\nStep 3: 对每个 split 切分为 1 min 子段（保留五类标签）...")
    train_sig, train_labels, train_parent_idx, train_chunk_idx = slice_parents_into_1min(
        signals, artifact_labels, parent_index, source_chunk_index, train_parents, seg_len=args.seg_len
    )
    val_sig, val_labels, val_parent_idx, val_chunk_idx = slice_parents_into_1min(
        signals, artifact_labels, parent_index, source_chunk_index, val_parents, seg_len=args.seg_len
    )
    test_sig, test_labels, test_parent_idx, test_chunk_idx = slice_parents_into_1min(
        signals, artifact_labels, parent_index, source_chunk_index, test_parents, seg_len=args.seg_len
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
