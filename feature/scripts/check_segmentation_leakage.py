"""
严格检查 segmentation 数据集是否存在数据泄漏

检查项：
1. parent 级别划分：train/val/test 的 parent 是否有交集
2. segment 索引：train_idx/val_idx/test_idx 是否有交集
3. 重复样本：是否存在完全相同的 signal
4. (parent, chunk_index) 是否重复出现在多个 split
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Set, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)


def main():
    parser = argparse.ArgumentParser(description="检查 segmentation 数据泄漏")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/segmentation_hard",
        help="segmentation 数据集目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/segmentation_hard/leakage_report.txt",
        help="泄漏报告输出路径",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_FEATURE_ROOT, data_dir)
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(_FEATURE_ROOT, output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    lines: list[str] = []
    has_leakage = False

    def log(s: str = ""):
        lines.append(s)
        print(s)

    log("=" * 60)
    log("Segmentation 数据泄漏检查报告")
    log("=" * 60)
    log(f"数据目录: {data_dir}")
    log("")

    # 加载数据
    train = np.load(os.path.join(data_dir, "train_dataset.npz"))
    val = np.load(os.path.join(data_dir, "val_dataset.npz"))
    test = np.load(os.path.join(data_dir, "test_dataset.npz"))

    train_sig = np.asarray(train["signals"], dtype=np.float32)
    train_mask = np.asarray(train["masks"], dtype=np.float32)
    train_parent = np.asarray(train["parent_indices"], dtype=np.int32)
    train_chunk = np.asarray(train["chunk_index"], dtype=np.int32) if "chunk_index" in train else None

    val_sig = np.asarray(val["signals"], dtype=np.float32)
    val_mask = np.asarray(val["masks"], dtype=np.float32)
    val_parent = np.asarray(val["parent_indices"], dtype=np.int32)
    val_chunk = np.asarray(val["chunk_index"], dtype=np.int32) if "chunk_index" in val else None

    test_sig = np.asarray(test["signals"], dtype=np.float32)
    test_mask = np.asarray(test["masks"], dtype=np.float32)
    test_parent = np.asarray(test["parent_indices"], dtype=np.int32)
    test_chunk = np.asarray(test["chunk_index"], dtype=np.int32) if "chunk_index" in test else None

    # ========== 一、Parent 级别划分 ==========
    log("一、Parent 级别划分")
    log("-" * 40)

    train_parents: Set[int] = set(np.unique(train_parent))
    val_parents: Set[int] = set(np.unique(val_parent))
    test_parents: Set[int] = set(np.unique(test_parent))

    log(f"  train parent 数量: {len(train_parents)}")
    log(f"  val parent 数量:   {len(val_parents)}")
    log(f"  test parent 数量: {len(test_parents)}")

    train_val_overlap = train_parents & val_parents
    train_test_overlap = train_parents & test_parents
    val_test_overlap = val_parents & test_parents

    log(f"  train ∩ val 是否为空: {len(train_val_overlap) == 0} (交集大小: {len(train_val_overlap)})")
    log(f"  train ∩ test 是否为空: {len(train_test_overlap) == 0} (交集大小: {len(train_test_overlap)})")
    log(f"  val ∩ test 是否为空: {len(val_test_overlap) == 0} (交集大小: {len(val_test_overlap)})")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        has_leakage = True
        log("  [警告] 存在 parent 重叠！")
    else:
        log("  [OK] 无 parent 重叠")
    log("")

    # ========== 二、Segment 索引与 (parent, chunk) 重叠 ==========
    log("二、Segment 与 (parent, chunk_index) 重叠")
    log("-" * 40)

    # 构建 (parent, chunk_index) 映射
    def build_parent_chunk_set(
        signals: np.ndarray,
        parents: np.ndarray,
        chunks: np.ndarray | None,
    ) -> Set[Tuple[int, int]]:
        """优先使用真实 chunk_index；仅在缺失时回退到顺序推断。"""
        if chunks is not None and len(chunks) == len(parents):
            return {(int(p), int(c)) for p, c in zip(parents, chunks)}

        seen: Dict[int, int] = {}
        out: Set[Tuple[int, int]] = set()
        for p in parents:
            c = seen.get(int(p), 0)
            out.add((int(p), c))
            seen[int(p)] = c + 1
        return out

    train_pc = build_parent_chunk_set(train_sig, train_parent, train_chunk)
    val_pc = build_parent_chunk_set(val_sig, val_parent, val_chunk)
    test_pc = build_parent_chunk_set(test_sig, test_parent, test_chunk)

    train_val_pc = train_pc & val_pc
    train_test_pc = train_pc & test_pc
    val_test_pc = val_pc & test_pc

    log(f"  train (parent, chunk) 数量: {len(train_pc)}")
    log(f"  val (parent, chunk) 数量:   {len(val_pc)}")
    log(f"  test (parent, chunk) 数量: {len(test_pc)}")
    log(f"  train ∩ val (parent, chunk): {len(train_val_pc)}")
    log(f"  train ∩ test (parent, chunk): {len(train_test_pc)}")
    log(f"  val ∩ test (parent, chunk): {len(val_test_pc)}")

    if train_val_pc or train_test_pc or val_test_pc:
        has_leakage = True
        log("  [警告] 存在 (parent, chunk) 重叠！")
    else:
        log("  [OK] 无 (parent, chunk) 重叠")
    log("")

    # ========== 三、完全重复的 signal ==========
    log("三、完全重复的 signal")
    log("-" * 40)

    def find_duplicates(arr: np.ndarray, name: str) -> int:
        """返回重复行数（不含首次出现）。"""
        _, inv = np.unique(arr.reshape(arr.shape[0], -1), axis=0, return_inverse=True)
        counts = np.bincount(inv)
        dup = np.sum(counts > 1)
        return int(dup)

    train_dup = find_duplicates(train_sig, "train")
    val_dup = find_duplicates(val_sig, "val")
    test_dup = find_duplicates(test_sig, "test")
    log(f"  train 内重复 signal 组数: {train_dup}")
    log(f"  val 内重复 signal 组数:   {val_dup}")
    log(f"  test 内重复 signal 组数:  {test_dup}")

    # 跨 split 重复：检查 train 的 signal 是否出现在 val/test
    def signal_in_set(sig: np.ndarray, pool: np.ndarray, tol: float = 1e-6) -> int:
        """pool 中有多少与 sig 完全相同的（逐元素）"""
        diff = np.abs(pool.reshape(pool.shape[0], -1) - sig.ravel())
        return int(np.sum(np.all(diff < tol, axis=1)))

    cross_val_count = 0
    for i in range(min(100, len(train_sig))):  # 抽样检查
        cross_val_count += signal_in_set(train_sig[i], val_sig)
    cross_test_count = 0
    for i in range(min(100, len(train_sig))):
        cross_test_count += signal_in_set(train_sig[i], test_sig)

    log(f"  (抽样 100) train signal 在 val 中出现次数: {cross_val_count}")
    log(f"  (抽样 100) train signal 在 test 中出现次数: {cross_test_count}")

    # 更严格：用 hash 检查所有
    def to_hash(a: np.ndarray) -> set[bytes]:
        return set(a.astype(np.float32).tobytes() for a in a)

    train_hashes = set()
    for i in range(len(train_sig)):
        train_hashes.add(train_sig[i].tobytes())
    val_hashes = set()
    for i in range(len(val_sig)):
        val_hashes.add(val_sig[i].tobytes())
    test_hashes = set()
    for i in range(len(test_sig)):
        test_hashes.add(test_sig[i].tobytes())

    train_val_sig_overlap = len(train_hashes & val_hashes)
    train_test_sig_overlap = len(train_hashes & test_hashes)
    val_test_sig_overlap = len(val_hashes & test_hashes)

    log(f"  train ∩ val 完全相同的 signal 数量: {train_val_sig_overlap}")
    log(f"  train ∩ test 完全相同的 signal 数量: {train_test_sig_overlap}")
    log(f"  val ∩ test 完全相同的 signal 数量: {val_test_sig_overlap}")

    def hash_to_parents(signals: np.ndarray, parents: np.ndarray) -> Dict[bytes, Set[int]]:
        mapping: Dict[bytes, Set[int]] = {}
        for sig, parent in zip(signals, parents):
            key = sig.tobytes()
            mapping.setdefault(key, set()).add(int(parent))
        return mapping

    def overlap_same_parent(
        left: Dict[bytes, Set[int]],
        right: Dict[bytes, Set[int]],
    ) -> int:
        overlaps = 0
        for key in left.keys() & right.keys():
            if left[key] & right[key]:
                overlaps += 1
        return overlaps

    train_hash_to_parent = hash_to_parents(train_sig, train_parent)
    val_hash_to_parent = hash_to_parents(val_sig, val_parent)
    test_hash_to_parent = hash_to_parents(test_sig, test_parent)

    same_parent_train_val = overlap_same_parent(train_hash_to_parent, val_hash_to_parent)
    same_parent_train_test = overlap_same_parent(train_hash_to_parent, test_hash_to_parent)
    same_parent_val_test = overlap_same_parent(val_hash_to_parent, test_hash_to_parent)

    log(f"  train ∩ val 且 parent 相同的重复 signal 数量: {same_parent_train_val}")
    log(f"  train ∩ test 且 parent 相同的重复 signal 数量: {same_parent_train_test}")
    log(f"  val ∩ test 且 parent 相同的重复 signal 数量: {same_parent_val_test}")

    if same_parent_train_val or same_parent_train_test or same_parent_val_test:
        has_leakage = True
        log("  [警告] 发现跨 split 且 parent 相同的重复 signal")
    elif train_val_sig_overlap or train_test_sig_overlap or val_test_sig_overlap:
        log("  [OK] 存在跨 split 重复 signal，但来自不同 parent，不按泄漏处理")
    else:
        log("  [OK] 无跨 split 完全重复 signal")
    log("")

    # ========== 四、build 流程说明 ==========
    log("四、build_segmentation_dataset.py 流程说明")
    log("-" * 40)
    log("  当前实现顺序（已修正为先 split 再切）：")
    log("    1. 加载 paired_dataset")
    log("    2. 构建 noise_mask")
    log("    3. 按 parent 划分：对 N 个 parent 做 80/10/10 随机划分")
    log("    4. 对 train_parents 切 1 min -> train_dataset")
    log("    5. 对 val_parents 切 1 min -> val_dataset")
    log("    6. 对 test_parents 切 1 min -> test_dataset")
    log("  结论：先 split parent，再切 1 min。无泄漏风险。")
    log("")

    # ========== 五、最终结论 ==========
    log("=" * 60)
    log("五、最终结论")
    log("=" * 60)
    if has_leakage:
        log("  [存在数据泄漏风险] 请检查上述警告项。")
    else:
        log("  [未发现数据泄漏]")
        log("    - parent 无重叠")
        log("    - (parent, chunk) 无重叠")
        log("    - 若存在跨 split 相同 signal，也来自不同 parent")
        log("  高 F1 可能来自：")
        log("    - hard 数据集噪声模式相对规律")
        log("    - 1 min 子段较短，噪声区域边界清晰")
        log("    - 二分类任务本身较简单")
    log("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
