"""
从已有 paired_dataset.npz 的 clean_signals 构建 hard 数据集

用法：无需 csv/fetal_dir，直接基于已有 easy 数据加噪。
  python scripts/build_hard_from_paired.py \
    --paired datasets/denoising_20min/paired_dataset.npz \
    --output_dir datasets/denoising_20min_hard
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)

from noise.noise_generator import NoiseGenerator
from scripts.analyze_noise_complexity import validate_hard_dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def main():
    parser = argparse.ArgumentParser(description="从 paired_dataset 构建 hard 数据集")
    parser.add_argument(
        "--paired",
        type=str,
        default="datasets/denoising_20min/paired_dataset.npz",
        help="已有 paired_dataset.npz 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/denoising_20min_hard",
        help="输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
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

    print("加载 paired_dataset...")
    data = np.load(paired_path)
    clean_signals = np.asarray(data["clean_signals"], dtype=np.float64)
    n = clean_signals.shape[0]
    print(f"  clean_signals: {clean_signals.shape}")

    # 从源目录加载原始 metadata，保留与 easy 一致的 sample_id（如 111_1_250110083031）
    sample_ids = np.array([f"sample_{i}" for i in range(n)], dtype="U64")
    paired_dir = os.path.dirname(paired_path)
    for meta_name in ("metadata.csv", "metadata_hard.csv"):
        meta_path = os.path.join(paired_dir, meta_name)
        if os.path.isfile(meta_path):
            df = pd.read_csv(meta_path)
            if "sample_id" in df.columns and len(df) >= n:
                sample_ids = np.array([str(df.loc[i, "sample_id"]) for i in range(n)], dtype="U64")
                print(f"  已加载原始 sample_id 从 {meta_name}")
            break

    noise_gen = NoiseGenerator(mode="hard", random_state=args.seed)
    noisy_signals = []
    artifact_labels_list = []
    metadata_list: List[Dict[str, Any]] = []

    print("注入 hard 噪声...")
    for i in tqdm(range(n), desc="Hard noise"):
        noisy, labels = noise_gen.generate_artifacts(clean_signals[i])
        noisy_signals.append(noisy)
        artifact_labels_list.append(labels)
        art_presence = [float(np.any(labels[:, k] > 0.5)) for k in range(5)]
        metadata_list.append({
            "sample_index": i,
            "sample_id": str(sample_ids[i]),
            "artifact_presence": art_presence,
        })

    noisy_signals = np.array(noisy_signals, dtype=np.float64)
    artifact_labels = np.array(artifact_labels_list, dtype=np.float32)
    reliability_scores = np.array(data["reliability_scores"], dtype=np.float32) if "reliability_scores" in data else np.ones(n, dtype=np.float32)

    print("\n保存...")
    np.savez_compressed(
        os.path.join(output_dir, "paired_dataset_hard.npz"),
        clean_signals=clean_signals,
        noisy_signals=noisy_signals,
        artifact_labels=artifact_labels,
        reliability_scores=reliability_scores,
        sample_ids=sample_ids,
    )
    if metadata_list:
        df = pd.DataFrame(metadata_list)
        df.to_csv(os.path.join(output_dir, "metadata_hard.csv"), index=False)
    print(f"  已保存到 {output_dir}")

    print("\n" + "=" * 60)
    print("Hard 数据集自动验证")
    print("=" * 60)
    stats, passes = validate_hard_dataset(
        artifact_labels,
        output_dir=os.path.join(output_dir, "validation"),
    )
    print(f"1. 每样本噪声类别数: 0类={stats['n_zero_class']}({stats['pct_zero']:.1f}%)")
    for k in range(1, 6):
        print(f"   {k}类={stats['hist_classes'][k]}({100*stats['hist_classes'][k]/stats['n_samples']:.1f}%)")
    print(f"2. 覆盖率: mean={stats['cov_mean']:.4f} median={stats['cov_median']:.4f} min={stats['cov_min']:.4f} max={stats['cov_max']:.4f}")
    print(f"   P25={stats['cov_p25']:.4f} P75={stats['cov_p75']:.4f} P90={stats['cov_p90']:.4f} P95={stats['cov_p95']:.4f}")
    print(f"3. 至少2类噪声: {stats['pct_ge2']:.1f}%")
    print(f"4. 至少3类噪声: {stats['pct_ge3']:.1f}%")
    print(f"5. 无噪声样本占比: {stats['pct_zero']:.1f}%")
    print(f"\nHard 标准检查: 无噪声=0, mean 10-20%, median>=8%, 至少2类≈100%")
    print(f"结果: {'✓ 通过' if passes else '✗ 未完全达到目标'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
