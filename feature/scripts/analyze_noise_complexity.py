"""
分析 paired_dataset / artifact_labels 的噪声复杂度

统计指标：
1. 每条样本包含几类噪声（0~5 类占比）
2. 每条样本的噪声覆盖比例（noise_mask.mean()）
3. 每类噪声的连续 region 长度分布
4. 每条样本的连续噪声段数量
5. 复合噪声占比（>=2 类、>=3 类）
6. 输出统计表和直方图到 results/noise_analysis/
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)

NOISE_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]


def get_continuous_regions_vec(mask: np.ndarray) -> np.ndarray:
    """
    向量化提取连续 1 的区间长度。
    返回长度数组。
    """
    m = (mask > 0.5).astype(np.int8)
    diff = np.diff(m, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts).astype(np.int32)


def get_region_lengths(mask: np.ndarray) -> np.ndarray:
    """返回该 mask 中所有连续 region 的长度数组。"""
    return get_continuous_regions_vec(mask)


def validate_hard_dataset(
    artifact_labels: np.ndarray,
    output_dir: Optional[str] = None,
    save_plots: bool = False,
) -> Tuple[dict, bool]:
    """
    对 artifact_labels 做 hard 验证统计，并检查是否满足 hard 标准。
    返回 (stats_dict, passes)：
    - stats_dict: 各类统计
    - passes: 是否满足无噪声=0、mean 10-20%、median>=8%、至少2类≈100%
    """
    N, L, K = artifact_labels.shape
    noise_mask = (artifact_labels.any(axis=-1)).astype(np.float32)
    n_classes_per_sample = np.sum(np.any(artifact_labels > 0.5, axis=1), axis=1)
    coverage_per_sample = noise_mask.mean(axis=1)

    # 1. 类别分布
    hist_classes, _ = np.histogram(n_classes_per_sample, bins=np.arange(7))
    total = N

    # 2. 覆盖统计
    cov_mean = float(np.mean(coverage_per_sample))
    cov_median = float(np.median(coverage_per_sample))
    cov_min = float(np.min(coverage_per_sample))
    cov_max = float(np.max(coverage_per_sample))

    # 3. region 长度（简化）
    all_lengths = {name: [] for name in NOISE_NAMES}
    for k, name in enumerate(NOISE_NAMES):
        for i in range(N):
            lens = get_region_lengths(artifact_labels[i, :, k])
            if len(lens) > 0:
                all_lengths[name].extend(lens.tolist())

    # 4. 段数
    m = (noise_mask > 0.5).astype(np.int8)
    padded = np.concatenate([np.zeros((N, 1), dtype=np.int8), m, np.zeros((N, 1), dtype=np.int8)], axis=1)
    n_segments = np.sum(np.diff(padded, axis=1) == 1, axis=1)

    n_ge2 = int(np.sum(n_classes_per_sample >= 2))
    n_ge3 = int(np.sum(n_classes_per_sample >= 3))
    n_zero = int(hist_classes[0])

    stats = {
        "n_samples": N,
        "n_zero_class": n_zero,
        "pct_zero": 100 * n_zero / max(total, 1),
        "hist_classes": hist_classes,
        "cov_mean": cov_mean,
        "cov_median": cov_median,
        "cov_min": cov_min,
        "cov_max": cov_max,
        "cov_p25": float(np.quantile(coverage_per_sample, 0.25)),
        "cov_p75": float(np.quantile(coverage_per_sample, 0.75)),
        "cov_p90": float(np.quantile(coverage_per_sample, 0.9)),
        "cov_p95": float(np.quantile(coverage_per_sample, 0.95)),
        "pct_ge2": 100 * n_ge2 / max(total, 1),
        "pct_ge3": 100 * n_ge3 / max(total, 1),
        "region_lengths": all_lengths,
        "n_segments_mean": float(np.mean(n_segments)),
        "n_segments_median": float(np.median(n_segments)),
    }

    # Hard 标准检查
    passes = (
        n_zero == 0
        and 0.10 <= cov_mean <= 0.25
        and cov_median >= 0.08
        and stats["pct_ge2"] >= 99.0
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "hard_validation.txt"), "w") as f:
            f.write(f"1. 每样本噪声类别数: 0类={n_zero}({stats['pct_zero']:.1f}%)\n")
            for k in range(1, 6):
                f.write(f"   {k}类={hist_classes[k]}({100*hist_classes[k]/total:.1f}%)\n")
            f.write(f"2. 覆盖率: mean={cov_mean:.4f} median={cov_median:.4f} min={cov_min:.4f} max={cov_max:.4f}\n")
            f.write(f"   P25={stats['cov_p25']:.4f} P75={stats['cov_p75']:.4f} P90={stats['cov_p90']:.4f} P95={stats['cov_p95']:.4f}\n")
            f.write(f"3. 至少2类: {n_ge2}/{total}={stats['pct_ge2']:.1f}%\n")
            f.write(f"4. 至少3类: {n_ge3}/{total}={stats['pct_ge3']:.1f}%\n")
            f.write(f"5. Hard 标准通过: {passes}\n")

    return stats, passes


def main():
    parser = argparse.ArgumentParser(description="分析噪声复杂度")
    parser.add_argument(
        "--paired",
        type=str,
        default="datasets/denoising_20min/paired_dataset.npz",
        help="paired_dataset.npz 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/noise_analysis",
        help="输出目录",
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
    artifact_labels = np.asarray(data["artifact_labels"], dtype=np.float32)
    N, L, K = artifact_labels.shape
    assert K == 5, f"期望 5 类噪声，得到 {K}"
    print(f"  artifact_labels: {artifact_labels.shape}")

    noise_mask = (artifact_labels.any(axis=-1)).astype(np.float32)

    # ========== 1. 每条样本包含几类噪声 ==========
    print("\n========== 1. 每条样本包含几类噪声 ==========")
    n_classes_per_sample = np.sum(
        np.any(artifact_labels > 0.5, axis=1), axis=1
    )  # [N]
    hist_classes, _ = np.histogram(n_classes_per_sample, bins=np.arange(7))
    total = N
    table_lines = ["类别数\t样本数\t占比"]
    for k in range(6):
        cnt = hist_classes[k]
        pct = 100 * cnt / total
        table_lines.append(f"{k}\t{cnt}\t{pct:.2f}%")
    table_str = "\n".join(table_lines)
    print(table_str)
    with open(os.path.join(output_dir, "1_classes_per_sample.txt"), "w") as f:
        f.write(table_str)

    # ========== 2. 每条样本的噪声覆盖比例 ==========
    print("\n========== 2. 每条样本的噪声覆盖比例 ==========")
    coverage_per_sample = noise_mask.mean(axis=1)  # [N]
    stats = {
        "mean": np.mean(coverage_per_sample),
        "median": np.median(coverage_per_sample),
        "min": np.min(coverage_per_sample),
        "max": np.max(coverage_per_sample),
        "std": np.std(coverage_per_sample),
    }
    quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    for q in quantiles:
        stats[f"p{int(q*100)}"] = np.quantile(coverage_per_sample, q)
    lines = [
        "指标\t值",
        f"平均值\t{stats['mean']:.4f}",
        f"中位数\t{stats['median']:.4f}",
        f"最小值\t{stats['min']:.4f}",
        f"最大值\t{stats['max']:.4f}",
        f"标准差\t{stats['std']:.4f}",
        f"P25\t{stats['p25']:.4f}",
        f"P50\t{stats['p50']:.4f}",
        f"P75\t{stats['p75']:.4f}",
        f"P90\t{stats['p90']:.4f}",
        f"P95\t{stats['p95']:.4f}",
        f"P99\t{stats['p99']:.4f}",
    ]
    cov_str = "\n".join(lines)
    print(cov_str)
    with open(os.path.join(output_dir, "2_coverage_per_sample.txt"), "w") as f:
        f.write(cov_str)

    # ========== 3. 每类噪声的连续 region 长度分布 ==========
    print("\n========== 3. 每类噪声的连续 region 长度分布 ==========")
    all_lengths = {name: [] for name in NOISE_NAMES}
    for k, name in enumerate(NOISE_NAMES):
        masks_k = artifact_labels[:, :, k]  # [N, L]
        for i in range(N):
            lens = get_region_lengths(masks_k[i])
            if len(lens) > 0:
                all_lengths[name].extend(lens.tolist())

    table_lines = ["噪声类型\tregion数\tmean\tmedian\tmin\tmax\tP90\tP95"]
    for name in NOISE_NAMES:
        arr = np.array(all_lengths[name])
        if len(arr) == 0:
            table_lines.append(f"{name}\t0\t-\t-\t-\t-\t-\t-")
        else:
            table_lines.append(
                f"{name}\t{len(arr)}\t{arr.mean():.1f}\t{np.median(arr):.1f}\t{arr.min()}\t{arr.max()}\t{np.quantile(arr,0.9):.1f}\t{np.quantile(arr,0.95):.1f}"
            )
    len_str = "\n".join(table_lines)
    print(len_str)
    with open(os.path.join(output_dir, "3_region_lengths_per_type.txt"), "w") as f:
        f.write(len_str)

    # ========== 4. 每条样本的连续噪声段数量 ==========
    print("\n========== 4. 每条样本的连续噪声段数量 ==========")
    m = (noise_mask > 0.5).astype(np.int8)
    padded = np.concatenate([np.zeros((N, 1), dtype=np.int8), m, np.zeros((N, 1), dtype=np.int8)], axis=1)
    n_segments_per_sample = np.sum(np.diff(padded, axis=1) == 1, axis=1).astype(np.int32)

    seg_hist, _ = np.histogram(n_segments_per_sample, bins=np.arange(0, n_segments_per_sample.max() + 2))
    seg_lines = ["段数\t样本数\t占比"]
    for s in range(min(21, len(seg_hist))):
        cnt = seg_hist[s]
        if cnt > 0 or s <= 10:
            seg_lines.append(f"{s}\t{cnt}\t{100*cnt/total:.2f}%")
    seg_str = "\n".join(seg_lines)
    print(seg_str)
    seg_stats = [
        f"平均值: {n_segments_per_sample.mean():.2f}",
        f"中位数: {np.median(n_segments_per_sample):.0f}",
        f"最大值: {n_segments_per_sample.max()}",
    ]
    print("\n".join(seg_stats))
    with open(os.path.join(output_dir, "4_segments_per_sample.txt"), "w") as f:
        f.write(seg_str + "\n\n" + "\n".join(seg_stats))

    # ========== 5. 复合噪声占比 ==========
    print("\n========== 5. 复合噪声占比 ==========")
    n_ge2 = np.sum(n_classes_per_sample >= 2)
    n_ge3 = np.sum(n_classes_per_sample >= 3)
    comp_lines = [
        f"至少 2 类噪声的样本: {n_ge2} / {N} = {100*n_ge2/N:.2f}%",
        f"至少 3 类噪声的样本: {n_ge3} / {N} = {100*n_ge3/N:.2f}%",
    ]
    comp_str = "\n".join(comp_lines)
    print(comp_str)
    with open(os.path.join(output_dir, "5_composite_noise.txt"), "w") as f:
        f.write(comp_str)

    # ========== 6. 直方图 ==========
    print("\n========== 6. 保存直方图 ==========")
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use("Agg")

        # 图1: 每样本噪声类别数
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(6), hist_classes, color="steelblue", edgecolor="black")
        ax.set_xlabel("Noise classes per sample")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of noise classes per sample")
        ax.set_xticks(range(6))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hist_classes_per_sample.png"), dpi=150)
        plt.close()

        # 图2: 每样本噪声覆盖比例
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(coverage_per_sample, bins=50, color="steelblue", edgecolor="black", alpha=0.8)
        ax.axvline(stats["median"], color="red", linestyle="--", label=f"median={stats['median']:.3f}")
        ax.set_xlabel("Noise coverage (noise_mask.mean())")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of noise coverage per sample")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hist_coverage.png"), dpi=150)
        plt.close()

        # 图3: 每类噪声 region 长度分布（箱线图）
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        for k, name in enumerate(NOISE_NAMES):
            arr = np.array(all_lengths[name])
            if len(arr) > 0:
                axes[k].hist(arr, bins=min(50, int(arr.max()) + 1), color="steelblue", edgecolor="black", alpha=0.8)
            axes[k].set_title(name)
            axes[k].set_xlabel("Region length (samples)")
        axes[5].axis("off")
        plt.suptitle("Region length distribution per noise type")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hist_region_lengths.png"), dpi=150)
        plt.close()

        # 图4: 每样本连续噪声段数量
        fig, ax = plt.subplots(figsize=(8, 5))
        max_show = min(30, n_segments_per_sample.max() + 1)
        ax.bar(range(max_show), seg_hist[:max_show], color="steelblue", edgecolor="black")
        ax.set_xlabel("Noise segments per sample")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of noise segments per sample")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "hist_segments_per_sample.png"), dpi=150)
        plt.close()

        print(f"  直方图已保存到 {output_dir}")
    except ImportError:
        print("  matplotlib 未安装，跳过直方图")

    # ========== 综合判断 ==========
    print("\n" + "=" * 60)
    print("综合判断：当前 synthetic dataset 是否过于简单？")
    print("=" * 60)

    pct_0class = 100 * hist_classes[0] / total
    pct_1class = 100 * hist_classes[1] / total
    median_coverage = stats["median"]
    median_segments = np.median(n_segments_per_sample)
    pct_single_segment = 100 * np.sum(n_segments_per_sample <= 1) / total

    conclusions = []
    if pct_0class > 10:
        conclusions.append(f"• 无噪声样本占比 {pct_0class:.1f}%，存在一定比例完全干净样本")
    if pct_1class > 50:
        conclusions.append(f"• 仅 1 类噪声样本占比 {pct_1class:.1f}%，多数样本噪声类型单一")
    if median_coverage < 0.02:
        conclusions.append(f"• 噪声覆盖中位数仅 {median_coverage*100:.2f}%，整体污染比例很低")
    if median_segments <= 1:
        conclusions.append(f"• 中位连续噪声段数 {median_segments:.0f}，多数样本噪声段很少")
    if pct_single_segment > 70:
        conclusions.append(f"• 仅 0~1 个噪声段的样本占 {pct_single_segment:.1f}%，存在「大多数样本只有一个很短噪声段」倾向")

    if conclusions:
        print("\n潜在问题（可能过于简单）：")
        for c in conclusions:
            print(c)
    else:
        print("\n未发现明显「过于简单」的迹象。")

    print("\n正面特征：")
    print(f"• 复合噪声（>=2 类）样本占 {100*n_ge2/N:.1f}%")
    print(f"• 至少 3 类噪声样本占 {100*n_ge3/N:.1f}%")
    print(f"• 覆盖比例 P90={stats['p90']*100:.2f}%，P95={stats['p95']*100:.2f}%")

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("噪声复杂度分析总结\n\n")
        f.write("潜在问题：\n")
        f.write("\n".join(conclusions) if conclusions else "无\n")
        f.write("\n\n正面特征：\n")
        f.write(f"复合噪声(>=2类)样本: {100*n_ge2/N:.1f}%\n")
        f.write(f"至少3类噪声样本: {100*n_ge3/N:.1f}%\n")

    print(f"\n分析完成，结果已保存到 {output_dir}")


if __name__ == "__main__":
    main()
