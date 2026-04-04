"""
干净 vs 加噪 信号对比可视化

流程：
1. 用 signal_quality 筛选相对干净的数据（reliability_percent >= 阈值）
2. 在干净信号上加入噪声（使用 noise/noise_generator）
3. 双子图对比：左=原始/干净，右=加噪

用法:
  python scripts/visualize_clean_vs_noisy.py --fetal_path /path/to/xxx.fetal --start 0 --end 4800
  python scripts/visualize_clean_vs_noisy.py --csv /scratch2/yzd/CTG/batch1_valid.xlsx --fetal_dir /scratch2/yzd/CTG/batch1/fetal --id_column 档案号 --min_reliability 90
  python scripts/visualize_clean_vs_noisy.py --csv ... --fetal_dir ... --segment_len 4800 --min_reliability 99 --max_samples 30 --output_dir results/clean_vs_noisy_20min

  # 从已有 paired_dataset 可视化（支持 easy / hard）
  python scripts/visualize_clean_vs_noisy.py --from_paired datasets/denoising_20min_hard/paired_dataset_hard.npz --max_samples 20 --output_dir results/clean_vs_noisy_hard

"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.io.fetal_reader import read_fetal
from ctg_pipeline.noise.noise_generator import NoiseGenerator
from ctg_pipeline.preprocessing.fhr_baseline_optimized import analyse_baseline_optimized, BaselineConfig
from ctg_pipeline.preprocessing.signal_quality import assess_signal_quality
from ctg_pipeline.utils.pathing import DENOISING_RESULTS_ROOT, resolve_repo_path
from experiments.denoising.scripts.extract_features import SAMPLE_RATE, extract_features_from_fetal_path


# 伪影类型名称与颜色（与 noise_generator 的 artifact_labels 列对应：0 halving, 1 doubling, 2 mhr, 3 missing, 4 spike）
ARTIFACT_NAMES = ['Halving', 'Doubling', 'MHR', 'Missing', 'Spike']
ARTIFACT_COLORS = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728', '#ff7f0e']  # 蓝 绿 紫 红 橙


def _compute_baseline_from_clean(clean_fhr: np.ndarray) -> np.ndarray:
    """从 clean_fhr 计算 FIGO 基线，与 easy 可视化一致。"""
    mask = np.zeros(len(clean_fhr), dtype=bool)
    config = BaselineConfig(
        window_size=1920,
        window_step=240,
        smoothing_window=240,
    )
    baseline = analyse_baseline_optimized(
        clean_fhr.astype(np.float64),
        config=config,
        mask=mask,
        sample_rate=SAMPLE_RATE,
    )
    return baseline.astype(np.float64)


def _get_segments_from_mask(mask: np.ndarray, thresh: float = 0.5):
    """从一维 0/1 mask 得到 (start, end) 区间列表。"""
    padded = np.concatenate(([0], (mask > thresh).astype(np.int32), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def _debug_doubling(
    clean_fhr: np.ndarray,
    noisy_fhr: np.ndarray,
    artifact_labels: np.ndarray,
    sample_name: str,
) -> None:
    """打印 doubling 区域的调试信息。"""
    if artifact_labels is None or artifact_labels.shape[1] < 2:
        return
    segs = _get_segments_from_mask(artifact_labels[:, 1])
    for i, (start, end) in enumerate(segs):
        c_seg = clean_fhr[start:end]
        n_seg = noisy_fhr[start:end]
        c_valid = c_seg[np.isfinite(c_seg)]
        n_valid = n_seg[np.isfinite(n_seg)]
        print(f"[DEBUG] {sample_name} Doubling region #{i}: start={start}, end={end}")
        print(f"  Clean segment: min={np.nanmin(c_seg):.1f}, max={np.nanmax(c_seg):.1f}, mean={np.nanmean(c_seg):.1f}")
        print(f"  Corrupted segment: min={np.nanmin(n_seg):.1f}, max={np.nanmax(n_seg):.1f}, mean={np.nanmean(n_seg):.1f}")
        print(f"  Plot uses: noisy_fhr (direct from NoiseGenerator)")


def _check_doubling_consistency(
    clean_fhr: np.ndarray,
    noisy_fhr: np.ndarray,
    artifact_labels: np.ndarray,
    sample_name: str,
) -> None:
    """校验 doubling 区域：corrupted 均值应显著高于 clean（非 NaN 主导时）。"""
    if artifact_labels is None or artifact_labels.shape[1] < 2:
        return
    segs = _get_segments_from_mask(artifact_labels[:, 1])
    for start, end in segs:
        c_seg = clean_fhr[start:end]
        n_seg = noisy_fhr[start:end]
        nan_ratio = np.mean(np.isnan(n_seg))
        if nan_ratio > 0.5:
            continue
        c_mean = np.nanmean(c_seg)
        n_mean = np.nanmean(n_seg)
        if c_mean > 0 and n_mean < 1.5 * c_mean:
            import warnings
            warnings.warn(
                f"Doubling label exists but corrupted signal does not show expected amplitude increase. "
                f"Sample={sample_name} region=[{start}:{end}] clean_mean={c_mean:.1f} corrupted_mean={n_mean:.1f}"
            )


def plot_clean_vs_noisy(
    clean_fhr: np.ndarray,
    baseline: np.ndarray,
    noisy_fhr: np.ndarray,
    sample_name: str,
    reliability_pct: float = None,
    artifact_labels: np.ndarray = None,
    save_path: str = None,
):
    """
    双子图：上=干净信号+基线，下=加噪信号+基线；加噪图用彩色区域标出伪影类型。
    artifact_labels: [L, 5]，列为 halving, doubling, mhr, missing, spike。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    x = np.arange(len(clean_fhr))

    # 上：原始/干净
    ax1.plot(x, clean_fhr, color='black', linewidth=0.5, alpha=0.8, label='Clean FHR (bpm)')
    ax1.plot(x, baseline, color='orange', linewidth=1.5, alpha=0.8, label='Baseline (bpm)')
    ax1.set_title('Clean (Original)')
    ax1.set_ylabel('FHR (bpm)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(50, 220)

    # 下：加噪 + 伪影类型标注（先画半透明区域，再画曲线）
    legend_handles = []
    if artifact_labels is not None and artifact_labels.ndim == 2 and artifact_labels.shape[1] >= 5:
        for k in range(5):
            segs = _get_segments_from_mask(artifact_labels[:, k])
            for start, end in segs:
                ax2.axvspan(start, end, color=ARTIFACT_COLORS[k], alpha=0.35)
            if segs:
                legend_handles.append(mpatches.Patch(color=ARTIFACT_COLORS[k], alpha=0.35, label=ARTIFACT_NAMES[k]))
    ax2.plot(x, noisy_fhr, color='black', linewidth=0.5, alpha=0.8, label='Corrupted FHR (bpm)')
    ax2.plot(x, baseline, color='orange', linewidth=1.5, alpha=0.8, label='Baseline (bpm)')
    if artifact_labels is not None and artifact_labels.ndim == 2 and artifact_labels.shape[1] >= 5 and legend_handles:
        line_handles, line_labels = ax2.get_legend_handles_labels()
        ax2.legend(handles=line_handles + legend_handles,
                   labels=line_labels + [p.get_label() for p in legend_handles],
                   loc='upper right', fontsize=8)
    else:
        ax2.legend(loc='upper right')

    ax2.set_title('Corrupted (with noise)')
    ax2.set_xlabel('Time (samples @4Hz)')
    ax2.set_ylabel('FHR (bpm)')
    ax2.grid(True, alpha=0.3)
    n_valid = noisy_fhr[np.isfinite(noisy_fhr)]
    y_max = max(220, np.max(n_valid) if len(n_valid) > 0 else 220) + 20
    ax2.set_ylim(50, y_max)

    title = f'Sample: {sample_name}'
    if reliability_pct is not None:
        title += f' | Reliability: {reliability_pct:.1f}%'
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _run_from_paired(args) -> None:
    """从 paired_dataset.npz 加载并可视化 clean vs noisy。"""
    paired_path = str(resolve_repo_path(args.from_paired))
    if not os.path.isfile(paired_path):
        print(f"错误：paired_dataset 不存在 {paired_path}")
        return

    data = np.load(paired_path)
    clean_signals = np.asarray(data["clean_signals"], dtype=np.float64)
    noisy_signals = np.asarray(data["noisy_signals"], dtype=np.float64)
    artifact_labels = np.asarray(data["artifact_labels"], dtype=np.float32)
    reliability_scores = np.asarray(data["reliability_scores"], dtype=np.float32) if "reliability_scores" in data else None
    sample_ids = data["sample_ids"] if "sample_ids" in data else None
    n = clean_signals.shape[0]
    max_samples = min(args.max_samples, n)

    # 随机选取样本
    rng = np.random.default_rng(42)
    indices = rng.choice(n, size=max_samples, replace=False)

    os.makedirs(args.output_dir, exist_ok=True)
    for idx in indices:
        clean_fhr = clean_signals[idx]
        noisy_fhr = noisy_signals[idx]
        labels = artifact_labels[idx]
        baseline = _compute_baseline_from_clean(clean_fhr)
        sample_name = str(sample_ids[idx]) if sample_ids is not None else f"sample_{idx}"
        save_path = os.path.join(args.output_dir, f"clean_vs_noisy_{sample_name}.png")
        rel_pct = float(reliability_scores[idx]) if reliability_scores is not None else None
        plot_clean_vs_noisy(
            clean_fhr=clean_fhr,
            baseline=baseline,
            noisy_fhr=noisy_fhr,
            sample_name=sample_name,
            reliability_pct=rel_pct,
            artifact_labels=labels,
            save_path=save_path,
        )
    print(f"已保存 {max_samples} 张到 {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='干净 vs 加噪 信号对比（双子图）')
    parser.add_argument('--fetal_path', type=str, default=None, help='单条 .fetal 文件路径')
    parser.add_argument('--start', type=int, default=0, help='片段起始样本点')
    parser.add_argument('--end', type=int, default=None, help='片段结束样本点')
    parser.add_argument('--sample_name', type=str, default=None, help='样本名')
    parser.add_argument('--label', type=int, default=0, help='标签')
    parser.add_argument('--csv', type=str, default=None, help='批量：CSV/Excel 路径')
    parser.add_argument('--fetal_dir', type=str, default=None, help='批量时 fetal 目录')
    parser.add_argument('--id_column', type=str, default='sample_name', help='CSV 中样本 ID 列名')
    parser.add_argument('--path_column', type=str, default=None, help='CSV 中 .fetal 路径列名')
    parser.add_argument('--segment_len', type=int, default=4800, help='批量时若无 start/end 则用 0~segment_len')
    parser.add_argument('--min_reliability', type=float, default=99.0, help='信号质量筛选阈值：reliability_percent >= 此值才视为干净')
    parser.add_argument('--output_dir', type=str, default=str(DENOISING_RESULTS_ROOT / 'clean_vs_noisy'), help='输出目录')
    parser.add_argument('--max_samples', type=int, default=30, help='批量时最多画多少条（仅从满足质量阈值的样本中取）')
    parser.add_argument('--halving_prob', type=float, default=0.05, help='NoiseGenerator: halving 伪影概率')
    parser.add_argument('--doubling_prob', type=float, default=0.05, help='NoiseGenerator: doubling 伪影概率')
    parser.add_argument('--mhr_prob', type=float, default=0.10, help='NoiseGenerator: MHR 伪影概率')
    parser.add_argument('--missing_prob', type=float, default=0.10, help='NoiseGenerator: missing 伪影概率')
    parser.add_argument('--spike_prob', type=float, default=0.10, help='NoiseGenerator: spike 伪影概率')
    parser.add_argument('--debug', action='store_true', help='打印 doubling 等区域的调试信息')
    parser.add_argument(
        '--from_paired',
        type=str,
        default=None,
        help='从已有 paired_dataset.npz 可视化（含 clean_signals, noisy_signals, artifact_labels），支持 easy/hard',
    )
    args = parser.parse_args()

    args.output_dir = str(resolve_repo_path(args.output_dir))

    # 从 paired_dataset 可视化（无需 csv/fetal）
    if args.from_paired:
        _run_from_paired(args)
        return

    noise_gen = NoiseGenerator(
        halving_prob=args.halving_prob,
        doubling_prob=args.doubling_prob,
        mhr_prob=args.mhr_prob,
        missing_prob=args.missing_prob,
        spike_prob=args.spike_prob,
        sample_rate_hz=SAMPLE_RATE,
        ensure_at_least_one=True, # 确保至少有一种伪影（flase则和论文一致，运行没有伪影）
    )

    def process_one(fetal_path: str, start: int, end: int, sample_name: str, label: int) -> bool:
        try:
            features, intermediates = extract_features_from_fetal_path(
                fetal_path,
                sample_name=sample_name,
                start_point=start,
                end_point=end,
                label=label,
                sample_rate=SAMPLE_RATE,
                return_intermediates=True,
            )
            raw_fhr = intermediates['raw_fhr']
            clean_fhr = intermediates['clean_fhr']
            baseline = intermediates['baseline']

            # 用 signal_quality 筛选：仅保留相对干净的数据
            mask, quality_stats = assess_signal_quality(raw_fhr, sample_rate=SAMPLE_RATE)
            reliability_pct = quality_stats['reliability_percent']
            if reliability_pct < args.min_reliability:
                return False

            # 在干净信号上加入噪声（使用 clean_fhr 作为基础），并拿到伪影标签用于标注
            noisy_fhr, artifact_labels = noise_gen.generate_artifacts(clean_fhr.astype(np.float64))

            if args.debug:
                _debug_doubling(clean_fhr, noisy_fhr, artifact_labels, sample_name)

            _check_doubling_consistency(clean_fhr, noisy_fhr, artifact_labels, sample_name)

            save_path = os.path.join(args.output_dir, f'clean_vs_noisy_{sample_name}.png')
            plot_clean_vs_noisy(
                clean_fhr=clean_fhr,
                baseline=baseline,
                noisy_fhr=noisy_fhr,
                sample_name=sample_name,
                reliability_pct=reliability_pct,
                artifact_labels=artifact_labels,
                save_path=save_path,
            )
            return True
        except Exception as e:
            print(f"  失败 {sample_name}: {e}")
            return False

    if args.fetal_path:
        if not os.path.isfile(args.fetal_path):
            print(f"错误：文件不存在 {args.fetal_path}")
            return
        sample_name = args.sample_name or os.path.splitext(os.path.basename(args.fetal_path))[0]
        if args.end is None:
            data = read_fetal(args.fetal_path)
            args.end = len(data.fhr)
        os.makedirs(args.output_dir, exist_ok=True)
        if process_one(args.fetal_path, args.start, args.end, sample_name, args.label):
            print(f"已保存: {args.output_dir}/clean_vs_noisy_{sample_name}.png")
        else:
            print(f"样本 {sample_name} 未通过质量筛选（reliability < {args.min_reliability}%），或处理失败")
        return

    if not args.csv:
        print("请指定 --fetal_path 或 --csv")
        return

    if args.csv.endswith('.xlsx') or args.csv.endswith('.xls'):
        df = pd.read_excel(args.csv)
    else:
        df = pd.read_csv(args.csv)

    path_col = args.path_column or ('fetal_path' if 'fetal_path' in df.columns else None) or ('path' if 'path' in df.columns else None)
    id_col = args.id_column if args.id_column in df.columns else ('档案号' if '档案号' in df.columns else list(df.columns)[0])
    start_col = 'start_point' if 'start_point' in df.columns else ('start' if 'start' in df.columns else None)
    end_col = 'end_point' if 'end_point' in df.columns else ('end' if 'end' in df.columns else None)

    rows = []
    for i, r in df.iterrows():
        if path_col:
            fp = r[path_col]
        else:
            if not args.fetal_dir:
                print("批量且无 path 列时需指定 --fetal_dir")
                return
            fp = os.path.join(args.fetal_dir, f"{r[id_col]}.fetal")
        if not os.path.isfile(fp):
            continue
        start = int(r[start_col]) if start_col else 0
        end = int(r[end_col]) if end_col else (start + args.segment_len)
        name = str(r.get(id_col, i))
        lab = int(r['label']) if 'label' in df.columns else 0
        rows.append((fp, start, end, name, lab))

    os.makedirs(args.output_dir, exist_ok=True)
    count = 0
    skipped = 0
    for idx, (fetal_path, start, end, sample_name, label) in enumerate(rows):
        if count >= args.max_samples:
            break
        if process_one(fetal_path, start, end, sample_name, label):
            count += 1
            print(f"[{count}/{args.max_samples}] 已保存: clean_vs_noisy_{sample_name}.png")
        else:
            skipped += 1
    print(f"完成：成功 {count} 张，跳过（未满足质量）{skipped} 条 -> {args.output_dir}")


if __name__ == '__main__':
    main()
