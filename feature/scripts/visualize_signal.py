"""
只画第一个子图：Raw FHR & Baseline 信号图

单图展示原始胎心率与基线，风格与 visualize_features.py 第一个子图一致。

用法:
  python scripts/visualize_signal.py --fetal_path /path/to/xxx.fetal --start 0 --end 4800 --sample_name xxx
  python scripts/visualize_signal.py --csv samples.csv --fetal_dir /path/to/fetal --id_column 档案号 --output_dir results/signal_plots
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)

from scripts.extract_features import extract_features_from_fetal_path, SAMPLE_RATE


def plot_signal_only(
    raw_fhr: np.ndarray,
    baseline: np.ndarray,
    sample_name: str,
    save_path: str = None,
    title: str = None,
    xlabel: str = "Time (samples @4Hz)",
    ylabel: str = "FHR (bpm)",
):
    """
    仅绘制 Raw FHR & Baseline 单图（与 visualize_features 第一个子图一致）。
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(raw_fhr, color='black', linewidth=0.5, alpha=0.8, label='Raw FHR (bpm)')
    ax.plot(baseline, color='orange', linewidth=1.5, alpha=0.8, label='Baseline (bpm)')
    ax.set_title(title or f'Raw FHR & Baseline | {sample_name}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 220)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='仅绘制 Raw FHR & Baseline 信号图')
    parser.add_argument('--fetal_path', type=str, default=None, help='单条 .fetal 文件路径')
    parser.add_argument('--start', type=int, default=0, help='片段起始样本点')
    parser.add_argument('--end', type=int, default=None, help='片段结束样本点（默认整条）')
    parser.add_argument('--sample_name', type=str, default=None, help='样本名')
    parser.add_argument('--label', type=int, default=0, help='标签（可选）')
    parser.add_argument('--csv', type=str, default=None, help='批量：CSV/Excel 路径')
    parser.add_argument('--fetal_dir', type=str, default=None, help='批量时 fetal 目录，与 CSV 的 ID 列拼接')
    parser.add_argument('--id_column', type=str, default='sample_name', help='CSV 中样本 ID 列名')
    parser.add_argument('--path_column', type=str, default=None, help='CSV 中 .fetal 路径列名')
    parser.add_argument('--segment_len', type=int, default=4800, help='批量时若无 start/end 则用 0~segment_len')
    parser.add_argument('--output_dir', type=str, default='results/signal_plots', help='输出目录')
    parser.add_argument('--max_samples', type=int, default=30, help='批量时最多画多少条')
    args = parser.parse_args()

    if args.fetal_path:
        if not os.path.isfile(args.fetal_path):
            print(f"错误：文件不存在 {args.fetal_path}")
            return
        sample_name = args.sample_name or os.path.splitext(os.path.basename(args.fetal_path))[0]
        if args.end is None:
            from ctg_io.fetal_reader import read_fetal
            data = read_fetal(args.fetal_path)
            args.end = len(data.fhr)
        features, intermediates = extract_features_from_fetal_path(
            args.fetal_path,
            sample_name=sample_name,
            start_point=args.start,
            end_point=args.end,
            label=args.label,
            sample_rate=SAMPLE_RATE,
            return_intermediates=True,
        )
        raw_fhr = intermediates['raw_fhr']
        baseline = intermediates['baseline']
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f'signal_{sample_name}.png')
        title = f'Raw FHR & Baseline | {sample_name}'
        if args.label is not None:
            title += f' | Label: {args.label}'
        plot_signal_only(raw_fhr, baseline, sample_name, save_path=save_path, title=title)
        print(f"已保存: {save_path}")
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

    n = min(args.max_samples, len(rows))
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, (fetal_path, start, end, sample_name, label) in enumerate(rows[:n]):
        try:
            _, intermediates = extract_features_from_fetal_path(
                fetal_path,
                sample_name=sample_name,
                start_point=start,
                end_point=end,
                label=label,
                sample_rate=SAMPLE_RATE,
                return_intermediates=True,
            )
            raw_fhr = intermediates['raw_fhr']
            baseline = intermediates['baseline']
            save_path = os.path.join(args.output_dir, f'signal_{sample_name}.png')
            title = f'Raw FHR & Baseline | {sample_name} | Label: {label}'
            plot_signal_only(raw_fhr, baseline, sample_name, save_path=save_path, title=title)
            print(f"[{idx+1}/{n}] 已保存: {save_path}")
        except Exception as e:
            print(f"[{idx+1}/{n}] 失败 {sample_name}: {e}")
    print(f"完成，共 {n} 张图 -> {args.output_dir}")


if __name__ == '__main__':
    main()
