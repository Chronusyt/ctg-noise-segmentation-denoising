"""
CTG 特征提取结果可视化

参照 CTG/ctg_analysis_demo/ctg_dl/visualize_data 的布局，对单条或批量 fetal 片段进行
预处理并绘制：Raw FHR & Baseline、Clean FHR & ACC/DEC & Baseline、STV、FMP、Mask、TOCO & UC。

用法:
  # 单条：指定 .fetal 路径与片段区间
  python scripts/visualize_features.py --fetal_path /path/to/xxx.fetal --start 0 --end 4800 --sample_name xxx

  # 批量：通过 CSV（列: fetal_path 或 path, start_point, end_point, sample_name[, label]）
  python scripts/visualize_features.py --csv samples.csv --output_dir results/visualizations

  # 从目录 + 档案号列表（需与 ctg_analysis_demo 类似的目录结构）
  python scripts/visualize_features.py --fetal_dir /data/batch1/fetal --csv batch1_valid.xlsx --id_column 档案号 --segment_len 4800
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


def _get_segments(mask: np.ndarray, thresh: float = 0.5):
    """从二值 mask 得到 (start, end) 区间列表。"""
    padded = np.concatenate(([0], (mask > thresh).astype(np.int32), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))


def plot_ctg_sample(
    intermediates: dict,
    sample_name: str,
    label: int = None,
    save_path: str = None,
    show_toco: bool = True,
):
    """
    绘制单条 CTG 样本的 3x2 子图（与 ctg_analysis_demo 风格一致）。

    子图:
      (0,0) Raw FHR & Baseline
      (0,1) Clean FHR & ACC/DEC & Baseline
      (1,0) STV (ms)
      (1,1) FMP
      (2,0) Mask (0=可靠, 1=不可靠)
      (2,1) TOCO & UC（若 show_toco=True）
    """
    raw_fhr = intermediates.get('raw_fhr')
    clean_fhr = intermediates.get('clean_fhr')
    baseline = intermediates.get('baseline')
    mask = intermediates.get('mask')
    acc_binary = intermediates.get('acc_binary')
    dec_binary = intermediates.get('dec_binary')
    stv = intermediates.get('stv')
    raw_fmp = intermediates.get('raw_fmp')
    denoised_toco = intermediates.get('denoised_toco')
    toco_baseline = intermediates.get('toco_baseline')
    uc_binary = intermediates.get('uc_binary')

    n = len(clean_fhr) if clean_fhr is not None else 0
    if n == 0:
        raise ValueError("intermediates 中缺少 clean_fhr 或长度为 0")

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    title = f'Sample: {sample_name}'
    if label is not None:
        title += f' | Label: {label}'
    fig.suptitle(title, fontsize=14)

    # (0,0) Raw FHR & Baseline
    ax = axes[0, 0]
    if raw_fhr is not None:
        ax.plot(raw_fhr, color='black', linewidth=0.5, alpha=0.8, label='Raw FHR (bpm)')
    if baseline is not None:
        ax.plot(baseline, color='orange', linewidth=1.5, alpha=0.8, label='Baseline (bpm)')
    ax.set_title('Raw FHR & Baseline')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 220)

    # (0,1) Clean FHR & ACC/DEC & Baseline
    ax = axes[0, 1]
    if clean_fhr is not None:
        ax.plot(clean_fhr, color='green', linewidth=0.8, label='Clean FHR (bpm)')
    if baseline is not None:
        ax.plot(baseline, color='orange', linewidth=1.5, alpha=0.8, label='Baseline (bpm)')

    if acc_binary is not None and clean_fhr is not None:
        acc_masked = clean_fhr.copy()
        acc_masked[acc_binary <= 0.5] = np.nan
        ax.plot(acc_masked, color='blue', linewidth=2, label='ACC')
        for start, end in _get_segments(acc_binary):
            ax.axvspan(start, end, color='blue', alpha=0.2)
            ax.text((start + end) / 2, np.nanmax(clean_fhr) + 2, f'{end - start}', color='blue', fontsize=8, ha='center', va='bottom')

    if dec_binary is not None and clean_fhr is not None:
        dec_masked = clean_fhr.copy()
        dec_masked[dec_binary <= 0.5] = np.nan
        ax.plot(dec_masked, color='brown', linewidth=2, label='DEC')
        for start, end in _get_segments(dec_binary):
            ax.axvspan(start, end, color='brown', alpha=0.2)
            ax.text((start + end) / 2, np.nanmin(clean_fhr) - 2, f'{end - start}', color='brown', fontsize=8, ha='center', va='top')

    ax.set_title('Clean FHR with ACC/DEC & Baseline')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(50, 220)

    # (1,0) STV
    ax = axes[1, 0]
    if stv is not None:
        valid_stv = np.where(np.isfinite(stv), stv, np.nan)
        ax.plot(valid_stv, color='purple', linewidth=0.8)
    ax.set_title('STV (ms)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 30)

    # (1,1) FMP
    ax = axes[1, 1]
    if raw_fmp is not None:
        fmp_plot = np.asarray(raw_fmp, dtype=np.float64)
        if fmp_plot.max() > 1.5:
            fmp_plot = (fmp_plot > 0).astype(np.float64)
        ax.plot(fmp_plot, color='green', linewidth=0.8)
    ax.set_title('FMP')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # (2,0) Mask (0=可靠, 1=不可靠；画成 0/1 便于看丢失段)
    ax = axes[2, 0]
    if mask is not None:
        ax.plot(mask.astype(np.float64), color='gray', linewidth=0.8)
    ax.set_title('Mask (0=reliable, 1=unreliable)')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    # (2,1) TOCO & UC 或隐藏
    ax = axes[2, 1]
    if show_toco and denoised_toco is not None and toco_baseline is not None:
        ax.plot(denoised_toco, color='black', linewidth=0.6, alpha=0.8, label='TOCO')
        ax.plot(toco_baseline, color='orange', linewidth=1, alpha=0.8, label='TOCO baseline')
        if uc_binary is not None:
            uc_plot = np.where(uc_binary > 0.5, np.nanmax(denoised_toco) * 1.02, np.nan)
            ax.plot(uc_plot, color='red', linewidth=1.5, alpha=0.7, label='UC')
        ax.set_title('TOCO & UC')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='CTG 特征提取结果可视化（参照 ctg_analysis_demo）')
    parser.add_argument('--fetal_path', type=str, default=None, help='单条 .fetal 文件路径')
    parser.add_argument('--start', type=int, default=0, help='片段起始样本点')
    parser.add_argument('--end', type=int, default=None, help='片段结束样本点（默认整条）')
    parser.add_argument('--sample_name', type=str, default=None, help='样本名，用于标题和文件名')
    parser.add_argument('--label', type=int, default=0, help='标签（可选，用于标题）')
    parser.add_argument('--csv', type=str, default=None, help='批量：CSV/Excel 路径，需含 fetal_path/path, start_point, end_point, sample_name')
    parser.add_argument('--fetal_dir', type=str, default=None, help='批量时：fetal 文件所在目录，CSV 中为档案号时拼接为 fetal_dir/{id}.fetal')
    parser.add_argument('--id_column', type=str, default='sample_name', help='CSV 中样本 ID 列名（与 fetal_dir 拼接时用）')
    parser.add_argument('--path_column', type=str, default=None, help='CSV 中 .fetal 路径列名（若有则优先于 fetal_dir+id）')
    parser.add_argument('--segment_len', type=int, default=4800, help='批量时若 CSV 无 start/end，则用 0 到 segment_len')
    parser.add_argument('--output_dir', type=str, default='results/visualizations', help='输出目录')
    parser.add_argument('--max_samples', type=int, default=30, help='批量时最多画多少条')
    parser.add_argument('--no_toco', action='store_true', help='不画 TOCO/UC 子图')
    args = parser.parse_args()

    if args.fetal_path:
        # 单条
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
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f'{sample_name}.png')
        plot_ctg_sample(
            intermediates,
            sample_name=sample_name,
            label=args.label,
            save_path=save_path,
            show_toco=not args.no_toco,
        )
        print(f"已保存: {save_path}")
        return

    if not args.csv:
        print("请指定 --fetal_path 或 --csv 进行批量可视化")
        return

    # 批量
    if args.csv.endswith('.xlsx') or args.csv.endswith('.xls'):
        df = pd.read_excel(args.csv)
    else:
        df = pd.read_csv(args.csv)

    path_col = args.path_column
    id_col = args.id_column
    if id_col not in df.columns and '档案号' in df.columns:
        id_col = '档案号'
    if path_col and path_col not in df.columns:
        path_col = None
    if path_col is None and 'fetal_path' in df.columns:
        path_col = 'fetal_path'
    if path_col is None and 'path' in df.columns:
        path_col = 'path'

    start_col = 'start_point' if 'start_point' in df.columns else 'start'
    end_col = 'end_point' if 'end_point' in df.columns else 'end'
    if start_col not in df.columns:
        start_col = None
    if end_col not in df.columns:
        end_col = None

    label_col = 'label' if 'label' in df.columns else None

    rows = []
    for i, r in df.iterrows():
        if path_col:
            fp = r[path_col]
        else:
            if not args.fetal_dir:
                print("批量且无 path 列时需指定 --fetal_dir")
                return
            sid = r[id_col]
            fp = os.path.join(args.fetal_dir, f"{sid}.fetal")
        if not os.path.isfile(fp):
            continue
        start = int(r[start_col]) if start_col is not None else 0
        end = int(r[end_col]) if end_col is not None else (start + args.segment_len)
        name = str(r.get(id_col, r.get('sample_name', i)))
        lab = int(r[label_col]) if label_col is not None else 0
        rows.append((fp, start, end, name, lab))

    n = min(args.max_samples, len(rows))
    os.makedirs(args.output_dir, exist_ok=True)
    for idx, (fetal_path, start, end, sample_name, label) in enumerate(rows[:n]):
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
            save_path = os.path.join(args.output_dir, f'{sample_name}.png')
            plot_ctg_sample(
                intermediates,
                sample_name=sample_name,
                label=label,
                save_path=save_path,
                show_toco=not args.no_toco,
            )
            print(f"[{idx+1}/{n}] 已保存: {save_path}")
        except Exception as e:
            print(f"[{idx+1}/{n}] 失败 {sample_name}: {e}")
    print(f"完成，共 {n} 张图 -> {args.output_dir}")


if __name__ == '__main__':
    main()
