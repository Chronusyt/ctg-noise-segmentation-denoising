"""
在真实原始 CTG 脏信号上运行两阶段推理并可视化。

流程:
1. 从原始 fetal 数据读取指定片段
2. 计算原始 reliability_percent，筛选低 reliability 样本
3. raw dirty signal -> multilabel segmentation -> predicted 5-class masks
4. concat(raw dirty, predicted masks) -> mask-guided denoiser -> reconstructed signal
5. 对 reconstructed signal 再做一次 signal quality assessment
6. 保存可视化、summary.csv、summary.txt

说明:
- 这是推理与可视化实验，不重新训练模型
- 预处理对齐 feature/scripts/visualize_signal.py:
  使用 extract_features_from_fetal_path(...) 获取与原脚本一致的 raw_fhr / baseline
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPERIMENT_ROOT = os.path.dirname(_SCRIPT_DIR)
_FEATURE_ROOT = os.path.join(_EXPERIMENT_ROOT, "feature")
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)

from ctg_io.fetal_reader import read_fetal
from ctg_preprocessing.signal_quality import assess_signal_quality
from models.unet1d_mask_guided_denoiser import UNet1DMaskGuidedDenoiser
from models.unet1d_multilabel_segmentation import UNet1DMultilabelSegmentation
from scripts.extract_features import SAMPLE_RATE, extract_features_from_fetal_path

CLASS_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]
CLASS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


@dataclass
class CandidateRow:
    sample_id: str
    fetal_path: str
    start: int
    end: int
    label: int
    selected_reason: str
    original_reliability: Optional[float] = None


def load_table(path: str) -> pd.DataFrame:
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def resolve_columns(df: pd.DataFrame, id_column: str) -> Dict[str, Optional[str]]:
    path_col = None
    for name in ("fetal_path", "path"):
        if name in df.columns:
            path_col = name
            break

    id_col = id_column if id_column in df.columns else ("档案号" if "档案号" in df.columns else df.columns[0])
    start_col = "start_point" if "start_point" in df.columns else ("start" if "start" in df.columns else None)
    end_col = "end_point" if "end_point" in df.columns else ("end" if "end" in df.columns else None)

    reliability_col = None
    for name in (
        "reliability_percent",
        "original_reliability",
        "reliability",
        "signal_reliability",
    ):
        if name in df.columns:
            reliability_col = name
            break

    return {
        "path_col": path_col,
        "id_col": id_col,
        "start_col": start_col,
        "end_col": end_col,
        "reliability_col": reliability_col,
    }


def build_fetal_path(row: pd.Series, fetal_dir: Optional[str], path_col: Optional[str], id_col: str) -> Optional[str]:
    if path_col:
        value = str(row[path_col])
        if value and value != "nan":
            return value
    if not fetal_dir:
        return None
    return os.path.join(fetal_dir, f"{row[id_col]}.fetal")


def ensure_abs(path: str, root: str) -> str:
    return path if os.path.isabs(path) else os.path.join(root, path)


def compute_reliability_from_raw(fetal_path: str, start: int, end: int) -> float:
    fetal_data = read_fetal(fetal_path)
    raw_fhr = np.asarray(fetal_data.fhr[start:end], dtype=np.float32)
    _, stats = assess_signal_quality(raw_fhr, sample_rate=SAMPLE_RATE)
    return float(stats["reliability_percent"])


def collect_candidates(
    df: pd.DataFrame,
    fetal_dir: Optional[str],
    id_column: str,
    segment_len: int,
    min_reliability: float,
    max_reliability: float,
    max_samples: int,
) -> List[CandidateRow]:
    cols = resolve_columns(df, id_column)
    candidates: List[CandidateRow] = []
    total_rows = len(df)

    print(f"开始筛选真实脏样本，共 {total_rows} 条记录...", flush=True)
    for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
        if row_idx == 1 or row_idx % 100 == 0:
            print(
                f"筛选进度: {row_idx}/{total_rows} | 已找到 {len(candidates)}/{max_samples} 个候选",
                flush=True,
            )

        sample_id = str(row.get(cols["id_col"], ""))
        fetal_path = build_fetal_path(row, fetal_dir, cols["path_col"], cols["id_col"])
        if not fetal_path or not os.path.isfile(fetal_path):
            continue

        start = int(row[cols["start_col"]]) if cols["start_col"] else 0
        end = int(row[cols["end_col"]]) if cols["end_col"] else (start + segment_len)
        label = int(row["label"]) if "label" in df.columns and not pd.isna(row["label"]) else 0

        reliability = None
        if cols["reliability_col"] is not None and not pd.isna(row[cols["reliability_col"]]):
            reliability = float(row[cols["reliability_col"]])
        else:
            try:
                reliability = compute_reliability_from_raw(fetal_path, start, end)
            except Exception:
                reliability = None

        if reliability is None:
            continue
        if reliability < min_reliability or reliability >= max_reliability:
            continue

        candidates.append(
            CandidateRow(
                sample_id=sample_id,
                fetal_path=fetal_path,
                start=start,
                end=end,
                label=label,
                selected_reason=f"{min_reliability:g} <= reliability < {max_reliability:g}",
                original_reliability=reliability,
            )
        )
        print(
            f"  命中候选 {len(candidates)}/{max_samples}: {sample_id} | reliability={reliability:.2f}%",
            flush=True,
        )
        if len(candidates) >= max_samples:
            print("已达到 max_samples，停止继续扫描。", flush=True)
            break

    return candidates


def load_segmentation_model(model_path: str, device: str) -> torch.nn.Module:
    model = UNet1DMultilabelSegmentation(in_channels=1, out_channels=5, base_channels=32, depth=3)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device).eval()


def load_denoiser_model(model_path: str, device: str) -> torch.nn.Module:
    model = UNet1DMaskGuidedDenoiser(in_channels=6, out_channels=1, base_channels=32, depth=3)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    return model.to(device).eval()


def predict_masks(model: torch.nn.Module, raw_signal: np.ndarray, device: str) -> np.ndarray:
    x = torch.from_numpy(raw_signal[np.newaxis, np.newaxis, :].astype(np.float32)).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return np.transpose(probs, (1, 0))


def reconstruct_signal(
    model: torch.nn.Module,
    raw_signal: np.ndarray,
    pred_masks: np.ndarray,
    device: str,
) -> np.ndarray:
    model_input = np.concatenate(
        [
            raw_signal[np.newaxis, np.newaxis, :].astype(np.float32),
            np.transpose(pred_masks[np.newaxis, :, :], (0, 2, 1)).astype(np.float32),
        ],
        axis=1,
    )
    x = torch.from_numpy(model_input).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0, 0]
    return pred


def sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def plot_case(
    sample_id: str,
    raw_signal: np.ndarray,
    baseline: np.ndarray,
    pred_masks: np.ndarray,
    reconstructed: np.ndarray,
    quality_mask_before: np.ndarray,
    quality_mask_after: np.ndarray,
    original_reliability: float,
    reconstructed_reliability: float,
    segmentation_model_path: str,
    denoiser_model_path: str,
    save_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过可视化")
        return

    x = np.arange(len(raw_signal))
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    axes[0].plot(x, raw_signal, color="black", linewidth=0.7, label="Raw dirty signal")
    axes[0].plot(x, baseline, color="orange", linewidth=1.2, alpha=0.8, label="Baseline")
    axes[0].fill_between(
        x,
        45,
        225,
        where=quality_mask_before.astype(bool),
        color="red",
        alpha=0.18,
        label="Original low-quality region",
    )
    axes[0].set_ylabel("FHR (bpm)")
    axes[0].set_ylim(45, 225)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"Raw Dirty Signal | sample_id={sample_id} | "
        f"orig_rel={original_reliability:.2f}% | recon_rel={reconstructed_reliability:.2f}%"
    )

    for idx, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        axes[1].fill_between(
            x,
            idx - 0.35,
            idx + 0.35,
            where=pred_masks[:, idx] >= 0.5,
            color=color,
            alpha=0.70,
        )
        axes[1].plot(x, pred_masks[:, idx] * 0.25 + (idx - 0.1), color=color, linewidth=0.6, alpha=0.9)
    axes[1].set_ylim(-0.7, 4.7)
    axes[1].set_yticks(range(len(CLASS_NAMES)))
    axes[1].set_yticklabels(CLASS_NAMES, fontsize=8)
    axes[1].set_ylabel("Pred masks")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Predicted 5-class masks (thresholded shading + probability trace)")

    axes[2].plot(x, reconstructed, color="#1f77b4", linewidth=0.8, label="Reconstructed signal")
    axes[2].fill_between(
        x,
        45,
        225,
        where=quality_mask_after.astype(bool),
        color="red",
        alpha=0.18,
        label="Reconstructed low-quality region",
    )
    axes[2].set_ylabel("FHR (bpm)")
    axes[2].set_ylim(45, 225)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Reconstructed signal from mask-guided denoiser")

    improvement = reconstructed_reliability - original_reliability
    axes[3].plot(x, raw_signal, color="black", linewidth=0.6, alpha=0.7, label="Raw dirty")
    axes[3].plot(x, reconstructed, color="#1f77b4", linewidth=0.8, alpha=0.9, label="Reconstructed")
    axes[3].plot(x, baseline, color="orange", linewidth=1.0, alpha=0.8, label="Baseline")
    axes[3].set_ylabel("FHR (bpm)")
    axes[3].set_xlabel("Time (samples @4Hz)")
    axes[3].set_ylim(45, 225)
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title(
        "Before/After overlay | "
        f"delta_reliability={improvement:+.2f}%"
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def write_summary_txt(
    rows: List[Dict[str, Any]],
    output_path: str,
    args: argparse.Namespace,
) -> None:
    processed = [r for r in rows if r["status"] == "processed"]
    skipped = [r for r in rows if r["status"] != "processed"]

    lines = []
    lines.append("真实脏信号两阶段推理实验汇总")
    lines.append("=" * 72)
    lines.append(f"csv: {args.csv}")
    lines.append(f"fetal_dir: {args.fetal_dir}")
    lines.append(f"segment_len: {args.segment_len}")
    lines.append(f"reliability 筛选: [{args.min_reliability}, {args.max_reliability})")
    lines.append(f"max_samples: {args.max_samples}")
    lines.append(f"segmentation_model: {args.segmentation_model}")
    lines.append(f"denoiser_model: {args.denoiser_model}")
    lines.append("")
    lines.append(f"候选并处理样本数: {len(processed)}")
    lines.append(f"跳过样本数: {len(skipped)}")
    if processed:
        orig = np.array([r["original_reliability"] for r in processed], dtype=np.float64)
        recon = np.array([r["reconstructed_reliability"] for r in processed], dtype=np.float64)
        delta = recon - orig
        lines.append("")
        lines.append("整体统计:")
        lines.append(f"  original_reliability mean: {orig.mean():.4f}%")
        lines.append(f"  reconstructed_reliability mean: {recon.mean():.4f}%")
        lines.append(f"  delta mean: {delta.mean():+.4f}%")
        lines.append(f"  delta median: {np.median(delta):+.4f}%")
        lines.append(f"  improved_count: {int(np.sum(delta > 0))}")
        lines.append(f"  worsened_count: {int(np.sum(delta < 0))}")
        lines.append(f"  unchanged_count: {int(np.sum(np.isclose(delta, 0.0)))}")
    if skipped:
        lines.append("")
        lines.append("跳过原因统计:")
        reason_counts: Dict[str, int] = {}
        for row in skipped:
            reason = str(row["skip_reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"  {reason}: {count}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="在真实原始 CTG 脏信号上运行两阶段推理并可视化")
    parser.add_argument("--csv", type=str, required=True, help="CSV/Excel 路径")
    parser.add_argument("--fetal_dir", type=str, default=None, help="原始 fetal 文件目录")
    parser.add_argument("--id_column", type=str, default="档案号", help="样本 ID 列名")
    parser.add_argument("--segment_len", type=int, default=4800, help="若无 start/end 列时默认截取长度")
    parser.add_argument("--min_reliability", type=float, default=0.0, help="最小 reliability（含）")
    parser.add_argument("--max_reliability", type=float, default=50.0, help="最大 reliability（不含）")
    parser.add_argument("--max_samples", type=int, default=20, help="最多处理多少条脏样本")
    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="feature/results/multilabel_segmentation_hard/best_model.pt",
        help="multilabel segmentation 模型路径",
    )
    parser.add_argument(
        "--denoiser_model",
        type=str,
        default="feature/results/multilabel_guided_denoising_hard_pred/best_model.pt",
        help="mask-guided denoiser 模型路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="real_dirty_inference/results/real_dirty_inference",
        help="输出目录",
    )
    parser.add_argument("--device", type=str, default="", help="推理设备")
    args = parser.parse_args()

    args.csv = ensure_abs(args.csv, _EXPERIMENT_ROOT)
    if args.fetal_dir:
        args.fetal_dir = ensure_abs(args.fetal_dir, _EXPERIMENT_ROOT)
    args.segmentation_model = ensure_abs(args.segmentation_model, _EXPERIMENT_ROOT)
    args.denoiser_model = ensure_abs(args.denoiser_model, _EXPERIMENT_ROOT)
    args.output_dir = ensure_abs(args.output_dir, _EXPERIMENT_ROOT)

    visual_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(visual_dir, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    df = load_table(args.csv)
    candidates = collect_candidates(
        df=df,
        fetal_dir=args.fetal_dir,
        id_column=args.id_column,
        segment_len=args.segment_len,
        min_reliability=args.min_reliability,
        max_reliability=args.max_reliability,
        max_samples=args.max_samples,
    )
    print(f"筛选出的脏样本数量: {len(candidates)}", flush=True)

    seg_model = load_segmentation_model(args.segmentation_model, device)
    denoiser_model = load_denoiser_model(args.denoiser_model, device)

    summary_rows: List[Dict[str, Any]] = []

    for idx, candidate in enumerate(candidates, start=1):
        print(f"开始推理样本 {idx}/{len(candidates)}: {candidate.sample_id}", flush=True)
        base_row: Dict[str, Any] = {
            "sample_id": candidate.sample_id,
            "fetal_path": candidate.fetal_path,
            "start_point": candidate.start,
            "end_point": candidate.end,
            "segment_len": candidate.end - candidate.start,
            "selected_reason": candidate.selected_reason,
            "status": "processed",
            "skip_reason": "",
            "original_reliability": candidate.original_reliability,
            "reconstructed_reliability": "",
            "reliability_delta": "",
            "visualization_path": "",
        }

        try:
            fetal_data = read_fetal(candidate.fetal_path)
            total_len = len(fetal_data.fhr)
            if candidate.end > total_len:
                base_row["status"] = "skipped"
                base_row["skip_reason"] = f"segment exceeds signal length ({total_len})"
                summary_rows.append(base_row)
                print(f"[{idx}/{len(candidates)}] 跳过 {candidate.sample_id}: {base_row['skip_reason']}")
                continue

            if candidate.end - candidate.start < args.segment_len:
                base_row["status"] = "skipped"
                base_row["skip_reason"] = f"segment shorter than segment_len ({args.segment_len})"
                summary_rows.append(base_row)
                print(f"[{idx}/{len(candidates)}] 跳过 {candidate.sample_id}: {base_row['skip_reason']}")
                continue

            _, intermediates = extract_features_from_fetal_path(
                candidate.fetal_path,
                sample_name=candidate.sample_id,
                start_point=candidate.start,
                end_point=candidate.end,
                label=candidate.label,
                sample_rate=SAMPLE_RATE,
                return_intermediates=True,
            )
            raw_signal = np.asarray(intermediates["raw_fhr"], dtype=np.float32)
            baseline = np.asarray(intermediates["baseline"], dtype=np.float32)
            quality_mask_before, quality_stats_before = assess_signal_quality(raw_signal, sample_rate=SAMPLE_RATE)
            pred_masks = predict_masks(seg_model, raw_signal, device)
            reconstructed = reconstruct_signal(denoiser_model, raw_signal, pred_masks, device).astype(np.float32)
            quality_mask_after, quality_stats_after = assess_signal_quality(reconstructed, sample_rate=SAMPLE_RATE)

            orig_rel = float(quality_stats_before["reliability_percent"])
            recon_rel = float(quality_stats_after["reliability_percent"])
            vis_name = f"{idx:03d}_{sanitize_filename(candidate.sample_id)}.png"
            vis_path = os.path.join(visual_dir, vis_name)
            plot_case(
                sample_id=candidate.sample_id,
                raw_signal=raw_signal,
                baseline=baseline,
                pred_masks=pred_masks,
                reconstructed=reconstructed,
                quality_mask_before=quality_mask_before,
                quality_mask_after=quality_mask_after,
                original_reliability=orig_rel,
                reconstructed_reliability=recon_rel,
                segmentation_model_path=args.segmentation_model,
                denoiser_model_path=args.denoiser_model,
                save_path=vis_path,
            )

            base_row["original_reliability"] = orig_rel
            base_row["reconstructed_reliability"] = recon_rel
            base_row["reliability_delta"] = recon_rel - orig_rel
            base_row["visualization_path"] = vis_path
            summary_rows.append(base_row)
            print(
                f"[{idx}/{len(candidates)}] 完成 {candidate.sample_id} | "
                f"orig={orig_rel:.2f}% -> recon={recon_rel:.2f}%"
            )
        except Exception as exc:
            base_row["status"] = "skipped"
            base_row["skip_reason"] = str(exc)
            summary_rows.append(base_row)
            print(f"[{idx}/{len(candidates)}] 失败 {candidate.sample_id}: {exc}")

    csv_path = os.path.join(args.output_dir, "summary.csv")
    fieldnames = [
        "sample_id",
        "fetal_path",
        "start_point",
        "end_point",
        "segment_len",
        "selected_reason",
        "status",
        "skip_reason",
        "original_reliability",
        "reconstructed_reliability",
        "reliability_delta",
        "visualization_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    txt_path = os.path.join(args.output_dir, "summary.txt")
    write_summary_txt(summary_rows, txt_path, args)
    print(f"summary.csv 已保存到 {csv_path}")
    print(f"summary.txt 已保存到 {txt_path}")
    print(f"visualizations 已保存到 {visual_dir}")


if __name__ == "__main__":
    main()
