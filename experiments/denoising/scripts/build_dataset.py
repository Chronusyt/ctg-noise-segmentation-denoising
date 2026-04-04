"""
Clean + Noisy 数据集构建 Pipeline

流程：
1. raw dataset: 原始 FHR 数据（从 fetal 文件读取）
2. clean dataset: 从 raw 中筛选 reliability >= 99% 的样本（工程上的干净子集）
3. noisy dataset: 对 clean dataset 注入噪声，得到成对 (clean_signal, noisy_signal, artifact_labels)

Segment 切段策略（影响样本数）：
- 1 fetal 文件 = 1 raw 样本（CSV 每行对应一个 fetal 文件）
- 每样本取 [start, end)，由 CSV 的 start_point/end_point 指定；若无则 [0, segment_len)
- 非重叠、非滑窗：同一文件只产出一个 segment

用于训练噪声修复/重建模型。

用法:
  python scripts/build_dataset.py --csv /scratch2/yzd/CTG/batch1_valid.xlsx --fetal_dir /scratch2/yzd/CTG/batch1/fetal --id_column 档案号 --output_dir datasets/denoising
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.noise.clinical_noise_generator import ClinicalNoiseConfig, ClinicalNoiseGenerator
from ctg_pipeline.noise.noise_generator import NoiseGenerator
from ctg_pipeline.preprocessing.signal_quality import assess_signal_quality
from ctg_pipeline.utils.pathing import DENOISING_DATASETS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path
from experiments.denoising.scripts.analyze_noise_complexity import validate_hard_dataset
from experiments.denoising.scripts.extract_features import SAMPLE_RATE, run_pipeline

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


NOISE_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]


# =============================================================================
# easy / hard 现有逻辑（保持兼容）
# =============================================================================

def build_clean_and_noisy_datasets(
    raw_fhr_list: List[np.ndarray],
    raw_fmp_list: List[np.ndarray],
    raw_toco_list: List[np.ndarray],
    sample_ids: List[str],
    noise_generator: NoiseGenerator,
    reliability_threshold: float = 99.0,
    sample_rate: float = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    从 raw 信号构建 clean + noisy 成对数据集。

    clean dataset 定义：reliability_percent >= reliability_threshold 的样本。
    noisy dataset：对 clean 调用 NoiseGenerator 注入噪声得到。
    """
    clean_signals = []
    noisy_signals = []
    artifact_labels_list = []
    metadata_list = []

    for raw_fhr, raw_fmp, raw_toco, sid in tqdm(
        zip(raw_fhr_list, raw_fmp_list, raw_toco_list, sample_ids),
        total=len(sample_ids),
        desc="Building dataset",
    ):
        raw_fhr = np.asarray(raw_fhr, dtype=np.float64)
        raw_fmp = np.asarray(raw_fmp, dtype=np.float64)
        raw_toco = np.asarray(raw_toco, dtype=np.float64)

        mask, quality_stats = assess_signal_quality(raw_fhr, sample_rate=sample_rate)
        reliability_pct = quality_stats["reliability_percent"]
        if reliability_pct < reliability_threshold:
            continue

        try:
            _, intermediates = run_pipeline(raw_fhr, raw_fmp, raw_toco, sample_rate)
        except Exception:
            continue
        clean_fhr = intermediates["clean_fhr"]

        noisy_fhr, art_labels = noise_generator.generate_artifacts(clean_fhr.astype(np.float64))

        clean_signals.append(clean_fhr)
        noisy_signals.append(noisy_fhr)
        artifact_labels_list.append(art_labels)

        art_presence = [float(np.any(art_labels[:, k] > 0.5)) for k in range(5)]
        metadata_list.append({
            "sample_index": len(clean_signals) - 1,
            "sample_id": sid,
            "reliability_percent": reliability_pct,
            "is_selected_for_clean": True,
            "artifact_presence": art_presence,
        })

    if not clean_signals:
        return np.array([]), np.array([]), np.array([]), []

    clean_signals = np.array(clean_signals, dtype=np.float64)
    noisy_signals = np.array(noisy_signals, dtype=np.float64)
    artifact_labels = np.array(artifact_labels_list, dtype=np.float32)
    return clean_signals, noisy_signals, artifact_labels, metadata_list


# =============================================================================
# clinical 新逻辑
# =============================================================================

def select_clean_segments(
    raw_fhr_list: List[np.ndarray],
    raw_fmp_list: List[np.ndarray],
    raw_toco_list: List[np.ndarray],
    sample_ids: List[str],
    reliability_threshold: float = 99.0,
    sample_rate: float = SAMPLE_RATE,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    从 raw 数据中筛选 clean_signals，供 clinical 模式后续加噪。
    clinical noisy 必须严格从 clean_signals 派生，而不是直接从 raw 派生。
    """
    clean_signals: List[np.ndarray] = []
    metadata_list: List[Dict[str, Any]] = []

    for raw_fhr, raw_fmp, raw_toco, sid in tqdm(
        zip(raw_fhr_list, raw_fmp_list, raw_toco_list, sample_ids),
        total=len(sample_ids),
        desc="Selecting clean dataset",
    ):
        raw_fhr = np.asarray(raw_fhr, dtype=np.float64)
        raw_fmp = np.asarray(raw_fmp, dtype=np.float64)
        raw_toco = np.asarray(raw_toco, dtype=np.float64)

        _, quality_stats = assess_signal_quality(raw_fhr, sample_rate=sample_rate)
        reliability_pct = float(quality_stats["reliability_percent"])
        if reliability_pct < reliability_threshold:
            continue

        try:
            _, intermediates = run_pipeline(raw_fhr, raw_fmp, raw_toco, sample_rate)
        except Exception:
            continue

        sample_index = len(clean_signals)
        clean_fhr = np.asarray(intermediates["clean_fhr"], dtype=np.float64)
        clean_signals.append(clean_fhr)
        metadata_list.append({
            "sample_index": sample_index,
            "sample_id": sid,
            "parent_index": sample_index,
            "chunk_index": 0,
            "reliability_percent": reliability_pct,
            "is_selected_for_clean": True,
        })

    if not clean_signals:
        return np.array([]), []
    return np.asarray(clean_signals, dtype=np.float64), metadata_list


def build_clinical_paired_dataset(
    clean_signals: np.ndarray,
    clean_metadata: List[Dict[str, Any]],
    clinical_generator: ClinicalNoiseGenerator,
    sample_rate: float = SAMPLE_RATE,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    基于现有 clean_signals 构建 clinical noisy dataset。
    返回 noisy_signals、artifact_labels、更新后的 metadata。
    """
    noisy_signals: List[np.ndarray] = []
    artifact_labels_list: List[np.ndarray] = []
    metadata_list: List[Dict[str, Any]] = []

    for i in tqdm(range(len(clean_signals)), total=len(clean_signals), desc="Generating clinical noise"):
        clean_signal = np.asarray(clean_signals[i], dtype=np.float64)
        noisy_signal, artifact_labels = clinical_generator.generate_artifacts(clean_signal)
        _, noisy_stats = assess_signal_quality(noisy_signal, sample_rate=sample_rate)
        post_rel = float(noisy_stats["reliability_percent"])

        meta = dict(clean_metadata[i])
        meta["artifact_presence"] = [float(np.any(artifact_labels[:, k] > 0.5)) for k in range(5)]
        meta["post_noise_reliability"] = post_rel
        metadata_list.append(meta)

        noisy_signals.append(np.asarray(noisy_signal, dtype=np.float64))
        artifact_labels_list.append(np.asarray(artifact_labels, dtype=np.float32))

    if not noisy_signals:
        return np.array([]), np.array([]), []

    return (
        np.asarray(noisy_signals, dtype=np.float64),
        np.asarray(artifact_labels_list, dtype=np.float32),
        metadata_list,
    )


# =============================================================================
# 通用统计 / 保存 / 校验
# =============================================================================

def _compute_noise_ratios(artifact_labels: np.ndarray) -> Dict[str, float]:
    n = artifact_labels.shape[0]
    if n == 0:
        return {name: 0.0 for name in NOISE_NAMES}
    ratios = {}
    for k, name in enumerate(NOISE_NAMES):
        present = np.sum(np.any(artifact_labels[:, :, k] > 0.5, axis=1))
        ratios[name] = float(present) / n
    return ratios


def save_dataset(
    clean_signals: np.ndarray,
    noisy_signals: np.ndarray,
    artifact_labels: np.ndarray,
    metadata_list: List[Dict],
    output_dir: str,
) -> None:
    """保存 easy / hard 数据集。"""
    os.makedirs(output_dir, exist_ok=True)
    reliability_scores = (
        np.array([m["reliability_percent"] for m in metadata_list], dtype=np.float32)
        if metadata_list
        else np.array([], dtype=np.float32)
    )
    np.savez_compressed(
        os.path.join(output_dir, "clean_dataset.npz"),
        signals=clean_signals,
        reliability_scores=reliability_scores,
    )
    np.savez_compressed(
        os.path.join(output_dir, "noisy_dataset.npz"),
        signals=noisy_signals,
        artifact_labels=artifact_labels,
    )
    np.savez_compressed(
        os.path.join(output_dir, "paired_dataset.npz"),
        clean_signals=clean_signals,
        noisy_signals=noisy_signals,
        artifact_labels=artifact_labels,
        reliability_scores=reliability_scores,
    )
    if metadata_list:
        pd.DataFrame(metadata_list).to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"已保存到 {output_dir} (clean_dataset.npz, noisy_dataset.npz, paired_dataset.npz, metadata.csv)")


def save_clinical_dataset(
    clean_signals: np.ndarray,
    noisy_signals: np.ndarray,
    artifact_labels: np.ndarray,
    metadata_list: List[Dict[str, Any]],
    output_dir: str,
) -> Tuple[str, str]:
    """保存 clinical paired dataset 与 metadata。"""
    os.makedirs(output_dir, exist_ok=True)
    sample_ids = np.array([str(m.get("sample_id", "")) for m in metadata_list], dtype="U128")
    parent_index = np.array([int(m.get("parent_index", i)) for i, m in enumerate(metadata_list)], dtype=np.int32)
    chunk_index = np.array([int(m.get("chunk_index", 0)) for m in metadata_list], dtype=np.int32)
    reliability_scores = np.array([float(m.get("reliability_percent", np.nan)) for m in metadata_list], dtype=np.float32)
    post_noise_reliability_scores = np.array(
        [float(m.get("post_noise_reliability", np.nan)) for m in metadata_list],
        dtype=np.float32,
    )

    paired_path = os.path.join(output_dir, "paired_dataset_clinical.npz")
    np.savez_compressed(
        paired_path,
        clean_signals=clean_signals,
        noisy_signals=noisy_signals,
        artifact_labels=artifact_labels,
        sample_ids=sample_ids,
        sample_id=sample_ids,
        parent_index=parent_index,
        chunk_index=chunk_index,
        reliability_scores=reliability_scores,
        post_noise_reliability_scores=post_noise_reliability_scores,
    )

    metadata_path = os.path.join(output_dir, "metadata_clinical.csv")
    if metadata_list:
        pd.DataFrame(metadata_list).to_csv(metadata_path, index=False)
    return paired_path, metadata_path


def print_dataset_stats(
    n_raw: int,
    n_clean: int,
    metadata_list: List[Dict],
    artifact_labels: np.ndarray,
    reliability_threshold: float = 99.0,
) -> None:
    print("\n" + "=" * 60)
    print("数据集构建统计")
    print("=" * 60)
    print(f"Total raw samples:        {n_raw}")
    print(f"Selected clean samples (reliability >= {reliability_threshold}%): {n_clean}")
    if n_raw > 0:
        print(f"Clean selection ratio:    {n_clean / n_raw * 100:.2f}%")
    print(f"Generated noisy samples:   {n_clean}")
    if n_clean > 0 and metadata_list:
        avg_rel = np.mean([m["reliability_percent"] for m in metadata_list])
        print(f"Average reliability:      {avg_rel:.2f}%")
    print(f"Rejected samples:         {n_raw - n_clean}")
    if artifact_labels.size > 0:
        ratios = _compute_noise_ratios(artifact_labels)
        print("\nNoisy dataset 五类噪声出现比例 (sample-level):")
        for name, r in ratios.items():
            print(f"  {name.capitalize():10s} ratio: {r * 100:.2f}%")
    print("=" * 60)


def verify_pairing(
    clean_signals: np.ndarray,
    noisy_signals: np.ndarray,
    artifact_labels: np.ndarray,
    metadata_list: List[Dict],
    n_check: int = 3,
) -> bool:
    """easy / hard 现有样本级配对检查。"""
    n = clean_signals.shape[0] if clean_signals.size > 0 else 0
    ok = (
        n == noisy_signals.shape[0]
        and n == artifact_labels.shape[0]
        and n == len(metadata_list)
        and (n == 0 or (clean_signals.shape[1] == noisy_signals.shape[1] == artifact_labels.shape[1]))
    )
    print("\n[样本级配对检查]")
    print(f"  clean_signals.shape:   {clean_signals.shape}")
    print(f"  noisy_signals.shape:   {noisy_signals.shape}")
    print(f"  artifact_labels.shape: {artifact_labels.shape}")
    print(f"  metadata 条数:         {len(metadata_list)}")
    print(f"  样本数一致: {'✓' if ok else '✗'}")
    if n > 0:
        L_c, L_n, L_a = clean_signals.shape[1], noisy_signals.shape[1], artifact_labels.shape[1]
        print(f"  信号长度一致 (L={L_c}): {'✓' if L_c == L_n == L_a else '✗'}")
        for i in range(min(n_check, n)):
            mid = metadata_list[i]
            print(
                f"  样本 {i}: sample_id={mid.get('sample_id', '?')}, "
                f"reliability={mid.get('reliability_percent', 0):.1f}%, "
                f"clean_len={len(clean_signals[i])} noisy_len={len(noisy_signals[i])} "
                f"labels_len={artifact_labels[i].shape[0]} "
                f"{'✓' if len(clean_signals[i]) == len(noisy_signals[i]) == artifact_labels[i].shape[0] else '✗'}"
            )
    return ok


def verify_clinical_pairing(
    clean_signals: np.ndarray,
    noisy_signals: np.ndarray,
    artifact_labels: np.ndarray,
    metadata_list: List[Dict[str, Any]],
    report_path: str,
) -> bool:
    """clinical 模式的严格配对检查，并输出报告。"""
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    n = clean_signals.shape[0] if clean_signals.size > 0 else 0
    sample_id_count_ok = len([m for m in metadata_list if "sample_id" in m]) == n
    parent_count_ok = len([m for m in metadata_list if "parent_index" in m]) == n
    chunk_count_ok = len([m for m in metadata_list if "chunk_index" in m]) == n
    sample_index_ok = all(int(m.get("sample_index", -1)) == i for i, m in enumerate(metadata_list)) if metadata_list else True
    shape_ok = clean_signals.shape == noisy_signals.shape
    label_shape_ok = (
        artifact_labels.ndim == 3
        and artifact_labels.shape[0] == n
        and (artifact_labels.shape[1] == clean_signals.shape[1] if n > 0 else True)
        and (artifact_labels.shape[2] == 5 if artifact_labels.size > 0 else True)
    )
    paired_length_ok = True
    for i in range(n):
        if clean_signals.shape[1] != noisy_signals.shape[1] or clean_signals.shape[1] != artifact_labels.shape[1]:
            paired_length_ok = False
            break

    ok = all([
        shape_ok,
        label_shape_ok,
        sample_id_count_ok,
        parent_count_ok,
        chunk_count_ok,
        sample_index_ok,
        paired_length_ok,
    ])

    lines = []
    lines.append("clinical pairing verification report")
    lines.append("=" * 72)
    lines.append(f"1. clean_signals.shape == noisy_signals.shape: {shape_ok}")
    lines.append(f"   clean_signals.shape: {clean_signals.shape}")
    lines.append(f"   noisy_signals.shape: {noisy_signals.shape}")
    lines.append(f"2. artifact_labels.shape == (N, L, 5): {label_shape_ok}")
    lines.append(f"   artifact_labels.shape: {artifact_labels.shape}")
    lines.append(f"3. sample_id 数量一致: {sample_id_count_ok}")
    lines.append(f"4. parent_index 数量一致: {parent_count_ok}")
    lines.append(f"5. chunk_index 数量一致: {chunk_count_ok}")
    lines.append(f"6. clean / noisy / labels 是否逐样本对应: {paired_length_ok and sample_index_ok}")
    lines.append(f"   sample_index 顺序一致: {sample_index_ok}")
    lines.append("")
    lines.append(f"最终结果: {'PASS' if ok else 'FAIL'}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[clinical pairing verification]")
    for line in lines:
        print(line)
    return ok


def _get_region_lengths(mask: np.ndarray) -> np.ndarray:
    mask = (np.asarray(mask) > 0.5).astype(np.int8)
    diff = np.diff(mask, prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts).astype(np.int32)


def compute_sample_reliabilities(signals: np.ndarray, sample_rate: float = SAMPLE_RATE) -> np.ndarray:
    rel = np.zeros(signals.shape[0], dtype=np.float32)
    for i in tqdm(range(signals.shape[0]), total=signals.shape[0], desc="Reliability stats"):
        _, stats = assess_signal_quality(signals[i], sample_rate=sample_rate)
        rel[i] = float(stats["reliability_percent"])
    return rel


def summarize_noise_statistics(
    artifact_labels: np.ndarray,
    reliabilities: np.ndarray,
) -> Dict[str, Any]:
    noise_mask = (artifact_labels.any(axis=-1)).astype(np.float32)
    sample_ratios = noise_mask.mean(axis=1) if noise_mask.size > 0 else np.array([], dtype=np.float32)
    coverage = {
        name: float(np.mean(artifact_labels[:, :, k] > 0.5)) if artifact_labels.size > 0 else 0.0
        for k, name in enumerate(NOISE_NAMES)
    }
    length_stats: Dict[str, Dict[str, float]] = {}
    for k, name in enumerate(NOISE_NAMES):
        lengths: List[int] = []
        for i in range(artifact_labels.shape[0]):
            region_lengths = _get_region_lengths(artifact_labels[i, :, k])
            if len(region_lengths) > 0:
                lengths.extend(region_lengths.tolist())
        if lengths:
            arr = np.asarray(lengths, dtype=np.float64)
            length_stats[name] = {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "max": float(np.max(arr)),
                "min": float(np.min(arr)),
            }
        else:
            length_stats[name] = {"mean": 0.0, "median": 0.0, "max": 0.0, "min": 0.0}

    if sample_ratios.size > 0:
        pollution = {
            "mean": float(np.mean(sample_ratios)),
            "median": float(np.median(sample_ratios)),
            "p25": float(np.quantile(sample_ratios, 0.25)),
            "p75": float(np.quantile(sample_ratios, 0.75)),
            "p90": float(np.quantile(sample_ratios, 0.90)),
            "p95": float(np.quantile(sample_ratios, 0.95)),
        }
    else:
        pollution = {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0}

    if reliabilities.size > 0:
        reliability_stats = {
            "mean": float(np.mean(reliabilities)),
            "median": float(np.median(reliabilities)),
        }
    else:
        reliability_stats = {"mean": 0.0, "median": 0.0}

    return {
        "coverage": coverage,
        "length_stats": length_stats,
        "pollution": pollution,
        "reliability": reliability_stats,
    }


def write_clinical_noise_statistics_report(stats: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    lines = []
    lines.append("clinical noise statistics")
    lines.append("=" * 72)
    lines.append("1. 各类噪声覆盖率")
    for name in NOISE_NAMES:
        lines.append(f"  {name} coverage: {stats['coverage'][name]:.6f}")
    lines.append("")
    lines.append("2. 单段噪声长度统计 (points)")
    for name in NOISE_NAMES:
        s = stats["length_stats"][name]
        lines.append(
            f"  {name}: mean={s['mean']:.4f} median={s['median']:.4f} max={s['max']:.0f} min={s['min']:.0f}"
        )
    lines.append("")
    lines.append("3. 每条样本污染比例分布")
    lines.append(
        "  mean={mean:.6f} median={median:.6f} P25={p25:.6f} P75={p75:.6f} P90={p90:.6f} P95={p95:.6f}".format(
            **stats["pollution"]
        )
    )
    lines.append("")
    lines.append("4. reliability 分布")
    lines.append(
        f"  mean reliability={stats['reliability']['mean']:.4f} median reliability={stats['reliability']['median']:.4f}"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def infer_hard_dataset_path(clinical_output_dir: str) -> Optional[str]:
    candidates = []
    if clinical_output_dir.endswith("_clinical"):
        candidates.append(clinical_output_dir[:-9] + "_hard")
    candidates.append(str(DENOISING_DATASETS_ROOT / "denoising_20min_hard"))
    for candidate_dir in candidates:
        path = os.path.join(candidate_dir, "paired_dataset_hard.npz")
        if os.path.isfile(path):
            return path
    return None


def write_clinical_vs_hard_comparison(
    clinical_stats: Dict[str, Any],
    clinical_output_dir: str,
    output_path: str,
    sample_rate: float = SAMPLE_RATE,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    hard_path = infer_hard_dataset_path(clinical_output_dir)
    lines = []
    lines.append("clinical vs hard comparison")
    lines.append("=" * 72)
    if hard_path is None:
        lines.append("未找到 hard dataset，跳过对比。")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return

    data = np.load(hard_path)
    hard_labels = np.asarray(data["artifact_labels"], dtype=np.float32)
    hard_noisy = np.asarray(data["noisy_signals"], dtype=np.float64)
    if "post_noise_reliability_scores" in data:
        hard_reliability = np.asarray(data["post_noise_reliability_scores"], dtype=np.float32)
    else:
        hard_reliability = compute_sample_reliabilities(hard_noisy, sample_rate=sample_rate)
    hard_stats = summarize_noise_statistics(hard_labels, hard_reliability)

    lines.append(f"hard dataset: {hard_path}")
    lines.append("")
    lines.append("平均污染比例对比")
    lines.append(
        f"  clinical mean pollution: {clinical_stats['pollution']['mean']:.6f} | hard mean pollution: {hard_stats['pollution']['mean']:.6f}"
    )
    lines.append(
        f"  clinical median pollution: {clinical_stats['pollution']['median']:.6f} | hard median pollution: {hard_stats['pollution']['median']:.6f}"
    )
    lines.append("")
    lines.append("单段噪声长度对比 (points)")
    for name in NOISE_NAMES:
        cs = clinical_stats["length_stats"][name]
        hs = hard_stats["length_stats"][name]
        lines.append(
            f"  {name}: clinical mean={cs['mean']:.4f} median={cs['median']:.4f} max={cs['max']:.0f} | "
            f"hard mean={hs['mean']:.4f} median={hs['median']:.4f} max={hs['max']:.0f}"
        )
    lines.append("")
    lines.append("reliability 对比")
    lines.append(
        f"  clinical mean={clinical_stats['reliability']['mean']:.4f} median={clinical_stats['reliability']['median']:.4f} | "
        f"hard mean={hard_stats['reliability']['mean']:.4f} median={hard_stats['reliability']['median']:.4f}"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# fetal 读取
# =============================================================================

def load_raw_from_fetal(
    csv_path: str,
    fetal_dir: str,
    id_column: str = "档案号",
    segment_len: int = 4800,
    start_column: Optional[str] = None,
    end_column: Optional[str] = None,
    path_column: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[str]]:
    """从 fetal 文件加载 raw 数据。"""
    from ctg_pipeline.io.fetal_reader import read_fetal

    if csv_path.endswith(".xlsx") or csv_path.endswith(".xls"):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)

    id_col = id_column if id_column in df.columns else "档案号"
    path_col = path_column or ("fetal_path" if "fetal_path" in df.columns else None) or ("path" if "path" in df.columns else None)
    start_col = start_column or ("start_point" if "start_point" in df.columns else None) or ("start" if "start" in df.columns else None)
    end_col = end_column or ("end_point" if "end_point" in df.columns else None) or ("end" if "end" in df.columns else None)

    raw_fhr_list = []
    raw_fmp_list = []
    raw_toco_list = []
    sample_ids = []

    rows = []
    for i, r in df.iterrows():
        if path_col:
            fp = r[path_col]
        else:
            fp = os.path.join(fetal_dir, f"{r[id_col]}.fetal")
        if not os.path.isfile(fp):
            continue
        start = int(r[start_col]) if start_col else 0
        end = int(r[end_col]) if end_col else (start + segment_len)
        sid = str(r.get(id_col, i))
        rows.append((fp, start, end, sid))

    if max_samples is not None:
        rows = rows[:max_samples]

    for fp, start, end, sid in tqdm(rows, desc="Loading raw"):
        try:
            data = read_fetal(fp)
            raw_fhr_list.append(data.fhr[start:end])
            raw_fmp_list.append(data.fmp[start:end])
            raw_toco_list.append(data.toco[start:end])
            sample_ids.append(sid)
        except Exception:
            continue

    return raw_fhr_list, raw_fmp_list, raw_toco_list, sample_ids


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="构建 Clean + Noisy 数据集（用于噪声修复模型训练）")
    parser.add_argument("--csv", type=str, required=True, help="原始数据 CSV/Excel 路径")
    parser.add_argument("--fetal_dir", type=str, required=True, help="fetal 文件目录")
    parser.add_argument("--id_column", type=str, default="档案号", help="样本 ID 列名")
    parser.add_argument("--segment_len", type=int, default=4800, help="片段长度（样本数）")
    parser.add_argument("--reliability_threshold", type=float, default=99.0, help="clean 筛选阈值")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DENOISING_DATASETS_ROOT / "denoising_20min"),
        help="输出目录",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="最多处理多少条 raw 样本（用于调试）")
    parser.add_argument("--use_paper_distribution", action="store_true", default=True, help="NoiseGenerator 使用论文分布")
    parser.add_argument("--no_paper_distribution", action="store_true", help="NoiseGenerator 使用旧默认分布")
    parser.add_argument(
        "--noise_mode",
        type=str,
        choices=["easy", "hard", "clinical"],
        default="easy",
        help="噪声模式：easy=原有逻辑；hard=高覆盖、多类、连续段主导；clinical=临床可修复约束模式（halving/doubling 使用 baseline-driven 注入）",
    )
    parser.add_argument(
        "--max_halving_segment_seconds",
        type=float,
        default=15.0,
        help="clinical halving 最长时长（秒，运行时会 clamp 到 15 秒）",
    )
    parser.add_argument(
        "--max_doubling_segment_seconds",
        type=float,
        default=15.0,
        help="clinical doubling 最长时长（秒，运行时会 clamp 到 15 秒）",
    )
    parser.add_argument("--max_mhr_segment_seconds", type=float, default=10.0)
    parser.add_argument("--max_missing_segment_seconds", type=float, default=8.0)
    parser.add_argument("--max_spike_segment_seconds", type=float, default=2.0)
    parser.add_argument("--max_total_noise_ratio_clinical", type=float, default=0.12)
    parser.add_argument("--min_post_noise_reliability", type=float, default=85.0)
    parser.add_argument("--clinical_max_retries", type=int, default=80)
    args = parser.parse_args()

    use_paper = getattr(args, "use_paper_distribution", True) and not getattr(args, "no_paper_distribution", False)
    noise_mode = getattr(args, "noise_mode", "easy")

    generated_files: List[str] = []
    report_dir = str(DENOISING_RESULTS_ROOT / "noise_generation_reports")
    os.makedirs(report_dir, exist_ok=True)

    if noise_mode == "hard":
        noise_gen = NoiseGenerator(
            use_paper_distribution=use_paper,
            ensure_at_least_one=False,
            mode="hard",
        )
        base = args.output_dir.rstrip("/")
        if not base.endswith("_hard"):
            args.output_dir = base + "_hard"
    elif noise_mode == "clinical":
        base = args.output_dir.rstrip("/")
        if not base.endswith("_clinical"):
            args.output_dir = base + "_clinical"
        clinical_config = ClinicalNoiseConfig(
            sample_rate_hz=SAMPLE_RATE,
            max_halving_segment_seconds=args.max_halving_segment_seconds,
            max_doubling_segment_seconds=args.max_doubling_segment_seconds,
            max_mhr_segment_seconds=args.max_mhr_segment_seconds,
            max_missing_segment_seconds=args.max_missing_segment_seconds,
            max_spike_segment_seconds=args.max_spike_segment_seconds,
            max_total_noise_ratio_clinical=args.max_total_noise_ratio_clinical,
            min_post_noise_reliability=args.min_post_noise_reliability,
            max_retries=args.clinical_max_retries,
            use_paper_distribution=use_paper,
        )
        noise_gen = ClinicalNoiseGenerator(config=clinical_config)
    else:
        noise_gen = NoiseGenerator(use_paper_distribution=use_paper, ensure_at_least_one=False)

    args.csv = str(resolve_repo_path(args.csv))
    args.fetal_dir = str(resolve_repo_path(args.fetal_dir))
    args.output_dir = str(resolve_repo_path(args.output_dir))

    print("Step 1: 读取原始 FHR 数据...")
    raw_fhr_list, raw_fmp_list, raw_toco_list, sample_ids = load_raw_from_fetal(
        args.csv,
        args.fetal_dir,
        id_column=args.id_column,
        segment_len=args.segment_len,
        max_samples=args.max_samples,
    )
    n_raw = len(raw_fhr_list)
    print(f"  加载 {n_raw} 条 raw 样本")
    mins = args.segment_len / (4 * 60)
    print(f"  Segment length: {args.segment_len} ({mins:.0f} minutes at 4 Hz)")
    print("  Segment strategy: 1 fetal file = 1 segment, [start, end) from CSV or [0, segment_len); non-overlapping, no sliding window")

    if n_raw == 0:
        print("错误：未加载到任何样本")
        return

    if noise_mode == "clinical":
        print("\nStep 2: 先从 raw 筛选 clean dataset（clinical noisy 将严格基于 clean_signals 生成）...")
        clean_signals, clean_metadata = select_clean_segments(
            raw_fhr_list,
            raw_fmp_list,
            raw_toco_list,
            sample_ids,
            reliability_threshold=args.reliability_threshold,
            sample_rate=SAMPLE_RATE,
        )
        n_clean = clean_signals.shape[0] if clean_signals.size > 0 else 0
        print(f"  clean_signals: {clean_signals.shape}")
        if n_clean == 0:
            print("错误：未筛选出任何 clean 样本")
            return

        print("\nStep 3: 基于 clean_signals 生成 clinical noisy dataset...")
        noisy_signals, artifact_labels, metadata_list = build_clinical_paired_dataset(
            clean_signals,
            clean_metadata,
            noise_gen,
            sample_rate=SAMPLE_RATE,
        )
        print_dataset_stats(n_raw, n_clean, metadata_list, artifact_labels, args.reliability_threshold)

        print("\nStep 4: clinical pairing verification...")
        pairing_ok = verify_pairing(clean_signals, noisy_signals, artifact_labels, metadata_list, n_check=3)
        pairing_report = os.path.join(report_dir, "clinical_pairing_check.txt")
        strict_pairing_ok = verify_clinical_pairing(
            clean_signals,
            noisy_signals,
            artifact_labels,
            metadata_list,
            pairing_report,
        )
        generated_files.append(pairing_report)

        print("\nStep 5: 保存 clinical dataset...")
        paired_path, metadata_path = save_clinical_dataset(
            clean_signals,
            noisy_signals,
            artifact_labels,
            metadata_list,
            args.output_dir,
        )
        generated_files.extend([paired_path, metadata_path])

        print("\nStep 6: 输出统计报告...")
        reliabilities = np.array([float(m.get("post_noise_reliability", np.nan)) for m in metadata_list], dtype=np.float32)
        stats = summarize_noise_statistics(artifact_labels, reliabilities)
        stats_report = os.path.join(report_dir, "clinical_noise_statistics.txt")
        write_clinical_noise_statistics_report(stats, stats_report)
        generated_files.append(stats_report)

        comparison_report = os.path.join(report_dir, "clinical_vs_hard_comparison.txt")
        write_clinical_vs_hard_comparison(stats, args.output_dir, comparison_report, sample_rate=SAMPLE_RATE)
        generated_files.append(comparison_report)

        print("\nStep 7: clinical 输出说明")
        print("新增或修改的文件列表:")
        for path in generated_files:
            print(f"  - {path}")
        print(f"clinical dataset 保存路径: {paired_path}")
        print(f"pairing verification 结果: {'PASS' if (pairing_ok and strict_pairing_ok) else 'FAIL'}")
        print(f"噪声统计报告路径: {stats_report}")
        print(f"clinical vs hard 对比报告路径: {comparison_report}")
        return

    print("\nStep 2–6: 筛选 clean、注入噪声、构建成对数据...")
    clean_signals, noisy_signals, artifact_labels, metadata_list = build_clean_and_noisy_datasets(
        raw_fhr_list,
        raw_fmp_list,
        raw_toco_list,
        sample_ids,
        noise_generator=noise_gen,
        reliability_threshold=args.reliability_threshold,
        sample_rate=SAMPLE_RATE,
    )

    n_clean = clean_signals.shape[0] if clean_signals.size > 0 else 0
    print_dataset_stats(n_raw, n_clean, metadata_list, artifact_labels, args.reliability_threshold)
    verify_pairing(clean_signals, noisy_signals, artifact_labels, metadata_list, n_check=3)

    print("\nStep 7: 保存数据集...")
    save_dataset(clean_signals, noisy_signals, artifact_labels, metadata_list, args.output_dir)

    if noise_mode == "hard" and artifact_labels.size > 0:
        print("\n" + "=" * 60)
        print("Hard 数据集自动验证")
        print("=" * 60)
        stats, passes = validate_hard_dataset(
            artifact_labels,
            output_dir=os.path.join(args.output_dir, "validation") if args.output_dir else None,
        )
        print(f"1. 每样本噪声类别数: 0类={stats['n_zero_class']}({stats['pct_zero']:.1f}%)")
        for k in range(1, 6):
            print(f"   {k}类={stats['hist_classes'][k]}({100 * stats['hist_classes'][k] / stats['n_samples']:.1f}%)")
        print(f"2. 覆盖率: mean={stats['cov_mean']:.4f} median={stats['cov_median']:.4f} min={stats['cov_min']:.4f} max={stats['cov_max']:.4f}")
        print(f"   P25={stats['cov_p25']:.4f} P75={stats['cov_p75']:.4f} P90={stats['cov_p90']:.4f} P95={stats['cov_p95']:.4f}")
        print(f"3. 至少2类噪声: {stats['pct_ge2']:.1f}%")
        print(f"4. 至少3类噪声: {stats['pct_ge3']:.1f}%")
        print(f"5. 无噪声样本占比: {stats['pct_zero']:.1f}%")
        print("\nHard 标准检查: 无噪声=0, mean 10-20%, median>=8%, 至少2类≈100%")
        print(f"结果: {'✓ 通过' if passes else '✗ 未完全达到目标（可调整 NoiseGenerator hard 参数）'}")
        print("=" * 60)

    print("\n成对数据 shape:")
    if n_clean > 0:
        print(f"  clean_signals:    {clean_signals.shape}")
        print(f"  noisy_signals:    {noisy_signals.shape}")
        print(f"  artifact_labels:  {artifact_labels.shape}")


# =============================================================================
# 最小可运行示例
# =============================================================================
SEGMENT_LEN_DEFAULT = 4800


def _make_clean_signal(L: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(
        120 + 15 * np.sin(2 * np.pi * np.arange(L) / 200) + rng.normal(0, 3, L),
        80, 180,
    ).astype(np.float64)


def _make_dirty_signal_dropout(L: int, seed: int, n_dropout: int = 200) -> np.ndarray:
    sig = _make_clean_signal(L, seed)
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(L, size=min(n_dropout, L), replace=False)
    sig[idx] = 0
    return sig


def _make_dirty_signal_spike(L: int, seed: int, n_spike: int = 200) -> np.ndarray:
    sig = _make_clean_signal(L, seed)
    rng = np.random.default_rng(seed + 2)
    idx = rng.choice(L, size=min(n_spike, L), replace=False)
    sig[idx] = 255
    return sig


def _make_dirty_signal_boundary(L: int, seed: int, n_bad: int = 150) -> np.ndarray:
    sig = _make_clean_signal(L, seed)
    rng = np.random.default_rng(seed + 3)
    idx = rng.choice(L, size=min(n_bad, L), replace=False)
    sig[idx] = rng.choice([30, 35, 230, 225], size=len(idx))
    return sig


def _make_dirty_signal_extreme_jump(L: int, seed: int, n_jumps: int = 50) -> np.ndarray:
    sig = _make_clean_signal(L, seed)
    rng = np.random.default_rng(seed + 4)
    for _ in range(n_jumps):
        i = rng.integers(1, L - 1)
        sig[i] = sig[i - 1] + rng.choice([-50, 50])
    return np.clip(sig, 80, 180)


def run_minimal_example():
    np.random.seed(42)
    n_clean_syn = 50
    n_dirty = 5
    L = SEGMENT_LEN_DEFAULT

    raw_fhr_list = []
    raw_fmp_list = []
    raw_toco_list = []
    sample_ids = []

    for i in range(n_clean_syn):
        raw_fhr_list.append(_make_clean_signal(L, i))
        raw_fmp_list.append(np.zeros(L))
        raw_toco_list.append(np.zeros(L))
        sample_ids.append(f"syn_clean_{i}")

    raw_fhr_list.append(_make_dirty_signal_dropout(L, 100))
    raw_fmp_list.append(np.zeros(L))
    raw_toco_list.append(np.zeros(L))
    sample_ids.append("syn_dirty_dropout")

    raw_fhr_list.append(_make_dirty_signal_spike(L, 101))
    raw_fmp_list.append(np.zeros(L))
    raw_toco_list.append(np.zeros(L))
    sample_ids.append("syn_dirty_spike")

    raw_fhr_list.append(_make_dirty_signal_boundary(L, 102))
    raw_fmp_list.append(np.zeros(L))
    raw_toco_list.append(np.zeros(L))
    sample_ids.append("syn_dirty_boundary")

    raw_fhr_list.append(_make_dirty_signal_extreme_jump(L, 103))
    raw_fmp_list.append(np.zeros(L))
    raw_toco_list.append(np.zeros(L))
    sample_ids.append("syn_dirty_jump")

    raw_fhr_list.append(_make_dirty_signal_dropout(L, 104, n_dropout=300))
    raw_fmp_list.append(np.zeros(L))
    raw_toco_list.append(np.zeros(L))
    sample_ids.append("syn_dirty_dropout_heavy")

    n_raw = len(raw_fhr_list)
    noise_mode = "clinical" if "--clinical" in sys.argv else ("hard" if "--hard" in sys.argv else "easy")
    if noise_mode == "clinical":
        clean_signals, clean_metadata = select_clean_segments(
            raw_fhr_list,
            raw_fmp_list,
            raw_toco_list,
            sample_ids,
            reliability_threshold=99.0,
        )
        clinical_gen = ClinicalNoiseGenerator()
        noisy_signals, artifact_labels, metadata_list = build_clinical_paired_dataset(
            clean_signals,
            clean_metadata,
            clinical_gen,
            sample_rate=SAMPLE_RATE,
        )
    else:
        noise_gen = NoiseGenerator(
            use_paper_distribution=True,
            ensure_at_least_one=False,
            mode=noise_mode,
        )
        clean_signals, noisy_signals, artifact_labels, metadata_list = build_clean_and_noisy_datasets(
            raw_fhr_list,
            raw_fmp_list,
            raw_toco_list,
            sample_ids,
            noise_generator=noise_gen,
            reliability_threshold=99.0,
        )

    n_clean = clean_signals.shape[0] if clean_signals.size > 0 else 0
    n_rejected = n_raw - n_clean

    print("\n[Minimal Example]")
    print(f"noise_mode: {noise_mode}")
    print(f"Input raw samples:     {n_raw} (含 {n_clean_syn} 干净 + {n_dirty} 脏)")
    print(f"Output clean samples:  {n_clean} (预期 ~{n_clean_syn}，脏样本应被筛掉)")
    print(f"Rejected:              {n_rejected} (预期 >= {n_dirty})")
    print(f"\nOutput shapes:")
    print(f"  clean_signals:   {clean_signals.shape}")
    print(f"  noisy_signals:   {noisy_signals.shape}")
    print(f"  artifact_labels: {artifact_labels.shape}")
    if metadata_list:
        print(f"Avg reliability (clean): {np.mean([m['reliability_percent'] for m in metadata_list]):.1f}%")
    if artifact_labels.size > 0:
        ratios = _compute_noise_ratios(artifact_labels)
        print("\nNoisy dataset 五类噪声出现比例:")
        for name, r in ratios.items():
            print(f"  {name.capitalize():10s} ratio: {r * 100:.2f}%")

    verify_pairing(clean_signals, noisy_signals, artifact_labels, metadata_list, n_check=5)


if __name__ == "__main__":
    if "--minimal" in sys.argv:
        run_minimal_example()
    else:
        main()
