import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from tqdm import tqdm
except ImportError:
    # Fallback: keep code runnable even if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

from ctg_pipeline.io.fetal_reader import read_fetal
from ctg_pipeline.preprocessing.signal_quality import assess_signal_quality
from ctg_pipeline.preprocessing.artifact_correction import (
    correct_artifacts, CorrectionMethod, CorrectionConfig
)
from ctg_pipeline.preprocessing.fhr_baseline_optimized import analyse_baseline_optimized, BaselineConfig
from ctg_pipeline.preprocessing.baseline_variability import (
    compute_baseline_variability, get_baseline_variability_statistics,
    BaselineVariabilityConfig, classify_baseline_variability
)
from ctg_pipeline.preprocessing.variability import (
    compute_stv, compute_stv_overall, get_stv_statistics, STVConfig,
    compute_ltv, compute_ltv_overall, get_ltv_statistics, LTVConfig,
    classify_stv, ms_to_bpm
)
from ctg_pipeline.preprocessing.acc_detection_figo_v2 import (
    detect_accelerations_figo, AccelerationConfig, AccelerationCriterion
)
from ctg_pipeline.preprocessing.dec_detection_figo_v2 import (
    detect_decelerations_figo, DecelerationConfig, DecelerationCriterion
)
from ctg_pipeline.preprocessing.toco_denoise import denoise_toco, TocoDenoiseConfig
from ctg_pipeline.preprocessing.toco_baseline_v2 import (
    estimate_baseline as analyse_toco_baseline_v2, TocoBaselineConfig
)
from ctg_pipeline.preprocessing.uc_detection_v2 import (
    detect_uc_v2, UcDetectionConfigV2, UcModelV2, contractions_to_binary
)



# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_baseline_features(
    clean_fhr: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Extract baseline-related features.
    
    Features:
    - baseline_mean: Mean baseline FHR (bpm)
    - baseline_std: Std of baseline
    - baseline_min: Min baseline
    - baseline_max: Max baseline
    - baseline_range: Range of baseline
    - fhr_baseline_deviation_mean: Mean absolute deviation from baseline
    - fhr_baseline_deviation_std: Std of deviation from baseline
    """
    # mask convention: 0 = reliable, 1 = unreliable
    valid_mask = (mask == 0)
    
    # Baseline statistics
    valid_baseline = baseline[valid_mask]
    if len(valid_baseline) == 0:
        valid_baseline = baseline
    
    baseline_mean = np.mean(valid_baseline)
    baseline_std = np.std(valid_baseline)
    baseline_min = np.min(valid_baseline)
    baseline_max = np.max(valid_baseline)
    baseline_range = baseline_max - baseline_min
    
    # Deviation from baseline
    deviation = np.abs(clean_fhr - baseline)
    valid_deviation = deviation[valid_mask] if np.any(valid_mask) else deviation
    deviation_mean = np.mean(valid_deviation)
    deviation_std = np.std(valid_deviation)
    
    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'baseline_min': baseline_min,
        'baseline_max': baseline_max,
        'baseline_range': baseline_range,
        'fhr_baseline_deviation_mean': deviation_mean,
        'fhr_baseline_deviation_std': deviation_std,
    }


def extract_variability_features(
    bv: np.ndarray,
    stv: np.ndarray,
    ltv: np.ndarray,
    stv_overall_ms: float,
    ltv_overall_ms: float,
) -> Dict[str, float]:
    """
    Extract variability-related features — all length-agnostic.

    Three complementary measures:

    Baseline Variability (BV):
        Robust amplitude (P95−P5) in 1-min windows.  Unit: bpm.
        Clinical: absent <2, reduced 2–5, normal 6–25, marked >25 bpm.

    Short-Term Variability (STV):
        Epoch-to-epoch change in mean pulse interval.  Unit: ms.
        Epoch = 3.75 s.  Dawes–Redman: abnormal <3, borderline 3–4, normal ≥4 ms.

    Long-Term Variability (LTV):
        Minute-by-minute range of epoch pulse intervals.  Unit: ms.
    """
    # --- Baseline Variability (bpm) ---
    valid_bv = bv[np.isfinite(bv)]
    if len(valid_bv) == 0:
        valid_bv = np.array([0.0])
    bv_mean = float(np.mean(valid_bv))
    bv_class = classify_baseline_variability(bv_mean)
    bv_class_code = {"absent": 0, "reduced": 1, "normal": 2, "marked": 3}.get(bv_class, -1)

    # --- STV (ms) ---
    valid_stv = stv[np.isfinite(stv)]
    if len(valid_stv) == 0:
        valid_stv = np.array([0.0])
    stv_mean = float(np.mean(valid_stv))
    stv_class = classify_stv(stv_overall_ms)
    stv_class_code = {"abnormal": 0, "borderline": 1, "normal": 2}.get(stv_class, -1)

    # --- LTV (ms) ---
    valid_ltv = ltv[np.isfinite(ltv)]
    if len(valid_ltv) == 0:
        valid_ltv = np.array([0.0])
    ltv_mean = float(np.mean(valid_ltv))

    # STV / LTV ratio (dimensionless)
    if (np.isfinite(stv_overall_ms) and np.isfinite(ltv_overall_ms)
            and ltv_overall_ms > 0):
        stv_ltv_ratio = stv_overall_ms / ltv_overall_ms
    else:
        stv_ltv_ratio = np.nan

    return {
        # Baseline variability (bpm)
        'bv_mean': bv_mean,
        'bv_std': float(np.std(valid_bv)),
        'bv_median': float(np.median(valid_bv)),
        'bv_p10': float(np.percentile(valid_bv, 10)),
        'bv_p90': float(np.percentile(valid_bv, 90)),
        'bv_iqr': float(np.percentile(valid_bv, 75) - np.percentile(valid_bv, 25)),
        'bv_classification': float(bv_class_code),
        # STV (ms)
        'stv_mean_ms': stv_mean,
        'stv_std_ms': float(np.std(valid_stv)),
        'stv_median_ms': float(np.median(valid_stv)),
        'stv_p10_ms': float(np.percentile(valid_stv, 10)),
        'stv_p90_ms': float(np.percentile(valid_stv, 90)),
        'stv_overall_ms': float(stv_overall_ms) if np.isfinite(stv_overall_ms) else np.nan,
        'stv_mean_bpm': float(ms_to_bpm(stv_mean)),
        'stv_classification': float(stv_class_code),
        # LTV (ms)
        'ltv_mean_ms': ltv_mean,
        'ltv_std_ms': float(np.std(valid_ltv)),
        'ltv_median_ms': float(np.median(valid_ltv)),
        'ltv_p10_ms': float(np.percentile(valid_ltv, 10)),
        'ltv_p90_ms': float(np.percentile(valid_ltv, 90)),
        'ltv_overall_ms': float(ltv_overall_ms) if np.isfinite(ltv_overall_ms) else np.nan,
        'ltv_mean_bpm': float(ms_to_bpm(ltv_mean)),
        # Cross-variability
        'stv_ltv_ratio': float(stv_ltv_ratio) if np.isfinite(stv_ltv_ratio) else np.nan,
    }


def extract_acceleration_features(
    accelerations: List,
    acc_binary: np.ndarray,
    signal_length: int,
    sample_rate: float = 4.0
) -> Dict[str, float]:
    """
    Extract acceleration-related features.
    
    Length-agnostic features (rates, means, ratios):
    - acc_rate_per_10min: Acceleration rate per 10 minutes
    - acc_mean_duration_sec: Mean duration of each acceleration
    - acc_mean_amplitude: Mean amplitude (bpm above baseline)
    - acc_max_amplitude: Max amplitude
    - acc_coverage_ratio: Ratio of signal covered by accelerations
    """
    duration_min = signal_length / sample_rate / 60.0
    
    acc_count = len(accelerations)
    acc_rate_per_10min = acc_count / duration_min * 10.0 if duration_min > 0 else 0
    
    # Total duration from binary mask
    acc_total_samples = np.sum(acc_binary)
    acc_total_duration_sec = acc_total_samples / sample_rate
    acc_coverage_ratio = acc_total_samples / signal_length if signal_length > 0 else 0
    
    # Per-acceleration statistics
    if acc_count > 0:
        durations = [acc.duration_sec for acc in accelerations]
        amplitudes = [acc.peak_amplitude for acc in accelerations]
        acc_mean_duration_sec = np.mean(durations)
        acc_mean_amplitude = np.mean(amplitudes)
        acc_max_amplitude = np.max(amplitudes)
    else:
        acc_mean_duration_sec = 0.0
        acc_mean_amplitude = 0.0
        acc_max_amplitude = 0.0
    
    return {
        'acc_rate_per_10min': acc_rate_per_10min,
        'acc_mean_duration_sec': acc_mean_duration_sec,
        'acc_mean_amplitude': acc_mean_amplitude,
        'acc_max_amplitude': acc_max_amplitude,
        'acc_coverage_ratio': acc_coverage_ratio,
    }


def extract_deceleration_features(
    decelerations: List,
    dec_binary: np.ndarray,
    signal_length: int,
    sample_rate: float = 4.0
) -> Dict[str, float]:
    """
    Extract deceleration-related features.
    
    Length-agnostic features (rates, means, ratios):
    - dec_rate_per_10min: Deceleration rate per 10 minutes
    - dec_mean_duration_sec: Mean duration of each deceleration
    - dec_mean_amplitude: Mean amplitude (bpm below baseline, positive value)
    - dec_max_amplitude: Max amplitude
    - dec_coverage_ratio: Ratio of signal covered by decelerations
    - dec_mean_area: Mean area per deceleration (amplitude × duration)
    """
    duration_min = signal_length / sample_rate / 60.0
    
    dec_count = len(decelerations)
    dec_rate_per_10min = dec_count / duration_min * 10.0 if duration_min > 0 else 0
    
    # Total duration from binary mask
    dec_total_samples = np.sum(dec_binary)
    dec_total_duration_sec = dec_total_samples / sample_rate
    dec_coverage_ratio = dec_total_samples / signal_length if signal_length > 0 else 0
    
    # Per-deceleration statistics
    if dec_count > 0:
        durations = [dec.duration_sec for dec in decelerations]
        amplitudes = [abs(dec.nadir_amplitude) for dec in decelerations]  # Make positive
        dec_mean_duration_sec = np.mean(durations)
        dec_mean_amplitude = np.mean(amplitudes)
        dec_max_amplitude = np.max(amplitudes)
        dec_mean_area = float(np.mean([d * a for d, a in zip(durations, amplitudes)]))
    else:
        dec_mean_duration_sec = 0.0
        dec_mean_amplitude = 0.0
        dec_max_amplitude = 0.0
        dec_mean_area = 0.0
    
    return {
        'dec_rate_per_10min': dec_rate_per_10min,
        'dec_mean_duration_sec': dec_mean_duration_sec,
        'dec_mean_amplitude': dec_mean_amplitude,
        'dec_max_amplitude': dec_max_amplitude,
        'dec_coverage_ratio': dec_coverage_ratio,
        'dec_mean_area': dec_mean_area,
    }


def extract_signal_quality_features(
    mask: np.ndarray,
    quality_stats: Dict
) -> Dict[str, float]:
    """
    Extract signal quality features.
    
    Features:
    - signal_loss_ratio: Ratio of invalid/lost signal
    - valid_signal_ratio: Ratio of valid signal
    """
    # mask convention: 0 = reliable, 1 = unreliable
    loss_ratio = float(np.mean(mask))          # mean of 1s = fraction unreliable
    valid_ratio = 1.0 - loss_ratio
    
    return {
        'signal_loss_ratio': loss_ratio,
        'valid_signal_ratio': valid_ratio,
    }


def extract_fmp_features(
    fmp: np.ndarray,
    sample_rate: float = 4.0
) -> Dict[str, float]:
    """
    Extract fetal movement pattern (FMP) features from a binary signal.

    FMP is a binary (0/1) signal where 1 indicates detected fetal movement.
    All features are length-agnostic (rates, ratios, means).

    Features:
    - fmp_coverage_ratio: Fraction of recording with movement detected
    - fmp_burst_rate_per_10min: Number of movement bursts per 10 minutes
    - fmp_mean_burst_duration_sec: Mean duration of each movement burst
    - fmp_std_burst_duration_sec: Std of burst durations
    - fmp_mean_gap_duration_sec: Mean gap between consecutive bursts
    - fmp_std_gap_duration_sec: Std of gap durations
    """
    fmp = np.asarray(fmp, dtype=np.int32)
    n = len(fmp)
    duration_min = n / sample_rate / 60.0

    # Coverage ratio
    fmp_coverage = float(np.mean(fmp > 0)) if n > 0 else 0.0

    # Detect bursts: contiguous runs of 1
    # Pad with 0 on both sides to catch bursts at boundaries
    padded = np.concatenate(([0], (fmp > 0).astype(np.int32), [0]))
    diff = np.diff(padded)
    burst_starts = np.where(diff == 1)[0]
    burst_ends = np.where(diff == -1)[0]

    n_bursts = len(burst_starts)
    burst_rate_per_10min = n_bursts / duration_min * 10.0 if duration_min > 0 else 0.0

    if n_bursts > 0:
        burst_durations = (burst_ends - burst_starts) / sample_rate  # seconds
        mean_burst_dur = float(np.mean(burst_durations))
        std_burst_dur = float(np.std(burst_durations))
    else:
        mean_burst_dur = 0.0
        std_burst_dur = 0.0

    # Gaps between consecutive bursts
    if n_bursts > 1:
        gap_durations = (burst_starts[1:] - burst_ends[:-1]) / sample_rate  # seconds
        mean_gap_dur = float(np.mean(gap_durations))
        std_gap_dur = float(np.std(gap_durations))
    else:
        mean_gap_dur = 0.0
        std_gap_dur = 0.0

    return {
        'fmp_coverage_ratio': fmp_coverage,
        'fmp_burst_rate_per_10min': burst_rate_per_10min,
        'fmp_mean_burst_duration_sec': mean_burst_dur,
        'fmp_std_burst_duration_sec': std_burst_dur,
        'fmp_mean_gap_duration_sec': mean_gap_dur,
        'fmp_std_gap_duration_sec': std_gap_dur,
    }


def extract_toco_features(
    denoised_toco: np.ndarray,
    toco_baseline: np.ndarray,
    toco_quality_mask: np.ndarray,
    sample_rate: float = 4.0,
) -> Dict[str, float]:
    """
    Extract TOCO signal features — all length-agnostic.

    Features capture resting uterine tone, tonus variability, and signal quality.

    - toco_baseline_mean: Mean resting uterine tone
    - toco_baseline_std: Variability of resting tone over time
    - toco_baseline_range: Range of baseline drift
    - toco_above_baseline_mean: Mean elevation above baseline (activity index)
    - toco_above_baseline_std: Std of elevation
    - toco_above_baseline_p90: 90th percentile of elevation (peak activity)
    - toco_signal_std: Overall variability of denoised TOCO
    - toco_valid_ratio: Fraction of TOCO signal that is valid
    """
    toco = np.asarray(denoised_toco, dtype=np.float64)
    bl = np.asarray(toco_baseline, dtype=np.float64)
    qm = np.asarray(toco_quality_mask)

    valid = (qm == 0)
    n_valid = int(np.sum(valid))

    # Baseline statistics
    if n_valid > 0:
        vbl = bl[valid]
        bl_mean = float(np.mean(vbl))
        bl_std = float(np.std(vbl))
        bl_range = float(np.max(vbl) - np.min(vbl))
    else:
        bl_mean = float(np.mean(bl))
        bl_std = float(np.std(bl))
        bl_range = float(np.max(bl) - np.min(bl))

    # Above-baseline (elevation) — captures contraction activity intensity
    above = np.maximum(toco - bl, 0.0)
    if n_valid > 0:
        va = above[valid]
    else:
        va = above
    ab_mean = float(np.mean(va))
    ab_std = float(np.std(va))
    ab_p90 = float(np.percentile(va, 90))

    # Overall signal variability
    if n_valid > 0:
        sig_std = float(np.std(toco[valid]))
    else:
        sig_std = float(np.std(toco))

    toco_valid_ratio = float(np.mean(valid)) if len(valid) > 0 else 0.0

    return {
        'toco_baseline_mean': bl_mean,
        'toco_baseline_std': bl_std,
        'toco_baseline_range': bl_range,
        'toco_above_baseline_mean': ab_mean,
        'toco_above_baseline_std': ab_std,
        'toco_above_baseline_p90': ab_p90,
        'toco_signal_std': sig_std,
        'toco_valid_ratio': toco_valid_ratio,
    }


def extract_uc_features(
    ucs: List,
    uc_binary: np.ndarray,
    signal_length: int,
    sample_rate: float = 4.0,
) -> Dict[str, float]:
    """
    Extract uterine contraction (UC) features — all length-agnostic.

    - uc_rate_per_10min: Contraction frequency (clinical: normal 3-5/10 min)
    - uc_coverage_ratio: Fraction of recording occupied by contractions
    - uc_mean_duration_sec: Mean contraction duration
    - uc_std_duration_sec: Variability in contraction duration
    - uc_mean_strength: Mean contraction strength (peak − baseline)
    - uc_max_strength: Max contraction strength
    - uc_mean_area: Mean area under contraction curve
    - uc_mean_rise_time_sec: Mean time from start to peak
    - uc_mean_fall_time_sec: Mean time from peak to end
    - uc_rise_fall_ratio: Mean rise/fall time ratio (asymmetry)
    - uc_mean_interval_sec: Mean interval between successive peaks
    - uc_std_interval_sec: Regularity of contractions
    """
    duration_min = signal_length / sample_rate / 60.0
    n_uc = len(ucs)

    uc_rate_per_10min = n_uc / duration_min * 10.0 if duration_min > 0 else 0.0

    # Coverage from binary mask
    uc_total = int(np.sum(uc_binary))
    uc_coverage = uc_total / signal_length if signal_length > 0 else 0.0

    if n_uc > 0:
        durations = [uc.duration_sec for uc in ucs]
        strengths = [uc.strength for uc in ucs]
        areas = [uc.area for uc in ucs]
        rises = [uc.rise_time_sec for uc in ucs]
        falls = [uc.fall_time_sec for uc in ucs]

        mean_dur = float(np.mean(durations))
        std_dur = float(np.std(durations))
        mean_str = float(np.mean(strengths))
        max_str = float(np.max(strengths))
        mean_area = float(np.mean(areas))
        mean_rise = float(np.mean(rises))
        mean_fall = float(np.mean(falls))

        # Rise/fall ratio (>1 = slow rise, <1 = slow fall)
        ratios = [r / f if f > 0 else np.nan for r, f in zip(rises, falls)]
        valid_ratios = [r for r in ratios if np.isfinite(r)]
        rf_ratio = float(np.mean(valid_ratios)) if valid_ratios else np.nan
    else:
        mean_dur = std_dur = 0.0
        mean_str = max_str = mean_area = 0.0
        mean_rise = mean_fall = 0.0
        rf_ratio = np.nan

    # Inter-contraction interval (peak-to-peak)
    if n_uc > 1:
        peaks_sorted = sorted(uc.peak_index for uc in ucs)
        intervals = np.diff(peaks_sorted) / sample_rate  # seconds
        mean_interval = float(np.mean(intervals))
        std_interval = float(np.std(intervals))
    else:
        mean_interval = 0.0
        std_interval = 0.0

    return {
        'uc_rate_per_10min': uc_rate_per_10min,
        'uc_coverage_ratio': uc_coverage,
        'uc_mean_duration_sec': mean_dur,
        'uc_std_duration_sec': std_dur,
        'uc_mean_strength': mean_str,
        'uc_max_strength': max_str,
        'uc_mean_area': mean_area,
        'uc_mean_rise_time_sec': mean_rise,
        'uc_mean_fall_time_sec': mean_fall,
        'uc_rise_fall_ratio': rf_ratio if np.isfinite(rf_ratio) else np.nan,
        'uc_mean_interval_sec': mean_interval,
        'uc_std_interval_sec': std_interval,
    }


def extract_fhr_distribution_features(
    clean_fhr: np.ndarray,
    mask: np.ndarray
) -> Dict[str, float]:
    """
    Extract FHR distribution features.
    
    Features:
    - fhr_mean: Mean FHR
    - fhr_std: Std of FHR
    - fhr_min: Min FHR
    - fhr_max: Max FHR
    - fhr_range: Range of FHR
    - fhr_percentile_5: 5th percentile
    - fhr_percentile_25: 25th percentile (Q1)
    - fhr_percentile_50: Median
    - fhr_percentile_75: 75th percentile (Q3)
    - fhr_percentile_95: 95th percentile
    - fhr_iqr: Interquartile range
    - fhr_skewness: Skewness of FHR distribution
    - fhr_kurtosis: Kurtosis of FHR distribution
    """
    # mask convention: 0 = reliable, 1 = unreliable
    valid_mask = (mask == 0)
    valid_fhr = clean_fhr[valid_mask] if np.any(valid_mask) else clean_fhr
    
    # Filter out zeros and invalid values
    valid_fhr = valid_fhr[(valid_fhr > 50) & (valid_fhr < 220)]
    if len(valid_fhr) == 0:
        valid_fhr = clean_fhr[(clean_fhr > 50) & (clean_fhr < 220)]
    if len(valid_fhr) == 0:
        valid_fhr = np.array([120.0])  # Default normal FHR
    
    fhr_mean = np.mean(valid_fhr)
    fhr_std = np.std(valid_fhr)
    fhr_min = np.min(valid_fhr)
    fhr_max = np.max(valid_fhr)
    fhr_range = fhr_max - fhr_min
    
    # Percentiles
    fhr_p5 = np.percentile(valid_fhr, 5)
    fhr_p25 = np.percentile(valid_fhr, 25)
    fhr_p50 = np.percentile(valid_fhr, 50)
    fhr_p75 = np.percentile(valid_fhr, 75)
    fhr_p95 = np.percentile(valid_fhr, 95)
    fhr_iqr = fhr_p75 - fhr_p25
    
    # Higher moments
    if len(valid_fhr) > 2 and fhr_std > 0:
        fhr_skewness = np.mean(((valid_fhr - fhr_mean) / fhr_std) ** 3)
        fhr_kurtosis = np.mean(((valid_fhr - fhr_mean) / fhr_std) ** 4) - 3
    else:
        fhr_skewness = 0.0
        fhr_kurtosis = 0.0
    
    return {
        'fhr_mean': fhr_mean,
        'fhr_std': fhr_std,
        'fhr_min': fhr_min,
        'fhr_max': fhr_max,
        'fhr_range': fhr_range,
        'fhr_percentile_5': fhr_p5,
        'fhr_percentile_25': fhr_p25,
        'fhr_percentile_50': fhr_p50,
        'fhr_percentile_75': fhr_p75,
        'fhr_percentile_95': fhr_p95,
        'fhr_iqr': fhr_iqr,
        'fhr_skewness': fhr_skewness,
        'fhr_kurtosis': fhr_kurtosis,
    }


def extract_clinical_category_features(
    baseline: np.ndarray,
    bv: np.ndarray,
    accelerations: List,
    decelerations: List,
    mask: np.ndarray,
    signal_length: int,
    sample_rate: float = 4.0
) -> Dict[str, float]:
    """
    Extract FIGO-based clinical category features.
    
    Features:
    - baseline_category: 0=normal(110-160), 1=mild abnormal, 2=severe abnormal
    - variability_category: 0=absent(<2), 1=reduced(2-5), 2=normal(6-25), 3=marked(>25 bpm)
    - has_accelerations: Binary - presence of accelerations
    - has_decelerations: Binary - presence of decelerations
    - bradycardia_ratio: Ratio of time with FHR < 110 bpm
    - tachycardia_ratio: Ratio of time with FHR > 160 bpm
    """
    # mask convention: 0 = reliable, 1 = unreliable
    valid_mask = (mask == 0)
    valid_baseline = baseline[valid_mask] if np.any(valid_mask) else baseline
    
    # Baseline category
    mean_baseline = np.mean(valid_baseline)
    if 110 <= mean_baseline <= 160:
        baseline_category = 0  # Normal
    elif 100 <= mean_baseline < 110 or 160 < mean_baseline <= 170:
        baseline_category = 1  # Mild abnormal
    else:
        baseline_category = 2  # Severe abnormal
    
    # Variability category based on baseline variability (bpm)
    valid_bv = bv[np.isfinite(bv)]
    mean_bv = float(np.mean(valid_bv)) if len(valid_bv) > 0 else 0.0
    
    if mean_bv < 2:
        variability_category = 0  # Absent
    elif mean_bv < 6:
        variability_category = 1  # Reduced/minimal
    elif mean_bv <= 25:
        variability_category = 2  # Normal/moderate
    else:
        variability_category = 3  # Marked
    
    # Accelerations and decelerations presence
    has_accelerations = 1.0 if len(accelerations) > 0 else 0.0
    has_decelerations = 1.0 if len(decelerations) > 0 else 0.0
    
    # Bradycardia and tachycardia ratios
    bradycardia_samples = np.sum((valid_baseline < 110))
    tachycardia_samples = np.sum((valid_baseline > 160))
    total_valid = len(valid_baseline)
    
    bradycardia_ratio = bradycardia_samples / total_valid if total_valid > 0 else 0
    tachycardia_ratio = tachycardia_samples / total_valid if total_valid > 0 else 0
    
    return {
        'baseline_category': float(baseline_category),
        'variability_category': float(variability_category),
        'has_accelerations': has_accelerations,
        'has_decelerations': has_decelerations,
        'bradycardia_ratio': bradycardia_ratio,
        'tachycardia_ratio': tachycardia_ratio,
    }


def extract_all_features(
    clean_fhr: np.ndarray,
    baseline: np.ndarray,
    bv: np.ndarray,
    stv: np.ndarray,
    ltv: np.ndarray,
    stv_overall_ms: float,
    ltv_overall_ms: float,
    accelerations: List,
    acc_binary: np.ndarray,
    decelerations: List,
    dec_binary: np.ndarray,
    fmp: np.ndarray,
    denoised_toco: np.ndarray,
    toco_baseline: np.ndarray,
    toco_quality_mask: np.ndarray,
    ucs: List,
    uc_binary: np.ndarray,
    mask: np.ndarray,
    quality_stats: Dict,
    sample_rate: float = 4.0
) -> Dict[str, float]:
    """
    Extract all ML features from CTG signals.
    
    All features are length-agnostic: means, rates, ratios, percentiles.
    
    Returns:
        Dictionary of feature_name -> feature_value
    """
    signal_length = len(clean_fhr)
    
    features = {}
    
    # 1. Baseline features
    features.update(extract_baseline_features(clean_fhr, baseline, mask))
    
    # 2. Variability features (BV + STV + LTV)
    features.update(extract_variability_features(
        bv, stv, ltv, stv_overall_ms, ltv_overall_ms
    ))
    
    # 3. Acceleration features
    features.update(extract_acceleration_features(
        accelerations, acc_binary, signal_length, sample_rate
    ))
    
    # 4. Deceleration features
    features.update(extract_deceleration_features(
        decelerations, dec_binary, signal_length, sample_rate
    ))
    
    # 5. Signal quality features
    features.update(extract_signal_quality_features(mask, quality_stats))
    
    # 6. FHR distribution features
    features.update(extract_fhr_distribution_features(clean_fhr, mask))
    
    # 7. Fetal movement features
    features.update(extract_fmp_features(fmp, sample_rate))
    
    # 8. TOCO signal features
    features.update(extract_toco_features(
        denoised_toco, toco_baseline, toco_quality_mask, sample_rate
    ))
    
    # 9. Uterine contraction features
    features.update(extract_uc_features(
        ucs, uc_binary, signal_length, sample_rate
    ))
    
    # 10. Clinical category features
    features.update(extract_clinical_category_features(
        baseline, bv, accelerations, decelerations, mask, signal_length, sample_rate
    ))
    
    return features


# =============================================================================
# Main Processing
# =============================================================================

SAMPLE_RATE = 4.0  # Hz


def _run_pipeline(
    raw_fhr: np.ndarray,
    raw_fmp: np.ndarray,
    raw_toco: np.ndarray,
    sample_rate: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Run the full CTG preprocessing pipeline and return (features, intermediates).
    intermediates contains arrays needed for visualization (clean_fhr, baseline, etc.).
    """
    sr = float(sample_rate)
    sr_int = max(1, int(round(sr)))

    def sec_to_samples(seconds: float, min_samples: int = 1, make_odd: bool = False) -> int:
        samples = max(min_samples, int(round(seconds * sr)))
        if make_odd and samples % 2 == 0:
            samples += 1
        return samples

    mask, quality_stats = assess_signal_quality(raw_fhr, sample_rate=sr)
    correction_config = CorrectionConfig(
        gap_threshold=sec_to_samples(30.0),
        fixed_value_min_run=sec_to_samples(5.0),
        smoothing_window=sec_to_samples(1.25, min_samples=1, make_odd=True),
        min_valid_neighbors=sec_to_samples(1.0),
        neighbor_search_window=sec_to_samples(120.0),
    )
    clean_fhr, _ = correct_artifacts(raw_fhr, mask, config=correction_config)

    baseline_config = BaselineConfig(
        window_size=sec_to_samples(8 * 60.0),
        window_step=sec_to_samples(60.0),
        smoothing_window=sec_to_samples(60.0),
    )
    baseline = analyse_baseline_optimized(clean_fhr, config=baseline_config, mask=mask, sample_rate=sr)

    acc_config = AccelerationConfig(
        sample_rate=sr_int,
        criterion=AccelerationCriterion.RULE_15_15
    )
    accelerations, acc_binary = detect_accelerations_figo(
        clean_fhr, baseline, mask, acc_config
    )

    dec_config = DecelerationConfig(
        sample_rate=sr_int,
        criterion=DecelerationCriterion.RULE_15_15
    )
    decelerations, dec_binary = detect_decelerations_figo(
        clean_fhr, baseline, mask, uc_models=None, config=dec_config
    )

    bv_config = BaselineVariabilityConfig(
        window_sec=60.0,
        sampling_rate=sr,
        min_valid_ratio=0.5,
        use_baseline_deviation=True,
        smooth_output=True,
        smooth_window_sec=15.0,
        interpolate_invalid=True,
    )
    bv = compute_baseline_variability(
        fhr=clean_fhr,
        baseline=baseline,
        acc_mask=acc_binary,
        dec_mask=dec_binary,
        quality_mask=mask,
        config=bv_config,
    )

    stv = compute_stv(
        fhr=clean_fhr,
        acc_mask=acc_binary,
        dec_mask=dec_binary,
        quality_mask=mask,
        config=STVConfig(sampling_rate=sr),
    )
    stv_overall_ms = compute_stv_overall(
        fhr=clean_fhr,
        acc_mask=acc_binary,
        dec_mask=dec_binary,
        quality_mask=mask,
        config=STVConfig(sampling_rate=sr),
    )

    ltv = compute_ltv(
        fhr=clean_fhr,
        acc_mask=acc_binary,
        dec_mask=dec_binary,
        quality_mask=mask,
        config=LTVConfig(sampling_rate=sr),
    )
    ltv_overall_ms = compute_ltv_overall(
        fhr=clean_fhr,
        acc_mask=acc_binary,
        dec_mask=dec_binary,
        quality_mask=mask,
        config=LTVConfig(sampling_rate=sr),
    )

    toco_denoise_config = TocoDenoiseConfig(
        sample_rate=sr_int,
        spike_window=sec_to_samples(1.25, min_samples=1, make_odd=True),
        dropout_max_interpolate=sec_to_samples(5.0),
        plateau_min_length=sec_to_samples(3.0),
        plateau_max_interpolate=sec_to_samples(10.0),
        median_window=sec_to_samples(1.25, min_samples=1, make_odd=True),
    )
    toco_denoise_result = denoise_toco(raw_toco.astype(np.float64), config=toco_denoise_config)
    denoised_toco = toco_denoise_result.signal.astype(np.float64)
    toco_quality_mask = toco_denoise_result.quality_mask

    toco_baseline_config = TocoBaselineConfig(sample_rate=sr_int)
    toco_bl_result = analyse_toco_baseline_v2(
        denoised_toco, quality_mask=toco_quality_mask, config=toco_baseline_config
    )
    toco_baseline = toco_bl_result.baseline

    uc_config = UcDetectionConfigV2(sample_rate=sr_int)
    uc_result = detect_uc_v2(
        denoised_toco, toco_baseline, toco_quality_mask, config=uc_config
    )
    ucs = uc_result.contractions
    uc_binary = contractions_to_binary(ucs, len(denoised_toco))

    signal_length = len(clean_fhr)
    features = extract_all_features(
        clean_fhr=clean_fhr,
        baseline=baseline,
        bv=bv,
        stv=stv,
        ltv=ltv,
        stv_overall_ms=stv_overall_ms,
        ltv_overall_ms=ltv_overall_ms,
        accelerations=accelerations,
        acc_binary=acc_binary,
        decelerations=decelerations,
        dec_binary=dec_binary,
        fmp=raw_fmp,
        denoised_toco=denoised_toco,
        toco_baseline=toco_baseline,
        toco_quality_mask=toco_quality_mask,
        ucs=ucs,
        uc_binary=uc_binary,
        mask=mask,
        quality_stats=quality_stats,
        sample_rate=sr,
    )

    intermediates = {
        'raw_fhr': raw_fhr,
        'clean_fhr': clean_fhr,
        'baseline': baseline,
        'mask': mask,
        'acc_binary': acc_binary,
        'dec_binary': dec_binary,
        'bv': bv,
        'stv': stv,
        'ltv': ltv,
        'raw_fmp': raw_fmp,
        'denoised_toco': denoised_toco,
        'toco_baseline': toco_baseline,
        'toco_quality_mask': toco_quality_mask,
        'uc_binary': uc_binary,
    }
    return features, intermediates


def run_pipeline(
    raw_fhr: np.ndarray,
    raw_fmp: np.ndarray,
    raw_toco: np.ndarray,
    sample_rate: float = SAMPLE_RATE,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    公开接口：对 raw 信号运行完整预处理，返回 (features, intermediates)。
    供 dataset 构建等 pipeline 复用。
    """
    return _run_pipeline(raw_fhr, raw_fmp, raw_toco, sample_rate)


def extract_features_from_fetal_path(
    fetal_path: str,
    sample_name: str,
    start_point: int,
    end_point: int,
    label: int,
    gestational_age_raw=None,
    sample_rate: float = SAMPLE_RATE,
    return_intermediates: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, Any]]]:
    """
    Extract one sample's features from a fetal file path and segment range.

    If return_intermediates=True, returns (features_dict, intermediates_dict)
    where intermediates_dict contains arrays for visualization (clean_fhr, baseline, etc.).
    """
    fetal_data = read_fetal(fetal_path)
    raw_fhr = fetal_data.fhr[start_point:end_point]
    raw_fmp = fetal_data.fmp[start_point:end_point]
    raw_toco = fetal_data.toco[start_point:end_point]

    features, intermediates = _run_pipeline(raw_fhr, raw_fmp, raw_toco, sample_rate)

    features['sample_name'] = sample_name
    features['label'] = label
    features['signal_length'] = len(raw_fhr)
    features['duration_min'] = len(raw_fhr) / float(sample_rate) / 60.0

    try:
        features['gestational_age'] = int(float(str(gestational_age_raw).split('+')[0]))
    except (ValueError, TypeError):
        print(f"  Warning: Invalid gestational age for {sample_name}, setting as NaN")
        features['gestational_age'] = np.nan

    if return_intermediates:
        return features, intermediates
    return features

