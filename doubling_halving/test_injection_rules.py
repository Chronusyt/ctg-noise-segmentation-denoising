from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from .config_injection import get_default_config
    from .inject_doubling_halving import compute_local_baseline, inject_one_signal
except ImportError:
    from config_injection import get_default_config
    from inject_doubling_halving import compute_local_baseline, inject_one_signal


def _collect_segments(metadata_list: List[dict], key: str) -> List[dict]:
    segments: List[dict] = []
    for metadata in metadata_list:
        segments.extend(metadata[key])
    return segments


def _distribution(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _choose_test_cases(clean_all: np.ndarray, cfg) -> List[tuple[int, str]]:
    target_unclipped = max(2, cfg.test_sample_count // 4)
    target_clipped = max(2, cfg.test_sample_count // 4)

    unclipped_candidates: List[int] = []
    clipped_candidates: List[int] = []

    scan_limit = len(clean_all)
    for idx in range(scan_limit):
        baseline, _ = compute_local_baseline(clean_all[idx], cfg)
        baseline_min = float(np.min(baseline))
        baseline_mean = float(np.mean(baseline))

        if baseline_min <= cfg.unclipped_double_baseline_max and len(unclipped_candidates) < target_unclipped:
            unclipped_candidates.append(idx)
        if (
            cfg.clipped_double_baseline_min <= baseline_mean <= cfg.clipped_double_baseline_max
            and len(clipped_candidates) < target_clipped
        ):
            clipped_candidates.append(idx)

        if len(unclipped_candidates) >= target_unclipped and len(clipped_candidates) >= target_clipped:
            break

    cases: List[tuple[int, str]] = []
    used = set()

    for idx in unclipped_candidates:
        if idx not in used:
            cases.append((idx, "unclipped"))
            used.add(idx)

    for idx in clipped_candidates:
        if idx not in used:
            cases.append((idx, "clipped"))
            used.add(idx)

    rng = np.random.default_rng(cfg.random_seed)
    while len(cases) < cfg.test_sample_count:
        idx = int(rng.integers(0, len(clean_all)))
        if idx in used:
            continue
        cases.append((idx, "mixed"))
        used.add(idx)

    return cases[:cfg.test_sample_count]


def _segment_label(segment: dict) -> str:
    if segment["region_type"] == "double":
        subtype = segment.get("double_subtype") or "unknown"
        prefix = f"double/{subtype}"
        ceiling_txt = f" ceiling={segment['ceiling_used']:.1f}" if segment.get("ceiling_used") is not None else ""
    else:
        prefix = "half"
        ceiling_txt = ""
    return (
        f"{prefix} dur={segment['duration_sec']:.2f}s "
        f"ratio_used={segment['ratio_used']:.3f} "
        f"actual={segment['actual_mean_ratio']:.3f}{ceiling_txt}"
    )


def _validate_segments(halving_segments: List[dict], doubling_segments: List[dict], noisy_signals: np.ndarray, metadata_list: List[dict], cfg) -> None:
    all_durations = [seg["duration_sec"] for seg in halving_segments + doubling_segments]
    if all_durations:
        assert max(all_durations) <= cfg.max_duration_sec + 1e-6, "Found segment longer than 15 sec"

    for seg in halving_segments:
        assert cfg.halving_ratio_range[0] - 0.02 <= seg["actual_mean_ratio"] <= cfg.halving_ratio_range[1] + 0.02, (
            f"Halving actual ratio out of range: {seg['actual_mean_ratio']}"
        )

    unclipped_doubling = [seg for seg in doubling_segments if seg.get("double_subtype") == "unclipped"]
    clipped_doubling = [seg for seg in doubling_segments if seg.get("double_subtype") == "clipped"]

    for seg in unclipped_doubling:
        assert seg["mean_baseline"] <= cfg.unclipped_double_baseline_max + 1e-6, "Unclipped doubling baseline too high"
        expected_ratio = min(seg["ratio_used"], cfg.valid_fhr_max / max(seg["mean_baseline"], 1.0))
        assert abs(seg["actual_mean_ratio"] - expected_ratio) <= 0.10, (
            f"Unclipped doubling actual ratio out of range: {seg['actual_mean_ratio']}"
        )

    for seg in clipped_doubling:
        assert cfg.clipped_double_baseline_min - 1e-6 <= seg["mean_baseline"] <= cfg.clipped_double_baseline_max + 1e-6, (
            "Clipped doubling baseline outside expected range"
        )
        expected_ratio = min(seg["ratio_used"], seg["ceiling_used"] / max(seg["mean_baseline"], 1.0))
        assert abs(seg["actual_mean_ratio"] - expected_ratio) <= 0.18, (
            f"Clipped doubling ratio inconsistent with ceiling: {seg['actual_mean_ratio']}"
        )
        assert cfg.double_ceiling_range[0] <= seg["ceiling_used"] <= cfg.double_ceiling_range[1], "Ceiling out of configured range"

    all_noisy = np.asarray(noisy_signals, dtype=np.float64)
    assert np.all(all_noisy >= cfg.valid_fhr_min - 1e-6), "Noisy signal below valid range"
    assert np.all(all_noisy <= cfg.valid_fhr_max + 1e-6), "Noisy signal above valid range"

    for metadata in metadata_list:
        baseline = np.asarray(metadata["baseline"], dtype=np.float64)
        assert np.all(np.isfinite(baseline)), "Baseline contains invalid values"


def _plot_examples(
    clean_signals: np.ndarray,
    noisy_signals: np.ndarray,
    metadata_list: List[dict],
    masks: np.ndarray,
    case_modes: List[str],
    plot_dir: Path,
    cfg,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(len(clean_signals)):
        clean = clean_signals[idx]
        noisy = noisy_signals[idx]
        baseline = metadata_list[idx]["baseline"]
        mask = masks[idx]

        t = np.arange(len(clean)) / cfg.fs
        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True, gridspec_kw={"height_ratios": [4, 1.5, 1]})
        ax_sig, ax_ratio, ax_mask = axes

        ax_sig.plot(t, clean, lw=0.8, color="#455A64", label="Clean")
        ax_sig.plot(t, noisy, lw=0.9, color="#1976D2", label="Noisy")
        ax_sig.plot(t, baseline, lw=1.0, ls="--", color="#212121", label="Baseline")
        ax_sig.legend(loc="upper right")
        ax_sig.set_ylabel("FHR (bpm)")
        ax_sig.set_ylim(45, 260)
        ax_sig.grid(True, alpha=0.25)

        ratio = noisy / np.maximum(baseline, 1.0)
        ax_ratio.plot(t, ratio, lw=0.9, color="#7B1FA2", label="Noisy / baseline")
        ax_ratio.axhline(0.5, color="#E65100", ls="--", lw=0.8)
        ax_ratio.axhline(2.0, color="#1B5E20", ls="--", lw=0.8)
        ax_ratio.set_ylabel("Ratio")
        ax_ratio.set_ylim(0.0, 2.4)
        ax_ratio.grid(True, alpha=0.25)

        ax_mask.step(t, mask, where="mid", color="#D32F2F", lw=1.0)
        ax_mask.set_ylabel("Mask")
        ax_mask.set_xlabel("Time (sec)")
        ax_mask.set_yticks([0, 1, 2])
        ax_mask.grid(True, alpha=0.25)

        labels = [_segment_label(seg) for seg in metadata_list[idx]["halving_segments"]]
        labels.extend(_segment_label(seg) for seg in metadata_list[idx]["doubling_segments"])
        title_suffix = " | ".join(labels) if labels else "clean"
        fig.suptitle(f"sample {idx} | mode={case_modes[idx]} | {title_suffix}", fontsize=11)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        fig.savefig(plot_dir / f"test_injection_{idx:02d}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    cfg = get_default_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    test_cfg = replace(cfg, halving_prob=1.0, doubling_prob=1.0)

    dataset = np.load(cfg.data_path, allow_pickle=True)
    clean_all = np.asarray(dataset["signals"], dtype=np.float64)

    rng = np.random.default_rng(cfg.random_seed)
    sample_count = min(cfg.test_sample_count, len(clean_all))
    test_cases = _choose_test_cases(clean_all, test_cfg)[:sample_count]

    clean_examples = []
    noisy_examples = []
    masks = []
    metadata_list: List[dict] = []
    case_modes: List[str] = []

    for idx, mode in test_cases:
        clean_signal = clean_all[idx]
        if mode == "unclipped":
            sample_cfg = replace(
                test_cfg,
                enable_unclipped_doubling=True,
                enable_clipped_doubling=False,
            )
        elif mode == "clipped":
            sample_cfg = replace(
                test_cfg,
                enable_unclipped_doubling=False,
                enable_clipped_doubling=True,
            )
        else:
            sample_cfg = test_cfg

        noisy_signal, artifact_mask, metadata = inject_one_signal(clean_signal, rng=rng, cfg=sample_cfg)
        clean_examples.append(clean_signal.astype(np.float32))
        noisy_examples.append(noisy_signal.astype(np.float32))
        masks.append(artifact_mask)
        metadata_list.append(metadata)
        case_modes.append(mode)

    clean_examples = np.asarray(clean_examples, dtype=np.float32)
    noisy_examples = np.asarray(noisy_examples, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.uint8)

    halving_segments = _collect_segments(metadata_list, "halving_segments")
    doubling_segments = _collect_segments(metadata_list, "doubling_segments")
    unclipped_doubling = [seg for seg in doubling_segments if seg.get("double_subtype") == "unclipped"]
    clipped_doubling = [seg for seg in doubling_segments if seg.get("double_subtype") == "clipped"]

    _validate_segments(halving_segments, doubling_segments, noisy_examples, metadata_list, test_cfg)

    plot_dir = cfg.output_dir / cfg.test_plot_dirname
    _plot_examples(clean_examples, noisy_examples, metadata_list, masks, case_modes, plot_dir, test_cfg)

    halving_durations = [seg["duration_sec"] for seg in halving_segments]
    doubling_durations = [seg["duration_sec"] for seg in doubling_segments]
    unclipped_durations = [seg["duration_sec"] for seg in unclipped_doubling]
    clipped_durations = [seg["duration_sec"] for seg in clipped_doubling]
    halving_ratios = [seg["actual_mean_ratio"] for seg in halving_segments]
    doubling_ratios = [seg["actual_mean_ratio"] for seg in doubling_segments]
    unclipped_ratios = [seg["actual_mean_ratio"] for seg in unclipped_doubling]
    clipped_ratios = [seg["actual_mean_ratio"] for seg in clipped_doubling]
    halving_ratio_used = [seg["ratio_used"] for seg in halving_segments]
    unclipped_ratio_used = [seg["ratio_used"] for seg in unclipped_doubling]
    clipped_ratio_used = [seg["ratio_used"] for seg in clipped_doubling]
    ceiling_values = [seg["ceiling_used"] for seg in clipped_doubling if seg.get("ceiling_used") is not None]

    print("=" * 68)
    print(f"Test signals sampled      : {sample_count}")
    print("Injection probabilities   : halving=1.0, doubling=1.0 (test mode)")
    print(f"Case modes                : {case_modes}")
    print(f"Halving segments injected : {len(halving_segments)}")
    print(f"Unclipped doubling count  : {len(unclipped_doubling)}")
    print(f"Clipped doubling count    : {len(clipped_doubling)}")
    print(f"Doubling segments injected: {len(doubling_segments)}")
    print(f"Halving duration stats    : {_distribution(halving_durations)}")
    print(f"Unclipped dur stats       : {_distribution(unclipped_durations)}")
    print(f"Clipped dur stats         : {_distribution(clipped_durations)}")
    print(f"Doubling duration stats   : {_distribution(doubling_durations)}")
    print(f"Halving ratio_used stats  : {_distribution(halving_ratio_used)}")
    print(f"Unclipped ratio_used      : {_distribution(unclipped_ratio_used)}")
    print(f"Clipped ratio_used        : {_distribution(clipped_ratio_used)}")
    print(f"Halving actual ratio      : {_distribution(halving_ratios)}")
    print(f"Unclipped actual ratio    : {_distribution(unclipped_ratios)}")
    print(f"Clipped actual ratio      : {_distribution(clipped_ratios)}")
    print(f"Doubling actual ratio     : {_distribution(doubling_ratios)}")
    print(f"Clipped ceiling stats     : {_distribution(ceiling_values)}")
    print(f"Plots saved to            : {plot_dir}")
    print("=" * 68)


if __name__ == "__main__":
    main()
