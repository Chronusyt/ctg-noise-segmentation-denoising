from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

import numpy as np

try:
    from .config_injection import get_default_config
    from .inject_doubling_halving import inject_one_signal
except ImportError:
    from config_injection import get_default_config
    from inject_doubling_halving import inject_one_signal


def _make_sample_ids(count: int) -> np.ndarray:
    return np.array([f"clean_{idx:06d}" for idx in range(count)], dtype=object)


def _metadata_row(sample_id: str, reliability_score: float, metadata: dict) -> dict:
    halving_segments = metadata["halving_segments"]
    doubling_segments = metadata["doubling_segments"]
    return {
        "sample_id": sample_id,
        "reliability_score": round(float(reliability_score), 6),
        "has_halving": int(bool(metadata["has_halving"])),
        "has_doubling": int(bool(metadata["has_doubling"])),
        "num_halving_segments": len(halving_segments),
        "num_doubling_segments": len(doubling_segments),
        "halving_segments_json": json.dumps(halving_segments, ensure_ascii=False),
        "doubling_segments_json": json.dumps(doubling_segments, ensure_ascii=False),
        "double_subtype_json": json.dumps(metadata["double_subtype"], ensure_ascii=False),
        "ratio_used_json": json.dumps(metadata["ratio_used"], ensure_ascii=False),
        "actual_mean_ratio_json": json.dumps(metadata["actual_mean_ratio"], ensure_ascii=False),
        "duration_sec_json": json.dumps(metadata["duration_sec"], ensure_ascii=False),
        "ceiling_used_json": json.dumps(metadata["ceiling_used"], ensure_ascii=False),
        "baseline_strategy": metadata["baseline_strategy"],
    }


def _save_metadata_csv(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "sample_id",
        "reliability_score",
        "has_halving",
        "has_doubling",
        "num_halving_segments",
        "num_doubling_segments",
        "halving_segments_json",
        "doubling_segments_json",
        "double_subtype_json",
        "ratio_used_json",
        "actual_mean_ratio_json",
        "duration_sec_json",
        "ceiling_used_json",
        "baseline_strategy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = get_default_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(cfg.data_path, allow_pickle=True)
    clean_signals = np.asarray(dataset["signals"], dtype=np.float64)
    reliability_scores = np.asarray(dataset["reliability_scores"], dtype=np.float32)
    sample_ids = _make_sample_ids(len(clean_signals))

    rng = np.random.default_rng(cfg.random_seed)

    noisy_signals = np.zeros_like(clean_signals, dtype=np.float32)
    artifact_masks = np.zeros(clean_signals.shape, dtype=np.uint8)
    metadata_rows: List[dict] = []

    total_halving = 0
    total_doubling = 0

    for idx, clean_signal in enumerate(clean_signals):
        noisy_signal, artifact_mask, metadata = inject_one_signal(clean_signal, rng=rng, cfg=cfg)
        noisy_signals[idx] = noisy_signal
        artifact_masks[idx] = artifact_mask

        total_halving += len(metadata["halving_segments"])
        total_doubling += len(metadata["doubling_segments"])

        metadata_rows.append(
            _metadata_row(
                sample_id=str(sample_ids[idx]),
                reliability_score=float(reliability_scores[idx]),
                metadata=metadata,
            )
        )

    dataset_path = cfg.output_dir / cfg.synthetic_dataset_name
    metadata_path = cfg.output_dir / cfg.synthetic_metadata_name

    np.savez_compressed(
        dataset_path,
        clean_signals=clean_signals.astype(np.float32),
        noisy_signals=noisy_signals,
        artifact_masks=artifact_masks,
        sample_ids=sample_ids,
        reliability_scores=reliability_scores,
    )
    _save_metadata_csv(metadata_path, metadata_rows)

    print("=" * 68)
    print(f"Input clean dataset      : {cfg.data_path}")
    print(f"Saved synthetic dataset  : {dataset_path}")
    print(f"Saved metadata csv       : {metadata_path}")
    print(f"Num samples              : {len(clean_signals):,}")
    print(f"Injected halving segments: {total_halving:,}")
    print(f"Injected doubling segs   : {total_doubling:,}")
    print("=" * 68)


if __name__ == "__main__":
    main()
