from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT_ROOT / "feature"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "doubling_halving" / "output"


def _first_existing_dir(*candidates: Optional[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_dir():
            return candidate
    return None


def discover_default_data_dir() -> Optional[Path]:
    env_dir = os.environ.get("CTG_BATCH1_FETAL_DIR")
    env_path = Path(env_dir).expanduser() if env_dir else None

    return _first_existing_dir(
        env_path,
        PROJECT_ROOT / "batch1" / "fetal",
        PROJECT_ROOT / "data" / "batch1" / "fetal",
        Path("/scratch2/yzd/CTG/batch1/fetal"),
        Path("/scratch2/yzd/CTG/ctg_batch1/fetal"),
    )


@dataclass(frozen=True)
class DetectionConfig:
    sample_rate: float = 4.0

    segment_minutes: int = 20
    min_segment_minutes: int = 5
    min_quality_percent: float = 80.0

    half_ratio_lo: float = 0.42
    half_ratio_hi: float = 0.62
    double_ratio_lo: float = 1.70
    double_ratio_hi: float = 2.35

    min_duration_sec: float = 2.0
    max_duration_sec: float = 300.0

    valid_fhr_min: int = 55
    valid_fhr_max: int = 210
    baseline_min: float = 80.0
    baseline_max: float = 200.0

    min_confidence: float = 0.35
    confidence_bins: tuple[float, ...] = (0.35, 0.50, 0.70, 0.85, 1.01)

    baseline_strategy: str = "feature"
    baseline_window_sec: float = 300.0
    feature_baseline_window_size: int = 1920
    feature_baseline_window_step: int = 240
    feature_baseline_smoothing_window: int = 240
    feature_baseline_variability_threshold: float = 25.0
    feature_baseline_min_valid_ratio: float = 0.5

    plot_dpi: int = 140

    @property
    def segment_samples(self) -> int:
        return int(self.segment_minutes * 60 * self.sample_rate)

    @property
    def min_segment_samples(self) -> int:
        return int(self.min_segment_minutes * 60 * self.sample_rate)


@dataclass(frozen=True)
class RuntimeConfig:
    data_dir: Optional[Path] = None
    output_dir: Path = DEFAULT_OUTPUT_DIR
    num_workers: int = max(1, (os.cpu_count() or 4) - 2)
    save_plots: bool = True
    export_candidates: bool = True


def resolve_runtime_config(
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    num_workers: Optional[int] = None,
    save_plots: bool = True,
    export_candidates: bool = True,
) -> RuntimeConfig:
    resolved_data_dir = Path(data_dir).expanduser() if data_dir else discover_default_data_dir()
    resolved_output_dir = Path(output_dir).expanduser() if output_dir else DEFAULT_OUTPUT_DIR
    resolved_workers = num_workers if num_workers is not None else max(1, (os.cpu_count() or 4) - 2)

    return RuntimeConfig(
        data_dir=resolved_data_dir,
        output_dir=resolved_output_dir,
        num_workers=max(1, int(resolved_workers)),
        save_plots=save_plots,
        export_candidates=export_candidates,
    )


def config_to_dict(runtime_cfg: RuntimeConfig, detect_cfg: DetectionConfig) -> Dict[str, Any]:
    runtime_dict = asdict(runtime_cfg)
    runtime_dict["data_dir"] = str(runtime_cfg.data_dir) if runtime_cfg.data_dir else None
    runtime_dict["output_dir"] = str(runtime_cfg.output_dir)
    return {
        "runtime": runtime_dict,
        "detection": asdict(detect_cfg),
    }
