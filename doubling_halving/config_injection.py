from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


# Resolve project root from this file so the repo can be moved without
# rewriting absolute paths in config.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "feature" / "datasets" / "denoising_20min" / "clean_dataset.npz"
OUTPUT_DIR = PROJECT_ROOT / "doubling_halving" / "synthetic_output"

FS = 4.0

VALID_FHR_MIN = 50.0
VALID_FHR_MAX = 255.0

ENABLE_HALVING = True
ENABLE_DOUBLING = True

HALVING_PROB = 0.08
DOUBLING_PROB = 0.03

MAX_DURATION_SEC = 15.0

HALVING_DURATION_MAIN = (3.0, 8.0)
HALVING_DURATION_TAIL = (8.0, 15.0)

DOUBLING_DURATION_MAIN = (3.0, 8.0)
DOUBLING_DURATION_TAIL = (8.0, 15.0)

HALVING_RATIO_RANGE = (0.47, 0.55)
DOUBLING_RATIO_RANGE = (1.90, 2.05)

HALVING_SIGMA_RANGE = (1.5, 3.0)
DOUBLING_SIGMA_RANGE = (2.0, 4.0)

DOUBLE_CEILING_RANGE = (225.0, 250.0)

ENABLE_UNCLIPPED_DOUBLING = True
ENABLE_CLIPPED_DOUBLING = True

DOUBLE_UNCLIPPED_WEIGHT = 0.4
DOUBLE_CLIPPED_WEIGHT = 0.6

UNCLIPPED_DOUBLE_BASELINE_MAX = 115.0
CLIPPED_DOUBLE_BASELINE_MIN = 110.0
CLIPPED_DOUBLE_BASELINE_MAX = 145.0

TRANSITION_LEN_POINTS = (1, 3)

RANDOM_SEED = 42

HALVING_MAIN_WEIGHT = 0.80
DOUBLING_MAIN_WEIGHT = 0.80
MAX_PLACEMENT_ATTEMPTS = 64

BASELINE_STRATEGY = "feature"
BASELINE_WINDOW_SEC = 300.0
FEATURE_BASELINE_WINDOW_SIZE = 1920
FEATURE_BASELINE_WINDOW_STEP = 240
FEATURE_BASELINE_SMOOTHING_WINDOW = 240
FEATURE_BASELINE_VARIABILITY_THRESHOLD = 25.0
FEATURE_BASELINE_MIN_VALID_RATIO = 0.5

SYNTHETIC_DATASET_NAME = "synthetic_doubling_halving_dataset.npz"
SYNTHETIC_METADATA_NAME = "synthetic_metadata.csv"
TEST_PLOT_DIRNAME = "test_plots"

TEST_SAMPLE_COUNT = 8


@dataclass(frozen=True)
class InjectionConfig:
    data_path: Path = DATA_PATH
    output_dir: Path = OUTPUT_DIR
    fs: float = FS
    valid_fhr_min: float = VALID_FHR_MIN
    valid_fhr_max: float = VALID_FHR_MAX
    enable_halving: bool = ENABLE_HALVING
    enable_doubling: bool = ENABLE_DOUBLING
    halving_prob: float = HALVING_PROB
    doubling_prob: float = DOUBLING_PROB
    max_duration_sec: float = MAX_DURATION_SEC
    halving_duration_main: tuple[float, float] = HALVING_DURATION_MAIN
    halving_duration_tail: tuple[float, float] = HALVING_DURATION_TAIL
    doubling_duration_main: tuple[float, float] = DOUBLING_DURATION_MAIN
    doubling_duration_tail: tuple[float, float] = DOUBLING_DURATION_TAIL
    halving_ratio_range: tuple[float, float] = HALVING_RATIO_RANGE
    doubling_ratio_range: tuple[float, float] = DOUBLING_RATIO_RANGE
    halving_sigma_range: tuple[float, float] = HALVING_SIGMA_RANGE
    doubling_sigma_range: tuple[float, float] = DOUBLING_SIGMA_RANGE
    double_ceiling_range: tuple[float, float] = DOUBLE_CEILING_RANGE
    enable_unclipped_doubling: bool = ENABLE_UNCLIPPED_DOUBLING
    enable_clipped_doubling: bool = ENABLE_CLIPPED_DOUBLING
    double_unclipped_weight: float = DOUBLE_UNCLIPPED_WEIGHT
    double_clipped_weight: float = DOUBLE_CLIPPED_WEIGHT
    unclipped_double_baseline_max: float = UNCLIPPED_DOUBLE_BASELINE_MAX
    clipped_double_baseline_min: float = CLIPPED_DOUBLE_BASELINE_MIN
    clipped_double_baseline_max: float = CLIPPED_DOUBLE_BASELINE_MAX
    transition_len_points: tuple[int, int] = TRANSITION_LEN_POINTS
    random_seed: int = RANDOM_SEED
    halving_main_weight: float = HALVING_MAIN_WEIGHT
    doubling_main_weight: float = DOUBLING_MAIN_WEIGHT
    max_placement_attempts: int = MAX_PLACEMENT_ATTEMPTS
    baseline_strategy: str = BASELINE_STRATEGY
    baseline_window_sec: float = BASELINE_WINDOW_SEC
    feature_baseline_window_size: int = FEATURE_BASELINE_WINDOW_SIZE
    feature_baseline_window_step: int = FEATURE_BASELINE_WINDOW_STEP
    feature_baseline_smoothing_window: int = FEATURE_BASELINE_SMOOTHING_WINDOW
    feature_baseline_variability_threshold: float = FEATURE_BASELINE_VARIABILITY_THRESHOLD
    feature_baseline_min_valid_ratio: float = FEATURE_BASELINE_MIN_VALID_RATIO
    synthetic_dataset_name: str = SYNTHETIC_DATASET_NAME
    synthetic_metadata_name: str = SYNTHETIC_METADATA_NAME
    test_plot_dirname: str = TEST_PLOT_DIRNAME
    test_sample_count: int = TEST_SAMPLE_COUNT

    @property
    def max_duration_points(self) -> int:
        return int(round(self.max_duration_sec * self.fs))


def get_default_config() -> InjectionConfig:
    return InjectionConfig()


def get_config(
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> InjectionConfig:
    cfg = InjectionConfig()
    updates = {}
    if data_path is not None:
        updates["data_path"] = Path(data_path).expanduser().resolve()
    if output_dir is not None:
        updates["output_dir"] = Path(output_dir).expanduser().resolve()
    if updates:
        cfg = replace(cfg, **updates)
    return cfg
