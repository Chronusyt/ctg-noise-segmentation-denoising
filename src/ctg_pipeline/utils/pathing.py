"""Repository path helpers shared by experiment entrypoints."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts"
DOCS_ROOT = REPO_ROOT / "docs"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
RUNS_ROOT = ARTIFACTS_ROOT / "runs"

DENOISING_DATASETS_ROOT = ARTIFACTS_ROOT / "datasets" / "denoising"
DENOISING_RESULTS_ROOT = ARTIFACTS_ROOT / "results" / "denoising"
REAL_WORLD_RESULTS_ROOT = ARTIFACTS_ROOT / "results" / "real_world_inference"
DOUBLING_HALVING_DATASETS_ROOT = ARTIFACTS_ROOT / "datasets" / "doubling_halving"
DOUBLING_HALVING_RESULTS_ROOT = ARTIFACTS_ROOT / "results" / "doubling_halving"


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a CLI path relative to the repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate

