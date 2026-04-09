"""Utility helpers for CTG experiment orchestration."""

from .dataset_split import resolve_parent_and_chunk_indices, split_parent_groups
from .editing import build_edit_gate_torch, compute_region_masks_torch, union_mask_from_multilabel_torch
from .pathing import (
    ARTIFACTS_ROOT,
    DENOISING_DATASETS_ROOT,
    DENOISING_RESULTS_ROOT,
    DOUBLING_HALVING_DATASETS_ROOT,
    DOUBLING_HALVING_RESULTS_ROOT,
    REAL_WORLD_RESULTS_ROOT,
    REPO_ROOT,
    RUNS_ROOT,
    SRC_ROOT,
    resolve_repo_path,
)

__all__ = [
    "ARTIFACTS_ROOT",
    "DENOISING_DATASETS_ROOT",
    "DENOISING_RESULTS_ROOT",
    "DOUBLING_HALVING_DATASETS_ROOT",
    "DOUBLING_HALVING_RESULTS_ROOT",
    "REAL_WORLD_RESULTS_ROOT",
    "REPO_ROOT",
    "RUNS_ROOT",
    "SRC_ROOT",
    "build_edit_gate_torch",
    "compute_region_masks_torch",
    "resolve_parent_and_chunk_indices",
    "resolve_repo_path",
    "split_parent_groups",
    "union_mask_from_multilabel_torch",
]
