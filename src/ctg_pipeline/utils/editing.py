"""Helpers for constrained residual editing and region diagnostics."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _normalize_kernel_size(kernel_size: int) -> int:
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def union_mask_from_multilabel_torch(multilabel_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert a 5-class multilabel mask to a single-channel union mask.

    Expected input shape: [B, 5, L]
    Output shape: [B, 1, L]
    """
    if multilabel_mask.ndim != 3:
        raise ValueError(f"Expected mask shape [B, 5, L], got {tuple(multilabel_mask.shape)}")
    return multilabel_mask.amax(dim=1, keepdim=True).clamp(0.0, 1.0)


def dilate_mask_torch(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(radius)
    if radius <= 0:
        return mask.clamp(0.0, 1.0)
    kernel = radius * 2 + 1
    return F.max_pool1d(mask, kernel_size=kernel, stride=1, padding=radius).clamp(0.0, 1.0)


def smooth_mask_torch(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = _normalize_kernel_size(kernel_size)
    if kernel_size <= 1:
        return mask.clamp(0.0, 1.0)
    return F.avg_pool1d(mask, kernel_size=kernel_size, stride=1, padding=kernel_size // 2).clamp(0.0, 1.0)


def build_edit_gate_torch(
    multilabel_mask: torch.Tensor,
    gate_mode: str,
    clean_gate_value: float = 0.1,
    dilation_radius: int = 5,
    smooth_kernel_size: int = 5,
) -> torch.Tensor:
    """
    Build a single-channel edit gate from multilabel masks.

    Gate semantics:
    - corrupted region: gate close to 1.0
    - clean region: gate close to clean_gate_value (not hard zero)
    """
    clean_gate_value = float(clean_gate_value)
    clean_gate_value = max(0.0, min(clean_gate_value, 1.0))
    union = union_mask_from_multilabel_torch(multilabel_mask)

    if gate_mode == "none":
        return torch.ones_like(union)

    if gate_mode == "union_soft":
        gate_source = smooth_mask_torch(union, smooth_kernel_size)
    elif gate_mode == "union_dilated_soft":
        gate_source = dilate_mask_torch(union, dilation_radius)
        gate_source = smooth_mask_torch(gate_source, smooth_kernel_size)
    else:
        raise ValueError(f"Unsupported gate_mode: {gate_mode}")

    return (clean_gate_value + (1.0 - clean_gate_value) * gate_source).clamp(0.0, 1.0)


def compute_region_masks_torch(multilabel_mask: torch.Tensor, boundary_k: int = 5) -> Dict[str, torch.Tensor]:
    """
    Split the timeline into:
    - corrupted
    - clean
    - boundary_near_clean
    - far_clean
    """
    union = union_mask_from_multilabel_torch(multilabel_mask)
    clean = (1.0 - union).clamp(0.0, 1.0)
    dilated = dilate_mask_torch(union, boundary_k)
    boundary_near_clean = (clean * dilated).clamp(0.0, 1.0)
    far_clean = (clean * (1.0 - dilated)).clamp(0.0, 1.0)
    return {
        "corrupted": union,
        "clean": clean,
        "boundary_near_clean": boundary_near_clean,
        "far_clean": far_clean,
    }
