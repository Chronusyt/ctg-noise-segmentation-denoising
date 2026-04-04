from __future__ import annotations

from typing import Tuple

import numpy as np


def resolve_parent_and_chunk_indices(
    data: np.lib.npyio.NpzFile,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve per-sample parent and chunk indices from a paired dataset.

    Priority:
    1. Use stored `parent_index` / `chunk_index` when available.
    2. Fall back to grouping by `sample_ids` or `sample_id`.
    3. Final fallback: treat each sample as its own parent.
    """
    if "parent_index" in data:
        raw_parent = np.asarray(data["parent_index"])
    elif "sample_ids" in data:
        raw_parent = np.asarray(data["sample_ids"]).astype(str)
    elif "sample_id" in data:
        raw_parent = np.asarray(data["sample_id"]).astype(str)
    else:
        raw_parent = np.arange(n_samples, dtype=np.int32)

    if raw_parent.shape[0] != n_samples:
        raise ValueError(
            f"parent metadata length mismatch: expected {n_samples}, got {raw_parent.shape[0]}"
        )

    _, parent_index = np.unique(raw_parent, return_inverse=True)
    parent_index = np.asarray(parent_index, dtype=np.int32)

    if "chunk_index" in data:
        chunk_index = np.asarray(data["chunk_index"], dtype=np.int32)
        if chunk_index.shape[0] != n_samples:
            raise ValueError(
                f"chunk metadata length mismatch: expected {n_samples}, got {chunk_index.shape[0]}"
            )
    else:
        chunk_index = np.zeros(n_samples, dtype=np.int32)

    return parent_index, chunk_index


def split_parent_groups(
    parent_index: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split unique parent groups into train/val/test."""
    unique_parents = np.unique(np.asarray(parent_index, dtype=np.int32))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique_parents)

    n_parents = len(perm)
    n_train = int(n_parents * train_ratio)
    n_val = int(n_parents * val_ratio)
    n_test = n_parents - n_train - n_val

    train_parents = perm[:n_train]
    val_parents = perm[n_train : n_train + n_val]
    test_parents = perm[n_train + n_val : n_train + n_val + n_test]
    return train_parents, val_parents, test_parents
