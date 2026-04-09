"""Dataset loader for clinical physiological multitask data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_FIELDS = (
    "noisy_signals",
    "clean_signals",
    "masks",
    "baseline_labels",
    "stv_labels",
    "ltv_labels",
    "baseline_variability_labels",
    "acc_labels",
    "dec_labels",
)

PRED_MASK_VARIANTS = ("soft", "hard")


def _load_npz_dict(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {name: np.asarray(data[name]) for name in data.files}


def _validate_pred_mask_cache(
    arrays: Dict[str, np.ndarray],
    pred_arrays: Dict[str, np.ndarray],
    pred_mask_variant: str,
    pred_mask_cache_path: str,
) -> np.ndarray:
    if pred_mask_variant not in PRED_MASK_VARIANTS:
        raise ValueError(f"Unsupported pred_mask_variant: {pred_mask_variant}")

    pred_key = "pred_masks_soft" if pred_mask_variant == "soft" else "pred_masks_hard"
    if pred_key not in pred_arrays:
        raise KeyError(f"Missing {pred_key} in pred-mask cache: {pred_mask_cache_path}")

    pred_masks = np.asarray(pred_arrays[pred_key])
    n = arrays["noisy_signals"].shape[0]
    expected_shape = (n, arrays["noisy_signals"].shape[1], arrays["masks"].shape[2])
    if pred_masks.shape != expected_shape:
        raise ValueError(
            f"Pred-mask cache shape mismatch: {pred_masks.shape} != {expected_shape} "
            f"({pred_mask_cache_path})"
        )

    for key in ("parent_index", "chunk_index"):
        if key in arrays and key in pred_arrays and not np.array_equal(arrays[key], pred_arrays[key]):
            raise ValueError(f"Pred-mask cache alignment mismatch for {key}: {pred_mask_cache_path}")
    return pred_masks


def load_multitask_arrays(
    path: str,
    pred_mask_cache_path: str | None = None,
    pred_mask_variant: str = "soft",
) -> Dict[str, np.ndarray]:
    """Load a multitask `.npz` file into memory and validate required fields."""
    arrays = _load_npz_dict(path)
    missing = [name for name in REQUIRED_FIELDS if name not in arrays]
    if missing:
        raise KeyError(f"Missing fields in {path}: {missing}")

    if pred_mask_cache_path:
        pred_arrays = _load_npz_dict(pred_mask_cache_path)
        arrays["pred_masks"] = _validate_pred_mask_cache(arrays, pred_arrays, pred_mask_variant, pred_mask_cache_path)

    return arrays


@dataclass
class ClinicalMultitaskDataset(Dataset):
    """
    Torch Dataset for clinical multitask reconstruction data.

    Returned tensors:
    - noisy_signal: float32 [1, L] by default, or [L] if add_channel_dim=False
    - clean_signal: float32 [1, L] by default, or [L] if add_channel_dim=False
    - mask / multilabel_mask: float32 [5, L]
    - pred_mask: float32 [5, L] when present in the `.npz`
    - baseline_label / stv_label / ltv_label: float32 scalar tensors
    - baseline_variability_label: float32 scalar tensor
    - baseline_variability_class_label: int64 scalar tensor when present
    - acc_label / dec_label: float32 [1, L] by default, or [L] if add_channel_dim=False
    """

    path: str
    add_channel_dim: bool = True
    pred_mask_cache_path: str | None = None
    pred_mask_variant: str = "soft"

    def __post_init__(self) -> None:
        arrays = load_multitask_arrays(
            self.path,
            pred_mask_cache_path=self.pred_mask_cache_path,
            pred_mask_variant=self.pred_mask_variant,
        )
        self.noisy = np.asarray(arrays["noisy_signals"], dtype=np.float32)
        self.clean = np.asarray(arrays["clean_signals"], dtype=np.float32)
        self.masks = np.asarray(arrays["masks"], dtype=np.float32)
        self.pred_masks = np.asarray(arrays["pred_masks"]) if "pred_masks" in arrays else None
        self.baseline = np.asarray(arrays["baseline_labels"], dtype=np.float32)
        self.stv = np.asarray(arrays["stv_labels"], dtype=np.float32)
        self.ltv = np.asarray(arrays["ltv_labels"], dtype=np.float32)
        self.baseline_variability = np.asarray(arrays["baseline_variability_labels"], dtype=np.float32)
        self.baseline_variability_class = (
            np.asarray(arrays["baseline_variability_class_labels"], dtype=np.int64)
            if "baseline_variability_class_labels" in arrays
            else None
        )
        self.acc_labels = np.asarray(arrays["acc_labels"], dtype=np.float32)
        self.dec_labels = np.asarray(arrays["dec_labels"], dtype=np.float32)

        n = self.noisy.shape[0]
        for name, arr in (
            ("clean_signals", self.clean),
            ("masks", self.masks),
            ("baseline_labels", self.baseline),
            ("stv_labels", self.stv),
            ("ltv_labels", self.ltv),
            ("baseline_variability_labels", self.baseline_variability),
            ("acc_labels", self.acc_labels),
            ("dec_labels", self.dec_labels),
        ):
            if arr.shape[0] != n:
                raise ValueError(f"{name} first dimension {arr.shape[0]} != noisy_signals {n}")
        if self.acc_labels.shape[1] != self.noisy.shape[1] or self.dec_labels.shape[1] != self.noisy.shape[1]:
            raise ValueError("acc_labels/dec_labels length must match signal length")

    def __len__(self) -> int:
        return int(self.noisy.shape[0])

    def _signal_tensor(self, arr: np.ndarray) -> torch.Tensor:
        out = torch.from_numpy(arr.astype(np.float32, copy=False))
        return out.unsqueeze(0) if self.add_channel_dim else out

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        mask = torch.from_numpy(np.transpose(self.masks[index], (1, 0)).astype(np.float32, copy=False))
        item = {
            "noisy_signal": self._signal_tensor(self.noisy[index]),
            "clean_signal": self._signal_tensor(self.clean[index]),
            "mask": mask,
            "multilabel_mask": mask,
            "baseline_label": torch.tensor(self.baseline[index], dtype=torch.float32),
            "stv_label": torch.tensor(self.stv[index], dtype=torch.float32),
            "ltv_label": torch.tensor(self.ltv[index], dtype=torch.float32),
            "baseline_variability_label": torch.tensor(self.baseline_variability[index], dtype=torch.float32),
            "acc_label": self._signal_tensor(self.acc_labels[index]),
            "dec_label": self._signal_tensor(self.dec_labels[index]),
        }
        if self.baseline_variability_class is not None:
            item["baseline_variability_class_label"] = torch.tensor(
                self.baseline_variability_class[index],
                dtype=torch.long,
            )
        if self.pred_masks is not None:
            item["pred_mask"] = torch.from_numpy(
                np.transpose(self.pred_masks[index], (1, 0)).astype(np.float32, copy=False)
            )
        return item
