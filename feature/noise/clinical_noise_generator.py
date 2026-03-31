"""
Clinical-constrained synthetic noise generator for CTG signals.

Design goals:
- Keep easy/hard generation logic untouched.
- Generate noisy samples strictly from existing clean signals.
- Constrain artifact duration and total corruption ratio so samples remain
  "dirty but still clinically repairable".
- Reject and resample noisy variants whose post-noise reliability is too low.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ctg_preprocessing.signal_quality import assess_signal_quality
from noise.noise_generator import NoiseGenerator

NOISE_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]


@dataclass
class ClinicalNoiseConfig:
    """Configuration for clinical-constrained noise generation."""

    sample_rate_hz: float = 4.0

    max_halving_segment_seconds: float = 18.0
    max_doubling_segment_seconds: float = 18.0
    max_mhr_segment_seconds: float = 10.0
    max_missing_segment_seconds: float = 8.0
    max_spike_segment_seconds: float = 2.0

    min_halving_segment_seconds: float = 3.0
    min_doubling_segment_seconds: float = 3.0
    min_mhr_segment_seconds: float = 4.0
    min_missing_segment_seconds: float = 2.0
    min_spike_segment_seconds: float = 1.0

    max_total_noise_ratio_clinical: float = 0.12
    min_post_noise_reliability: float = 85.0
    max_retries: int = 80

    num_artifacts_min: int = 1
    num_artifacts_max: int = 3

    halving_prob: float = 0.35
    doubling_prob: float = 0.35
    mhr_prob: float = 0.25
    missing_prob: float = 0.25
    spike_prob: float = 0.45

    missing_fill_mode: str = "zero"
    use_paper_distribution: bool = True
    random_state: Optional[int] = None

    def max_seconds_by_type(self) -> Dict[int, float]:
        return {
            0: self.max_halving_segment_seconds,
            1: self.max_doubling_segment_seconds,
            2: self.max_mhr_segment_seconds,
            3: self.max_missing_segment_seconds,
            4: self.max_spike_segment_seconds,
        }

    def min_seconds_by_type(self) -> Dict[int, float]:
        return {
            0: self.min_halving_segment_seconds,
            1: self.min_doubling_segment_seconds,
            2: self.min_mhr_segment_seconds,
            3: self.min_missing_segment_seconds,
            4: self.min_spike_segment_seconds,
        }


class ClinicalNoiseGenerator:
    """
    Generate clinically repairable synthetic noise from clean CTG signals.

    The generated labels always follow the fixed channel order:
    [halving, doubling, mhr, missing, spike]
    """

    def __init__(self, config: Optional[ClinicalNoiseConfig] = None):
        self.config = config if config is not None else ClinicalNoiseConfig()
        self._rng = np.random.RandomState(self.config.random_state)
        self._backend = NoiseGenerator(
            use_paper_distribution=self.config.use_paper_distribution,
            ensure_at_least_one=False,
            mode="easy",
            missing_fill_mode=self.config.missing_fill_mode,
            random_state=self.config.random_state,
        )
        self._weights = np.array(
            [
                self.config.halving_prob,
                self.config.doubling_prob,
                self.config.mhr_prob,
                self.config.missing_prob,
                self.config.spike_prob,
            ],
            dtype=np.float64,
        )
        if np.all(self._weights <= 0):
            self._weights[:] = 1.0

    def _seconds_to_samples(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.config.sample_rate_hz)))

    def _segment_bounds(self, art_type: int, signal_length: int) -> Tuple[int, int]:
        min_len = min(self._seconds_to_samples(self.config.min_seconds_by_type()[art_type]), signal_length)
        max_len = min(self._seconds_to_samples(self.config.max_seconds_by_type()[art_type]), signal_length)
        if max_len < min_len:
            max_len = min_len
        return min_len, max_len

    def _sample_segment(self, art_type: int, signal_length: int) -> Tuple[int, int]:
        min_len, max_len = self._segment_bounds(art_type, signal_length)
        seg_len = self._rng.randint(min_len, max_len + 1)
        start = self._rng.randint(0, max(1, signal_length - seg_len + 1))
        end = min(signal_length, start + seg_len)
        return start, end

    def _weighted_choice_without_replacement(self, count: int) -> List[int]:
        choices = list(range(5))
        weights = self._weights.astype(np.float64).copy()
        selected: List[int] = []
        for _ in range(min(count, len(choices))):
            total = float(np.sum(weights))
            if total <= 0:
                idx = self._rng.randint(0, len(choices))
            else:
                probs = weights / total
                idx = int(self._rng.choice(np.arange(len(choices)), p=probs))
            selected.append(choices.pop(idx))
            weights = np.delete(weights, idx)
        return selected

    def _can_apply_union(self, union_mask: np.ndarray, candidate_mask: np.ndarray) -> bool:
        new_mask = np.maximum(union_mask, candidate_mask.astype(np.uint8))
        return float(np.mean(new_mask)) <= self.config.max_total_noise_ratio_clinical

    def _main_region_is_free(self, labels: np.ndarray, start: int, end: int) -> bool:
        return not np.any(labels[start:end, :4] > 0.5)

    def _spike_region_is_free_of_missing(self, labels: np.ndarray, start: int, end: int) -> bool:
        return not np.any(labels[start:end, 3] > 0.5)

    def _apply_one_artifact(
        self,
        corrupted: np.ndarray,
        original: np.ndarray,
        labels: np.ndarray,
        union_mask: np.ndarray,
        art_type: int,
        start: int,
        end: int,
    ) -> bool:
        candidate_mask = np.zeros(len(corrupted), dtype=np.uint8)
        trial = corrupted.copy()
        trial_labels = labels.copy()

        if art_type == 0:
            if not self._main_region_is_free(trial_labels, start, end):
                return False
            trial = self._backend.inject_halving(trial, start, end, base_signal=original)
            trial_labels[start:end, 0] = 1.0
            candidate_mask[start:end] = 1
        elif art_type == 1:
            if not self._main_region_is_free(trial_labels, start, end):
                return False
            trial = self._backend.inject_doubling(trial, start, end, base_signal=original)
            trial_labels[start:end, 1] = 1.0
            candidate_mask[start:end] = 1
        elif art_type == 2:
            if not self._main_region_is_free(trial_labels, start, end):
                return False
            trial = self._backend.inject_mhr(trial, start, end)
            trial_labels[start:end, 2] = 1.0
            candidate_mask[start:end] = 1
        elif art_type == 3:
            if not self._main_region_is_free(trial_labels, start, end):
                return False
            trial = self._backend.inject_missing(trial, start, end)
            trial_labels[start:end, 3] = 1.0
            candidate_mask[start:end] = 1
        else:
            if not self._spike_region_is_free_of_missing(trial_labels, start, end):
                return False
            trial, spike_labels = self._backend._inject_sparse_spikes(trial, start, end)
            if not np.any(spike_labels > 0.5):
                return False
            trial_labels[:, 4] = np.maximum(trial_labels[:, 4], spike_labels)
            candidate_mask = np.maximum(candidate_mask, (spike_labels > 0.5).astype(np.uint8))

        if candidate_mask.mean() <= 0:
            return False
        if not self._can_apply_union(union_mask, candidate_mask):
            return False

        corrupted[:] = trial
        labels[:] = trial_labels
        union_mask[:] = np.maximum(union_mask, candidate_mask)
        return True

    def _compute_reliability(self, signal: np.ndarray) -> float:
        _, stats = assess_signal_quality(signal.astype(np.float64), sample_rate=self.config.sample_rate_hz)
        return float(stats["reliability_percent"])

    def _generate_candidate(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        clean = np.asarray(signal, dtype=np.float64)
        corrupted = clean.copy()
        labels = np.zeros((len(clean), 5), dtype=np.float32)
        union_mask = np.zeros(len(clean), dtype=np.uint8)

        num_artifacts = self._rng.randint(
            self.config.num_artifacts_min,
            self.config.num_artifacts_max + 1,
        )
        artifact_types = self._weighted_choice_without_replacement(num_artifacts)

        for art_type in artifact_types:
            for _ in range(20):
                start, end = self._sample_segment(art_type, len(clean))
                if self._apply_one_artifact(corrupted, clean, labels, union_mask, art_type, start, end):
                    break

        return corrupted, labels

    def _generate_safe_fallback(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministic low-risk fallback used only after repeated failed retries.
        """
        clean = np.asarray(signal, dtype=np.float64)
        corrupted = clean.copy()
        labels = np.zeros((len(clean), 5), dtype=np.float32)
        union_mask = np.zeros(len(clean), dtype=np.uint8)

        fallback_types = [0, 4]
        for art_type in fallback_types:
            min_len, max_len = self._segment_bounds(art_type, len(clean))
            seg_len = min(min_len, max_len, max(1, len(clean) // 20))
            start = max(0, len(clean) // 3)
            end = min(len(clean), start + seg_len)
            if self._apply_one_artifact(corrupted, clean, labels, union_mask, art_type, start, end):
                reliability = self._compute_reliability(corrupted)
                if reliability >= self.config.min_post_noise_reliability:
                    return corrupted, labels
        raise RuntimeError("clinical noise fallback failed to satisfy post-noise reliability")

    def generate_artifacts(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one clinically constrained noisy sample from one clean signal.
        """
        best_candidate: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        for _ in range(self.config.max_retries):
            corrupted, labels = self._generate_candidate(signal)
            if not np.any(labels > 0.5):
                continue
            reliability = self._compute_reliability(corrupted)
            if best_candidate is None or reliability > best_candidate[2]:
                best_candidate = (corrupted.copy(), labels.copy(), reliability)
            if reliability >= self.config.min_post_noise_reliability:
                return corrupted, labels

        if best_candidate is not None and best_candidate[2] >= self.config.min_post_noise_reliability:
            return best_candidate[0], best_candidate[1]
        return self._generate_safe_fallback(signal)

    def generate_batch(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch version of generate_artifacts.
        """
        arr = np.asarray(signals, dtype=np.float64)
        if arr.ndim == 1:
            corrupted, labels = self.generate_artifacts(arr)
            return corrupted, labels

        noisy_batch = []
        labels_batch = []
        for i in range(arr.shape[0]):
            noisy, labels = self.generate_artifacts(arr[i])
            noisy_batch.append(noisy)
            labels_batch.append(labels)
        return np.asarray(noisy_batch, dtype=np.float64), np.asarray(labels_batch, dtype=np.float32)
