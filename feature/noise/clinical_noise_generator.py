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
from scipy.ndimage import median_filter

from ctg_preprocessing.fhr_baseline_optimized import BaselineConfig, analyse_baseline_optimized
from ctg_preprocessing.signal_quality import assess_signal_quality
from noise.noise_generator import NoiseGenerator

NOISE_NAMES = ["halving", "doubling", "mhr", "missing", "spike"]

_VALID_FHR_MIN = 50.0
_VALID_FHR_MAX = 255.0
_HALVING_RATIO_RANGE = (0.47, 0.55)
_DOUBLING_RATIO_RANGE = (1.90, 2.05)
_HALVING_SIGMA_RANGE = (1.5, 3.0)
_DOUBLING_SIGMA_RANGE = (2.0, 4.0)
_DOUBLE_CEILING_RANGE = (225.0, 250.0)
_UNCLIPPED_DOUBLE_BASELINE_TIERS = (105.0, 110.0, 115.0)
_CLIPPED_DOUBLE_BASELINE_MIN = 110.0
_CLIPPED_DOUBLE_BASELINE_MAX = 145.0
_TRANSITION_LEN_POINTS = (1, 3)
_MAIN_DURATION_SECONDS = (3.0, 8.0)
_TAIL_DURATION_SECONDS = (8.0, 15.0)
_MAIN_DURATION_WEIGHT = 0.80
_DOUBLE_UNCLIPPED_WEIGHT = 0.4
_DOUBLE_CLIPPED_WEIGHT = 0.6
_MAX_DH_SECONDS = 15.0
_MAX_PLACEMENT_ATTEMPTS = 64
_FEATURE_BASELINE_WINDOW_SEC = 300.0
_FEATURE_BASELINE_WINDOW_SIZE = 1920
_FEATURE_BASELINE_WINDOW_STEP = 240
_FEATURE_BASELINE_SMOOTHING_WINDOW = 240
_FEATURE_BASELINE_VARIABILITY_THRESHOLD = 25.0
_FEATURE_BASELINE_MIN_VALID_RATIO = 0.5


@dataclass
class ClinicalNoiseConfig:
    """Configuration for clinical-constrained noise generation."""

    sample_rate_hz: float = 4.0

    max_halving_segment_seconds: float = 15.0
    max_doubling_segment_seconds: float = 15.0
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
            0: min(self.max_halving_segment_seconds, _MAX_DH_SECONDS),
            1: min(self.max_doubling_segment_seconds, _MAX_DH_SECONDS),
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

    def _interpolate_or_fill(self, signal: np.ndarray, fill_value: float = 140.0) -> np.ndarray:
        out = np.asarray(signal, dtype=np.float64).copy()
        nan_mask = ~np.isfinite(out)
        if not np.any(nan_mask):
            return out

        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) == 0:
            out[:] = fill_value
            return out
        if len(valid_idx) == 1:
            out[nan_mask] = out[valid_idx[0]]
            return out

        out[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, out[valid_idx])
        return out

    def _compute_local_baseline(self, clean_signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(clean_signal, dtype=np.float64)
        zero_mask = np.zeros(len(signal), dtype=np.uint8)
        try:
            baseline_cfg = BaselineConfig(
                window_size=_FEATURE_BASELINE_WINDOW_SIZE,
                window_step=_FEATURE_BASELINE_WINDOW_STEP,
                smoothing_window=_FEATURE_BASELINE_SMOOTHING_WINDOW,
                variability_threshold=_FEATURE_BASELINE_VARIABILITY_THRESHOLD,
                min_valid_ratio=_FEATURE_BASELINE_MIN_VALID_RATIO,
            )
            baseline = analyse_baseline_optimized(
                signal,
                config=baseline_cfg,
                mask=zero_mask,
                sample_rate=self.config.sample_rate_hz,
            ).astype(np.float64)
            if len(baseline) == len(signal) and np.all(np.isfinite(baseline)):
                return baseline
        except Exception:
            pass

        fill_ready = self._interpolate_or_fill(signal)
        win = int(_FEATURE_BASELINE_WINDOW_SEC * self.config.sample_rate_hz)
        if win % 2 == 0:
            win += 1
        return median_filter(fill_ready, size=max(3, win), mode="reflect").astype(np.float64)

    def _smooth_noise(self, noise: np.ndarray) -> np.ndarray:
        if len(noise) <= 2:
            return noise
        kernel_len = min(5, len(noise))
        kernel = np.ones(kernel_len, dtype=np.float64) / kernel_len
        return np.convolve(noise, kernel, mode="same")

    def _sample_transition_len(self) -> int:
        low, high = _TRANSITION_LEN_POINTS
        return int(self._rng.randint(low, high + 1))

    def _sample_dh_duration_points(self, art_type: int) -> int:
        min_seconds = min(self.config.min_seconds_by_type()[art_type], self.config.max_seconds_by_type()[art_type])
        max_seconds = self.config.max_seconds_by_type()[art_type]
        if max_seconds <= _MAIN_DURATION_SECONDS[1]:
            low, high = min_seconds, max(min_seconds, max_seconds)
        else:
            if self._rng.rand() < _MAIN_DURATION_WEIGHT:
                low, high = min_seconds, _MAIN_DURATION_SECONDS[1]
            else:
                low, high = _TAIL_DURATION_SECONDS[0], max(_TAIL_DURATION_SECONDS[0], max_seconds)
        duration_sec = float(self._rng.uniform(low, high))
        return self._seconds_to_samples(duration_sec)

    def _apply_short_transition(
        self,
        segment_values: np.ndarray,
        original_signal: np.ndarray,
        start_idx: int,
        end_idx: int,
        transition_len: int,
    ) -> np.ndarray:
        blended = np.asarray(segment_values, dtype=np.float64).copy()
        seg_len = len(blended)
        if seg_len == 0:
            return blended

        transition_len = max(0, min(transition_len, seg_len // 2))
        if transition_len == 0:
            return blended

        left_ref = original_signal[start_idx - 1] if start_idx > 0 else blended[0]
        right_ref = original_signal[end_idx] if end_idx < len(original_signal) else blended[-1]

        for i in range(transition_len):
            alpha = (i + 1) / float(transition_len + 1)
            blended[i] = (1.0 - alpha) * left_ref + alpha * blended[i]

        for i in range(transition_len):
            alpha = (transition_len - i) / float(transition_len + 1)
            pos = seg_len - transition_len + i
            blended[pos] = alpha * blended[pos] + (1.0 - alpha) * right_ref

        return blended

    def _find_baseline_window(
        self,
        baseline: np.ndarray,
        clean_signal: np.ndarray,
        labels: np.ndarray,
        union_mask: np.ndarray,
        duration_points: int,
        transition_len: int,
        baseline_min: Optional[float] = None,
        baseline_max: Optional[float] = None,
        preferred_window: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        n = len(clean_signal)
        if duration_points >= n:
            return None

        valid_baseline = (
            np.isfinite(baseline)
            & (baseline >= 80.0)
            & (baseline <= 200.0)
            & np.isfinite(clean_signal)
            & (clean_signal >= _VALID_FHR_MIN)
            & (clean_signal <= _VALID_FHR_MAX)
        )
        padding = max(transition_len, 1)
        max_start = n - duration_points

        def is_valid_window(start: int, end: int) -> bool:
            start = max(0, min(start, n))
            end = max(start, min(end, n))
            if end <= start:
                return False
            win_start = max(0, start - padding)
            win_end = min(n, end + padding)
            if np.any(union_mask[win_start:win_end] > 0):
                return False
            if np.any(labels[win_start:win_end, :4] > 0.5):
                return False
            if not np.all(valid_baseline[start:end]):
                return False
            mean_baseline = float(np.mean(baseline[start:end]))
            if baseline_min is not None and mean_baseline < baseline_min:
                return False
            if baseline_max is not None and mean_baseline > baseline_max:
                return False
            candidate_mask = np.zeros(n, dtype=np.uint8)
            candidate_mask[start:end] = 1
            return self._can_apply_union(union_mask, candidate_mask)

        if preferred_window is not None:
            start, end = preferred_window
            if is_valid_window(start, end):
                return start, end

        for _ in range(_MAX_PLACEMENT_ATTEMPTS):
            start = int(self._rng.randint(0, max_start + 1))
            end = start + duration_points
            if is_valid_window(start, end):
                return start, end
        return None

    def _commit_main_segment(
        self,
        corrupted: np.ndarray,
        labels: np.ndarray,
        union_mask: np.ndarray,
        start: int,
        end: int,
        label_idx: int,
        segment_values: np.ndarray,
    ) -> bool:
        if start >= end or len(segment_values) != (end - start):
            return False
        if not self._main_region_is_free(labels, start, end):
            return False
        candidate_mask = np.zeros(len(corrupted), dtype=np.uint8)
        candidate_mask[start:end] = 1
        if not self._can_apply_union(union_mask, candidate_mask):
            return False

        corrupted[start:end] = segment_values
        labels[start:end, label_idx] = 1.0
        union_mask[:] = np.maximum(union_mask, candidate_mask)
        return True

    def _apply_baseline_driven_halving(
        self,
        corrupted: np.ndarray,
        original: np.ndarray,
        baseline: np.ndarray,
        labels: np.ndarray,
        union_mask: np.ndarray,
        preferred_window: Optional[Tuple[int, int]] = None,
        allow_random_fallback: bool = True,
    ) -> bool:
        transition_len = self._sample_transition_len()
        if preferred_window is not None:
            preferred_start, preferred_end = preferred_window
            duration_points = max(1, preferred_end - preferred_start)
        else:
            duration_points = self._sample_dh_duration_points(0)

        window = self._find_baseline_window(
            baseline,
            original,
            labels,
            union_mask,
            duration_points,
            transition_len,
            preferred_window=preferred_window,
        )
        if window is None and preferred_window is not None and allow_random_fallback:
            duration_points = self._sample_dh_duration_points(0)
            window = self._find_baseline_window(
                baseline,
                original,
                labels,
                union_mask,
                duration_points,
                transition_len,
            )
        if window is None:
            return False

        start, end = window
        baseline_segment = baseline[start:end]
        ratio_used = float(self._rng.uniform(*_HALVING_RATIO_RANGE))
        sigma_used = float(self._rng.uniform(*_HALVING_SIGMA_RANGE))
        target = baseline_segment * ratio_used
        noise = self._smooth_noise(self._rng.normal(0.0, sigma_used, size=len(target)))
        segment_values = target + noise
        segment_values = self._apply_short_transition(segment_values, original, start, end, transition_len)
        segment_values = np.clip(segment_values, _VALID_FHR_MIN, _VALID_FHR_MAX)
        return self._commit_main_segment(corrupted, labels, union_mask, start, end, 0, segment_values)

    def _apply_unclipped_doubling(
        self,
        corrupted: np.ndarray,
        original: np.ndarray,
        baseline: np.ndarray,
        labels: np.ndarray,
        union_mask: np.ndarray,
    ) -> bool:
        for baseline_max in _UNCLIPPED_DOUBLE_BASELINE_TIERS:
            duration_points = self._sample_dh_duration_points(1)
            transition_len = self._sample_transition_len()
            window = self._find_baseline_window(
                baseline,
                original,
                labels,
                union_mask,
                duration_points,
                transition_len,
                baseline_max=baseline_max,
            )
            if window is None:
                continue

            start, end = window
            baseline_segment = baseline[start:end]
            local_baseline = float(np.mean(baseline_segment))
            max_ratio_allowed = _VALID_FHR_MAX / max(local_baseline, 1.0)
            if max_ratio_allowed < _DOUBLING_RATIO_RANGE[0]:
                continue

            ratio_high = min(_DOUBLING_RATIO_RANGE[1], max_ratio_allowed)
            if ratio_high < _DOUBLING_RATIO_RANGE[0]:
                continue

            ratio_used = float(self._rng.uniform(_DOUBLING_RATIO_RANGE[0], ratio_high))
            sigma_used = float(self._rng.uniform(*_DOUBLING_SIGMA_RANGE))
            target = baseline_segment * ratio_used
            noise = self._smooth_noise(self._rng.normal(0.0, sigma_used, size=len(target)))
            segment_values = target + noise
            segment_values = self._apply_short_transition(segment_values, original, start, end, transition_len)
            segment_values = np.clip(segment_values, _VALID_FHR_MIN, _VALID_FHR_MAX)
            if self._commit_main_segment(corrupted, labels, union_mask, start, end, 1, segment_values):
                return True
        return False

    def _apply_clipped_doubling(
        self,
        corrupted: np.ndarray,
        original: np.ndarray,
        baseline: np.ndarray,
        labels: np.ndarray,
        union_mask: np.ndarray,
    ) -> bool:
        for _ in range(4):
            duration_points = self._sample_dh_duration_points(1)
            transition_len = self._sample_transition_len()
            window = self._find_baseline_window(
                baseline,
                original,
                labels,
                union_mask,
                duration_points,
                transition_len,
                baseline_min=_CLIPPED_DOUBLE_BASELINE_MIN,
                baseline_max=_CLIPPED_DOUBLE_BASELINE_MAX,
            )
            if window is None:
                continue

            start, end = window
            baseline_segment = baseline[start:end]
            ratio_used = float(self._rng.uniform(*_DOUBLING_RATIO_RANGE))
            sigma_used = float(self._rng.uniform(*_DOUBLING_SIGMA_RANGE))
            candidate_target = baseline_segment * ratio_used
            target_peak = float(np.percentile(candidate_target, 80))
            if target_peak < _DOUBLE_CEILING_RANGE[0]:
                continue

            ceiling_upper = min(_DOUBLE_CEILING_RANGE[1], max(_DOUBLE_CEILING_RANGE[0], target_peak))
            ceiling_used = float(self._rng.uniform(_DOUBLE_CEILING_RANGE[0], ceiling_upper))

            noise = self._smooth_noise(self._rng.normal(0.0, sigma_used, size=len(candidate_target)))
            segment_values = candidate_target + noise
            over_ceiling = segment_values >= ceiling_used
            if np.any(over_ceiling):
                ceiling_band = float(
                    self._rng.uniform(max(1.5, sigma_used * 0.7), max(3.5, sigma_used * 1.6))
                )
                downward_jitter = np.abs(
                    self._smooth_noise(
                        self._rng.normal(
                            loc=ceiling_band * 0.45,
                            scale=max(0.6, ceiling_band * 0.35),
                            size=len(candidate_target),
                        )
                    )
                )
                segment_values = segment_values.copy()
                segment_values[over_ceiling] = ceiling_used - np.clip(
                    downward_jitter[over_ceiling],
                    0.0,
                    ceiling_band,
                )

                if np.mean(over_ceiling) > 0.4:
                    exact_hit_prob = float(self._rng.uniform(0.12, 0.30))
                    exact_hits = over_ceiling & (self._rng.rand(len(candidate_target)) < exact_hit_prob)
                    segment_values[exact_hits] = ceiling_used

            segment_values = self._apply_short_transition(segment_values, original, start, end, transition_len)
            segment_values = np.clip(segment_values, _VALID_FHR_MIN, _VALID_FHR_MAX)
            if self._commit_main_segment(corrupted, labels, union_mask, start, end, 1, segment_values):
                return True
        return False

    def _apply_baseline_driven_doubling(
        self,
        corrupted: np.ndarray,
        original: np.ndarray,
        baseline: np.ndarray,
        labels: np.ndarray,
        union_mask: np.ndarray,
    ) -> bool:
        names = np.array(["unclipped", "clipped"], dtype=object)
        weights = np.array([_DOUBLE_UNCLIPPED_WEIGHT, _DOUBLE_CLIPPED_WEIGHT], dtype=np.float64)
        weights = weights / np.sum(weights)
        preferred = str(names[int(self._rng.choice(np.arange(len(names)), p=weights))])
        ordered = [preferred] + [name for name in names.tolist() if name != preferred]

        for subtype in ordered:
            if subtype == "unclipped":
                if self._apply_unclipped_doubling(corrupted, original, baseline, labels, union_mask):
                    return True
            else:
                if self._apply_clipped_doubling(corrupted, original, baseline, labels, union_mask):
                    return True
        return False

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

        if art_type == 2:
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

    def _generate_candidate(self, signal: np.ndarray, baseline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        clean = np.asarray(signal, dtype=np.float64)
        corrupted = clean.copy()
        labels = np.zeros((len(clean), 5), dtype=np.float32)
        union_mask = np.zeros(len(clean), dtype=np.uint8)

        num_artifacts = self._rng.randint(
            self.config.num_artifacts_min,
            self.config.num_artifacts_max + 1,
        )
        artifact_types = self._weighted_choice_without_replacement(num_artifacts)
        ordered_main_types = [art_type for art_type in artifact_types if art_type != 4]
        ordered_spike_types = [art_type for art_type in artifact_types if art_type == 4]

        for art_type in ordered_main_types:
            if art_type == 0:
                self._apply_baseline_driven_halving(corrupted, clean, baseline, labels, union_mask)
                continue
            if art_type == 1:
                self._apply_baseline_driven_doubling(corrupted, clean, baseline, labels, union_mask)
                continue
            for _ in range(20):
                start, end = self._sample_segment(art_type, len(clean))
                if self._apply_one_artifact(corrupted, clean, labels, union_mask, art_type, start, end):
                    break

        for art_type in ordered_spike_types:
            for _ in range(20):
                start, end = self._sample_segment(art_type, len(clean))
                if self._apply_one_artifact(corrupted, clean, labels, union_mask, art_type, start, end):
                    break

        return corrupted, labels

    def _generate_safe_fallback(self, signal: np.ndarray, baseline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deterministic low-risk fallback used only after repeated failed retries.
        """
        clean = np.asarray(signal, dtype=np.float64)
        corrupted = clean.copy()
        labels = np.zeros((len(clean), 5), dtype=np.float32)
        union_mask = np.zeros(len(clean), dtype=np.uint8)

        halving_len = min(
            self._seconds_to_samples(self.config.min_halving_segment_seconds),
            self._seconds_to_samples(self.config.max_seconds_by_type()[0]),
            max(1, len(clean) // 20),
        )
        halving_start = max(0, len(clean) // 3)
        halving_end = min(len(clean), halving_start + halving_len)
        if self._apply_baseline_driven_halving(
            corrupted,
            clean,
            baseline,
            labels,
            union_mask,
            preferred_window=(halving_start, halving_end),
            allow_random_fallback=False,
        ):
            reliability = self._compute_reliability(corrupted)
            if reliability >= self.config.min_post_noise_reliability:
                return corrupted, labels

        spike_len = min(
            self._seconds_to_samples(self.config.min_spike_segment_seconds),
            self._seconds_to_samples(self.config.max_seconds_by_type()[4]),
            max(1, len(clean) // 24),
        )
        spike_start = max(0, len(clean) // 3)
        spike_end = min(len(clean), spike_start + spike_len)
        if self._apply_one_artifact(corrupted, clean, labels, union_mask, 4, spike_start, spike_end):
            reliability = self._compute_reliability(corrupted)
            if reliability >= self.config.min_post_noise_reliability:
                return corrupted, labels
        raise RuntimeError("clinical noise fallback failed to satisfy post-noise reliability")

    def generate_artifacts(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate one clinically constrained noisy sample from one clean signal.
        """
        clean = np.asarray(signal, dtype=np.float64)
        baseline = self._compute_local_baseline(clean)
        best_candidate: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        for _ in range(self.config.max_retries):
            corrupted, labels = self._generate_candidate(clean, baseline)
            if not np.any(labels > 0.5):
                continue
            reliability = self._compute_reliability(corrupted)
            if best_candidate is None or reliability > best_candidate[2]:
                best_candidate = (corrupted.copy(), labels.copy(), reliability)
            if reliability >= self.config.min_post_noise_reliability:
                return corrupted, labels

        if best_candidate is not None and best_candidate[2] >= self.config.min_post_noise_reliability:
            return best_candidate[0], best_candidate[1]
        return self._generate_safe_fallback(clean, baseline)

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
