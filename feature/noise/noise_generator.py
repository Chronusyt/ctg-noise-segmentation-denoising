"""
合成噪声生成模块 - CleanCTG 论文一致性实现

实现 5 种 CTG 伪影类型的注入：halving, doubling, mhr, missing, spike。
支持多标签、复合模式、可复现随机、污染比例约束。

[本文件为 CTG_test 自包含副本，源自 CTG/ctg_cleanCTG/utils/noise_generator.py]
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Literal, Union, Dict, Any, List

# 0 halving, 1 doubling, 2 mhr, 3 missing, 4 spike
ARTIFACT_MAIN = (0, 1, 2)  # 主伪影，支持 compound missing
ARTIFACT_ALL = (0, 1, 2, 3, 4)


class NoiseGenerator:
    """
    噪声生成器 - 按 CleanCTG 论文实现 5 种 CTG 伪影类型注入。

    论文约束：
    - 总污染比例 <= max_corruption_ratio (默认 50%)
    - 单段最大长度 <= max_segment_ratio (默认 5%)
    - 允许输出完全 clean 的样本（不强制至少一种噪声）
    - 支持复合模式：主伪影前后可自动插入 missing 段

    mode="hard" 时：每条样本至少 2 类噪声、目标覆盖率 8%-20%、连续段主导、spike 降权；
    主噪声（halving/doubling/mhr/missing）互斥，spike 可与 halving/doubling/mhr 叠加，不可与 missing 叠加。
    """

    def __init__(
        self,
        # 概率默认设为 None，由 use_paper_distribution 决定采用哪套默认值
        halving_prob: Optional[float] = None,
        doubling_prob: Optional[float] = None,
        mhr_prob: Optional[float] = None,
        missing_prob: Optional[float] = None,
        spike_prob: Optional[float] = None,
        use_paper_distribution: bool = True,
        max_corruption_ratio: float = 0.5,
        max_segment_ratio: float = 0.05,
        sample_rate_hz: float = 4.0,
        compound_prob: float = 0.3,
        missing_fill_mode: Literal["nan", "zero"] = "nan",
        spike_density: float = 0.1,
        mhr_noise_std: float = 2.0,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        ensure_at_least_one: bool = False,
        # hard mode 参数（mode="hard" 时生效）
        mode: Literal["easy", "hard"] = "easy",
        num_artifacts_min: int = 2,
        num_artifacts_max: int = 4,
        target_coverage_min: float = 0.08,
        target_coverage_max: float = 0.20,
        max_segment_ratio_hard: float = 0.12,
        compound_prob_hard: float = 0.7,
        spike_weight_hard: float = 0.25,
        spike_burst_len_min: int = 2,
        spike_burst_len_max: int = 5,
        max_corruption_ratio_hard: float = 0.35,
    ):
        """
        Args:
            halving_prob: Halving 伪影采样概率。None 时由 use_paper_distribution 决定默认值。
            doubling_prob: Doubling 伪影采样概率。None 时由 use_paper_distribution 决定默认值。
            mhr_prob: MHR 伪影采样概率。None 时由 use_paper_distribution 决定默认值。
            missing_prob: Missing 伪影采样概率。None 时由 use_paper_distribution 决定默认值。
            spike_prob: Spike 伪影采样概率。None 时由 use_paper_distribution 决定默认值。
            use_paper_distribution: 若 True，默认概率参考 CleanCTG 论文 Table 1 (synthetic CTG) 的噪声比例；
                若 False，None 概率会退回到工程旧默认（0.05/0.05/0.10/0.10/0.10）。
                注意：论文 Table 1 的比例是 synthetic CTG 的统计结果（样本级包含比例/覆盖统计），不是严格的"注入概率"。
                这里作为工程默认的采样概率，用于逼近论文分布；小样本下不保证精确匹配。
            max_corruption_ratio: 最大污染比例，所有噪声污染点总数不超过此比例（默认 0.5）
            max_segment_ratio: 单个伪影段最大长度占整段比例（默认 0.05）
            sample_rate_hz: 采样率 Hz（默认 4.0）
            compound_prob: 主伪影（halving/doubling/mhr）前后插入 missing 的概率（默认 0.3）
            missing_fill_mode: Missing 填充模式，"nan" 或 "zero"（论文表述为 null values，默认 "nan"）
            spike_density: Spike 污染区间内实际 spike 点占比（默认 0.1，即 10% 稀疏）
            mhr_noise_std: MHR 波形小随机扰动标准差 bpm（默认 2.0）
            random_state: 随机种子或 RandomState，用于可复现
            ensure_at_least_one: 若 True，保证至少注入一种噪声（用于可视化等场景；训练时默认 False 以符合论文）
            mode: "easy" 或 "hard"，hard 下覆盖率更高、多类噪声、连续段主导
            num_artifacts_min/max: hard 下每样本噪声类别数范围（默认 2~4）
            target_coverage_min/max: hard 下目标覆盖率范围（默认 8%~20%）
            max_segment_ratio_hard: hard 下单段最大长度比例（默认 12%）
            compound_prob_hard: hard 下 compound missing 概率（默认 0.7）
            spike_weight_hard: hard 下 spike 被选中的相对权重（降低主导性）
            spike_burst_len_min/max: hard 下 spike 改为短 burst，长度范围
            max_corruption_ratio_hard: hard 下总污染比例上限（默认 35%）
        """
        # CleanCTG 论文 Table 1 synthetic CTG 中的噪声比例（样本级统计结果），用作默认采样概率以逼近论文分布。
        paper_probs = {
            "halving_prob": 0.0339,
            "doubling_prob": 0.0340,
            "mhr_prob": 0.0328,
            "missing_prob": 0.4410,
            "spike_prob": 0.7600,
        }
        legacy_probs = {
            "halving_prob": 0.05,
            "doubling_prob": 0.05,
            "mhr_prob": 0.10,
            "missing_prob": 0.10,
            "spike_prob": 0.10,
        }
        default_probs = paper_probs if use_paper_distribution else legacy_probs

        # 若用户显式传入某个概率参数，则优先使用用户传入；否则按 default_probs（论文/旧默认）补齐
        self.halving_prob = float(default_probs["halving_prob"] if halving_prob is None else halving_prob)
        self.doubling_prob = float(default_probs["doubling_prob"] if doubling_prob is None else doubling_prob)
        self.mhr_prob = float(default_probs["mhr_prob"] if mhr_prob is None else mhr_prob)
        self.missing_prob = float(default_probs["missing_prob"] if missing_prob is None else missing_prob)
        self.spike_prob = float(default_probs["spike_prob"] if spike_prob is None else spike_prob)
        self.use_paper_distribution = bool(use_paper_distribution)
        self.distribution_source = (
            "CleanCTG Table 1 synthetic CTG" if use_paper_distribution else "custom/legacy"
        )
        self.max_corruption_ratio = max_corruption_ratio
        self.max_segment_ratio = max_segment_ratio
        self.sample_rate_hz = float(sample_rate_hz)
        self.compound_prob = compound_prob
        self.missing_fill_mode = missing_fill_mode
        self.spike_density = spike_density
        self.mhr_noise_std = mhr_noise_std
        self.ensure_at_least_one = ensure_at_least_one
        self._rng = np.random.RandomState(random_state) if isinstance(random_state, (int, type(None))) else random_state
        # hard mode
        self.mode = str(mode).lower()
        self.num_artifacts_min = int(num_artifacts_min)
        self.num_artifacts_max = int(num_artifacts_max)
        self.target_coverage_min = float(target_coverage_min)
        self.target_coverage_max = float(target_coverage_max)
        self.max_segment_ratio_hard = float(max_segment_ratio_hard)
        self.compound_prob_hard = float(compound_prob_hard)
        self.spike_weight_hard = float(spike_weight_hard)
        self.spike_burst_len_min = int(spike_burst_len_min)
        self.spike_burst_len_max = int(spike_burst_len_max)
        self.max_corruption_ratio_hard = float(max_corruption_ratio_hard)

    def get_config_summary(self) -> Dict[str, Any]:
        """返回当前概率配置摘要（用于记录/调试）。"""
        return {
            "halving_prob": self.halving_prob,
            "doubling_prob": self.doubling_prob,
            "mhr_prob": self.mhr_prob,
            "missing_prob": self.missing_prob,
            "spike_prob": self.spike_prob,
            "distribution_source": self.distribution_source,
            "use_paper_distribution": self.use_paper_distribution,
            "max_corruption_ratio": self.max_corruption_ratio,
            "max_segment_ratio": self.max_segment_ratio,
            "compound_prob": self.compound_prob,
            "missing_fill_mode": self.missing_fill_mode,
            "spike_density": self.spike_density,
            "mode": self.mode,
            "target_coverage_min": self.target_coverage_min,
            "target_coverage_max": self.target_coverage_max,
            "max_segment_ratio_hard": self.max_segment_ratio_hard,
            "compound_prob_hard": self.compound_prob_hard,
        }

    def estimate_noise_distribution(
        self,
        signals: np.ndarray,
        n_samples: int = 1000,
        random_state: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        经验统计：估计每种噪声在样本级的出现比例（用于近似验证论文分布）。

        统计方式：对每条样本，若 labels[:, k] 存在至少一个 1，则视为"该样本包含该噪声"。
        注意：由于复合模式、段长度约束、总污染比例约束、随机性与多标签共存，
        实际统计结果不会与理论采样概率完全一致；该函数用于快速近似验证。
        """
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]
        total = min(int(n_samples), int(signals.shape[0]))
        if total <= 0:
            return {
                "halving_ratio": 0.0,
                "doubling_ratio": 0.0,
                "mhr_ratio": 0.0,
                "missing_ratio": 0.0,
                "spike_ratio": 0.0,
            }

        rng = np.random.RandomState(random_state)
        idx = (
            rng.choice(signals.shape[0], size=total, replace=False)
            if signals.shape[0] > total
            else np.arange(total)
        )

        present = np.zeros(5, dtype=np.int32)
        for i in idx:
            _, labels = self.generate_artifacts(signals[i])
            for k in range(5):
                if np.any(labels[:, k] > 0.5):
                    present[k] += 1

        denom = float(total)
        return {
            "halving_ratio": present[0] / denom,
            "doubling_ratio": present[1] / denom,
            "mhr_ratio": present[2] / denom,
            "missing_ratio": present[3] / denom,
            "spike_ratio": present[4] / denom,
        }

    def _sample_segment(
        self,
        L: int,
        min_samples: int = 1,
        max_samples: Optional[int] = None,
        use_hard_ratio: bool = False,
    ) -> Tuple[int, int]:
        """
        在 [0, L) 内随机采样一个区间 [start, end)，长度在 [min_samples, max_samples]。
        对应论文：单段最大长度不超过 L * max_segment_ratio。
        use_hard_ratio: 若 True，使用 max_segment_ratio_hard。
        """
        ratio = self.max_segment_ratio_hard if use_hard_ratio else self.max_segment_ratio
        if max_samples is None:
            max_samples = max(min_samples, int(L * ratio))
        max_samples = min(max_samples, L)
        segment_length = self._rng.randint(min_samples, max_samples + 1)
        segment_length = min(segment_length, L)
        start = self._rng.randint(0, max(1, L - segment_length + 1))
        end = min(start + segment_length, L)
        return start, end

    def _can_apply_region(
        self,
        corruption_mask: np.ndarray,
        start: int,
        end: int,
        max_ratio: Optional[float] = None,
    ) -> bool:
        """
        检查在 [start, end) 注入后，总污染比例是否仍不超过 max_ratio。
        使用 union 后的 mask，重叠不重复计数。
        """
        if max_ratio is None:
            max_ratio = self.max_corruption_ratio_hard if self.mode == "hard" else self.max_corruption_ratio
        new_mask = corruption_mask.copy()
        new_mask[start:end] = 1
        return new_mask.mean() <= max_ratio

    def _is_region_non_overlapping_main(self, artifact_labels: np.ndarray, start: int, end: int) -> bool:
        """
        检查 [start, end) 是否与主噪声（halving/doubling/mhr/missing）无重叠。
        主噪声互斥：同一时间点只能有一种。
        """
        return not np.any(artifact_labels[start:end, :4] > 0.5)

    def _is_region_non_overlapping_missing(self, artifact_labels: np.ndarray, start: int, end: int) -> bool:
        """
        检查 [start, end) 是否与 missing 无重叠。
        spike 可与 halving/doubling/mhr 叠加，但不可与 missing 叠加（missing 段无有效值）。
        """
        return np.sum(artifact_labels[start:end, 3] > 0.5) == 0

    def _update_corruption_mask(
        self,
        corruption_mask: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        """就地更新 corruption_mask，标记 [start, end) 为已污染。"""
        corruption_mask[start:end] = 1

    def _apply_missing(
        self,
        corrupted: np.ndarray,
        labels: np.ndarray,
        corruption_mask: np.ndarray,
        start: int,
        end: int,
        require_non_overlap: bool = False,
    ) -> bool:
        """
        在 [start, end) 注入 missing，并更新 labels[:, 3] 和 corruption_mask。
        不覆盖已有 halving/doubling 区域，避免 doubling 被 missing 抹掉。
        spike 与 missing 互斥：不向已有 spike 的位置填充 missing。
        返回是否成功注入（若会超比例则跳过）。
        require_non_overlap: hard 模式下为 True，要求区间与已有主噪声无重叠。
        """
        start = max(0, min(start, len(corrupted)))
        end = max(start, min(end, len(corrupted)))
        if start >= end:
            return False
        if require_non_overlap and not self._is_region_non_overlapping_main(labels, start, end):
            return False
        scaling_mask = (labels[start:end, 0] > 0.5) | (labels[start:end, 1] > 0.5)
        spike_mask = labels[start:end, 4] > 0.5
        fill_mask = ~scaling_mask & ~spike_mask
        if not np.any(fill_mask):
            return False
        if not self._can_apply_region(corruption_mask, start, end):
            return False

        fill_val = np.nan if self.missing_fill_mode == "nan" else 0.0
        corrupted[start:end][fill_mask] = fill_val
        labels[start:end, 3][fill_mask] = 1.0
        corruption_mask[start:end] = np.maximum(corruption_mask[start:end], fill_mask.astype(np.uint8))
        return True

    def _apply_compound_missing(
        self,
        corrupted: np.ndarray,
        labels: np.ndarray,
        corruption_mask: np.ndarray,
        main_start: int,
        main_end: int,
    ) -> None:
        """
        在主伪影 [main_start, main_end) 前后各插入 short missing 段。
        对应论文复合模式：missing + 主伪影 + missing。
        子段长度 1~5 秒，换算为样本数。
        """
        L = len(corrupted)
        min_sec, max_sec = 1.0, 5.0
        min_samp = max(1, int(min_sec * self.sample_rate_hz))
        max_samp = max(min_samp, int(max_sec * self.sample_rate_hz))

        # 左 missing: [main_start - left_len, main_start)
        require_no_overlap = self.mode == "hard"
        left_len = self._rng.randint(min_samp, max_samp + 1)
        left_start = max(0, main_start - left_len)
        left_end = main_start
        if left_end > left_start and self._can_apply_region(corruption_mask, left_start, left_end):
            self._apply_missing(corrupted, labels, corruption_mask, left_start, left_end, require_non_overlap=require_no_overlap)

        # 右 missing: [main_end, main_end + right_len)
        right_len = self._rng.randint(min_samp, max_samp + 1)
        right_start = main_end
        right_end = min(L, main_end + right_len)
        if right_end > right_start and self._can_apply_region(corruption_mask, right_start, right_end):
            self._apply_missing(corrupted, labels, corruption_mask, right_start, right_end, require_non_overlap=require_no_overlap)

    def inject_halving(
        self,
        signal: np.ndarray,
        start: int,
        end: int,
        base_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        注入 Halving：局部 scaling artifact，整段缩放为 0.5 倍。
        对应论文：halving 作为 scaling artefact。
        若提供 base_signal，则用 base_signal 缩放；否则用 signal，避免被其他伪影覆盖后抵消。
        """
        corrupted = signal.copy().astype(np.float64)
        start = max(0, min(start, len(corrupted)))
        end = max(start, min(end, len(corrupted)))
        if start >= end:
            return corrupted
        base = base_signal if base_signal is not None else signal
        scaled = np.clip(base[start:end].astype(np.float64) * 0.5, 50, 255)
        corrupted[start:end] = np.where(
            np.isfinite(base[start:end]),
            scaled,
            corrupted[start:end],
        )
        return corrupted

    def inject_doubling(
        self,
        signal: np.ndarray,
        start: int,
        end: int,
        base_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        注入 Doubling：局部 scaling artifact，整段缩放为 2.0 倍。
        对应论文：doubling 作为 scaling artefact。
        若提供 base_signal，则用 base_signal 缩放；否则用 signal，避免被其他伪影覆盖后抵消。
        """
        corrupted = signal.copy().astype(np.float64)
        start = max(0, min(start, len(corrupted)))
        end = max(start, min(end, len(corrupted)))
        if start >= end:
            return corrupted
        base = base_signal if base_signal is not None else signal
        scaled = np.clip(base[start:end].astype(np.float64) * 2.0, 50, 255)
        corrupted[start:end] = np.where(
            np.isfinite(base[start:end]),
            scaled,
            corrupted[start:end],
        )
        return corrupted

    def inject_mhr(self, signal: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        注入 MHR（母体心率）伪影：用合成母体心率模式完全替换原片段。
        论文：MHR 范围 70–110 bpm，平滑、低频、近似生理的波形。
        基线从 [70, 110] 均匀采样，小幅振荡 2–8 bpm，小噪声 std 可配置，clip 到 [60, 120]。
        """
        corrupted = signal.copy().astype(np.float64)
        start = max(0, min(start, len(corrupted)))
        end = max(start, min(end, len(corrupted)))
        if start >= end:
            return corrupted

        segment_length = end - start
        mhr_baseline = self._rng.uniform(70, 110)
        oscillation_amp = self._rng.uniform(2, 8)
        period = max(1, int(60.0 / mhr_baseline * self.sample_rate_hz))
        t = np.arange(segment_length, dtype=np.float64)
        mhr_signal = (
            mhr_baseline
            + oscillation_amp * np.sin(2 * np.pi * t / period)
            + self._rng.normal(0, self.mhr_noise_std, segment_length)
        )
        mhr_signal = np.clip(mhr_signal, 60, 120)
        corrupted[start:end] = mhr_signal
        return corrupted

    def inject_missing(self, signal: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        注入 Missing：真实缺失，用 np.nan 或 0 填充，不再做线性插值。
        论文：null values，默认 np.nan。
        """
        corrupted = signal.copy().astype(np.float64)
        start = max(0, min(start, len(corrupted)))
        end = max(start, min(end, len(corrupted)))
        if start >= end:
            return corrupted
        fill_val = np.nan if self.missing_fill_mode == "nan" else 0.0
        corrupted[start:end] = fill_val
        return corrupted

    def _inject_sparse_spikes(
        self,
        signal: np.ndarray,
        start: int,
        end: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        在 [start, end) 内注入稀疏离散 spike，不整段统一平移。
        每个 spike 点独立随机方向，幅值 ±5~±40 bpm。
        只标 spike 实际发生的位置，不标整个候选区间。
        返回 (corrupted_signal, spike_labels_for_this_segment)。
        """
        corrupted = signal.copy().astype(np.float64)
        start = max(0, min(start, len(corrupted)))
        end = max(start, min(end, len(corrupted)))
        segment_len = end - start
        if segment_len <= 0:
            return corrupted, np.zeros(len(corrupted), dtype=np.float32)

        n_spikes = max(1, int(segment_len * self.spike_density))
        n_spikes = min(n_spikes, segment_len)
        spike_positions = self._rng.choice(segment_len, size=n_spikes, replace=False)
        spike_positions = np.sort(spike_positions)

        spike_labels = np.zeros(len(corrupted), dtype=np.float32)
        for idx in spike_positions:
            i = start + idx
            if i >= len(corrupted):
                continue
            if not np.isfinite(corrupted[i]):
                continue
            direction = self._rng.choice([-1, 1])
            magnitude = self._rng.uniform(5, 40)
            corrupted[i] = corrupted[i] + direction * magnitude
            corrupted[i] = np.clip(corrupted[i], 50, 255)
            spike_labels[i] = 1.0

        return corrupted, spike_labels

    def _inject_spike_burst(
        self,
        signal: np.ndarray,
        start: int,
        end: int,
        n_bursts: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        hard 模式：在 [start, end) 内注入若干短 burst（每 burst 2~5 点），而非单点。
        减少 region 数量，使 spike 不再主导统计。
        """
        corrupted = signal.copy().astype(np.float64)
        start = max(0, min(start, len(corrupted)))
        end = max(start, min(end, len(corrupted)))
        segment_len = end - start
        spike_labels = np.zeros(len(corrupted), dtype=np.float32)
        if segment_len <= 0:
            return corrupted, spike_labels

        burst_len_min = min(self.spike_burst_len_min, segment_len)
        burst_len_max = min(self.spike_burst_len_max, segment_len)
        if burst_len_max < burst_len_min:
            burst_len_max = burst_len_min

        n_bursts = min(n_bursts, max(1, segment_len // burst_len_min))
        for _ in range(n_bursts):
            burst_len = self._rng.randint(burst_len_min, burst_len_max + 1)
            max_start = start + segment_len - burst_len
            if max_start < start:
                continue
            burst_start = self._rng.randint(start, max_start + 1)
            burst_end = min(burst_start + burst_len, len(corrupted))
            for i in range(burst_start, burst_end):
                if i >= len(corrupted):
                    break
                if not np.isfinite(corrupted[i]):
                    continue
                direction = self._rng.choice([-1, 1])
                magnitude = self._rng.uniform(5, 40)
                corrupted[i] = corrupted[i] + direction * magnitude
                corrupted[i] = np.clip(corrupted[i], 50, 255)
                spike_labels[i] = 1.0
        return corrupted, spike_labels

    def generate_artifacts(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成带伪影的信号和伪影标签。
        mode="easy"：不强制至少注入一种噪声，允许输出完全 clean 的样本。
        mode="hard"：至少 2 类噪声、目标覆盖率 8%-20%、连续段主导。
        """
        if self.mode == "hard":
            return self._generate_artifacts_hard(signal)
        return self._generate_artifacts_easy(signal)

    def _generate_artifacts_easy(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """easy 模式：原有逻辑。"""
        corrupted = np.asarray(signal, dtype=np.float64).copy()
        original = corrupted.copy()
        L = len(signal)
        artifact_labels = np.zeros((L, 5), dtype=np.float32)
        corruption_mask = np.zeros(L, dtype=np.uint8)

        artifact_types = []
        if self._rng.random() < self.halving_prob:
            artifact_types.append(0)
        if self._rng.random() < self.doubling_prob:
            artifact_types.append(1)
        if self._rng.random() < self.mhr_prob:
            artifact_types.append(2)
        if self._rng.random() < self.missing_prob:
            artifact_types.append(3)
        if self._rng.random() < self.spike_prob:
            artifact_types.append(4)

        if not artifact_types and self.ensure_at_least_one:
            artifact_types = [self._rng.randint(0, 5)]

        random_order = self._rng.permutation(len(artifact_types))
        artifact_types = [artifact_types[i] for i in random_order]

        for art_type in artifact_types:
            if corruption_mask.mean() >= self.max_corruption_ratio:
                break

            max_seg = max(1, int(L * self.max_segment_ratio))
            min_seg = 1
            if art_type == 2:
                min_seg = max(1, int(2 * self.sample_rate_hz))
            start, end = self._sample_segment(L, min_samples=min_seg, max_samples=max_seg)
            if start >= end:
                continue

            if not self._can_apply_region(corruption_mask, start, end):
                continue

            use_compound = (
                art_type in (0, 1, 2)
                and self._rng.random() < self.compound_prob
            )

            if use_compound:
                self._apply_compound_missing(corrupted, artifact_labels, corruption_mask, start, end)

            if not self._can_apply_region(corruption_mask, start, end):
                continue

            if art_type == 0:
                corrupted = self.inject_halving(corrupted, start, end, base_signal=original)
                artifact_labels[start:end, 0] = 1.0
                self._update_corruption_mask(corruption_mask, start, end)
            elif art_type == 1:
                corrupted = self.inject_doubling(corrupted, start, end, base_signal=original)
                artifact_labels[start:end, 1] = 1.0
                self._update_corruption_mask(corruption_mask, start, end)
            elif art_type == 2:
                corrupted = self.inject_mhr(corrupted, start, end)
                artifact_labels[start:end, 2] = 1.0
                self._update_corruption_mask(corruption_mask, start, end)
            elif art_type == 3:
                if self._apply_missing(corrupted, artifact_labels, corruption_mask, start, end):
                    pass
            elif art_type == 4:
                corrupted, spike_lab = self._inject_sparse_spikes(corrupted, start, end)
                artifact_labels[:, 4] = np.maximum(artifact_labels[:, 4], spike_lab)
                for i in range(L):
                    if spike_lab[i] > 0:
                        corruption_mask[i] = 1

        return corrupted, artifact_labels

    def _generate_artifacts_hard(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        hard 模式：至少 2 类噪声、目标覆盖率 8%-20%、连续段主导、spike 降权。
        """
        corrupted = np.asarray(signal, dtype=np.float64).copy()
        original = corrupted.copy()
        L = len(signal)
        artifact_labels = np.zeros((L, 5), dtype=np.float32)
        corruption_mask = np.zeros(L, dtype=np.uint8)
        max_ratio = self.max_corruption_ratio_hard
        target = self._rng.uniform(self.target_coverage_min, self.target_coverage_max)
        compound_prob = self.compound_prob_hard

        # 1. 采样噪声类别数 2~4，并选择类别（spike 降权：以 spike_weight_hard 概率加入）
        num_art = self._rng.randint(
            self.num_artifacts_min,
            min(self.num_artifacts_max + 1, 6),
        )
        include_spike = self._rng.random() < self.spike_weight_hard
        if include_spike and num_art >= 2:
            num_main = num_art - 1
            main_types = self._rng.choice([0, 1, 2, 3], size=num_main, replace=False)
            artifact_types = main_types.tolist() + [4]
        else:
            artifact_types = self._rng.choice([0, 1, 2, 3], size=num_art, replace=False).tolist()
        self._rng.shuffle(artifact_types)

        # 2. 先注入每类至少一段（主噪声互斥，spike 可与 halving/doubling/mhr 叠加）
        for art_type in artifact_types:
            if corruption_mask.mean() >= max_ratio:
                break
            max_seg = max(1, int(L * self.max_segment_ratio_hard))
            min_seg = 8 if art_type == 2 else 2
            start, end = self._sample_segment(L, min_samples=min_seg, max_samples=max_seg, use_hard_ratio=True)
            if start >= end:
                continue
            if not self._can_apply_region(corruption_mask, start, end):
                continue
            if art_type == 4:
                if not self._is_region_non_overlapping_missing(artifact_labels, start, end):
                    continue
            else:
                if not self._is_region_non_overlapping_main(artifact_labels, start, end):
                    continue

            use_compound = art_type in ARTIFACT_MAIN and self._rng.random() < compound_prob
            if use_compound:
                self._apply_compound_missing(corrupted, artifact_labels, corruption_mask, start, end)
            if not self._can_apply_region(corruption_mask, start, end):
                continue
            if art_type == 4:
                if not self._is_region_non_overlapping_missing(artifact_labels, start, end):
                    continue
            else:
                if not self._is_region_non_overlapping_main(artifact_labels, start, end):
                    continue

            self._apply_one_artifact(
                corrupted, original, artifact_labels, corruption_mask,
                art_type, start, end, L, use_spike_burst=True,
            )

        # 3. 持续注入直到达到目标覆盖率（主噪声互斥，spike 可叠加）
        max_extra = 20
        for _ in range(max_extra):
            if corruption_mask.mean() >= target or corruption_mask.mean() >= max_ratio:
                break
            art_type = self._rng.choice(artifact_types)
            max_seg = max(1, int(L * self.max_segment_ratio_hard))
            min_seg = 8 if art_type == 2 else 2
            start, end = self._sample_segment(L, min_samples=min_seg, max_samples=max_seg, use_hard_ratio=True)
            if start >= end or not self._can_apply_region(corruption_mask, start, end):
                continue
            if art_type == 4:
                if not self._is_region_non_overlapping_missing(artifact_labels, start, end):
                    continue
            else:
                if not self._is_region_non_overlapping_main(artifact_labels, start, end):
                    continue
            use_compound = art_type in ARTIFACT_MAIN and self._rng.random() < compound_prob
            if use_compound:
                self._apply_compound_missing(corrupted, artifact_labels, corruption_mask, start, end)
            if not self._can_apply_region(corruption_mask, start, end):
                continue
            if art_type == 4:
                if not self._is_region_non_overlapping_missing(artifact_labels, start, end):
                    continue
            else:
                if not self._is_region_non_overlapping_main(artifact_labels, start, end):
                    continue
            self._apply_one_artifact(
                corrupted, original, artifact_labels, corruption_mask,
                art_type, start, end, L, use_spike_burst=True,
            )

        return corrupted, artifact_labels

    def _apply_one_artifact(
        self,
        corrupted: np.ndarray,
        original: np.ndarray,
        artifact_labels: np.ndarray,
        corruption_mask: np.ndarray,
        art_type: int,
        start: int,
        end: int,
        L: int,
        use_spike_burst: bool = False,
    ) -> None:
        """注入单段噪声，就地修改 corrupted。"""
        if art_type == 0:
            corrupted[:] = self.inject_halving(corrupted, start, end, base_signal=original)
            artifact_labels[start:end, 0] = 1.0
            self._update_corruption_mask(corruption_mask, start, end)
        elif art_type == 1:
            corrupted[:] = self.inject_doubling(corrupted, start, end, base_signal=original)
            artifact_labels[start:end, 1] = 1.0
            self._update_corruption_mask(corruption_mask, start, end)
        elif art_type == 2:
            corrupted[:] = self.inject_mhr(corrupted, start, end)
            artifact_labels[start:end, 2] = 1.0
            self._update_corruption_mask(corruption_mask, start, end)
        elif art_type == 3:
            self._apply_missing(corrupted, artifact_labels, corruption_mask, start, end)
        elif art_type == 4:
            if use_spike_burst:
                c2, spike_lab = self._inject_spike_burst(corrupted, start, end, n_bursts=3)
                corrupted[:] = c2
            else:
                corrupted[:], spike_lab = self._inject_sparse_spikes(corrupted, start, end)
            artifact_labels[:, 4] = np.maximum(artifact_labels[:, 4], spike_lab)
            for i in range(L):
                if spike_lab[i] > 0:
                    corruption_mask[i] = 1

    def generate_batch(
        self,
        signals: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量生成伪影。

        Args:
            signals: [B, L] 或 [L]

        Returns:
            corrupted_signals: [B, L]
            artifact_labels: [B, L, 5] 或 [L, 5]
        """
        if signals.ndim == 1:
            signals = signals[np.newaxis, :]

        B, L = signals.shape
        corrupted_batch = []
        labels_batch = []

        for i in range(B):
            corrupted, labels = self.generate_artifacts(signals[i])
            corrupted_batch.append(corrupted)
            labels_batch.append(labels)

        corrupted_batch = np.array(corrupted_batch)
        labels_batch = np.array(labels_batch)

        if signals.ndim == 1:
            corrupted_batch = corrupted_batch[0]
            labels_batch = labels_batch[0]

        return corrupted_batch, labels_batch
