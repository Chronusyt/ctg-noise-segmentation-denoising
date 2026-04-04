# Figure Sources

论文图表建议直接从以下目录取材：

- 分割训练与评估：
  `artifacts/results/denoising/segmentation_*`
  `artifacts/results/denoising/multilabel_segmentation_*`
- 去噪训练、评估与对比：
  `artifacts/results/denoising/denoising_baseline_*`
  `artifacts/results/denoising/multilabel_guided_denoising_*`
  `artifacts/results/denoising/comparisons/`
- 原始样本与 clean/noisy 可视化：
  `artifacts/results/denoising/raw_features/`
  `artifacts/results/denoising/clean_vs_noisy_*`
- 真实样本推理：
  `artifacts/results/real_world_inference/`
- doubling / halving 案例：
  `artifacts/results/doubling_halving/candidate_mining/`
  `artifacts/results/doubling_halving/review_samples/`

如果某张图被正式用于论文，建议把最终导出版复制到 `docs/examples/` 或 `docs/figures/` 做固定引用。

