# Real Dirty Inference

这个目录用于在真实原始 CTG 脏信号上复用已经训练好的两阶段模型做推理与可视化。

## 文件

- `run_real_dirty_inference.py`
  在原始低 reliability 样本上运行：
  `raw dirty -> multilabel segmentation -> predicted masks -> mask-guided denoiser -> reconstructed`

## 默认输出

- `results/real_dirty_inference/visualizations/`
- `results/real_dirty_inference/summary.csv`
- `results/real_dirty_inference/summary.txt`

## 默认模型

- segmentation:
  `feature/results/multilabel_segmentation_hard/best_model.pt`
- denoiser:
  `feature/results/multilabel_guided_denoising_hard_pred/best_model.pt`
