# Real-World Inference

本目录用于把已经训练好的两阶段模型直接应用到真实低质量 CTG 片段。

## 入口

- `run_real_world_inference.py`

## 默认输入与输出

- 默认模型：
  `artifacts/results/denoising/multilabel_segmentation_hard/best_model.pt`
  `artifacts/results/denoising/multilabel_guided_denoising_hard_pred/best_model.pt`
- 默认输出：
  `artifacts/results/real_world_inference/real_world_inference/`

## 运行示例

```bash
python3 experiments/real_world_inference/run_real_world_inference.py \
  --csv /scratch2/yzd/CTG/batch1_valid.xlsx \
  --fetal_dir /scratch2/yzd/CTG/batch1/fetal \
  --id_column 档案号 \
  --output_dir artifacts/results/real_world_inference/hard
```

