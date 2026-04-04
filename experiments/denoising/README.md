# Denoising Experiments

本目录只保留 denoising 主线实验入口，所有共享实现已经迁移到 `src/ctg_pipeline/`，所有产物默认写到 `artifacts/`。

## 目录职责

- `scripts/build_dataset.py`
  构建 easy / hard / clinical paired dataset。
- `scripts/build_segmentation_dataset.py`
  构建 binary segmentation 数据。
- `scripts/build_multilabel_segmentation_dataset.py`
  构建五类逐点分割数据。
- `scripts/build_denoising_dataset.py`
  构建 direct denoising baseline 数据。
- `scripts/build_mask_guided_denoising_dataset.py`
  构建两阶段 mask-guided denoising 数据。

## 默认产物目录

- 数据集：
  `artifacts/datasets/denoising/`
- 训练与评估：
  `artifacts/results/denoising/`

## 典型流程

### 1. paired dataset

```bash
python3 experiments/denoising/scripts/build_dataset.py \
  --csv /scratch2/yzd/CTG/batch1_valid.xlsx \
  --fetal_dir /scratch2/yzd/CTG/batch1/fetal \
  --id_column 档案号 \
  --output_dir artifacts/datasets/denoising/denoising_20min
```

### 2. segmentation

```bash
python3 experiments/denoising/scripts/build_segmentation_dataset.py \
  --paired artifacts/datasets/denoising/denoising_20min_hard/paired_dataset_hard.npz \
  --output_dir artifacts/datasets/denoising/segmentation_hard

python3 experiments/denoising/scripts/train_segmentation.py \
  --data_dir artifacts/datasets/denoising/segmentation_hard \
  --output_dir artifacts/results/denoising/segmentation_hard
```

### 3. multilabel segmentation

```bash
python3 experiments/denoising/scripts/build_multilabel_segmentation_dataset.py \
  --paired artifacts/datasets/denoising/denoising_20min_hard/paired_dataset_hard.npz \
  --output_dir artifacts/datasets/denoising/multilabel_segmentation_hard
```

### 4. denoising baseline

```bash
python3 experiments/denoising/scripts/build_denoising_dataset.py \
  --paired artifacts/datasets/denoising/denoising_20min_hard/paired_dataset_hard.npz \
  --output_dir artifacts/datasets/denoising/denoising_baseline_hard
```

### 5. mask-guided denoising

```bash
python3 experiments/denoising/scripts/build_mask_guided_denoising_dataset.py \
  --source_dir artifacts/datasets/denoising/denoising_baseline_hard \
  --mask_source pred \
  --segmentation_model artifacts/results/denoising/multilabel_segmentation_hard/best_model.pt
```

完整 rerun 推荐直接使用顶层 `scripts/run_full_pipeline.sh`。

