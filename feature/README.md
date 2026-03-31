# CTG_test/feature - 自包含目录结构

本目录为 CTG 特征提取与数据集构建的**自包含**实现，不依赖 CTG 外部路径。

## 目录结构

```
feature/
├── README.md                    # 本文件
├── scripts/                     # 所有可运行脚本
│   ├── build_dataset.py         # 数据集构建 pipeline（clean + noisy）
│   ├── extract_features.py      # 特征提取主逻辑
│   ├── visualize_features.py   # 特征可视化
│   ├── visualize_signal.py      # 单图信号可视化
│   └── visualize_clean_vs_noisy.py  # 干净 vs 加噪对比可视化
│
├── ctg_preprocessing/           # 信号预处理模块
│
├── ctg_io/                      # fetal 数据读取
│   ├── __init__.py
│   └── fetal_reader.py
│
├── noise/                       # 噪声生成模块
│   ├── __init__.py
│   └── noise_generator.py
│
├── datasets/                    # 构建的数据集输出目录
│   └── denoising/
│
└── results/                     # 可视化输出目录
```

## 依赖说明

- **ctg_io**: fetal 文件读取，自包含副本
- **noise**: 噪声生成器（CleanCTG 论文一致性实现），自包含副本
- **ctg_preprocessing**: 信号预处理，本地实现

## 运行方式

```bash
cd /home/yt/CTG_test/feature

# 最小示例
python scripts/build_dataset.py --minimal

# 完整数据集构建
python scripts/build_dataset.py --csv /path/to/batch1_valid.xlsx --fetal_dir /path/to/fetal --id_column 档案号 --output_dir datasets/denoising

python scripts/build_dataset.py \
  --csv /scratch2/yzd/CTG/batch1_valid.xlsx \
  --fetal_dir /scratch2/yzd/CTG/batch1/fetal \
  --id_column 档案号 \
  --segment_len 4800 \
  --reliability_threshold 99 \
  --output_dir /home/yt/CTG_test/feature/datasets/denoising_20min

# 可视化
python scripts/visualize_clean_vs_noisy.py --csv /scratch2/yzd/CTG/batch1_valid.xlsx --fetal_dir /scratch2/yzd/CTG/batch1/fetal --id_column 档案号 --segment_len 4800 --min_reliability 99 --max_samples 30 --output_dir results/clean_vs_noisy

python scripts/visualize_signal.py \
  --csv /scratch2/yzd/CTG/batch1_valid.xlsx \
  --fetal_dir /scratch2/yzd/CTG/batch1/fetal \
  --id_column 档案号 \
  --segment_len 4800 \
  --max_samples 30 \
  --output_dir results/signal_plots


```

```bash
cd /home/yt/CTG_test/feature

# 最小示例
python scripts/build_dataset.py --minimal

# 完整数据集构建
python scripts/build_dataset.py \
  --csv /scratch2/yzd/CTG/batch1_valid.xlsx \
  --fetal_dir /scratch2/yzd/CTG/batch1/fetal \
  --id_column 档案号 \
  --segment_len 4800 \
  --reliability_threshold 99 \
  --output_dir datasets/denoising_20min

# 干净 vs 加噪可视化
python scripts/visualize_clean_vs_noisy.py \
  --csv /scratch2/yzd/CTG/batch1_valid.xlsx \
  --fetal_dir /scratch2/yzd/CTG/batch1/fetal \
  --id_column 档案号 \
  --segment_len 4800 \
  --min_reliability 99 \
  --max_samples 30 \
  --output_dir results/clean_vs_noisy

# 原始信号可视化
python scripts/visualize_signal.py \
  --csv /scratch2/yzd/CTG/batch1_valid.xlsx \
  --fetal_dir /scratch2/yzd/CTG/batch1/fetal \
  --id_column 档案号 \
  --segment_len 4800 \
  --max_samples 30 \
  --output_dir results/signal_plots

```

---

## 噪声实验（四套并行）

本目录提供**四套独立**实验，用于论文与汇报对比：

| 实验 | 任务 | 输出 | 数据目录 | 结果目录 |
|------|------|------|----------|----------|
| **Binary segmentation** | 有噪声 / 无噪声（二分类） | 1 通道 mask | `datasets/segmentation_hard/` | `results/segmentation_hard/` |
| **Multilabel segmentation** | 五类逐点分割 | 5 通道 mask | `datasets/multilabel_segmentation_hard/` | `results/multilabel_segmentation_hard/` |
| **Direct denoising baseline** | 直接去噪（noisy→clean） | reconstructed signal | `datasets/denoising_baseline_hard/` | `results/denoising_baseline_hard/` |
| **Mask-guided denoising** | 两阶段：segmentation→denoising | reconstructed signal | `datasets/multilabel_guided_denoising_hard_*` | `results/multilabel_guided_denoising_hard_*` |

四套实验**互不覆盖**。Mask-guided 含 pred（真实可部署）与 gt（oracle 上限）两种版本。

### Binary segmentation（有噪声/无噪声）

- 脚本：`build_segmentation_dataset.py`、`train_segmentation.py`、`evaluate_segmentation.py`
- 模型：`models/unet1d_segmentation.py`
- 输出：单通道二值 mask（noise vs clean）

### Multilabel segmentation（五类逐点分割）

- 脚本：`build_multilabel_segmentation_dataset.py`、`train_multilabel_segmentation.py`、`evaluate_multilabel_segmentation.py`
- 模型：`models/unet1d_multilabel_segmentation.py`
- 输出：5 通道独立 mask（sigmoid，非 softmax，支持多类共存如 spike+halving）

```bash
# 构建 multilabel 数据集
python scripts/build_multilabel_segmentation_dataset.py \
  --paired datasets/denoising_20min_hard/paired_dataset_hard.npz \
  --output_dir datasets/multilabel_segmentation_hard

# 训练
python scripts/train_multilabel_segmentation.py \
  --data_dir datasets/multilabel_segmentation_hard \
  --output_dir results/multilabel_segmentation_hard

# 评估
python scripts/evaluate_multilabel_segmentation.py \
  --data_dir datasets/multilabel_segmentation_hard \
  --model_path results/multilabel_segmentation_hard/best_model.pt \
  --output_dir results/multilabel_segmentation_hard
```

### Direct denoising baseline（直接去噪）

- 脚本：`build_denoising_dataset.py`、`train_denoiser.py`、`evaluate_denoiser.py`
- 模型：`models/unet1d_denoiser.py`
- 输出：reconstructed signal（noisy→clean）
- 评估：overall_mse、corrupted_region_mse、clean_region_mse，含 identity baseline 对比

```bash
# 构建 denoising 数据集
python scripts/build_denoising_dataset.py \
  --paired datasets/denoising_20min_hard/paired_dataset_hard.npz \
  --output_dir datasets/denoising_baseline_hard

# 训练
python scripts/train_denoiser.py \
  --data_dir datasets/denoising_baseline_hard \
  --output_dir results/denoising_baseline_hard

# 评估（含 identity baseline）
python scripts/evaluate_denoiser.py \
  --data_dir datasets/denoising_baseline_hard \
  --model_path results/denoising_baseline_hard/best_model.pt \
  --output_dir results/denoising_baseline_hard
```

### Mask-guided denoising（两阶段：segmentation → denoising）

- 脚本：`build_mask_guided_denoising_dataset.py`、`train_mask_guided_denoiser.py`、`evaluate_mask_guided_denoiser.py`、`compare_denoising_results.py`
- 模型：`models/unet1d_mask_guided_denoiser.py`
- 输入：noisy + 5 类 mask（pred 或 gt）
- 输出：reconstructed signal
- pred 版本：真实可部署；gt 版本：oracle 上限

```bash
# 构建 pred mask 数据集（需先训练 multilabel segmentation）
python scripts/build_mask_guided_denoising_dataset.py \
  --source_dir datasets/denoising_baseline_hard \
  --mask_source pred \
  --segmentation_model results/multilabel_segmentation_hard/best_model.pt

# 构建 GT mask 数据集
python scripts/build_mask_guided_denoising_dataset.py \
  --source_dir datasets/denoising_baseline_hard \
  --mask_source gt

# 训练 pred mask guided denoiser
python scripts/train_mask_guided_denoiser.py --mask_source pred

# 训练 GT mask oracle denoiser
python scripts/train_mask_guided_denoiser.py --mask_source gt

# 评估
python scripts/evaluate_mask_guided_denoiser.py --mask_source pred
python scripts/evaluate_mask_guided_denoiser.py --mask_source gt

# 对比三种方法
python scripts/compare_denoising_results.py
```
