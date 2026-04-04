# Repository Layout

## 顶层结构

- `src/ctg_pipeline/`
  共享实现。所有实验脚本统一从这里导入模块。
- `experiments/`
  只保留实验入口脚本、配置与实验说明。
- `scripts/`
  顶层 orchestrator。
- `artifacts/`
  所有数据集、训练结果、推理结果、历史 runs。
- `docs/`
  仓库导航、实验说明与论文图表索引。

## 目录映射

- `feature/ctg_io` -> `src/ctg_pipeline/io`
- `feature/ctg_preprocessing` -> `src/ctg_pipeline/preprocessing`
- `feature/models` -> `src/ctg_pipeline/models`
- `feature/noise` -> `src/ctg_pipeline/noise`
- `feature/scripts` -> `experiments/denoising/scripts`
- `real_dirty_inference` -> `experiments/real_world_inference`
- `doubling_halving` -> `experiments/doubling_halving`

## 产物约定

- `artifacts/datasets/denoising/`
  paired dataset、segmentation dataset、denoising dataset。
- `artifacts/results/denoising/`
  训练结果、评估结果、汇总表、可视化。
- `artifacts/results/real_world_inference/`
  真实脏信号推理结果。
- `artifacts/results/doubling_halving/`
  候选挖掘、人工复核样本。
- `artifacts/datasets/doubling_halving/`
  doubling / halving 合成数据。
- `artifacts/runs/`
  完整 rerun 输出。

## 迁移原则

- 新脚本默认只写 `artifacts/`。
- 源码目录不再保存 `datasets/`、`results/`、`runs/`。
- 默认 import 统一为 `ctg_pipeline.*`。
- 历史产物已并入 `artifacts/` 下的新结构，不再保留旧目录作为运行入口。

