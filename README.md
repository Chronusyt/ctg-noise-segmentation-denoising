# CTG_test

论文实验仓已整理为三层结构：

- `src/ctg_pipeline/`: 共享代码，包含 IO、预处理、模型、噪声与通用工具。
- `experiments/`: 各实验专题入口。
- `artifacts/`: 数据集、训练结果、推理结果、历史 rerun 产物。

当前主实验入口：

- `experiments/denoising/`: 合成噪声数据构建、分割、去噪与对比。
- `experiments/real_world_inference/`: 真实低质量 CTG 上的两阶段推理。
- `experiments/doubling_halving/`: doubling / halving 候选挖掘与合成注入。

常用文档：

- `docs/repo_layout.md`: 新目录结构与迁移说明。
- `docs/figure_sources.md`: 论文图表建议对应的产物目录。

常用入口：

- `scripts/run_full_pipeline.sh`: 完整实验编排脚本。
- `experiments/denoising/scripts/build_dataset.py`: paired dataset 构建。
- `experiments/real_world_inference/run_real_world_inference.py`: 真实样本推理。
- `experiments/doubling_halving/check_half_and_double.py`: 候选挖掘。

默认情况下，所有新产物都会写入 `artifacts/`，源码目录不再混放结果文件。

