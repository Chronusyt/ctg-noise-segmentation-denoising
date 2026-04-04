# Synthetic Doubling / Halving Injection

## 测试注入规则

```bash
python3 experiments/doubling_halving/test_injection_rules.py
```

## 构建合成数据

```bash
python3 experiments/doubling_halving/build_synthetic_doubling_halving_dataset.py \
  --data_path artifacts/datasets/denoising/denoising_20min/clean_dataset.npz \
  --output_dir artifacts/datasets/doubling_halving/synthetic_output
```

默认输入：

`artifacts/datasets/denoising/denoising_20min/clean_dataset.npz`

默认输出：

`artifacts/datasets/doubling_halving/synthetic_output/`

