# Synthetic Doubling / Halving Injection

本模块用于在 clean FHR 信号上注入短时 synthetic halving / doubling 伪影，服务于小于 15 秒的短时伪影修复任务。

## 设计依据

- halving 和 doubling 都是连续错误计数区间，不是单点 spike。
- halving 的目标 ratio 贴近真实样本，范围放宽到 `0.47-0.55`。
- doubling 拆成两类：`unclipped doubling` 和 `clipped doubling`。
- doubling 的目标 ratio 仍以 `1.90-2.05` 为中心，但只有 clipped doubling 才会进入 ceiling 附近的高平台形态。
- 区间内部以局部 baseline 为主体生成相对平稳的平台，并加入小幅抖动。
- 进入和退出边界只做 `1-3` 个采样点的短过渡，模拟突变进入 / 突变退出。

## 为什么限制到 15 秒以内

当前 synthetic 数据集只服务于“短时 doubling / halving 伪影修复”任务。根据现阶段策略，小于 15 秒的伪影由 AI 修复，因此本阶段不生成超过 15 秒的 halving / doubling。

## 规则概述

- `halving`
  - 持续时间主分布 `3-8 秒`，少量长尾 `8-15 秒`
  - 目标值约为 `baseline_local * ratio_half`
  - 区间内加入小幅高斯抖动
  - 最终限制在 `[50, 220]`

- `unclipped doubling`
  - 持续时间主分布 `3-8 秒`，少量长尾 `8-15 秒`
  - 优先放在较低 baseline 区域，默认 `baseline <= 115`
  - 目标值约为 `baseline_local * ratio_double`
  - 区间内加入小幅高斯抖动
  - 不主动做 ceiling 压缩，尽量保留 `actual_mean_ratio ≈ 2.0`
  - 最终限制在 `[50, 220]`

- `clipped doubling`
  - 持续时间主分布 `3-8 秒`，少量长尾 `8-15 秒`
  - 优先放在中等偏高 baseline 区域，默认 `110 <= baseline <= 145`
  - 目标值约为 `baseline_local * ratio_double`
  - 区间内加入小幅高斯抖动
  - 再应用 `225-250` 的随机 ceiling，产生高平台 / clipped effect
  - ceiling 不是整段常数硬截，而是 ceiling 附近的窄带压缩，并保留少量 exact ceiling 命中
  - 最终限制在 `[50, 255]`

## Synthetic 注入比例

- 默认使用：
  - `HALVING_PROB = 0.08`
  - `DOUBLING_PROB = 0.03`
- 这个比例不会机械复现真实检测统计，而是为了让训练数据里 doubling 也有足够样本可学。
- 原则是：`halving > doubling`，但 doubling 不能稀少到模型几乎见不到。

## 运行方式

测试注入规则：

```bash
python /home/yt/CTG_test/doubling_halving/test_injection_rules.py
```

构建 synthetic dataset：

```bash
python /home/yt/CTG_test/doubling_halving/build_synthetic_doubling_halving_dataset.py
```

默认输入：

`/home/yt/CTG_test/feature/datasets/denoising_20min/clean_dataset.npz`

默认输出：

`/home/yt/CTG_test/doubling_halving/synthetic_output/`
