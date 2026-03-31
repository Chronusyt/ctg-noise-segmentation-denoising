"""
对比三种去噪方法的评估结果

1. Direct denoising baseline: noisy -> clean
2. Multilabel predicted-mask guided: noisy + predicted masks -> clean
3. Multilabel GT-mask oracle: noisy + GT masks -> clean

输出表格对比 overall_mse, corrupted_region_mse, clean_region_mse, overall_mae, corrupted_region_mae, clean_region_mae
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_FEATURE_ROOT = os.path.dirname(_SCRIPT_DIR)
if _FEATURE_ROOT not in sys.path:
    sys.path.insert(0, _FEATURE_ROOT)


def load_metrics(result_dir: str) -> dict | None:
    """从 test_metrics.json 加载指标。"""
    path = os.path.join(result_dir, "test_metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="对比三种去噪方法")
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="结果根目录",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/denoising_comparison.txt",
        help="对比表输出路径",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        choices=["hard", "clinical"],
        default="hard",
        help="实验标签，用于选择 *_hard 或 *_clinical 结果目录",
    )
    args = parser.parse_args()

    if not os.path.isabs(args.results_root):
        args.results_root = os.path.join(_FEATURE_ROOT, args.results_root)
    if not os.path.isabs(args.output):
        args.output = os.path.join(_FEATURE_ROOT, args.output)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    methods = [
        ("direct_denoising", "Direct denoising baseline", f"denoising_baseline_{args.experiment_tag}"),
        ("pred_mask_guided", "Multilabel pred-mask guided", f"multilabel_guided_denoising_{args.experiment_tag}_pred"),
        ("gt_mask_oracle", "Multilabel GT-mask oracle", f"multilabel_guided_denoising_{args.experiment_tag}_gt"),
    ]

    metrics_keys = [
        "overall_mse", "corrupted_region_mse", "clean_region_mse",
        "overall_mae", "corrupted_region_mae", "clean_region_mae",
    ]

    rows = []
    for key, name, subdir in methods:
        result_dir = os.path.join(args.results_root, subdir)
        data = load_metrics(result_dir)
        if data is None:
            rows.append((name, {k: float("nan") for k in metrics_keys}))
            continue
        if "learned_denoiser" in data:
            overall = data["learned_denoiser"]["overall"]  # direct denoising
        elif "overall" in data:
            overall = data["overall"]  # mask-guided
        else:
            overall = {}
        rows.append((name, {k: overall.get(k, float("nan")) for k in metrics_keys}))

    lines = []
    lines.append("=" * 80)
    lines.append("去噪方法对比")
    lines.append("=" * 80)
    lines.append(f"实验标签: {args.experiment_tag}")
    lines.append("")
    header = f"{'方法':<35} " + " ".join(f"{k:>18}" for k in metrics_keys)
    lines.append(header)
    lines.append("-" * len(header))
    for name, m in rows:
        vals = " ".join(f"{m[k]:>18.4f}" if not (isinstance(m[k], float) and m[k] != m[k]) else f"{'N/A':>18}" for k in metrics_keys)
        lines.append(f"{name:<35} {vals}")
    lines.append("")
    lines.append("说明:")
    lines.append("  - Direct denoising: noisy -> clean，无 mask 引导")
    lines.append("  - Pred-mask guided: noisy + predicted 5-class masks -> clean，真实可部署方案")
    lines.append("  - GT-mask oracle: noisy + GT 5-class masks -> clean，理论上限")
    lines.append("")
    lines.append("结论:")
    lines.append("  1. 若 pred-mask 优于 direct：知道噪声类型有助于去噪")
    lines.append("  2. 若 GT-mask 优于 pred-mask：系统瓶颈在 segmentation")
    lines.append("  3. 若 GT-mask 优于 direct：denoising 阶段有提升空间")
    lines.append("=" * 80)

    out_text = "\n".join(lines)
    print(out_text)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"\n对比表已保存到 {args.output}")

    json_out = args.output.replace(".txt", ".json")
    json_data = {
        "methods": [r[0] for r in rows],
        "metrics": {k: [r[1][k] for r in rows] for k in metrics_keys},
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存到 {json_out}")


if __name__ == "__main__":
    main()
