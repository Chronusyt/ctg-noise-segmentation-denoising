"""
对比三种去噪方法的评估结果

1. Direct denoising baseline: noisy -> clean
2. Multilabel predicted-mask guided: noisy + predicted masks -> clean
3. Multilabel GT-mask oracle: noisy + GT masks -> clean

输出表格对比重建误差与 baseline/STV/LTV feature-preservation 指标。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_RESULTS_ROOT, resolve_repo_path


def load_metrics(result_dir: str) -> dict | None:
    """从 test_metrics.json 加载指标。"""
    candidates = [
        os.path.join(result_dir, "test_metrics.json"),
        os.path.join(result_dir, "eval", "test_metrics.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def resolve_result_dir(results_roots: list[str], subdir: str) -> tuple[str, dict | None]:
    """Find the first result directory containing test metrics."""
    for root in results_roots:
        result_dir = os.path.join(root, subdir)
        data = load_metrics(result_dir)
        if data is not None:
            return result_dir, data
    return os.path.join(results_roots[0], subdir), None


def extract_method_metrics(data: dict | None, metrics_keys: list[str]) -> dict:
    if data is None:
        return {k: float("nan") for k in metrics_keys}

    if "learned_denoiser" in data:
        overall = data["learned_denoiser"].get("overall", {})
        features = data["learned_denoiser"].get("feature_preservation", {})
    else:
        overall = data.get("overall", {})
        features = data.get("feature_preservation", {})

    merged = dict(overall)
    merged.update(features)
    return {k: merged.get(k, float("nan")) for k in metrics_keys}


def _derive_json_output_path(output_path: str) -> str:
    root, ext = os.path.splitext(output_path)
    return f"{root}.json" if ext else f"{output_path}.json"


def _derive_text_output_path(output_path: str) -> str:
    root, ext = os.path.splitext(output_path)
    if ext.lower() == ".txt":
        return output_path
    return f"{root}.txt" if ext else f"{output_path}.txt"


def _derive_markdown_output_path(output_path: str) -> str:
    root, ext = os.path.splitext(output_path)
    return f"{root}.md" if ext else f"{output_path}.md"


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and value != value


def build_markdown_table(rows: list[tuple[str, str, dict]], metrics_keys: list[str]) -> str:
    headers = ["method", *metrics_keys]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for name, _source_dir, metric_row in rows:
        vals = [name]
        vals.extend("N/A" if _is_nan(metric_row[k]) else f"{metric_row[k]:.4f}" for k in metrics_keys)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="对比三种去噪方法")
    parser.add_argument(
        "--results_root",
        type=str,
        default=str(DENOISING_RESULTS_ROOT),
        help="结果根目录",
    )
    parser.add_argument(
        "--fallback_results_root",
        action="append",
        default=None,
        help="额外结果根目录；可重复传入。用于兼容 artifacts/runs/results/denoising 等历史产物。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DENOISING_RESULTS_ROOT / "comparisons" / "denoising_comparison.txt"),
        help="对比表输出路径",
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        choices=["hard", "clinical"],
        default="hard",
        help="实验标签，用于选择 *_hard 或 *_clinical 结果目录",
    )
    parser.add_argument(
        "--markdown_output",
        type=str,
        default=None,
        help="Markdown 表输出路径；默认由 --output 派生 .md",
    )
    args = parser.parse_args()

    args.results_root = str(resolve_repo_path(args.results_root))
    args.output = str(resolve_repo_path(args.output))
    default_fallback = str(ARTIFACTS_ROOT / "runs" / "results" / "denoising")
    fallback_roots = args.fallback_results_root or [default_fallback]
    results_roots = [args.results_root]
    for root in fallback_roots:
        resolved = str(resolve_repo_path(root))
        if resolved not in results_roots:
            results_roots.append(resolved)
    args.markdown_output = str(resolve_repo_path(args.markdown_output)) if args.markdown_output else _derive_markdown_output_path(args.output)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.markdown_output) or ".", exist_ok=True)

    methods = [
        ("direct_denoising", "Direct denoising baseline", f"denoising_baseline_{args.experiment_tag}"),
        ("pred_mask_guided", "Multilabel pred-mask guided", f"multilabel_guided_denoising_{args.experiment_tag}_pred"),
        ("gt_mask_oracle", "Multilabel GT-mask oracle", f"multilabel_guided_denoising_{args.experiment_tag}_gt"),
    ]

    metrics_keys = [
        "overall_mse", "corrupted_region_mse", "clean_region_mse",
        "overall_mae", "corrupted_region_mae", "clean_region_mae",
        "baseline_mae", "stv_mae", "ltv_mae",
        "baseline_bias_mean", "baseline_bias_median",
        "stv_bias_mean", "stv_bias_median",
        "ltv_bias_mean", "ltv_bias_median",
    ]

    rows = []
    for key, name, subdir in methods:
        result_dir, data = resolve_result_dir(results_roots, subdir)
        rows.append((name, result_dir, extract_method_metrics(data, metrics_keys)))

    lines = []
    lines.append("=" * 80)
    lines.append("去噪方法对比")
    lines.append("=" * 80)
    lines.append(f"实验标签: {args.experiment_tag}")
    lines.append("结果检索根目录:")
    for root in results_roots:
        lines.append(f"  - {root}")
    lines.append("")
    header = f"{'方法':<35} " + " ".join(f"{k:>18}" for k in metrics_keys)
    lines.append(header)
    lines.append("-" * len(header))
    for name, source_dir, m in rows:
        vals = " ".join(f"{m[k]:>18.4f}" if not _is_nan(m[k]) else f"{'N/A':>18}" for k in metrics_keys)
        lines.append(f"{name:<35} {vals}")
        lines.append(f"{'source':<35} {source_dir}")
    lines.append("")
    lines.append("说明:")
    lines.append("  - Direct denoising: noisy -> clean，无 mask 引导")
    lines.append("  - Pred-mask guided: noisy + predicted 5-class masks -> clean，真实可部署方案")
    lines.append("  - GT-mask oracle: noisy + GT 5-class masks -> clean，mask 条件的诊断参考；不保证所有误差/feature 指标都最优")
    lines.append("")
    lines.append("结论:")
    lines.append("  1. 若 pred-mask 优于 direct：知道噪声类型有助于去噪")
    lines.append("  2. 若 GT-mask 在 MSE 上优于 pred-mask：更准确的 mask 有进一步提升重建误差的潜力")
    lines.append("  3. 若 GT-mask 在 feature-preservation 上弱于 pred-mask：需要检查 GT-mask 训练稳定性、mask 分布或 loss 设计")
    lines.append("=" * 80)

    out_text = "\n".join(lines)
    print(out_text)
    output_ext = os.path.splitext(args.output)[1].lower()
    if output_ext == ".csv":
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "source_dir", *metrics_keys])
            for name, source_dir, metric_row in rows:
                writer.writerow([name, source_dir, *[metric_row[k] for k in metrics_keys]])
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_text)
    print(f"\n对比表已保存到 {args.output}")

    txt_out = _derive_text_output_path(args.output)
    if txt_out != args.output:
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(out_text)
        print(f"文本摘要已保存到 {txt_out}")

    markdown = build_markdown_table(rows, metrics_keys)
    with open(args.markdown_output, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Markdown 表已保存到 {args.markdown_output}")

    json_out = _derive_json_output_path(args.output)
    json_data = {
        "results_roots": results_roots,
        "methods": [r[0] for r in rows],
        "source_dirs": [r[1] for r in rows],
        "metrics": {k: [r[2][k] for r in rows] for k in metrics_keys},
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存到 {json_out}")


if __name__ == "__main__":
    main()
