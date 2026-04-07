"""Check clinical direct/pred/GT dataset alignment and write a report."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (_REPO_ROOT, _SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ctg_pipeline.utils.pathing import ARTIFACTS_ROOT, DENOISING_DATASETS_ROOT, resolve_repo_path

CHECK_KEYS = ("noisy_signals", "clean_signals", "artifact_labels", "parent_index", "chunk_index")


def load_split(datasets_root: str, method: str, split: str) -> np.lib.npyio.NpzFile:
    if method == "direct":
        path = os.path.join(datasets_root, "denoising_baseline_clinical", f"{split}_dataset_denoising.npz")
    else:
        path = os.path.join(
            datasets_root,
            f"multilabel_guided_denoising_clinical_{method}",
            f"{split}_dataset_mask_guided.npz",
        )
    return np.load(path)


def compare_split(datasets_root: str, split: str) -> tuple[list[str], bool]:
    lines = [f"[{split}]"]
    direct = load_split(datasets_root, "direct", split)
    pred = load_split(datasets_root, "pred", split)
    gt = load_split(datasets_root, "gt", split)
    ok = True

    for key in CHECK_KEYS:
        if key not in direct.files or key not in pred.files or key not in gt.files:
            lines.append(f"{key}: missing in at least one dataset")
            ok = False
            continue
        shape_equal = direct[key].shape == pred[key].shape == gt[key].shape
        pred_equal = np.array_equal(direct[key], pred[key])
        gt_equal = np.array_equal(direct[key], gt[key])
        lines.append(
            f"{key}: shapes direct/pred/gt={direct[key].shape}/{pred[key].shape}/{gt[key].shape}, "
            f"shape_equal={shape_equal}, direct==pred={pred_equal}, direct==gt={gt_equal}"
        )
        ok = ok and shape_equal and pred_equal and gt_equal

    lines.append(f"pred masks shape: {pred['masks'].shape if 'masks' in pred.files else 'missing'}")
    lines.append(f"gt masks shape: {gt['masks'].shape if 'masks' in gt.files else 'missing'}")
    lines.append(f"sample_mismatch_found: {not ok}")
    return lines, ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Check clinical main dataset alignment")
    parser.add_argument("--datasets_root", type=str, default=str(DENOISING_DATASETS_ROOT))
    parser.add_argument(
        "--report_path",
        type=str,
        default=str(ARTIFACTS_ROOT / "results" / "summary" / "clinical_main" / "self_check_report.txt"),
    )
    parser.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    args = parser.parse_args()

    datasets_root = str(resolve_repo_path(args.datasets_root))
    report_path = str(resolve_repo_path(args.report_path))
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)

    all_ok = True
    lines = [
        "Clinical main self-check report",
        f"datasets_root: {datasets_root}",
        f"splits_checked: {', '.join(args.splits)}",
        "",
    ]
    for split in args.splits:
        split_lines, split_ok = compare_split(datasets_root, split)
        lines.extend(split_lines)
        lines.append("")
        all_ok = all_ok and split_ok
    lines.append(f"overall_sample_mismatch_found: {not all_ok}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Self-check report saved to {report_path}")


if __name__ == "__main__":
    main()
