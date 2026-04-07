#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

CSV="${CSV:-/scratch2/yzd/CTG/batch1_valid.xlsx}"
FETAL_DIR="${FETAL_DIR:-/scratch2/yzd/CTG/batch1/fetal}"
ID_COL="${ID_COL:-档案号}"
DEVICE="${DEVICE:-cuda}"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/artifacts/datasets/denoising}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/artifacts/results/denoising}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/artifacts/runs/logs_hard}"

mkdir -p "$DATA_ROOT" "$RESULT_ROOT/comparisons" "$LOG_ROOT"

run() {
  local name="$1"
  shift
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$name"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
}

run build_easy \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_dataset.py" \
  --csv "$CSV" \
  --fetal_dir "$FETAL_DIR" \
  --id_column "$ID_COL" \
  --output_dir "$DATA_ROOT/denoising_20min" \
  --noise_mode easy

run build_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_hard_from_paired.py" \
  --paired "$DATA_ROOT/denoising_20min/paired_dataset.npz" \
  --output_dir "$DATA_ROOT/denoising_20min_hard"

run build_seg_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_segmentation_dataset.py" \
  --paired "$DATA_ROOT/denoising_20min_hard/paired_dataset_hard.npz" \
  --output_dir "$DATA_ROOT/segmentation_hard"

run train_seg_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_segmentation.py" \
  --data_dir "$DATA_ROOT/segmentation_hard" \
  --output_dir "$RESULT_ROOT/segmentation_hard" \
  --device "$DEVICE"

run eval_seg_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_segmentation.py" \
  --data_dir "$DATA_ROOT/segmentation_hard" \
  --model_path "$RESULT_ROOT/segmentation_hard/best_model.pt" \
  --output_dir "$RESULT_ROOT/segmentation_hard/eval" \
  --device "$DEVICE"

run build_mseg_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_multilabel_segmentation_dataset.py" \
  --paired "$DATA_ROOT/denoising_20min_hard/paired_dataset_hard.npz" \
  --output_dir "$DATA_ROOT/multilabel_segmentation_hard"

run train_mseg_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_multilabel_segmentation.py" \
  --data_dir "$DATA_ROOT/multilabel_segmentation_hard" \
  --output_dir "$RESULT_ROOT/multilabel_segmentation_hard" \
  --device "$DEVICE"

run eval_mseg_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_multilabel_segmentation.py" \
  --data_dir "$DATA_ROOT/multilabel_segmentation_hard" \
  --model_path "$RESULT_ROOT/multilabel_segmentation_hard/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_segmentation_hard/eval" \
  --device "$DEVICE"

run build_denoise_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_denoising_dataset.py" \
  --paired "$DATA_ROOT/denoising_20min_hard/paired_dataset_hard.npz" \
  --output_dir "$DATA_ROOT/denoising_baseline_hard"

run train_denoise_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_denoiser.py" \
  --data_dir "$DATA_ROOT/denoising_baseline_hard" \
  --output_dir "$RESULT_ROOT/denoising_baseline_hard" \
  --device "$DEVICE"

run eval_denoise_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_denoiser.py" \
  --data_dir "$DATA_ROOT/denoising_baseline_hard" \
  --model_path "$RESULT_ROOT/denoising_baseline_hard/best_model.pt" \
  --output_dir "$RESULT_ROOT/denoising_baseline_hard/eval" \
  --device "$DEVICE"

run build_mgd_hard_pred \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_mask_guided_denoising_dataset.py" \
  --source_dir "$DATA_ROOT/denoising_baseline_hard" \
  --mask_source pred \
  --segmentation_model "$RESULT_ROOT/multilabel_segmentation_hard/best_model.pt" \
  --output_dir "$DATA_ROOT/multilabel_guided_denoising_hard_pred" \
  --device "$DEVICE"

run train_mgd_hard_pred \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_mask_guided_denoiser.py" \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_hard_pred" \
  --mask_source pred \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_hard_pred" \
  --device "$DEVICE"

run eval_mgd_hard_pred \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_mask_guided_denoiser.py" \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_hard_pred" \
  --mask_source pred \
  --model_path "$RESULT_ROOT/multilabel_guided_denoising_hard_pred/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_hard_pred/eval" \
  --device "$DEVICE"

run build_mgd_hard_gt \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_mask_guided_denoising_dataset.py" \
  --source_dir "$DATA_ROOT/denoising_baseline_hard" \
  --mask_source gt \
  --output_dir "$DATA_ROOT/multilabel_guided_denoising_hard_gt"

run train_mgd_hard_gt \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_mask_guided_denoiser.py" \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_hard_gt" \
  --mask_source gt \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_hard_gt" \
  --device "$DEVICE"

run eval_mgd_hard_gt \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_mask_guided_denoiser.py" \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_hard_gt" \
  --mask_source gt \
  --model_path "$RESULT_ROOT/multilabel_guided_denoising_hard_gt/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_hard_gt/eval" \
  --device "$DEVICE"

run compare_hard \
  python3 "$REPO_ROOT/experiments/denoising/scripts/compare_denoising_results.py" \
  --results_root "$RESULT_ROOT" \
  --experiment_tag hard \
  --output "$RESULT_ROOT/comparisons/denoising_comparison_hard.csv"

printf '\nDone. Logs are in %s\n' "$LOG_ROOT"
