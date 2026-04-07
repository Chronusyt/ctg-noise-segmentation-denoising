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
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/artifacts/runs/logs_clinical}"

mkdir -p "$DATA_ROOT" "$RESULT_ROOT/comparisons" "$LOG_ROOT"

run() {
  local name="$1"
  shift
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$name"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
}

run build_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_dataset.py" \
  --csv "$CSV" \
  --fetal_dir "$FETAL_DIR" \
  --id_column "$ID_COL" \
  --output_dir "$DATA_ROOT/denoising_20min_clinical" \
  --noise_mode clinical \
  --max_halving_segment_seconds 15 \
  --max_doubling_segment_seconds 15 \
  --max_mhr_segment_seconds 10 \
  --max_missing_segment_seconds 8 \
  --max_spike_segment_seconds 2 \
  --max_total_noise_ratio_clinical 0.12 \
  --min_post_noise_reliability 85 \
  --clinical_max_retries 80

run build_seg_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_segmentation_dataset.py" \
  --paired "$DATA_ROOT/denoising_20min_clinical/paired_dataset_clinical.npz" \
  --output_dir "$DATA_ROOT/segmentation_clinical"

run train_seg_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_segmentation.py" \
  --data_dir "$DATA_ROOT/segmentation_clinical" \
  --output_dir "$RESULT_ROOT/segmentation_clinical" \
  --device "$DEVICE"

run eval_seg_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_segmentation.py" \
  --data_dir "$DATA_ROOT/segmentation_clinical" \
  --model_path "$RESULT_ROOT/segmentation_clinical/best_model.pt" \
  --output_dir "$RESULT_ROOT/segmentation_clinical/eval" \
  --device "$DEVICE"

run build_mseg_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_multilabel_segmentation_dataset.py" \
  --paired "$DATA_ROOT/denoising_20min_clinical/paired_dataset_clinical.npz" \
  --output_dir "$DATA_ROOT/multilabel_segmentation_clinical"

run train_mseg_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_multilabel_segmentation.py" \
  --data_dir "$DATA_ROOT/multilabel_segmentation_clinical" \
  --output_dir "$RESULT_ROOT/multilabel_segmentation_clinical" \
  --device "$DEVICE"

run eval_mseg_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_multilabel_segmentation.py" \
  --data_dir "$DATA_ROOT/multilabel_segmentation_clinical" \
  --model_path "$RESULT_ROOT/multilabel_segmentation_clinical/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_segmentation_clinical/eval" \
  --device "$DEVICE"

run build_denoise_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_denoising_dataset.py" \
  --paired "$DATA_ROOT/denoising_20min_clinical/paired_dataset_clinical.npz" \
  --output_dir "$DATA_ROOT/denoising_baseline_clinical"

run train_denoise_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_denoiser.py" \
  --data_dir "$DATA_ROOT/denoising_baseline_clinical" \
  --output_dir "$RESULT_ROOT/denoising_baseline_clinical" \
  --device "$DEVICE"

run eval_denoise_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_denoiser.py" \
  --data_dir "$DATA_ROOT/denoising_baseline_clinical" \
  --model_path "$RESULT_ROOT/denoising_baseline_clinical/best_model.pt" \
  --output_dir "$RESULT_ROOT/denoising_baseline_clinical/eval" \
  --device "$DEVICE"

run build_mgd_clinical_pred \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_mask_guided_denoising_dataset.py" \
  --source_dir "$DATA_ROOT/denoising_baseline_clinical" \
  --mask_source pred \
  --segmentation_model "$RESULT_ROOT/multilabel_segmentation_clinical/best_model.pt" \
  --output_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_pred" \
  --device "$DEVICE"

run train_mgd_clinical_pred \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_mask_guided_denoiser.py" \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_pred" \
  --mask_source pred \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_clinical_pred" \
  --device "$DEVICE"

run eval_mgd_clinical_pred \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_mask_guided_denoiser.py" \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_pred" \
  --mask_source pred \
  --model_path "$RESULT_ROOT/multilabel_guided_denoising_clinical_pred/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_clinical_pred/eval" \
  --device "$DEVICE"

run build_mgd_clinical_gt \
  python3 "$REPO_ROOT/experiments/denoising/scripts/build_mask_guided_denoising_dataset.py" \
  --source_dir "$DATA_ROOT/denoising_baseline_clinical" \
  --mask_source gt \
  --output_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_gt"

run train_mgd_clinical_gt \
  python3 "$REPO_ROOT/experiments/denoising/scripts/train_mask_guided_denoiser.py" \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_gt" \
  --mask_source gt \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_clinical_gt" \
  --device "$DEVICE"

run eval_mgd_clinical_gt \
  python3 "$REPO_ROOT/experiments/denoising/scripts/evaluate_mask_guided_denoiser.py" \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_gt" \
  --mask_source gt \
  --model_path "$RESULT_ROOT/multilabel_guided_denoising_clinical_gt/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_clinical_gt/eval" \
  --device "$DEVICE"

run compare_clinical \
  python3 "$REPO_ROOT/experiments/denoising/scripts/compare_denoising_results.py" \
  --results_root "$RESULT_ROOT" \
  --experiment_tag clinical \
  --output "$RESULT_ROOT/comparisons/denoising_comparison_clinical.csv"

printf '\nDone. Logs are in %s\n' "$LOG_ROOT"
