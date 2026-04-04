#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RUN_TAG="${RUN_TAG:-$(date +%F)_full_pipeline}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/artifacts/runs/$RUN_TAG}"
CSV="${CSV:-/scratch2/yzd/CTG/batch1_valid.xlsx}"
FETAL_DIR="${FETAL_DIR:-/scratch2/yzd/CTG/batch1/fetal}"
ID_COL="${ID_COL:-档案号}"
DEFAULT_DEVICE="${DEFAULT_DEVICE:-cuda}"

DATA_ROOT="$RUN_ROOT/datasets/denoising"
RESULT_ROOT="$RUN_ROOT/results/denoising"
REAL_WORLD_ROOT="$RUN_ROOT/results/real_world_inference"
DOUBLE_HALF_RESULT_ROOT="$RUN_ROOT/results/doubling_halving"
DOUBLE_HALF_DATA_ROOT="$RUN_ROOT/datasets/doubling_halving"
LOG_ROOT="$RUN_ROOT/logs"
MASTER_LOG="$LOG_ROOT/_master.log"

mkdir -p \
  "$DATA_ROOT" \
  "$RESULT_ROOT/comparisons" \
  "$REAL_WORLD_ROOT" \
  "$DOUBLE_HALF_RESULT_ROOT" \
  "$DOUBLE_HALF_DATA_ROOT" \
  "$LOG_ROOT"

log_master() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$MASTER_LOG"
}

run() {
  local name="$1"
  shift
  log_master "START $name"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
  log_master "DONE  $name"
}

log_master "Run root: $RUN_ROOT"
log_master "CSV: $CSV"
log_master "FETAL_DIR: $FETAL_DIR"

run raw_features \
  python3 experiments/denoising/scripts/visualize_features.py \
  --csv "$CSV" \
  --fetal_dir "$FETAL_DIR" \
  --id_column "$ID_COL" \
  --segment_len 4800 \
  --max_samples 20 \
  --output_dir "$RESULT_ROOT/raw_features"

run build_easy \
  python3 experiments/denoising/scripts/build_dataset.py \
  --csv "$CSV" \
  --fetal_dir "$FETAL_DIR" \
  --id_column "$ID_COL" \
  --output_dir "$DATA_ROOT/denoising_20min" \
  --noise_mode easy

run vis_easy \
  python3 experiments/denoising/scripts/visualize_clean_vs_noisy.py \
  --from_paired "$DATA_ROOT/denoising_20min/paired_dataset.npz" \
  --max_samples 20 \
  --output_dir "$RESULT_ROOT/clean_vs_noisy_easy"

run build_hard \
  python3 experiments/denoising/scripts/build_hard_from_paired.py \
  --paired "$DATA_ROOT/denoising_20min/paired_dataset.npz" \
  --output_dir "$DATA_ROOT/denoising_20min_hard"

run vis_hard \
  python3 experiments/denoising/scripts/visualize_clean_vs_noisy.py \
  --from_paired "$DATA_ROOT/denoising_20min_hard/paired_dataset_hard.npz" \
  --max_samples 20 \
  --output_dir "$RESULT_ROOT/clean_vs_noisy_hard"

run build_clinical \
  python3 experiments/denoising/scripts/build_dataset.py \
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

run vis_clinical \
  python3 experiments/denoising/scripts/visualize_clean_vs_noisy.py \
  --from_paired "$DATA_ROOT/denoising_20min_clinical/paired_dataset_clinical.npz" \
  --max_samples 20 \
  --output_dir "$RESULT_ROOT/clean_vs_noisy_clinical"

run build_seg_hard \
  python3 experiments/denoising/scripts/build_segmentation_dataset.py \
  --paired "$DATA_ROOT/denoising_20min_hard/paired_dataset_hard.npz" \
  --output_dir "$DATA_ROOT/segmentation_hard"

run train_seg_hard \
  python3 experiments/denoising/scripts/train_segmentation.py \
  --data_dir "$DATA_ROOT/segmentation_hard" \
  --output_dir "$RESULT_ROOT/segmentation_hard" \
  --device "$DEFAULT_DEVICE"

run eval_seg_hard \
  python3 experiments/denoising/scripts/evaluate_segmentation.py \
  --data_dir "$DATA_ROOT/segmentation_hard" \
  --model_path "$RESULT_ROOT/segmentation_hard/best_model.pt" \
  --output_dir "$RESULT_ROOT/segmentation_hard/eval" \
  --device "$DEFAULT_DEVICE"

run build_seg_clinical \
  python3 experiments/denoising/scripts/build_segmentation_dataset.py \
  --paired "$DATA_ROOT/denoising_20min_clinical/paired_dataset_clinical.npz" \
  --output_dir "$DATA_ROOT/segmentation_clinical"

run train_seg_clinical \
  python3 experiments/denoising/scripts/train_segmentation.py \
  --data_dir "$DATA_ROOT/segmentation_clinical" \
  --output_dir "$RESULT_ROOT/segmentation_clinical" \
  --device "$DEFAULT_DEVICE"

run eval_seg_clinical \
  python3 experiments/denoising/scripts/evaluate_segmentation.py \
  --data_dir "$DATA_ROOT/segmentation_clinical" \
  --model_path "$RESULT_ROOT/segmentation_clinical/best_model.pt" \
  --output_dir "$RESULT_ROOT/segmentation_clinical/eval" \
  --device "$DEFAULT_DEVICE"

run build_mseg_hard \
  python3 experiments/denoising/scripts/build_multilabel_segmentation_dataset.py \
  --paired "$DATA_ROOT/denoising_20min_hard/paired_dataset_hard.npz" \
  --output_dir "$DATA_ROOT/multilabel_segmentation_hard"

run train_mseg_hard \
  python3 experiments/denoising/scripts/train_multilabel_segmentation.py \
  --data_dir "$DATA_ROOT/multilabel_segmentation_hard" \
  --output_dir "$RESULT_ROOT/multilabel_segmentation_hard" \
  --device "$DEFAULT_DEVICE"

run eval_mseg_hard \
  python3 experiments/denoising/scripts/evaluate_multilabel_segmentation.py \
  --data_dir "$DATA_ROOT/multilabel_segmentation_hard" \
  --model_path "$RESULT_ROOT/multilabel_segmentation_hard/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_segmentation_hard/eval" \
  --device "$DEFAULT_DEVICE"

run build_mseg_clinical \
  python3 experiments/denoising/scripts/build_multilabel_segmentation_dataset.py \
  --paired "$DATA_ROOT/denoising_20min_clinical/paired_dataset_clinical.npz" \
  --output_dir "$DATA_ROOT/multilabel_segmentation_clinical"

run train_mseg_clinical \
  python3 experiments/denoising/scripts/train_multilabel_segmentation.py \
  --data_dir "$DATA_ROOT/multilabel_segmentation_clinical" \
  --output_dir "$RESULT_ROOT/multilabel_segmentation_clinical" \
  --device "$DEFAULT_DEVICE"

run eval_mseg_clinical \
  python3 experiments/denoising/scripts/evaluate_multilabel_segmentation.py \
  --data_dir "$DATA_ROOT/multilabel_segmentation_clinical" \
  --model_path "$RESULT_ROOT/multilabel_segmentation_clinical/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_segmentation_clinical/eval" \
  --device "$DEFAULT_DEVICE"

run build_denoise_hard \
  python3 experiments/denoising/scripts/build_denoising_dataset.py \
  --paired "$DATA_ROOT/denoising_20min_hard/paired_dataset_hard.npz" \
  --output_dir "$DATA_ROOT/denoising_baseline_hard"

run train_denoise_hard \
  python3 experiments/denoising/scripts/train_denoiser.py \
  --data_dir "$DATA_ROOT/denoising_baseline_hard" \
  --output_dir "$RESULT_ROOT/denoising_baseline_hard" \
  --device "$DEFAULT_DEVICE"

run eval_denoise_hard \
  python3 experiments/denoising/scripts/evaluate_denoiser.py \
  --data_dir "$DATA_ROOT/denoising_baseline_hard" \
  --model_path "$RESULT_ROOT/denoising_baseline_hard/best_model.pt" \
  --output_dir "$RESULT_ROOT/denoising_baseline_hard/eval" \
  --device "$DEFAULT_DEVICE"

run build_denoise_clinical \
  python3 experiments/denoising/scripts/build_denoising_dataset.py \
  --paired "$DATA_ROOT/denoising_20min_clinical/paired_dataset_clinical.npz" \
  --output_dir "$DATA_ROOT/denoising_baseline_clinical"

run train_denoise_clinical \
  python3 experiments/denoising/scripts/train_denoiser.py \
  --data_dir "$DATA_ROOT/denoising_baseline_clinical" \
  --output_dir "$RESULT_ROOT/denoising_baseline_clinical" \
  --device "$DEFAULT_DEVICE"

run eval_denoise_clinical \
  python3 experiments/denoising/scripts/evaluate_denoiser.py \
  --data_dir "$DATA_ROOT/denoising_baseline_clinical" \
  --model_path "$RESULT_ROOT/denoising_baseline_clinical/best_model.pt" \
  --output_dir "$RESULT_ROOT/denoising_baseline_clinical/eval" \
  --device "$DEFAULT_DEVICE"

run build_mgd_hard_pred \
  python3 experiments/denoising/scripts/build_mask_guided_denoising_dataset.py \
  --source_dir "$DATA_ROOT/denoising_baseline_hard" \
  --mask_source pred \
  --segmentation_model "$RESULT_ROOT/multilabel_segmentation_hard/best_model.pt" \
  --output_dir "$DATA_ROOT/multilabel_guided_denoising_hard_pred" \
  --device "$DEFAULT_DEVICE"

run train_mgd_hard_pred \
  python3 experiments/denoising/scripts/train_mask_guided_denoiser.py \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_hard_pred" \
  --mask_source pred \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_hard_pred" \
  --device "$DEFAULT_DEVICE"

run eval_mgd_hard_pred \
  python3 experiments/denoising/scripts/evaluate_mask_guided_denoiser.py \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_hard_pred" \
  --mask_source pred \
  --model_path "$RESULT_ROOT/multilabel_guided_denoising_hard_pred/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_hard_pred/eval" \
  --device "$DEFAULT_DEVICE"

run build_mgd_hard_gt \
  python3 experiments/denoising/scripts/build_mask_guided_denoising_dataset.py \
  --source_dir "$DATA_ROOT/denoising_baseline_hard" \
  --mask_source gt \
  --output_dir "$DATA_ROOT/multilabel_guided_denoising_hard_gt"

run train_mgd_hard_gt \
  python3 experiments/denoising/scripts/train_mask_guided_denoiser.py \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_hard_gt" \
  --mask_source gt \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_hard_gt" \
  --device "$DEFAULT_DEVICE"

run eval_mgd_hard_gt \
  python3 experiments/denoising/scripts/evaluate_mask_guided_denoiser.py \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_hard_gt" \
  --mask_source gt \
  --model_path "$RESULT_ROOT/multilabel_guided_denoising_hard_gt/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_hard_gt/eval" \
  --device "$DEFAULT_DEVICE"

run build_mgd_clinical_pred \
  python3 experiments/denoising/scripts/build_mask_guided_denoising_dataset.py \
  --source_dir "$DATA_ROOT/denoising_baseline_clinical" \
  --mask_source pred \
  --segmentation_model "$RESULT_ROOT/multilabel_segmentation_clinical/best_model.pt" \
  --output_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_pred" \
  --device "$DEFAULT_DEVICE"

run train_mgd_clinical_pred \
  python3 experiments/denoising/scripts/train_mask_guided_denoiser.py \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_pred" \
  --mask_source pred \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_clinical_pred" \
  --device "$DEFAULT_DEVICE"

run eval_mgd_clinical_pred \
  python3 experiments/denoising/scripts/evaluate_mask_guided_denoiser.py \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_pred" \
  --mask_source pred \
  --model_path "$RESULT_ROOT/multilabel_guided_denoising_clinical_pred/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_clinical_pred/eval" \
  --device "$DEFAULT_DEVICE"

run build_mgd_clinical_gt \
  python3 experiments/denoising/scripts/build_mask_guided_denoising_dataset.py \
  --source_dir "$DATA_ROOT/denoising_baseline_clinical" \
  --mask_source gt \
  --output_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_gt"

run train_mgd_clinical_gt \
  python3 experiments/denoising/scripts/train_mask_guided_denoiser.py \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_gt" \
  --mask_source gt \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_clinical_gt" \
  --device "$DEFAULT_DEVICE"

run eval_mgd_clinical_gt \
  python3 experiments/denoising/scripts/evaluate_mask_guided_denoiser.py \
  --data_dir "$DATA_ROOT/multilabel_guided_denoising_clinical_gt" \
  --mask_source gt \
  --model_path "$RESULT_ROOT/multilabel_guided_denoising_clinical_gt/best_model.pt" \
  --output_dir "$RESULT_ROOT/multilabel_guided_denoising_clinical_gt/eval" \
  --device "$DEFAULT_DEVICE"

run compare_hard \
  python3 experiments/denoising/scripts/compare_denoising_results.py \
  --results_root "$RESULT_ROOT" \
  --experiment_tag hard \
  --output "$RESULT_ROOT/comparisons/denoising_comparison_hard.csv"

run compare_clinical \
  python3 experiments/denoising/scripts/compare_denoising_results.py \
  --results_root "$RESULT_ROOT" \
  --experiment_tag clinical \
  --output "$RESULT_ROOT/comparisons/denoising_comparison_clinical.csv"

run infer_real_hard \
  python3 experiments/real_world_inference/run_real_world_inference.py \
  --csv "$CSV" \
  --fetal_dir "$FETAL_DIR" \
  --id_column "$ID_COL" \
  --segmentation_model "$RESULT_ROOT/multilabel_segmentation_hard/best_model.pt" \
  --denoiser_model "$RESULT_ROOT/multilabel_guided_denoising_hard_pred/best_model.pt" \
  --output_dir "$REAL_WORLD_ROOT/hard" \
  --device "$DEFAULT_DEVICE"

run infer_real_clinical \
  python3 experiments/real_world_inference/run_real_world_inference.py \
  --csv "$CSV" \
  --fetal_dir "$FETAL_DIR" \
  --id_column "$ID_COL" \
  --segmentation_model "$RESULT_ROOT/multilabel_segmentation_clinical/best_model.pt" \
  --denoiser_model "$RESULT_ROOT/multilabel_guided_denoising_clinical_pred/best_model.pt" \
  --output_dir "$REAL_WORLD_ROOT/clinical" \
  --device "$DEFAULT_DEVICE"

run mine_double_half \
  python3 experiments/doubling_halving/check_half_and_double.py \
  --data-dir "$FETAL_DIR" \
  --output-dir "$DOUBLE_HALF_RESULT_ROOT/candidate_mining" \
  --workers 8

run synth_double_half \
  python3 experiments/doubling_halving/build_synthetic_doubling_halving_dataset.py \
  --data_path "$DATA_ROOT/denoising_20min/clean_dataset.npz" \
  --output_dir "$DOUBLE_HALF_DATA_ROOT/synthetic_output"

log_master "ALL DONE"
