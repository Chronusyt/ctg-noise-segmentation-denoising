# Clinical Physiological Multitask Dataset v1

This document describes the clinical multitask data closure used for the next
physiological multitask stage. It does not introduce a new model or training
pipeline.

## Scope

- Data line: clinical synthetic denoising data only.
- Source split: existing clinical train / val / test split.
- Label source: physiological labels are computed from `clean_signals`, not
  from noisy signals or reconstructed outputs.
- Output directory:
  `artifacts/datasets/denoising/clinical_multitask_physiology_v1/`

## Fields

- `noisy_signals`: float32, shape `[N, 240]`, noisy FHR input.
- `clean_signals`: float32, shape `[N, 240]`, clean reconstruction target.
- `masks`: float32, shape `[N, 240, 5]`, GT multilabel artifact masks.
- `pred_masks`: float32, shape `[N, 240, 5]`, predicted multilabel artifact masks when available.
- `artifact_labels`: float32, shape `[N, 240, 5]`, GT artifact labels.
- `baseline_labels`: float32, shape `[N]`, mean optimized baseline in bpm.
- `stv_labels`: float32, shape `[N]`, event-excluded STV in ms.
- `ltv_labels`: float32, shape `[N]`, event-excluded LTV in ms.
- `baseline_variability_labels`: float32, shape `[N]`, continuous mean baseline-variability amplitude in bpm.
- `baseline_variability_class_labels`: int32, shape `[N]`, auxiliary class id: `-1 unknown`, `0 absent`, `1 minimal/reduced`, `2 moderate/normal`, `3 marked`.
- `acc_labels`: uint8, shape `[N, 240]`, pointwise acceleration labels.
- `dec_labels`: uint8, shape `[N, 240]`, pointwise deceleration labels.
- `acc_counts`: int32, shape `[N]`, number of detected acceleration events.
- `dec_counts`: int32, shape `[N]`, number of detected deceleration events.
- `parent_index`: int32, shape `[N]`, parent 20-minute segment index.
- `chunk_index`: int32, shape `[N]`, one-minute chunk index within parent.

## Label Definitions

- Reconstruction target: `clean_signals`.
- Baseline: mean of the optimized FHR baseline trace from clean FHR.
- STV / LTV: existing pulse-interval STV/LTV implementation, computed after
  excluding detected acceleration/deceleration regions.
- Baseline variability: v1 uses a continuous regression label, defined as the
  mean robust one-minute baseline-deviation amplitude in bpm. The class label is
  saved as auxiliary metadata rather than used as the main v1 target.
- Acceleration / deceleration: pointwise binary labels generated from clean FHR
  with the existing FIGO 15x15 detectors.

## Loader

Use `ctg_pipeline.data.multitask_dataset.ClinicalMultitaskDataset`.

Each item returns at least:

- `noisy_signal`: float32 `[1, L]`
- `clean_signal`: float32 `[1, L]`
- `mask` / `multilabel_mask`: float32 `[5, L]`
- `pred_mask`: float32 `[5, L]` when present
- `baseline_label`: float32 scalar
- `stv_label`: float32 scalar
- `ltv_label`: float32 scalar
- `baseline_variability_label`: float32 scalar
- `baseline_variability_class_label`: int64 scalar when present
- `acc_label`: float32 `[1, L]`
- `dec_label`: float32 `[1, L]`

## Build And Check

Build:

```bash
python3 experiments/denoising/scripts/build_clinical_multitask_dataset.py
```

Check:

```bash
python3 experiments/denoising/scripts/check_clinical_multitask_dataset.py
```

Reports are written under:

```text
artifacts/results/summary/clinical_main/
```
