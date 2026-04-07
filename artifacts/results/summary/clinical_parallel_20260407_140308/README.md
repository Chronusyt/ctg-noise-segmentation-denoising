# Clinical Parallel Rerun Snapshot - 2026-04-07 14:03:08

This directory freezes the completed clinical main rerun with run id:

`clinical_parallel_20260407_140308`

Run window:

- Start: 2026-04-07 14:03:08 CST
- End: 2026-04-07 14:38:20 CST

Pipeline:

- Rebuilt pred-mask guided clinical dataset from `denoising_baseline_clinical`
- Rebuilt GT-mask guided clinical dataset from `denoising_baseline_clinical`
- Trained and evaluated direct denoising
- Trained and evaluated pred-mask guided denoising
- Trained and evaluated GT-mask oracle denoising
- Regenerated clinical comparison table
- Regenerated clinical main table and comparison figure

Frozen outputs:

- `denoising_comparison_clinical.csv`
- `denoising_comparison_clinical.md`
- `denoising_comparison_clinical.json`
- `denoising_comparison_clinical.txt`
- `clinical_main_results.csv`
- `clinical_main_results.md`
- `clinical_main_results.json`
- `clinical_main_comparison_sample_30649.png`
- `direct_test_metrics.json`
- `pred_mask_test_metrics.json`
- `gt_mask_test_metrics.json`
- `clinical_parallel_20260407_140308_main.log`
- `clinical_parallel_20260407_140308_direct.log`
- `clinical_parallel_20260407_140308_pred.log`
- `clinical_parallel_20260407_140308_gt.log`

Key metrics:

| method | overall_mse | corrupted_region_mse | clean_region_mse | overall_mae | corrupted_region_mae | clean_region_mae | baseline_mae | stv_mae | ltv_mae |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct denoising baseline | 0.1966 | 10.2344 | 0.1209 | 0.2059 | 2.1309 | 0.1914 | 0.0977 | 0.1334 | 0.6881 |
| Multilabel pred-mask guided | 0.1822 | 9.5503 | 0.1116 | 0.1856 | 2.0771 | 0.1713 | 0.0896 | 0.0961 | 0.4921 |
| Multilabel GT-mask oracle | 0.1281 | 8.9142 | 0.0618 | 0.2166 | 1.9886 | 0.2032 | 0.0632 | 0.1437 | 0.6943 |

Notes:

- This snapshot should be treated as the stable record for this rerun.
- Later live outputs under `artifacts/results/denoising/` may be overwritten by new runs.
- The snapshot includes the run logs so checkpoint selection and training dynamics can be inspected later.
