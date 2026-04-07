| method | overall_mse | corrupted_region_mse | clean_region_mse | overall_mae | corrupted_region_mae | clean_region_mae | baseline_mae | stv_mae | ltv_mae | baseline_bias_mean | baseline_bias_median | stv_bias_mean | stv_bias_median | ltv_bias_mean | ltv_bias_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Direct denoising baseline | 0.1966 | 10.2344 | 0.1209 | 0.2059 | 2.1309 | 0.1914 | 0.0977 | 0.1334 | 0.6881 | 0.0064 | 0.0000 | -0.0963 | -0.0465 | -0.5256 | -0.4073 |
| Multilabel pred-mask guided | 0.1822 | 9.5503 | 0.1116 | 0.1856 | 2.0771 | 0.1713 | 0.0896 | 0.0961 | 0.4921 | -0.0175 | 0.0000 | 0.0034 | 0.0091 | -0.0296 | 0.0383 |
| Multilabel GT-mask oracle | 0.1281 | 8.9142 | 0.0618 | 0.2166 | 1.9886 | 0.2032 | 0.0632 | 0.1437 | 0.6943 | 0.0095 | 0.0000 | 0.0871 | 0.0847 | 0.4815 | 0.4777 |
