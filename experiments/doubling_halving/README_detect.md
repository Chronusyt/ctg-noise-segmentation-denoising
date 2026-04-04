# Real Doubling / Halving Candidate Mining

```bash
python3 experiments/doubling_halving/check_half_and_double.py \
  --data-dir /scratch2/yzd/CTG/batch1/fetal \
  --output-dir artifacts/results/doubling_halving/candidate_mining \
  --workers 8
```

输出内容包括：

- `summary.csv`
- `candidate_segments.npz`
- `plots/`
- `run_config.json`

