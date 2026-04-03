# Real Doubling / Halving Candidate Mining

本目录用于从真实 `batch1` 原始 `.fetal` 数据中自动挖掘 doubling / halving 候选片段。

默认输出：

- `summary.csv`
- `candidate_segments.npz`
- `batch_stats.json`
- `run_config.json`
- `plots/`

运行示例：

```bash
python /home/yt/CTG_test/doubling_halving/check_half_and_double.py \
  --file /path/to/sample.fetal \
  --output-dir /tmp/dh_single
```

```bash
python /home/yt/CTG_test/doubling_halving/check_half_and_double.py \
  --data-dir /scratch2/yzd/CTG/batch1/fetal \
  --output-dir /home/yt/CTG_test/doubling_halving/output \
  --workers 8
```

如果环境里设置了 `CTG_BATCH1_FETAL_DIR`，批处理可直接不传 `--data-dir`。

`candidate_segments.npz` 需使用 `allow_pickle=True` 读取，因为片段数组按候选保存为 object array：

```python
data = np.load("candidate_segments.npz", allow_pickle=True)
raw_seg = data["raw_segment"][0]
```
