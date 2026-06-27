# Calibration/Scale Sweep

This experiment reuses the already trained EfficientNet models from
`results/cps/full_2`. It does not retrain models.

For each seed, the saved `scale` and `cal` predictions are merged into one
post-training CP pool. The sweep then changes how much of that pool is used for
conformal calibration.

Default sweep:

```text
cal fractions: 7.5%, 10%, 12.5%, 15%, 17.5%, 20%, 22.5%, 25%
KNN total budget: 25% = scale + cal
```

Method usage:

```text
SCP:     uses cal_fraction only
AS-MCP:  uses cal_fraction only
KNN-NCP: uses cal_fraction for conformal scores and the remaining KNN budget for scale
```

Run the default full sweep:

```bash
python -m sweep_test.run
```

Run a fast smoke check:

```bash
python -m sweep_test.run --seeds 0 --cal-fractions 0.075 0.10 --confidences 0.85 --no-save-intervals --output-dir /tmp/sweep_test_smoke
```

Outputs are written under `sweep_test/results/full_2` by default:

```text
config.json
metrics_long.csv
metrics_summary.csv
runtime_long.csv
runtime_summary.csv
subset_counts.csv
skipped.csv
intervals/intervals_seed_XX.parquet
```

Build CSV/LaTeX metric and runtime tables:

```bash
python -m sweep_test.tables
```

Build figures:

```bash
python -m sweep_test.plots
```

The table outputs live under `tables/`; figures live under `figures/`.

Run the complete sweep + tables + figures:

```bash
./sweep_test.sh
```

By default, interval files are skipped to keep the run lighter. To also save
per-seed interval files:

```bash
SAVE_INTERVALS=1 ./sweep_test.sh
```
