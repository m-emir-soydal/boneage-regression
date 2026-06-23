# Uncertainty Quantification (UQ)

This folder is an **additive** layer on top of the core pediatric bone age
regression project. The point-regression model (EfficientNet-B3 + sex) stays
exactly as-is; UQ methods wrap inference around a trained checkpoint and are
all compared on the **same metrics** so different approaches can be ranked
fairly.

The implemented method is **Monte Carlo (MC) Dropout**. More methods
(Bayesian neural network, conformal prediction, deep ensembles, ...) plug in
the same way. A `compare.py` aggregates every method's results into one ranked
table on the shared metrics.

## Why a separate folder

- Keeps the production point-regression pipeline (`train.py`, `model.py`,
  `data_loader.py`, `metrics.py`) untouched.
- Groups all UQ code, shared utilities, and results in one place.
- Every method emits the same metrics row schema, so results are directly
  comparable and a future `compare.py` can concatenate them into one table.

## Layout

```
UQ/
├── README.md                  # this file
├── compare.py                 # aggregate all methods into one ranked table
├── common/                    # shared helpers reused by every method
│   ├── paths.py               # puts project root on sys.path; results dirs
│   ├── uq_metrics.py          # PICP@90/95, MPIW@90/95 + point metrics
│   └── inference.py           # enable_dropout, gaussian_intervals, denormalize
├── mc_dropout/
│   └── run_mc_dropout.py      # MC Dropout entry point
└── results/
    ├── mc_dropout/            # grouped CSV outputs for this method
    └── comparison_<ts>.csv    # cross-method comparison table
```

Each method lives in its own folder (`UQ/<method>/`), reuses `common/`, and
writes its outputs to `UQ/results/<method>/`.

## Metrics

All metrics are computed in **de-normalized month space** (the model predicts
a normalized age which is multiplied back by `max_age`).

| Metric | Meaning | Goal |
|--------|---------|------|
| **PICP@90** | Prediction Interval Coverage Probability at 90% | close to 0.90 |
| **PICP@95** | Coverage at 95% | close to 0.95 |
| **MPIW@90** | Mean Prediction Interval Width at 90% (months) | smaller is better (given good coverage) |
| **MPIW@95** | Mean width at 95% (months) | smaller is better |
| MAE / RMSE / R2 | Point-accuracy of the mean prediction | for reference / parity with base model |

A method is "well calibrated" when PICP is near the nominal level; among
methods with comparable coverage, smaller MPIW is preferable (tighter, more
informative intervals).

## Monte Carlo Dropout

MC Dropout keeps `nn.Dropout` layers **active at inference** and runs `T`
stochastic forward passes. The per-sample mean is the point prediction and the
per-sample standard deviation drives Gaussian prediction intervals
(`mean +/- z * std`, with z = 1.645 for 90% and 1.96 for 95%).

> Note: the regression model has a single `nn.Dropout(0.5)` layer in its head
> (the EfficientNet classifier was replaced with `nn.Identity`). MC Dropout
> variance is therefore driven by that one layer. The model is intentionally
> left unchanged to keep the "same regression model" comparison valid.

### Run it

Reuse a checkpoint produced by the base `train.py`:

```bash
conda activate <env>   # the env used for training (has torch/torchvision)

python -m UQ.mc_dropout.run_mc_dropout \
    --checkpoint outputs/<run_name>/best_<run_name>_epXX_valX.XXX.pth \
    --samples 30 \
    --split test
```

Arguments:

- `--checkpoint` (required): path to a trained `.pth` model.
- `--samples` (default 30): number of stochastic forward passes `T`.
- `--split` (default `test`): one of `train` / `val` / `calib` / `test`. The
  same deterministic split as training is reproduced.
- `--output-name` (default `mc_dropout`): output file prefix.

### Outputs

Written to `UQ/results/mc_dropout/`:

- `mc_dropout_<split>_predictions_<timestamp>.csv`
  columns: `id, sex, true_age, pred_mean, pred_std, lower90, upper90,
  lower95, upper95, covered90, covered95`
- `mc_dropout_<split>_metrics_<timestamp>.csv`
  one comparison row (see schema below).

## Comparing methods

Once two or more methods have written metrics, aggregate them:

```bash
python -m UQ.compare              # all methods, all splits
python -m UQ.compare --split test # restrict to one split
```

`compare.py` scans `UQ/results/*/*_metrics_*.csv`, keeps the most recent row
per `(method, split)`, prints a ranked table (best-calibrated first — smallest
`|PICP@90 - 0.90|` — then tightest `MPIW@90`), and saves
`UQ/results/comparison_<timestamp>.csv`.

## Comparison row schema

Every method's metrics CSV shares this schema, which is the key to comparing
methods:

```
method, split, MAE, RMSE, MSE, R2, pct_within_6mo, pct_within_12mo,
PICP_90, PICP_95, MPIW_90, MPIW_95, n, <method-specific extras>, timestamp
```

`<method-specific extras>` are optional (for MC Dropout: `samples`,
`checkpoint`). A future `UQ/compare.py` can simply concatenate every
`UQ/results/*/*_metrics_*.csv` into a single table.

## Adding a new UQ method

1. Create `UQ/<method>/` with a `run_<method>.py` entry point.
2. Reuse `UQ/common/` for path setup, interval construction, and metrics.
3. Reuse the unchanged core model/data via `model.py` and `data_loader.py`.
4. Write outputs to `UQ/results/<method>/`.
5. Emit the shared metrics row via `common.uq_metrics.summarize(...)` so the
   results stay comparable.

## Roadmap

- **Bayesian neural network** — to be reimplemented (a previous variational
  last-layer head was removed pending a better implementation).
- **Conformal prediction** — distribution-free intervals calibrated on the
  currently-unused `calib_df` split from `data_loader.load_data()`.
- **Deep ensembles** — train multiple seeds (the project already supports
  per-seed runs) and aggregate predictive variance.

Implemented: **MC Dropout**.
