# Uncertainty Quantification (UQ)

This folder is an **additive** layer on top of the core pediatric bone age
regression project. The point-regression model (EfficientNet-B3 + sex) stays
exactly as-is; UQ methods wrap inference around a trained checkpoint and are
all compared on the **same metrics** so different approaches can be ranked
fairly.

Currently implemented methods: **Monte Carlo (MC) Dropout** and
**Heteroscedastic Regression**. More methods (Bayesian neural network,
conformal prediction, deep ensembles, ...) plug in the same way. A
`compare.py` aggregates every method's results into one ranked table on the
shared metrics.

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
│   ├── inference.py           # enable_dropout, gaussian_intervals, denormalize
│   └── calibration.py         # post-hoc std temperature (fit/apply/save/load)
├── mc_dropout/
│   ├── run_mc_dropout.py      # MC Dropout evaluation (+ post-hoc calibration)
│   └── train_mc_dropout.py    # UQ-aware training loop (calibration-aware)
├── heteroscedastic/
│   ├── hetero_model.py        # EfficientNet-B3 + sex, with mean + log_var heads
│   ├── losses.py              # Gaussian NLL (+ optional SmoothL1 on the mean)
│   ├── train_hetero.py        # Heteroscedastic training (single-pass UQ model)
│   ├── run_hetero.py          # Single-pass evaluation (+ post-hoc calibration)
│   └── run_hetero.sh          # conda-activating train+calib+test wrapper
└── results/
    ├── mc_dropout/            # grouped CSV outputs for this method
    ├── heteroscedastic/       # grouped CSV outputs for this method
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
- `--fit-calibration`: fit a post-hoc std temperature on **this** split (use
  with `--split calib`) and save it for later reuse. See below.
- `--calibration PATH|auto`: apply a previously saved temperature before
  building intervals (`auto` picks the most recent one).

### Countering low coverage: post-hoc calibration

Raw MC Dropout intervals here **under-cover** (PICP@90 ≈ 0.79 instead of 0.90)
because the single head-dropout layer produces a `std` that is too small. The
cheapest fix is **temperature scaling**: learn one scalar `s` on the held-out
`calib` split so that `mean ± z * (s * std)` reaches the nominal coverage, then
apply that same `s` to the test split.

`s` is the maximum-likelihood Gaussian scale of the standardized residuals,
`s = sqrt(mean(((y - mean) / std)^2))` (computed in months). `s = 1` means the
raw std already matched the residuals; `s > 1` widens the intervals.

```bash
# 1. Fit the temperature on the calibration split (saves a JSON under
#    UQ/results/mc_dropout/calibration_<ts>.json)
python -m UQ.mc_dropout.run_mc_dropout \
    --checkpoint outputs/<run>/best_*.pth --split calib --fit-calibration

# 2. Evaluate the test split with that temperature applied
python -m UQ.mc_dropout.run_mc_dropout \
    --checkpoint outputs/<run>/best_*.pth --split test --calibration auto
```

The metrics row records `std_scale` and `calibrated` so calibrated and
uncalibrated runs are distinguishable in `compare.py`.

### Outputs

Written to `UQ/results/mc_dropout/`:

- `mc_dropout_<split>_predictions_<timestamp>.csv`
  columns: `id, sex, true_age, pred_mean, pred_std, std_scale, lower90,
  upper90, lower95, upper95, covered90, covered95` (intervals reflect the
  applied `std_scale`; `pred_std` is the raw dropout std)
- `calibration_<timestamp>.json` (when `--fit-calibration` is used): the saved
  std temperature plus metadata, reusable via `--calibration`
- `mc_dropout_<split>_metrics_<timestamp>.csv`
  one comparison row (see schema below).

### UQ-aware training (Option 1)

Post-hoc calibration pins coverage to the nominal level, but it cannot make the
intervals *tighter* — that is decided at training time. `train_mc_dropout.py` is
a sibling of the base `train.py` that trains the **same** model while optimizing
for uncertainty quality:

- `--dropout` is configurable (the base trainer hard-codes 0.5), since MC
  variance is driven entirely by the head dropout layer.
- After each epoch it runs MC Dropout on the `calib` split, fits the temperature
  above, and scores the epoch by the **calibrated** interval quality instead of
  plain val MAE. Because calibration already fixes coverage, the default
  `combo` criterion minimizes calibrated `MPIW@90` (tighter intervals) plus a
  small MAE penalty so point accuracy does not regress.

```bash
python -m UQ.mc_dropout.train_mc_dropout \
    --output-name uqtrain --dropout 0.5 --epochs 50 \
    --select-by combo --mc-eval-samples 10 --mc-eval-every 1
```

Key arguments:

- `--select-by` (default `combo`): `val_mae` (base behavior), `picp90`
  (minimize `|PICP@90 - 0.90|`), `mpiw90` (tightest calibrated intervals), or
  `combo` (`MPIW@90 + mae_weight * MAE`).
- `--mae-weight` (default 0.01): weight on calib MAE (months) in `combo`.
- `--mc-eval-samples` (default 10): MC passes for the per-epoch calib eval
  (kept small for speed; the final calibration uses `--final-samples`, 30).
- `--mc-eval-every` (default 1): run the (expensive) MC calib eval every k
  epochs.

On completion it restores the best checkpoint, fits the definitive temperature
on `calib`, and writes it to `UQ/results/mc_dropout/` so the evaluation step
below can reuse it with `--calibration auto`:

```bash
python -m UQ.mc_dropout.run_mc_dropout \
    --checkpoint outputs/uqtrain_seed42/best_*.pth \
    --split test --calibration auto
```

## Heteroscedastic Regression

Heteroscedastic regression keeps the **same backbone** (EfficientNet-B3 + sex)
but replaces the single regression head with two heads:

- `mean_out` predicts the normalized bone age (same target as the base model).
- `log_var_out` predicts the log-variance of a per-sample Gaussian
  likelihood, so `sigma(x) = exp(0.5 * log_var)` varies with the input.

Trained with the Gaussian negative log-likelihood

```
L = 0.5 * ( exp(-log_var) * (y - mean)^2 + log_var )
```

(optionally combined with a small SmoothL1 term on the mean), it learns
**input-dependent aleatoric uncertainty**: wider intervals on harder
radiographs, tighter on easier ones. Inference is a **single deterministic
forward pass** — no MC sampling required — so it is much cheaper at eval time
than MC Dropout.

> Note: this method models aleatoric (data) uncertainty only. It does not
> capture epistemic uncertainty about the model weights themselves. MC
> Dropout / deep ensembles / BNNs cover that space. The methods are
> complementary, not redundant.

The package is fully self-contained under [`heteroscedastic/`](heteroscedastic/);
removing that folder plus `results/heteroscedastic/` removes the method
without touching the core training pipeline or other UQ methods.

### Run it

```bash
conda activate rsna-boneage   # same env as base train.py

# 1. Train (writes checkpoint under OUTPUT_DIR/<run_name>/)
python -m UQ.heteroscedastic.train_hetero \
    --output-name hetero --seed 42 --epochs 50

# 2. Fit calibration on the held-out calib split
python -m UQ.heteroscedastic.run_hetero \
    --checkpoint outputs/hetero_seed42/best_*.pth \
    --split calib --fit-calibration

# 3. Evaluate the test split with that calibration applied
python -m UQ.heteroscedastic.run_hetero \
    --checkpoint outputs/hetero_seed42/best_*.pth \
    --split test --calibration auto
```

Or one shot via the wrapper (activates conda, trains, fits, evaluates):

```bash
bash UQ/heteroscedastic/run_hetero.sh --output-name hetero --seed 42 --epochs 50
```

### Training arguments

- `--output-name` (default `hetero`): run prefix; checkpoints saved under
  `outputs/<output-name>_seed<seed>/`.
- `--seed` (default 42), `--epochs` (default 50), `--lr` (default 1e-4),
  `--dropout` (default 0.5): same semantics as the base trainer.
- `--quick-test`: 1% data, 2 epochs (smoke test).
- `--mae-weight` (default 0.01): weight of an auxiliary SmoothL1 term on the
  mean head added to NLL. `0` is pure NLL. A small positive value preserves
  point accuracy when NLL alone would let MAE drift.
- `--select-by` (default `val_mae`): checkpoint selection criterion. Set to
  `mpiw90` or `combo` to select for calibrated interval quality on the
  calib split (mirrors the MC Dropout UQ-aware trainer).
- `--calib-eval-every` (default 1): only used when `--select-by != val_mae`.

After training, the final calibration on the calib split is written to
`UQ/results/heteroscedastic/calibration_<ts>.json` so the evaluation step
can pick it up with `--calibration auto`.

### Evaluation arguments

Same shape as `run_mc_dropout.py`:

- `--checkpoint` (required): trained heteroscedastic `.pth`.
- `--split` (default `test`): `train` / `val` / `calib` / `test`.
- `--fit-calibration`: fit a fresh temperature on **this** split (use with
  `--split calib`).
- `--calibration PATH|auto`: apply a saved temperature before building
  intervals (`auto` = newest one).

### Outputs

Written to `UQ/results/heteroscedastic/`:

- `heteroscedastic_<split>_predictions_<ts>.csv` — same columns as the
  MC Dropout predictions CSV (`id, sex, true_age, pred_mean, pred_std,
  std_scale, lower90, upper90, lower95, upper95, covered90, covered95`).
- `heteroscedastic_<split>_metrics_<ts>.csv` — one comparison row using the
  shared schema below.
- `calibration_<ts>.json` (with `--fit-calibration`).

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
`checkpoint`, `std_scale`, `calibrated`). A future `UQ/compare.py` can simply
concatenate every `UQ/results/*/*_metrics_*.csv` into a single table.

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

Implemented: **MC Dropout**, **Heteroscedastic Regression**.
