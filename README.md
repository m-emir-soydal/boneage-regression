# RSNA Pediatric Bone Age Estimation

Modular PyTorch project for pediatric bone age regression using a multi-input
architecture (EfficientNet-B3 + sex). An optional **Uncertainty Quantification
(UQ)** layer under [`UQ/`](UQ/) adds MC Dropout and heteroscedastic regression,
compared on shared PICP/MPIW metrics.

## Project structure

```
boneage-regression/
├── config.py              # Paths (from .env), image size, batch size
├── data_loader.py         # CSV loading, train/val/calib/test splits, DataLoaders
├── model.py               # EfficientNet-B3 multi-input point-regression model
├── metrics.py             # Point metrics + CSV export
├── train.py               # Main training entry point
├── test.py                # Evaluate a checkpoint on the held-out test set
├── run_training.sh        # Conda wrapper for train.py
├── requirements.txt       # Python dependencies
├── .envexample            # Template for .env (copy to .env)
├── tests/
│   └── test_smoke.py      # Quick model forward-pass smoke test
└── UQ/                    # Additive UQ methods (see UQ/README.md)
    ├── compare.py         # Rank all methods on PICP/MPIW
    ├── common/            # Shared UQ metrics, intervals, calibration
    ├── mc_dropout/        # Monte Carlo Dropout
    └── heteroscedastic/   # Heteroscedastic regression (mean + log-variance)
```

## Methodology

### Data split and preprocessing

- **Training source:** `train.csv` + images under `data/train/`.
- **Held-out test:** `test.csv` + images under `data/test/` (RSNA external test set; skipped if missing).
- **Internal split (from training data only):** 50% train / 25% validation / 25% calibration, stratified by sex (`male`).
- **Images:** Resized to 300×300, ImageNet normalization, light augmentation on train.
- **Targets:** Bone age normalized by `max_age` from the training split only.
- **Sex:** Encoded as `0.0` / `1.0`.

### Point-regression model

1. **Image branch:** EfficientNet-B3 (ImageNet weights), classifier replaced with identity → feature vector.
2. **Sex branch:** `Linear(1 → 32)` + ReLU.
3. **Head:** Concatenate → `Linear → 256` + ReLU + Dropout(0.5) → `Linear → 1` (normalized age).
4. **Training:** Adam (lr `1e-4`), SmoothL1 (Huber) loss, checkpoint on best validation MAE.

### Uncertainty quantification (optional)

UQ methods live under [`UQ/`](UQ/) and do **not** modify the core pipeline. Both implemented methods report the same metrics (PICP@90/95, MPIW@90/95, MAE, etc.) and can be ranked with:

```bash
python -m UQ.compare --split test
```

See [`UQ/README.md`](UQ/README.md) for MC Dropout and heteroscedastic regression usage.

---

## Setup on a new machine (after clone)

### 1. Clone the repository

```bash
git clone https://github.com/m-emir-soydal/boneage-regression.git
cd boneage-regression
git checkout emir-uq   # or your working branch
```

### 2. Create and activate the conda environment

```bash
conda create -n rsna-boneage python=3.11 -y
conda activate rsna-boneage
```

### 3. Install PyTorch (GPU or CPU)

Pick the build that matches your machine from [pytorch.org](https://pytorch.org/get-started/locally/).

**GPU (example — adjust CUDA version):**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**CPU only:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare the dataset

Place the RSNA data so the layout matches:

```
data/
├── train.csv
├── train/
│   └── <id>.png
├── test.csv          # optional; required for external test evaluation
└── test/
    └── <id>.png
```

`train.csv` columns: `id`, `boneage`, `male`.  
`test.csv` columns: `Image ID`, `Bone Age (months)`, `male`.

### 6. Configure environment paths

```bash
cp .envexample .env
```

Edit `.env` if needed:

```
DATA_DIR=data
OUTPUT_DIR=outputs
```

Paths can be absolute or relative to the project root.

### 7. Verify the install

```bash
pytest tests/test_smoke.py -q
```

---

## Quickstart — point-regression training

### Quick smoke test (1% data, 2 epochs)

```bash
chmod +x run_training.sh
./run_training.sh --quick-test --output-name smoke_test
```

Or directly:

```bash
conda activate rsna-boneage
python train.py --quick-test --output-name smoke_test --seed 42
```

### Full training

```bash
./run_training.sh --output-name full_train --seed 42 --epochs 50
```

Or:

```bash
python train.py --output-name full_train --seed 42 --epochs 50
```

### Evaluate on the held-out test set

```bash
python test.py \
  --checkpoint outputs/full_train_seed42/best_full_train_seed42_epXX_valX.XXX.pth \
  --output-name full_train
```

### `run_training.sh` options

| Flag | Description |
|------|-------------|
| `--quick-test` | 1% of data, 2 epochs |
| `--output-name NAME` | Run prefix for checkpoints and logs |
| `--seed N` | Torch seed (default `123`) |
| `--env NAME` | Conda env name (default `rsna-boneage`) |

---

## Quickstart — UQ methods

All UQ commands assume `conda activate rsna-boneage` from the project root.

### MC Dropout (wraps an existing point-regression checkpoint)

```bash
python -m UQ.mc_dropout.run_mc_dropout \
  --checkpoint outputs/full_train_seed42/best_*.pth \
  --split calib --fit-calibration

python -m UQ.mc_dropout.run_mc_dropout \
  --checkpoint outputs/full_train_seed42/best_*.pth \
  --split test --calibration auto --samples 30
```

### Heteroscedastic regression (trains its own model)

```bash
python -m UQ.heteroscedastic.train_hetero --output-name hetero --seed 42 --epochs 50

python -m UQ.heteroscedastic.run_hetero \
  --checkpoint outputs/hetero_seed42/best_*.pth \
  --split test --calibration auto
```

One-shot wrapper (train + calib + test):

```bash
bash UQ/heteroscedastic/run_hetero.sh --output-name hetero --seed 42 --epochs 50
```

### Compare methods

```bash
python -m UQ.compare --split test
```

Writes a ranked table to `UQ/results/comparison_<timestamp>.csv`.

---

## Outputs

### Point regression (`OUTPUT_DIR/`, default `outputs/`)

Per run (`<output-name>_seed<seed>/`):

- `best_<run>_epXX_valX.XXX.pth` — best checkpoint (validation MAE)
- `history_<run>.pkl` — training history
- `loss_plot_<run>.png` — loss curves (if matplotlib/seaborn installed)
- `<run>_val_predictions_<ts>.csv`, `<run>_val_metrics_<ts>.csv`
- `<run>_test_predictions_<ts>.csv`, `<run>_test_metrics_<ts>.csv` (if `test.csv` exists)

### UQ (`UQ/results/<method>/`)

- `*_predictions_*.csv` — per-sample means, stds, intervals
- `*_metrics_*.csv` — one row per run (PICP, MPIW, MAE, …)
- `calibration_*.json` — post-hoc temperature scaling (when fitted)

---

## Removing UQ (optional)

The core pipeline does not depend on UQ. To remove it:

```bash
rm -rf UQ/heteroscedastic UQ/results/heteroscedastic
# or remove the entire UQ layer:
rm -rf UQ
```

Point-regression training (`train.py`, `model.py`) continues to work unchanged.
