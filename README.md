# Refactored RSNA Pediatric Bone Age Estimation

This project is a modularized version of the original monolithic notebook for pediatric bone age estimation using a multi-input architecture (EfficientNet-B3 + Sex information).

## Project Structure
```
refactored/
├── config.py         # Handles environment variables and base settings.
├── data_loader.py    # Loads dataset, performs train/val split, creates tf.data sets.
├── model.py          # Defines the EfficientNet-B3 multi-input architecture.
├── metrics.py        # Evaluates the model, de-normalizes predictions, and exports results.
├── train.py          # Main entry point. Handles the training loop and callbacks.
├── run_training.sh   # Bash script for easy execution.
├── tests/            # Contains smoke tests.
│   └── test_smoke.py
├── .env              # Environment configurations (data paths).
└── requirements.txt  # Project dependencies.
```

## Methodology

This project performs **Conformalized Quantile Regression (CQR)** for bone-age
estimation: the network predicts a set of quantiles, and a held-out calibration
split is used to conformalize the resulting intervals to a target coverage.

### Data Split & Preprocessing
- **Dataset:** The source `train.csv` is split **50 / 25 / 12.5 / 12.5** into
  `train` / `val` / `scale` / `cal`, stratified by the `male` column (sex). The
  source `val.csv` is the held-out **TEST** set. `scale` is kept for layout
  compatibility but is unused by CQR.
- **Per-seed splits:** With `PER_SEED_SPLITS = True` (config), each seed gets its
  own stratified partition (`random_state = seed`).
- **Calibration:** `cal` is the conformal calibration split used to derive the
  CQR correction `q_hat` per confidence level.
- **Image Preprocessing:** Images are resized to **300x300** with standard
  EfficientNet transforms.
- **Normalization:** The target (`boneage`) is divided by the max age of the
  training split; quantile regression and conformalization happen in this
  normalized space and intervals are de-normalized to months for reporting.

### Model Architecture
A multi-input quantile-regression network:
1. **Image Branch:** **EfficientNet-B3** (ImageNet weights) extracts visual features.
2. **Tabular Branch:** The normalized `sex` input passes through a small dense layer.
3. **Fusion & Quantile Head:** Image + sex features are concatenated, passed
   through a `Linear(256) + ReLU + Dropout(0.5)` block, then a final
   `Linear(n_quantiles)` head outputs one value per quantile level.
4. **Confidence levels:** `CONFIDENCES = [0.85, 0.90, 0.95]` → quantiles
   `[0.025, 0.05, 0.075, 0.5, 0.925, 0.95, 0.975]` (median included).
5. **Optimization:** **Adam** (lr 1e-4) minimizing the **pinball (quantile)**
   loss; the best checkpoint is selected on validation pinball loss.

### Conformalization
After training, on the `cal` split the conformity score
`E = max(q_lo - y, y - q_hi)` is computed per confidence level, and
`q_hat` is its `(1-α)(1 + 1/n)` empirical quantile. The calibrated interval is
`[q_lo - q_hat, q_hi + q_hat]`, de-normalized and clipped to `[Y_MIN, Y_MAX]`.
Reported metrics: median MAE/RMSE/R², plus per-level empirical coverage and
mean interval width.

## Setup & Installation

1. This project requires an existing conda environment. By default, it expects `rsna-boneage`. If you do not have it, create it and install the requirements:
   ```bash
   conda create -n rsna-boneage python=3.11
   conda activate rsna-boneage
   pip install -r requirements.txt
   ```

2. Configure your environment paths in `.env`:
   - `DATA_DIR`: Path to the raw data folder containing `train.csv` and images.
   - `OUTPUT_DIR`: Directory where model weights and evaluation metrics will be saved.

## Quickstart

You can use the `run_training.sh` shell script to start the training process. 

### Quick Test
To verify everything works correctly on a very small subset of data (1% sample, 2 epochs):
```bash
chmod +x run_training.sh
./run_training.sh --quick-test --output-name smoke_test

python test.py --checkpoint <path.pth> --output-name run_v1
```

### Full Training
To run the full training pipeline:
```bash
./run_training.sh --output-name full_train_v1
```

### Options
- `--quick-test` : Run a lightweight test pipeline.
- `--output-name <NAME>` : Prefix assigned to the output checkpoints, histories, and CSV files.
- `--env <ENV_NAME>` : Override the default conda environment (`rsna-boneage`).

## Outputs

After training, the script generates the following under `OUTPUT_DIR/<NAME>/`:
- `best_<NAME>_epXX_valX.XXXX.pth`: Best checkpoint by validation pinball loss.
- `history_<NAME>.pkl`: Train/val pinball-loss history.
- `q_hat_<NAME>.json`: CQR corrections per confidence level (normalized + months).
- `<NAME>_<split>_predictions_<TS>.csv`: Per-sample median prediction plus
  `lo_<c>` / `hi_<c>` / `covered_<c>` interval columns for each confidence.
- `<NAME>_<split>_point_metrics_<TS>.csv`: MAE / RMSE / MSE / R² on the median.
- `<NAME>_<split>_interval_metrics_<TS>.csv`: Coverage and mean width per level.

To evaluate a checkpoint on TEST (re-calibrates on the matching seed's `cal` split):
```bash
python test.py --checkpoint <path.pth> --output-name run_v1 --seed <SEED>
```
