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

### Data Split & Preprocessing
- **Dataset:** The dataset is split into **80% Training** and **20% Validation**.
- **Stratification:** The split is stratified based on the `male` column (sex) to ensure gender balance across both sets.
- **Image Preprocessing:** Images are resized to **300x300** and preprocessed using standard EfficientNet specific transformations.
- **Normalization:** 
  - The target variable (`boneage`) is normalized by dividing by the maximum age in the training set to stabilize regression.
  - The `sex` input is represented as a numeric float (0.0 or 1.0).

### Model Architecture
The problem is approached as a regression task using a multi-input architecture:
1. **Image Branch:** An **EfficientNet-B3** base model (initialized with ImageNet weights, excluding top layers) extracts visual features, followed by a `GlobalAveragePooling2D` layer.
2. **Tabular Branch:** The normalized `sex` input is explicitly passed as an auxiliary input.
3. **Fusion & Prediction Head:** 
   - The extracted image features and the sex feature are concatenated.
   - The combined vector is passed through a `Dense` layer (256 units, ReLU activation) and a `Dropout` layer (rate: 0.5).
   - Finally, a single-unit `Dense` layer (linear activation) outputs the predicted continuous bone age.
4. **Optimization:** The model is optimized using **Adam** (learning rate: 1e-4) with **Mean Absolute Error (MAE)** as the primary loss function.

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

After training, the script will generate the following in your `OUTPUT_DIR` (default: `outputs/`):
- `best_<NAME>_XX-XXX.h5`: The best model checkpoint based on validation MAE.
- `history_<NAME>.pkl`: Saved training metrics.
- `<NAME>_val_predictions_<TIMESTAMP>.csv`: DataFrame with actual vs. predicted ages for the validation set.
- `<NAME>_val_metrics_<TIMESTAMP>.csv`: Key metrics summary (MAE, MSE, RMSE, R², etc.).
