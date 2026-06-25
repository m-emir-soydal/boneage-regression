"""Heteroscedastic regression evaluation for UQ.

Loads a trained heteroscedastic checkpoint (see ``train_hetero.py``), runs a
**single deterministic forward pass** that returns ``(mean, log_var)`` for
each sample, builds Gaussian prediction intervals ``mean +/- z * sigma(x)``,
optionally applies a saved post-hoc temperature, and writes the same shared
metrics row schema as the other UQ methods so ``UQ.compare`` can rank them
on identical PICP/MPIW columns.

Example:
    # Fit calibration on the held-out calib split (saves a JSON in
    # UQ/results/heteroscedastic/ for later auto-reuse).
    python -m UQ.heteroscedastic.run_hetero \\
        --checkpoint outputs/hetero_seed42/best_*.pth \\
        --split calib --fit-calibration

    # Evaluate the test split using the saved temperature.
    python -m UQ.heteroscedastic.run_hetero \\
        --checkpoint outputs/hetero_seed42/best_*.pth \\
        --split test --calibration auto
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Make both the project root and the UQ package importable whether run as a
# module or as a plain script. Mirrors run_mc_dropout.py.
_THIS = Path(__file__).resolve()
_UQ_DIR = _THIS.parent.parent
_PROJECT_ROOT = _UQ_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_UQ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from common.calibration import (  # noqa: E402
    apply_temperature,
    fit_temperature,
    resolve_calibration,
    save_calibration,
)
from common.inference import denormalize, gaussian_intervals  # noqa: E402
from common.paths import ensure_project_on_path, method_results_dir  # noqa: E402
from common.uq_metrics import summarize  # noqa: E402

ensure_project_on_path()

from data_loader import build_eval_loader, load_data  # noqa: E402

from .hetero_model import build_hetero_model  # noqa: E402
from .losses import clamp_log_var  # noqa: E402

METHOD_NAME = "heteroscedastic"


def select_split(split, train_df, val_df, calib_df, test_df):
    mapping = {
        "train": train_df,
        "val": val_df,
        "calib": calib_df,
        "test": test_df,
    }
    if split not in mapping:
        raise ValueError(f"Unknown split '{split}'; expected one of {sorted(mapping)}")
    return mapping[split]


@torch.no_grad()
def hetero_forward(model, loader, device):
    """Single deterministic pass; returns ``(mean, std)`` arrays (normalized).

    Unlike MC Dropout, we only need to pass each input through the network once
    because the network outputs the standard deviation (from log-variance)
    directly as part of its predictions.
    """
    model.eval()
    means = []
    stds = []
    for batch in loader:
        inputs, _ = batch
        img = inputs["image_input"].to(device)
        sex = inputs["sex_input"].to(device)
        
        # 1. Predict mean and log-variance
        mean, log_var = model(img, sex)
        
        # 2. Clamp log-variance to avoid numerical instability
        log_var = clamp_log_var(log_var)
        
        # 3. Convert log-variance to standard deviation: std = exp(0.5 * log_var)
        std = torch.exp(0.5 * log_var)
        
        means.append(mean.cpu().numpy().reshape(-1))
        stds.append(std.cpu().numpy().reshape(-1))
    return np.concatenate(means), np.concatenate(stds)


def main():
    parser = argparse.ArgumentParser(
        description="Heteroscedastic regression UQ evaluation"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained heteroscedastic .pth checkpoint")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "calib", "test"],
                        help="Which data split to evaluate on")
    parser.add_argument("--output-name", type=str, default=METHOD_NAME,
                        help="Prefix for output files")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate used at training time (must match "
                             "the checkpoint architecture; doesn't affect "
                             "eval because dropout is off in eval mode)")
    parser.add_argument("--fit-calibration", action="store_true",
                        help="Fit a post-hoc std temperature on THIS split "
                             "(run with --split calib) and save it for reuse")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Apply a saved std temperature before building "
                             "intervals: a path to a calibration JSON or "
                             "'auto' to use the most recent one")
    args = parser.parse_args()

    # 1. Load splits and select the target split for evaluation
    print("Loading data...")
    train_df, val_df, calib_df, test_df, max_age = load_data(sample_frac=1.0)
    df = select_split(args.split, train_df, val_df, calib_df, test_df)
    print(f"Evaluating on '{args.split}' split: {len(df)} samples (max_age={max_age})")

    loader = build_eval_loader(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize the model and load the weights
    print(f"Loading model weights from {args.checkpoint}")
    model = build_hetero_model(dropout=args.dropout)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    # 3. Perform a single deterministic forward pass to get mean and standard deviation
    print("Running single deterministic forward pass...")
    mean_norm, std_norm = hetero_forward(model, loader, device)

    # 4. De-normalize predictions from normalized [0, 1] space to months
    pred_mean = denormalize(mean_norm, max_age)
    pred_std = denormalize(std_norm, max_age)

    # Ground truth labels (unshuffled)
    y_true = df["boneage_norm"].values * max_age

    out_dir = method_results_dir(METHOD_NAME)

    # 5. Apply the calibration temperature.
    # Raw predicted standard deviations are often over- or under-confident.
    # The temperature scale multiplier resolves this by scaling standard deviation:
    # `std_eff = pred_std * std_scale`
    std_scale = 1.0
    if args.calibration:
        payload, cal_path = resolve_calibration(args.calibration, out_dir)
        std_scale = float(payload["std_scale"])
        print(f"Applying calibration temperature {std_scale:.4f} from {cal_path}")
    std_eff = apply_temperature(pred_std, std_scale)

    # 6. Generate Gaussian prediction intervals: lower/upper = mean +/- Z * std_eff
    lower90, upper90 = gaussian_intervals(pred_mean, std_eff, 90)
    lower95, upper95 = gaussian_intervals(pred_mean, std_eff, 95)

    # 7. Fit a new calibration temperature on this split if requested
    if args.fit_calibration:
        fitted_scale = fit_temperature(y_true, pred_mean, pred_std)
        cal_path = save_calibration(
            out_dir, fitted_scale, split=args.split, n=len(df),
            extra={
                "method": METHOD_NAME,
                "source": "fit",
                "checkpoint": Path(args.checkpoint).name,
            },
        )
        print(f"Fitted std temperature {fitted_scale:.4f} on '{args.split}' "
              f"({len(df)} samples) -> saved to {cal_path}")

    # 8. Compute point accuracy (MAE/RMSE/R2) and UQ metrics (PICP/MPIW)
    row = summarize(
        METHOD_NAME, y_true, pred_mean,
        lower90, upper90, lower95, upper95,
        split=args.split,
        extra={
            "checkpoint": Path(args.checkpoint).name,
            "std_scale": std_scale,
            "calibrated": int(std_scale != 1.0),
        },
    )

    print(f"\n--- Heteroscedastic ({args.split}) Metrics ---")
    print(f"MAE:      {row['MAE']:.2f} months")
    print(f"RMSE:     {row['RMSE']:.2f} months")
    print(f"R2:       {row['R2']:.3f}")
    print(f"PICP@90:  {row['PICP_90']*100:.1f}%  (target 90%)")
    print(f"PICP@95:  {row['PICP_95']*100:.1f}%  (target 95%)")
    print(f"MPIW@90:  {row['MPIW_90']:.2f} months")
    print(f"MPIW@95:  {row['MPIW_95']:.2f} months")
    print("------------------------------------------\n")

    # 9. Save predictions and summary metrics as CSV files
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    covered90 = (y_true >= lower90) & (y_true <= upper90)
    covered95 = (y_true >= lower95) & (y_true <= upper95)
    predictions_df = pd.DataFrame({
        "id": df["id"].values,
        "sex": df["male"].values.astype(int),
        "true_age": y_true,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "std_scale": std_scale,
        "lower90": lower90,
        "upper90": upper90,
        "lower95": lower95,
        "upper95": upper95,
        "covered90": covered90.astype(int),
        "covered95": covered95.astype(int),
    })
    pred_path = out_dir / f"{args.output_name}_{args.split}_predictions_{ts}.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path} ({len(predictions_df)} rows)")

    metrics_df = pd.DataFrame([row])
    metrics_path = out_dir / f"{args.output_name}_{args.split}_metrics_{ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to:     {metrics_path}")

    print("Heteroscedastic evaluation completed.")


if __name__ == "__main__":
    main()
