"""Monte Carlo Dropout uncertainty quantification.

Reuses an existing trained regression checkpoint (.pth), runs T stochastic
forward passes with dropout active at inference, builds Gaussian prediction
intervals, and reports PICP@90/95 and MPIW@90/95 alongside the usual point
metrics. Outputs are written under ``UQ/results/mc_dropout/``.

Example:
    python -m UQ.mc_dropout.run_mc_dropout \\
        --checkpoint outputs/run_seed123/best_run_seed123_ep12_val0.123.pth \\
        --samples 30 --split test
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Make both the project root (for model/data_loader/config) and the UQ package
# importable whether run as a module or as a plain script.
_THIS = Path(__file__).resolve()
_UQ_DIR = _THIS.parent.parent
_PROJECT_ROOT = _UQ_DIR.parent
for _p in (str(_PROJECT_ROOT), str(_UQ_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

from common.paths import method_results_dir, ensure_project_on_path  # noqa: E402
from common.inference import (  # noqa: E402
    enable_dropout,
    gaussian_intervals,
    denormalize,
    mc_forward_passes,
)
from common.uq_metrics import summarize  # noqa: E402
from common.calibration import (  # noqa: E402
    fit_temperature,
    apply_temperature,
    save_calibration,
    resolve_calibration,
)

ensure_project_on_path()

from data_loader import load_data, build_eval_loader  # noqa: E402
from model import build_multi_input_model  # noqa: E402

METHOD_NAME = "mc_dropout"


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


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Dropout UQ over a trained bone age model"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained .pth model checkpoint")
    parser.add_argument("--samples", type=int, default=30,
                        help="Number of stochastic forward passes (T)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "calib", "test"],
                        help="Which data split to evaluate on")
    parser.add_argument("--output-name", type=str, default=METHOD_NAME,
                        help="Prefix for output files")
    parser.add_argument("--fit-calibration", action="store_true",
                        help="Fit a post-hoc std temperature on THIS split "
                             "(run with --split calib) and save it for reuse")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Apply a saved std temperature before building "
                             "intervals: a path to a calibration JSON or "
                             "'auto' to use the most recent one")
    args = parser.parse_args()

    print("Loading data...")
    train_df, val_df, calib_df, test_df, max_age = load_data(sample_frac=1.0)
    df = select_split(args.split, train_df, val_df, calib_df, test_df)
    print(f"Evaluating on '{args.split}' split: {len(df)} samples (max_age={max_age})")

    loader = build_eval_loader(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model weights from {args.checkpoint}")
    model = build_multi_input_model()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)

    # Keep dropout active at inference for MC sampling.
    enable_dropout(model)

    print(f"Running {args.samples} stochastic forward passes...")
    preds_norm, _ = mc_forward_passes(model, loader, device, args.samples)

    # Per-sample statistics across the T passes (normalized space), then de-normalize.
    mean_norm = preds_norm.mean(axis=0)
    std_norm = preds_norm.std(axis=0)

    pred_mean = denormalize(mean_norm, max_age)
    pred_std = denormalize(std_norm, max_age)

    # Ground truth is read directly from the (unshuffled) dataframe to avoid
    # any ambiguity in target ordering.
    y_true = df["boneage_norm"].values * max_age

    out_dir = method_results_dir(METHOD_NAME)

    # Apply a previously-fitted temperature (widens/tightens the raw dropout
    # std so the intervals actually reach their nominal coverage).
    std_scale = 1.0
    if args.calibration:
        payload, cal_path = resolve_calibration(args.calibration, out_dir)
        std_scale = float(payload["std_scale"])
        print(f"Applying calibration temperature {std_scale:.4f} from {cal_path}")
    std_eff = apply_temperature(pred_std, std_scale)

    lower90, upper90 = gaussian_intervals(pred_mean, std_eff, 90)
    lower95, upper95 = gaussian_intervals(pred_mean, std_eff, 95)

    # Fit a fresh temperature on this split (intended for --split calib). This
    # is fit on the RAW std so the saved scale is independent of any already
    # applied calibration.
    if args.fit_calibration:
        fitted_scale = fit_temperature(y_true, pred_mean, pred_std)
        cal_path = save_calibration(
            out_dir, fitted_scale, split=args.split, n=len(df),
            extra={
                "method": METHOD_NAME,
                "source": "fit",
                "samples": args.samples,
                "checkpoint": Path(args.checkpoint).name,
            },
        )
        print(f"Fitted std temperature {fitted_scale:.4f} on '{args.split}' "
              f"({len(df)} samples) -> saved to {cal_path}")

    row = summarize(
        METHOD_NAME, y_true, pred_mean,
        lower90, upper90, lower95, upper95,
        split=args.split,
        extra={
            "samples": args.samples,
            "checkpoint": Path(args.checkpoint).name,
            "std_scale": std_scale,
            "calibrated": int(std_scale != 1.0),
        },
    )

    print(f"\n--- MC Dropout ({args.split}) Metrics ---")
    print(f"MAE:      {row['MAE']:.2f} months")
    print(f"RMSE:     {row['RMSE']:.2f} months")
    print(f"R2:       {row['R2']:.3f}")
    print(f"PICP@90:  {row['PICP_90']*100:.1f}%  (target 90%)")
    print(f"PICP@95:  {row['PICP_95']*100:.1f}%  (target 95%)")
    print(f"MPIW@90:  {row['MPIW_90']:.2f} months")
    print(f"MPIW@95:  {row['MPIW_95']:.2f} months")
    print("----------------------------------\n")

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

    print("MC Dropout evaluation completed.")


if __name__ == "__main__":
    main()
