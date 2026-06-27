import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

from config import OUTPUT_DIR, CONFIDENCES, Y_MIN, Y_MAX
from cqr_utils import predict_quantiles, build_intervals, median_prediction


def evaluate_and_save_metrics(model, loader, df, max_age, q_hat,
                              run_name="run", device="cpu", split="val"):
    """
    Evaluate the CQR model on a dataset: de-normalize the median prediction and
    the calibrated quantile intervals, compute point metrics (MAE/RMSE/R2) plus
    per-confidence coverage and mean interval width, and save predictions and
    metrics to CSV.

    `q_hat` is the dict of CQR corrections from cqr_utils.calibrate.
    """
    print(f"\n[{run_name}] Evaluating CQR model on {split} set...")

    preds, y_norm_true = predict_quantiles(model, loader, device)
    y_true = y_norm_true * max_age
    y_med = median_prediction(preds, max_age)
    intervals = build_intervals(preds, q_hat, max_age, y_min=Y_MIN, y_max=Y_MAX)

    # Point metrics on the median prediction.
    mae = mean_absolute_error(y_true, y_med)
    mse = mean_squared_error(y_true, y_med)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_med)
    acc6 = np.mean(np.abs(y_med - y_true) <= 6.0) * 100
    acc12 = np.mean(np.abs(y_med - y_true) <= 12.0) * 100

    print(f"\n--- {split.capitalize()} Point Metrics (median) ---")
    print(f"MAE:             {mae:.2f} months")
    print(f"RMSE:            {rmse:.2f} months")
    print(f"R2:              {r2:.3f}")
    print(f"% within +/-6mo:  {acc6:.1f}%")
    print(f"% within +/-12mo: {acc12:.1f}%")

    # Interval metrics per confidence level.
    # PICP  = Prediction Interval Coverage Probability (empirical coverage).
    # PINAW = Prediction Interval Normalized Average Width (mean width / target range).
    y_range = float(y_true.max() - y_true.min())
    print(f"--- {split.capitalize()} CQR Interval Metrics ---")
    interval_rows = []
    for conf in CONFIDENCES:
        lo, hi = intervals[conf]
        covered = (y_true >= lo) & (y_true <= hi)
        picp = float(np.mean(covered))
        width = float(np.mean(hi - lo))
        pinaw = width / y_range if y_range > 0 else float("nan")
        print(f"  conf={conf:.2f} -> PICP={picp:.3f} | "
              f"mean width={width:.2f} mo | PINAW={pinaw:.3f} | "
              f"q_hat={q_hat[conf] * max_age:.2f} mo")
        interval_rows.append({
            "run_name": run_name,
            "split": split,
            "confidence": conf,
            "PICP": picp,
            "mean_width_months": width,
            "PINAW": pinaw,
            "q_hat_months": q_hat[conf] * max_age,
        })
    print("--------------------------\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample predictions, with interval bounds for every confidence level.
    predictions_df = pd.DataFrame({
        "id": df["id"].values,
        "sex": df["male"].values.astype(int),
        "true_age": y_true,
        "pred_median": y_med,
    })
    predictions_df["abs_error"] = (predictions_df["pred_median"] - predictions_df["true_age"]).abs()
    for conf in CONFIDENCES:
        lo, hi = intervals[conf]
        tag = f"{int(round(conf * 100))}"
        predictions_df[f"lo_{tag}"] = lo
        predictions_df[f"hi_{tag}"] = hi
        predictions_df[f"covered_{tag}"] = ((y_true >= lo) & (y_true <= hi)).astype(int)

    pred_path = run_dir / f"{run_name}_{split}_predictions_{ts}.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path} ({len(predictions_df)} rows)")

    # Metrics summary: one point-metrics row + per-confidence interval rows.
    point_row = {
        "run_name": run_name,
        "split": split,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2,
        "pct_within_6mo": acc6,
        "pct_within_12mo": acc12,
        "n": len(predictions_df),
        "timestamp": ts,
    }
    point_df = pd.DataFrame([point_row])
    point_path = run_dir / f"{run_name}_{split}_point_metrics_{ts}.csv"
    point_df.to_csv(point_path, index=False)

    interval_df = pd.DataFrame(interval_rows)
    interval_path = run_dir / f"{run_name}_{split}_interval_metrics_{ts}.csv"
    interval_df.to_csv(interval_path, index=False)
    print(f"Saved metrics to:     {point_path}")
    print(f"                      {interval_path}")

    return point_df, interval_df
