import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from config import OUTPUT_DIR

def evaluate_and_save_metrics(model, ds_val, val_df, max_age, run_name="run"):
    """
    Evaluates the model on the validation dataset, de-normalizes predictions,
    calculates metrics (MAE, RMSE, MSE, R2, accuracy), and saves them to CSV.
    """
    print(f"\n[{run_name}] Evaluating model on validation set...")
    
    # Get predictions
    y_norm_pred = model.predict(ds_val, verbose=1).flatten()
    
    # De-normalize
    y_pred = y_norm_pred * max_age
    y_true = val_df['boneage_norm'].values * max_age
    
    # Calculate metrics
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    r2    = r2_score(y_true, y_pred)
    acc6  = np.mean(np.abs(y_pred - y_true) <= 6.0)  * 100
    acc12 = np.mean(np.abs(y_pred - y_true) <= 12.0) * 100

    print("\n--- Validation Metrics ---")
    print(f"MAE:             {mae:.2f} months")
    print(f"RMSE:            {rmse:.2f} months")
    print(f"MSE:             {mse:.2f}")
    print(f"R²:              {r2:.3f}")
    print(f"% within ±6 mo:  {acc6:.1f}%")
    print(f"% within ±12 mo: {acc12:.1f}%")
    print("--------------------------\n")

    # Generate timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save Predictions DataFrame
    predictions_df = pd.DataFrame({
        "id":        val_df["id"].values,
        "sex":       val_df["male"].values.astype(int),
        "true_age":  y_true,
        "pred_age":  y_pred,
    })
    predictions_df["abs_error"] = (predictions_df["pred_age"] - predictions_df["true_age"]).abs()
    
    pred_path = OUTPUT_DIR / f"{run_name}_val_predictions_{ts}.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to: {pred_path} ({len(predictions_df)} rows)")

    # Save Metrics DataFrame
    metrics_df = pd.DataFrame([{
        "run_name":       run_name,
        "MAE":            mae,
        "RMSE":           rmse,
        "MSE":            mse,
        "R2":             r2,
        "pct_within_6mo": acc6,
        "pct_within_12mo": acc12,
        "n_val":          len(predictions_df),
        "timestamp":      ts,
    }])
    
    metrics_path = OUTPUT_DIR / f"{run_name}_val_metrics_{ts}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to:     {metrics_path}")
    
    return metrics_df
