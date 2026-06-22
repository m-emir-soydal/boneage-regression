import numpy as np
import pandas as pd

import cps_config as C
from conformal.metrics import mae, median_absolute_error, rmse
from cps.io_utils import read_table


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def _markdown_table(df):
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def _point_prediction_summary():
    path = C.RUN_OUTPUT_DIR / "all_predictions.parquet"
    df = read_table(path)
    test_df = df[df["split"] == "test"]
    rows = []
    for seed, part in test_df.groupby("seed"):
        rows.append({
            "seed": seed,
            "MAE": mae(part["y_true"], part["y_pred"]),
            "RMSE": rmse(part["y_true"], part["y_pred"]),
            "MedAE": median_absolute_error(part["y_true"], part["y_pred"]),
        })
    point = pd.DataFrame(rows)
    return {
        "MAE": _mean_std(point["MAE"]),
        "RMSE": _mean_std(point["RMSE"]),
        "MedAE": _mean_std(point["MedAE"]),
    }


def _bcp_summary(conformal_dir):
    metrics_path = conformal_dir / "metrics_long.csv"
    if not metrics_path.exists():
        return "BCP metrics were not found."

    metrics = pd.read_csv(metrics_path)
    bcp = metrics[metrics["method"] == "BCP"].copy()
    if bcp.empty:
        return "BCP was not included in this run."

    per_seed = bcp.groupby("seed").first().reset_index()
    width = _mean_std(per_seed["bcp_interval_width_months"])
    coverage = _mean_std(100.0 * per_seed["bcp_estimated_coverage"])
    miscoverage = _mean_std(100.0 * per_seed["bcp_estimated_miscoverage"])
    return (
        f"- Fixed interval width: {width} months\n"
        f"- Calibration-estimated coverage: {coverage}%\n"
        f"- Calibration-estimated miscoverage: {miscoverage}%"
    )


def main():
    conformal_dir = C.RUN_OUTPUT_DIR / "conformal"
    table_dir = conformal_dir / "tables"
    fig_dir = conformal_dir / "figures"
    report_path = conformal_dir / "report.md"

    point = _point_prediction_summary()
    bcp_text = _bcp_summary(conformal_dir)
    main_table = pd.read_csv(table_dir / "main_table_mean.csv")
    runtime_table = pd.read_csv(table_dir / "runtime_table.csv")
    suffix = str(int(round(C.selected_plot_confidence() * 100)))

    figure_paths = [
        fig_dir / "calibration_curve.png",
        fig_dir / "boxplot_gap_pinaw.png",
        fig_dir / f"width_distribution_{suffix}.png",
        fig_dir / f"subgroup_coverage_{suffix}.png",
        fig_dir / f"width_vs_error_SCP_{suffix}.png",
        fig_dir / f"width_vs_error_KNN-NCP_{suffix}.png",
        fig_dir / f"width_vs_error_AS-MCP_{suffix}.png",
        fig_dir / f"width_vs_error_BCP_{suffix}.png",
    ]

    text = f"""# Conformal Prediction Report

Run mode: {C.RUN_MODE}
Seeds: {C.SEEDS}
Data fraction: {C.DATA_FRACTION}
Epochs: {C.EPOCHS}

## Point prediction

Mean ± std over seeds:

- MAE: {point["MAE"]}
- RMSE: {point["RMSE"]}
- Median absolute error: {point["MedAE"]}

## Main conformal results

{_markdown_table(main_table)}

## Runtime

{_markdown_table(runtime_table)}

## Backward CP diagnostics

{bcp_text}

## Generated figures

""" + "\n".join(f"- {path}" for path in figure_paths) + """

## Interpretation notes

- Debug mode uses a very small calibration/scale/test sample, so KNN-NCP can
  reduce to a global scale estimate and AS-MCP can fall back to the global SCP
  threshold. In that case, method curves may overlap exactly.
- SCP gives the global baseline interval.
- KNN-NCP should produce adaptive widths based on embedding-space local residuals.
- AS-MCP should improve calibration across sex and predicted-age groups.
- BCP fixes interval width first and estimates the resulting calibration coverage,
  so it should not be interpreted as targeting each desired coverage level directly.
- Best method should not be chosen by highest PICP alone.
- Prefer methods with PICP close to target, low PINAW, low interval score, and small subgroup coverage gaps.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
