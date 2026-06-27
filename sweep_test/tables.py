import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sweep_test.config as S


METHOD_ORDER = ["SCP", "AS-MCP", "KNN-NCP"]


def _conf_label(confidence):
    return f"{int(round(confidence * 100))}%"


def _mean_std(series, decimals=3):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return ""
    std = series.std(ddof=1)
    if np.isnan(std):
        std = 0.0
    return f"{series.mean():.{decimals}f} ± {std:.{decimals}f}"


def _mean(series, decimals=3):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return ""
    return f"{series.mean():.{decimals}f}"


def _save_csv_tex(df, csv_path, tex_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{" + "l" * len(df.columns) + "}\n")
        f.write("\\hline\n")
        f.write(" & ".join(df.columns) + " \\\\\n")
        f.write("\\hline\n")
        for _, row in df.iterrows():
            f.write(" & ".join(str(row[col]) for col in df.columns) + " \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def _sort_key(row):
    method_rank = METHOD_ORDER.index(row["Method"]) if row["Method"] in METHOD_ORDER else 99
    return method_rank, row["cal_fraction"], row["scale_fraction"]


def _metric_table(metrics, confidences, formatter):
    rows = []
    for (method, cal_fraction, scale_fraction), _ in metrics.groupby(
        ["method", "cal_fraction", "scale_fraction"], dropna=False
    ):
        row = {
            "Method": method,
            "cal_fraction": cal_fraction,
            "scale_fraction": scale_fraction,
            "cp_total_fraction": cal_fraction + scale_fraction,
        }
        part_budget = metrics[
            (metrics["method"] == method)
            & (metrics["cal_fraction"].round(6) == round(cal_fraction, 6))
            & (metrics["scale_fraction"].round(6) == round(scale_fraction, 6))
        ]
        for confidence in confidences:
            part = part_budget[
                part_budget["confidence"].round(6) == round(confidence, 6)
            ]
            label = _conf_label(confidence)
            row[f"{label} PICP"] = formatter(part["picp_percent"])
            row[f"{label} gap_pp"] = formatter(part["coverage_gap_pp"])
            row[f"{label} PINAW"] = formatter(part["pinaw"])
            row[f"{label} interval_score"] = formatter(part["interval_score"])
            row[f"{label} mean_width"] = formatter(part["mean_width_months"])
        rows.append(row)

    rows = sorted(rows, key=_sort_key)
    return pd.DataFrame(rows)


def _runtime_table(runtime):
    rows = []
    for (method, cal_fraction, scale_fraction), part in runtime.groupby(
        ["method", "cal_fraction", "scale_fraction"], dropna=False
    ):
        rows.append({
            "Method": method,
            "cal_fraction": cal_fraction,
            "scale_fraction": scale_fraction,
            "cp_total_fraction": cal_fraction + scale_fraction,
            "fit_time_sec": _mean_std(part["fit_time_sec"]),
            "predict_time_sec": _mean_std(part["predict_time_sec"]),
            "predict_ms_per_sample_conf": _mean_std(part["predict_ms_per_sample_conf"]),
            "n_cal": _mean(part["n_cal"], decimals=1),
            "n_scale": _mean(part["n_scale"], decimals=1),
            "n_test": _mean(part["n_test"], decimals=1),
        })
    rows = sorted(rows, key=_sort_key)
    return pd.DataFrame(rows)


def _best_budget_table(metrics, confidences):
    rows = []
    grouped = metrics.groupby(
        ["method", "confidence", "cal_fraction", "scale_fraction"], dropna=False
    ).agg(
        picp_mean=("picp", "mean"),
        picp_std=("picp", "std"),
        gap_abs_mean=("coverage_gap_pp", lambda s: float(np.mean(np.abs(s)))),
        pinaw_mean=("pinaw", "mean"),
        interval_score_mean=("interval_score", "mean"),
        mean_width_months_mean=("mean_width_months", "mean"),
    ).reset_index()

    for method in METHOD_ORDER:
        for confidence in confidences:
            part = grouped[
                (grouped["method"] == method)
                & (grouped["confidence"].round(6) == round(confidence, 6))
            ].copy()
            if part.empty:
                continue
            part = part.sort_values(["gap_abs_mean", "pinaw_mean", "interval_score_mean"])
            best = part.iloc[0]
            rows.append({
                "Method": method,
                "confidence": confidence,
                "cal_fraction": best["cal_fraction"],
                "scale_fraction": best["scale_fraction"],
                "picp_mean": best["picp_mean"],
                "picp_std": best["picp_std"],
                "abs_gap_pp_mean": best["gap_abs_mean"],
                "pinaw_mean": best["pinaw_mean"],
                "interval_score_mean": best["interval_score_mean"],
                "mean_width_months_mean": best["mean_width_months_mean"],
            })
    return pd.DataFrame(rows)


def build_tables(input_dir, table_confidences):
    input_dir = Path(input_dir)
    table_dir = input_dir / "tables"
    metrics = pd.read_csv(input_dir / "metrics_long.csv")
    runtime = pd.read_csv(input_dir / "runtime_long.csv")

    _save_csv_tex(
        _metric_table(metrics, table_confidences, _mean),
        table_dir / "metrics_mean_by_budget.csv",
        table_dir / "metrics_mean_by_budget.tex",
    )
    _save_csv_tex(
        _metric_table(metrics, table_confidences, _mean_std),
        table_dir / "metrics_mean_std_by_budget.csv",
        table_dir / "metrics_mean_std_by_budget.tex",
    )
    _save_csv_tex(
        _runtime_table(runtime),
        table_dir / "runtime_by_budget.csv",
        table_dir / "runtime_by_budget.tex",
    )
    _best_budget_table(metrics, table_confidences).to_csv(
        table_dir / "best_budget_by_method_confidence.csv", index=False
    )
    print(f"Saved sweep tables to {table_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build CSV/LaTeX tables for sweep_test results.")
    parser.add_argument("--input-dir", type=Path, default=S.OUTPUT_DIR)
    parser.add_argument("--table-confidences", type=float, nargs="+", default=S.TABLE_CONFIDENCES)
    args = parser.parse_args()
    build_tables(args.input_dir, args.table_confidences)


if __name__ == "__main__":
    main()

