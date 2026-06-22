import pandas as pd

import cps_config as C


METHOD_ORDER = ["SCP", "KNN-NCP", "AS-MCP", "BCP"]


def _conf_label(confidence):
    return f"δ = {int(round(confidence * 100))}%"


def _mean_std(series, decimals=3):
    return f"{series.mean():.{decimals}f} ± {series.std():.{decimals}f}"


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


def _main_table_mean(metrics):
    rows = []
    for method in METHOD_ORDER:
        row = {"Method": method}
        for confidence in C.TABLE_CONFIDENCES:
            part = metrics[
                (metrics["method"] == method)
                & (metrics["confidence"].round(6) == round(confidence, 6))
            ]
            label = _conf_label(confidence)
            row[f"{label} PICP"] = f"{part['picp_percent'].mean():.3f}"
            row[f"{label} PINAW"] = f"{part['pinaw'].mean():.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _main_table_mean_std(metrics):
    rows = []
    for method in METHOD_ORDER:
        row = {"Method": method}
        for confidence in C.TABLE_CONFIDENCES:
            part = metrics[
                (metrics["method"] == method)
                & (metrics["confidence"].round(6) == round(confidence, 6))
            ]
            label = _conf_label(confidence)
            row[f"{label} PICP"] = _mean_std(part["picp_percent"])
            row[f"{label} PINAW"] = _mean_std(part["pinaw"])
        rows.append(row)
    return pd.DataFrame(rows)


def _interval_score_table(metrics):
    rows = []
    for method in METHOD_ORDER:
        row = {"Method": method}
        for confidence in C.TABLE_CONFIDENCES:
            part = metrics[
                (metrics["method"] == method)
                & (metrics["confidence"].round(6) == round(confidence, 6))
            ]
            row[f"{int(round(confidence * 100))}%"] = _mean_std(part["interval_score"])
        rows.append(row)
    return pd.DataFrame(rows)


def _runtime_table(runtime):
    rows = []
    for method in METHOD_ORDER:
        part = runtime[runtime["method"] == method]
        rows.append({
            "Method": method,
            "fit_time_sec": _mean_std(part["fit_time_sec"]),
            "predict_ms_per_sample": _mean_std(part["predict_ms_per_sample"]),
        })
    return pd.DataFrame(rows)


def main():
    conformal_dir = C.RUN_OUTPUT_DIR / "conformal"
    table_dir = conformal_dir / "tables"
    metrics = pd.read_csv(conformal_dir / "metrics_long.csv")
    runtime = pd.read_csv(conformal_dir / "runtime_long.csv")

    _save_csv_tex(
        _main_table_mean(metrics),
        table_dir / "main_table_mean.csv",
        table_dir / "main_table_mean.tex",
    )
    _save_csv_tex(
        _main_table_mean_std(metrics),
        table_dir / "main_table_mean_std.csv",
        table_dir / "main_table_mean_std.tex",
    )
    _save_csv_tex(
        _interval_score_table(metrics),
        table_dir / "interval_score_table.csv",
        table_dir / "interval_score_table.tex",
    )
    _save_csv_tex(
        _runtime_table(runtime),
        table_dir / "runtime_table.csv",
        table_dir / "runtime_table.tex",
    )
    print(f"Saved tables to {table_dir}")


if __name__ == "__main__":
    main()
