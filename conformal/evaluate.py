import time

import numpy as np
import pandas as pd

import cps_config as C
from conformal.methods import (
    AgeSexQuantileMondrianCP,
    BackwardFixedWidthCP,
    KNNNormalizedCP,
    StandardSplitCP,
)
from conformal.metrics import (
    coverage_gap_pp,
    interval_score,
    mae,
    median_absolute_error,
    miscoverage_rate,
    mpiw,
    picp,
    pinaw,
    rmse,
)
from cps.io_utils import read_table, write_table


def _method_instances():
    return [
        ("SCP", StandardSplitCP()),
        (
            "KNN-NCP",
            KNNNormalizedCP(
                k=C.KNN_K,
                metric=C.KNN_METRIC,
                eps=C.KNN_EPS,
                standardize=C.KNN_STANDARDIZE,
            ),
        ),
        (
            "AS-MCP",
            AgeSexQuantileMondrianCP(
                n_bins=C.N_AGE_QUANTILE_BINS,
                quantiles=C.AGE_QUANTILES,
                min_bin_size=C.MIN_MONDRIAN_BIN_SIZE,
            ),
        ),
        ("BCP", BackwardFixedWidthCP(interval_width_months=C.BCP_INTERVAL_WIDTH_MONTHS)),
    ]


def _fit_method(method_name, method, scale_df, cal_df, z_scale, z_cal):
    if method_name == "SCP":
        return method.fit(cal_df["y_true"].values, cal_df["y_pred"].values)
    if method_name == "KNN-NCP":
        return method.fit(
            scale_df["y_true"].values,
            scale_df["y_pred"].values,
            z_scale,
            cal_df["y_true"].values,
            cal_df["y_pred"].values,
            z_cal,
        )
    if method_name == "AS-MCP":
        return method.fit(
            cal_df["y_true"].values,
            cal_df["y_pred"].values,
            cal_df["sex"].values,
        )
    if method_name == "BCP":
        return method.fit(cal_df["y_true"].values, cal_df["y_pred"].values)
    raise ValueError(f"Unknown method: {method_name}")


def _predict_method(method_name, method, test_df, z_test, confidence):
    if method_name == "KNN-NCP":
        lower, upper, _ = method.predict_interval(
            test_df["y_pred"].values, z_test, confidence
        )
        return lower, upper
    if method_name == "AS-MCP":
        return method.predict_interval(
            test_df["y_pred"].values, test_df["sex"].values, confidence
        )
    if method_name == "BCP":
        return method.predict_interval(test_df["y_pred"].values, confidence)
    return method.predict_interval(test_df["y_pred"].values, confidence)


def _method_diagnostics(method, confidence):
    base = {
        "bcp_interval_width_months": np.nan,
        "bcp_estimated_coverage": np.nan,
        "bcp_estimated_miscoverage": np.nan,
        "bcp_trust_at_confidence": np.nan,
    }
    if hasattr(method, "diagnostics"):
        base.update(method.diagnostics(confidence))
    return base


def _metric_row(seed, method_name, method, confidence, test_df, lower, upper):
    y_true = test_df["y_true"].values
    y_pred = test_df["y_pred"].values
    width = upper - lower
    picp_value = picp(y_true, lower, upper)
    alpha = 1.0 - confidence

    row = {
        "seed": seed,
        "method": method_name,
        "confidence": confidence,
        "picp": picp_value,
        "picp_percent": 100.0 * picp_value,
        "mpiw": mpiw(lower, upper),
        "pinaw": pinaw(lower, upper, C.Y_MIN, C.Y_MAX),
        "coverage_gap_pp": coverage_gap_pp(y_true, lower, upper, confidence),
        "miscoverage_rate": miscoverage_rate(y_true, lower, upper),
        "interval_score": interval_score(y_true, lower, upper, alpha),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "mean_width_months": float(np.mean(width)),
        "median_width_months": float(np.median(width)),
        "n_test": len(test_df),
    }
    row.update(_method_diagnostics(method, confidence))
    return row


def _interval_rows(seed, method_name, confidence, test_df, lower, upper):
    out = test_df[["id", "seed", "y_true", "y_pred", "sex"]].copy()
    out.insert(2, "method", method_name)
    out.insert(3, "confidence", confidence)
    out.insert(4, "split", "test")
    out["lower"] = lower
    out["upper"] = upper
    out["width"] = upper - lower
    out["abs_error"] = (out["y_true"] - out["y_pred"]).abs()
    return out


def _summarize(df, group_cols):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(group_cols)
    summary = df.groupby(group_cols)[list(numeric_cols)].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def main():
    out_dir = C.RUN_OUTPUT_DIR / "conformal"
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    runtime_rows = []
    interval_tables = []

    for seed in C.SEEDS:
        seed_dir = C.RUN_OUTPUT_DIR / f"seed_{seed:02d}"
        pred_df = read_table(seed_dir / "predictions.parquet")

        scale_df = pred_df[pred_df["split"] == "scale"].reset_index(drop=True)
        cal_df = pred_df[pred_df["split"] == "cal"].reset_index(drop=True)
        test_df = pred_df[pred_df["split"] == "test"].reset_index(drop=True)

        z_scale = np.load(seed_dir / "embeddings_scale.npy")
        z_cal = np.load(seed_dir / "embeddings_cal.npy")
        z_test = np.load(seed_dir / "embeddings_test.npy")

        for method_name, method in _method_instances():
            fit_start = time.perf_counter()
            _fit_method(method_name, method, scale_df, cal_df, z_scale, z_cal)
            fit_time = time.perf_counter() - fit_start

            predict_time = 0.0
            for confidence in C.CONFIDENCES:
                pred_start = time.perf_counter()
                lower, upper = _predict_method(method_name, method, test_df, z_test, confidence)
                predict_time += time.perf_counter() - pred_start

                metric_rows.append(
                    _metric_row(seed, method_name, method, confidence, test_df, lower, upper)
                )
                if C.SAVE_INTERVALS:
                    interval_tables.append(
                        _interval_rows(seed, method_name, confidence, test_df, lower, upper)
                    )

            denom = max(1, len(test_df) * len(C.CONFIDENCES))
            runtime_rows.append({
                "seed": seed,
                "method": method_name,
                "fit_time_sec": fit_time,
                "predict_time_sec": predict_time,
                "predict_ms_per_sample": 1000.0 * predict_time / denom,
                "n_test": len(test_df),
            })

    metrics_long = pd.DataFrame(metric_rows)
    runtime_long = pd.DataFrame(runtime_rows)
    metrics_long.to_csv(out_dir / "metrics_long.csv", index=False)
    runtime_long.to_csv(out_dir / "runtime_long.csv", index=False)

    metrics_summary = _summarize(metrics_long, ["method", "confidence"])
    runtime_summary = _summarize(runtime_long, ["method"])
    metrics_summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    runtime_summary.to_csv(out_dir / "runtime_summary.csv", index=False)

    if C.SAVE_INTERVALS:
        intervals = pd.concat(interval_tables, ignore_index=True)
        write_table(intervals, out_dir / "intervals_long.parquet")

    print(f"Saved conformal results to {out_dir}")


if __name__ == "__main__":
    main()
