import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sweep_test.config as S
from conformal.methods import AgeSexQuantileMondrianCP, KNNNormalizedCP, StandardSplitCP
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
import cps_config as C


METHODS = ("SCP", "AS-MCP", "KNN-NCP")


def _parse_float_list(values):
    return [float(v) for v in values]


def _discover_seeds(source_run_dir):
    seeds = []
    for seed_dir in sorted(Path(source_run_dir).glob("seed_*")):
        try:
            seeds.append(int(seed_dir.name.split("_")[-1]))
        except ValueError:
            continue
    return seeds


def _count_from_fraction(n_pool, fraction, available_fraction):
    if fraction < 0:
        return 0
    ratio = fraction / available_fraction
    return int(round(n_pool * ratio))


def _load_seed_artifacts(source_run_dir, seed):
    seed_dir = Path(source_run_dir) / f"seed_{seed:02d}"
    pred_path = seed_dir / "predictions.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions: {pred_path}")

    pred_df = read_table(pred_path)
    scale_df = pred_df[pred_df["split"] == "scale"].reset_index(drop=True)
    cal_df = pred_df[pred_df["split"] == "cal"].reset_index(drop=True)
    test_df = pred_df[pred_df["split"] == "test"].reset_index(drop=True)

    z_scale = np.load(seed_dir / "embeddings_scale.npy")
    z_cal = np.load(seed_dir / "embeddings_cal.npy")
    z_test = np.load(seed_dir / "embeddings_test.npy")

    cp_df = pd.concat([scale_df, cal_df], ignore_index=True)
    z_cp = np.vstack([z_scale, z_cal])
    if len(cp_df) != len(z_cp):
        raise ValueError(f"CP rows/embeddings mismatch for seed {seed}")

    return cp_df, z_cp, test_df, z_test


def _fit_method(method_name, cal_df, z_cal, scale_df=None, z_scale=None):
    if method_name == "SCP":
        return StandardSplitCP().fit(cal_df["y_true"].values, cal_df["y_pred"].values)

    if method_name == "AS-MCP":
        return AgeSexQuantileMondrianCP(
            n_bins=C.N_AGE_QUANTILE_BINS,
            quantiles=C.AGE_QUANTILES,
            min_bin_size=C.MIN_MONDRIAN_BIN_SIZE,
        ).fit(cal_df["y_true"].values, cal_df["y_pred"].values, cal_df["sex"].values)

    if method_name == "KNN-NCP":
        if scale_df is None or z_scale is None or len(scale_df) == 0:
            raise ValueError("KNN-NCP requires a non-empty scale set")
        return KNNNormalizedCP(
            k=C.KNN_K,
            metric=C.KNN_METRIC,
            eps=C.KNN_EPS,
            standardize=C.KNN_STANDARDIZE,
        ).fit(
            scale_df["y_true"].values,
            scale_df["y_pred"].values,
            z_scale,
            cal_df["y_true"].values,
            cal_df["y_pred"].values,
            z_cal,
        )

    raise ValueError(f"Unknown method: {method_name}")


def _predict_method(method_name, method, test_df, z_test, confidence):
    if method_name == "KNN-NCP":
        lower, upper, _ = method.predict_interval(test_df["y_pred"].values, z_test, confidence)
        return lower, upper
    if method_name == "AS-MCP":
        return method.predict_interval(test_df["y_pred"].values, test_df["sex"].values, confidence)
    return method.predict_interval(test_df["y_pred"].values, confidence)


def _metric_row(
    seed,
    method_name,
    confidence,
    cal_fraction,
    scale_fraction,
    cp_total_fraction,
    n_cal,
    n_scale,
    test_df,
    lower,
    upper,
):
    y_true = test_df["y_true"].values
    y_pred = test_df["y_pred"].values
    width = upper - lower
    alpha = 1.0 - confidence
    coverage = picp(y_true, lower, upper)

    return {
        "seed": seed,
        "method": method_name,
        "confidence": confidence,
        "cal_fraction": cal_fraction,
        "scale_fraction": scale_fraction,
        "cp_total_fraction": cp_total_fraction,
        "n_cal": n_cal,
        "n_scale": n_scale,
        "n_test": len(test_df),
        "picp": coverage,
        "picp_percent": 100.0 * coverage,
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
    }


def _interval_rows(seed, method_name, confidence, cal_fraction, scale_fraction, test_df, lower, upper):
    out = test_df[["id", "seed", "y_true", "y_pred", "sex"]].copy()
    out.insert(2, "method", method_name)
    out.insert(3, "confidence", confidence)
    out.insert(4, "cal_fraction", cal_fraction)
    out.insert(5, "scale_fraction", scale_fraction)
    out["lower"] = lower
    out["upper"] = upper
    out["width"] = upper - lower
    out["covered"] = ((out["y_true"] >= out["lower"]) & (out["y_true"] <= out["upper"])).astype(int)
    out["abs_error"] = (out["y_true"] - out["y_pred"]).abs()
    return out


def _summarize(df):
    group_cols = ["method", "confidence", "cal_fraction", "scale_fraction", "cp_total_fraction"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(group_cols)
    summary = df.groupby(group_cols, dropna=False)[list(numeric_cols)].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def _runtime_summarize(df):
    group_cols = ["method", "cal_fraction", "scale_fraction"]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(group_cols)
    summary = df.groupby(group_cols, dropna=False)[list(numeric_cols)].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def _write_config(args, out_dir, seeds, cal_fractions, confidences):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_run_dir": str(Path(args.source_run_dir).resolve()),
        "output_dir": str(out_dir.resolve()),
        "seeds": seeds,
        "cal_fractions": cal_fractions,
        "confidences": confidences,
        "available_cp_fraction": args.available_cp_fraction,
        "knn_total_fraction": args.knn_total_fraction,
        "methods": METHODS,
        "notes": [
            "SCP and AS-MCP use cal_fraction only.",
            "KNN-NCP uses the same cal subset plus the remaining KNN budget as scale.",
            "All methods reuse existing full_2 predictions/embeddings; no retraining is performed.",
        ],
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run(args):
    source_run_dir = Path(args.source_run_dir)
    out_dir = Path(args.output_dir)
    seeds = args.seeds if args.seeds is not None else _discover_seeds(source_run_dir)
    cal_fractions = args.cal_fractions
    confidences = args.confidences

    _write_config(args, out_dir, seeds, cal_fractions, confidences)

    metric_rows = []
    runtime_rows = []
    subset_rows = []
    skipped_rows = []

    for seed in seeds:
        print(f"\n=== Sweep seed {seed:02d} ===")
        cp_df, z_cp, test_df, z_test = _load_seed_artifacts(source_run_dir, seed)
        n_pool = len(cp_df)
        seed_interval_tables = []

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_pool)
        n_knn_total = min(
            n_pool,
            _count_from_fraction(n_pool, args.knn_total_fraction, args.available_cp_fraction),
        )
        knn_pool_idx = perm[:n_knn_total]

        for cal_fraction in cal_fractions:
            n_cal = min(
                n_pool,
                _count_from_fraction(n_pool, cal_fraction, args.available_cp_fraction),
            )
            if n_cal <= 0:
                skipped_rows.append({
                    "seed": seed,
                    "cal_fraction": cal_fraction,
                    "method": "ALL",
                    "reason": "empty calibration subset",
                })
                continue

            cal_idx = perm[:n_cal]
            cal_df = cp_df.iloc[cal_idx].reset_index(drop=True)
            z_cal = z_cp[cal_idx]

            subset_rows.append({
                "seed": seed,
                "cal_fraction": cal_fraction,
                "method_family": "SCP/AS-MCP",
                "n_cal": len(cal_df),
                "n_scale": 0,
                "n_cp_pool": n_pool,
            })

            method_specs = [
                ("SCP", cal_df, z_cal, None, None, 0.0),
                ("AS-MCP", cal_df, z_cal, None, None, 0.0),
            ]

            if n_cal < n_knn_total:
                scale_idx = knn_pool_idx[n_cal:n_knn_total]
                scale_df = cp_df.iloc[scale_idx].reset_index(drop=True)
                z_scale = z_cp[scale_idx]
                scale_fraction = max(0.0, args.knn_total_fraction - cal_fraction)
                method_specs.append(("KNN-NCP", cal_df, z_cal, scale_df, z_scale, scale_fraction))
                subset_rows.append({
                    "seed": seed,
                    "cal_fraction": cal_fraction,
                    "method_family": "KNN-NCP",
                    "n_cal": len(cal_df),
                    "n_scale": len(scale_df),
                    "n_cp_pool": n_pool,
                })
            else:
                skipped_rows.append({
                    "seed": seed,
                    "cal_fraction": cal_fraction,
                    "method": "KNN-NCP",
                    "reason": "no scale samples remain for KNN-NCP",
                })

            for method_name, method_cal_df, method_z_cal, scale_df, z_scale, scale_fraction in method_specs:
                fit_start = time.perf_counter()
                method = _fit_method(method_name, method_cal_df, method_z_cal, scale_df, z_scale)
                fit_time = time.perf_counter() - fit_start

                predict_time = 0.0
                for confidence in confidences:
                    pred_start = time.perf_counter()
                    lower, upper = _predict_method(method_name, method, test_df, z_test, confidence)
                    predict_time += time.perf_counter() - pred_start

                    cp_total_fraction = cal_fraction + scale_fraction
                    metric_rows.append(_metric_row(
                        seed=seed,
                        method_name=method_name,
                        confidence=confidence,
                        cal_fraction=cal_fraction,
                        scale_fraction=scale_fraction,
                        cp_total_fraction=cp_total_fraction,
                        n_cal=len(method_cal_df),
                        n_scale=0 if scale_df is None else len(scale_df),
                        test_df=test_df,
                        lower=lower,
                        upper=upper,
                    ))

                    if args.save_intervals:
                        seed_interval_tables.append(_interval_rows(
                            seed,
                            method_name,
                            confidence,
                            cal_fraction,
                            scale_fraction,
                            test_df,
                            lower,
                            upper,
                        ))

                runtime_rows.append({
                    "seed": seed,
                    "method": method_name,
                    "cal_fraction": cal_fraction,
                    "scale_fraction": scale_fraction,
                    "fit_time_sec": fit_time,
                    "predict_time_sec": predict_time,
                    "predict_ms_per_sample_conf": (
                        1000.0 * predict_time / max(1, len(test_df) * len(confidences))
                    ),
                    "n_cal": len(method_cal_df),
                    "n_scale": 0 if scale_df is None else len(scale_df),
                    "n_test": len(test_df),
                })

        if args.save_intervals and seed_interval_tables:
            interval_dir = out_dir / "intervals"
            write_table(
                pd.concat(seed_interval_tables, ignore_index=True),
                interval_dir / f"intervals_seed_{seed:02d}.parquet",
            )

    metrics_long = pd.DataFrame(metric_rows)
    runtime_long = pd.DataFrame(runtime_rows)
    subset_counts = pd.DataFrame(subset_rows)
    skipped = pd.DataFrame(skipped_rows)

    metrics_long.to_csv(out_dir / "metrics_long.csv", index=False)
    runtime_long.to_csv(out_dir / "runtime_long.csv", index=False)
    subset_counts.to_csv(out_dir / "subset_counts.csv", index=False)
    skipped.to_csv(out_dir / "skipped.csv", index=False)

    if not metrics_long.empty:
        _summarize(metrics_long).to_csv(out_dir / "metrics_summary.csv", index=False)
    if not runtime_long.empty:
        _runtime_summarize(runtime_long).to_csv(out_dir / "runtime_summary.csv", index=False)

    print(f"\nSaved sweep results to {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep SCP/AS-MCP/KNN-NCP calibration and scale budgets using an existing CP run."
    )
    parser.add_argument("--source-run-dir", type=Path, default=S.SOURCE_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=S.OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--cal-fractions", type=float, nargs="+", default=S.CAL_FRACTIONS)
    parser.add_argument("--confidences", type=float, nargs="+", default=S.CONFIDENCES)
    parser.add_argument("--available-cp-fraction", type=float, default=S.AVAILABLE_CP_FRACTION)
    parser.add_argument("--knn-total-fraction", type=float, default=S.KNN_TOTAL_FRACTION)
    parser.add_argument("--save-intervals", action=argparse.BooleanOptionalAction, default=S.SAVE_INTERVALS)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
