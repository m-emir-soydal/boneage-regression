"""Aggregate the pipeline outputs and build the report artifacts.

Collects every ``comparison/<Method>/seed_XX/outputs/metrics.csv`` (BNN +
MC-Dropout) and writes both a combined report (both methods) and a separate
self-contained report per method:

    comparison/metrics_long.csv          both methods x seeds x confidences
    comparison/metrics_summary.csv       mean/std over seeds per (method, conf)
    comparison/point_metrics_summary.csv MAE/RMSE/MedAE per method
    comparison/tables/*.csv / *.tex      main + full + point tables
    comparison/figures/*.png / *.pdf     reliability / PINAW / gap / ... plots
    comparison/<Method>/report/...       the same set, that method only

This pipeline does NOT read or merge any conformal results — that comparison is
assembled separately. Can be run standalone:  ``python -m UQ.pipeline.report``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import pconfig as C

INTERVAL_METRIC_COLS = ["picp", "picp_percent", "mpiw", "pinaw", "coverage_gap_pp",
                        "miscoverage_rate", "interval_score", "mae", "rmse", "medae"]

METHOD_ORDER = [C.METHOD_DISPLAY_NAMES[m] for m in C.METHODS]
COLORS = {"BNN": "#d62728", "MC-Dropout": "#9467bd"}


# --------------------------------------------------------------------------- #
# Collect
# --------------------------------------------------------------------------- #
def collect_uq_long() -> pd.DataFrame:
    frames = []
    for method in C.METHODS:
        display = C.METHOD_DISPLAY_NAMES[method]
        for mfile in sorted((C.OUTPUT_ROOT / display.replace(" ", "_")).glob(
                "seed_*/outputs/metrics.csv")):
            frames.append(pd.read_csv(mfile))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["family"] = "UQ"
    return df


def build_long() -> pd.DataFrame:
    """Long table of BNN + MC-Dropout metrics only (no conformal merge —
    the conformal comparison is assembled separately by the user)."""
    long = collect_uq_long()
    if long.empty:
        return long
    cols = ["family", "method", "seed", "confidence"] + INTERVAL_METRIC_COLS + \
           ["mean_width_months", "median_width_months", "n_test"]
    for c in cols:
        if c not in long.columns:
            long[c] = np.nan
    return long[cols]


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
def build_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    g = long_df.groupby(["method", "confidence"])
    agg = g[INTERVAL_METRIC_COLS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = g.size()
    agg = agg.reset_index()
    agg["__o"] = agg["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return agg.sort_values(["__o", "confidence"]).drop(columns="__o").reset_index(drop=True)


def build_point_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    one = long_df.drop_duplicates(["method", "seed"])
    g = one.groupby("method")
    agg = g[["mae", "rmse", "medae"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = g.size()
    agg = agg.reset_index()
    agg["__o"] = agg["method"].map({m: i for i, m in enumerate(METHOD_ORDER)})
    return agg.sort_values("__o").drop(columns="__o").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _fmt(mean, std, n, dec=3):
    if pd.isna(mean):
        return "--"
    if n is None or n <= 1 or pd.isna(std):
        return f"{mean:.{dec}f}"
    return f"{mean:.{dec}f} ± {std:.{dec}f}"


def _present(summary):
    return [m for m in METHOD_ORDER if m in set(summary["method"])]


def main_tables(summary):
    rows_ms, rows_mean = [], []
    for method in _present(summary):
        sub = summary[summary["method"] == method]
        r_ms, r_mean = {"Method": method}, {"Method": method}
        for conf in C.CONFIDENCES:
            pct = int(round(conf * 100))
            r = sub[sub["confidence"] == round(conf, 2)]
            if r.empty:
                r_ms[f"δ={pct}% PICP"] = r_ms[f"δ={pct}% PINAW"] = "--"
                continue
            r = r.iloc[0]; n = r["n_seeds"]
            r_ms[f"δ={pct}% PICP"] = _fmt(r["picp_percent_mean"], r["picp_percent_std"], n)
            r_ms[f"δ={pct}% PINAW"] = _fmt(r["pinaw_mean"], r["pinaw_std"], n)
            r_mean[f"δ={pct}% PICP"] = round(r["picp_percent_mean"], 3)
            r_mean[f"δ={pct}% PINAW"] = round(r["pinaw_mean"], 3)
        rows_ms.append(r_ms); rows_mean.append(r_mean)
    return pd.DataFrame(rows_ms), pd.DataFrame(rows_mean)


def full_table(summary):
    specs = [("PICP (%)", "picp_percent", 3), ("MPIW", "mpiw", 3),
             ("PINAW", "pinaw", 3), ("Cov. gap (pp)", "coverage_gap_pp", 3),
             ("Miscov.", "miscoverage_rate", 4), ("Int. score", "interval_score", 3)]
    rows = []
    for method in _present(summary):
        sub = summary[summary["method"] == method]
        for conf in C.CONFIDENCES:
            r = sub[sub["confidence"] == round(conf, 2)]
            if r.empty:
                continue
            r = r.iloc[0]; n = r["n_seeds"]
            row = {"Method": method, "Conf.": f"{int(round(conf*100))}%"}
            for label, col, dec in specs:
                row[label] = _fmt(r[f"{col}_mean"], r[f"{col}_std"], n, dec)
            rows.append(row)
    return pd.DataFrame(rows)


def point_table(point_summary):
    rows = []
    for _, r in point_summary.iterrows():
        n = r["n_seeds"]
        rows.append({"Method": r["method"],
                     "MAE": _fmt(r["mae_mean"], r["mae_std"], n),
                     "RMSE": _fmt(r["rmse_mean"], r["rmse_std"], n),
                     "MedAE": _fmt(r["medae_mean"], r["medae_std"], n)})
    return pd.DataFrame(rows)


def save_table(df, tables_dir, name, caption):
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    latex = df.to_latex(index=False, escape=True, caption=caption, label=f"tab:{name}",
                        column_format="l" + "r" * (df.shape[1] - 1))
    (tables_dir / f"{name}.tex").write_text(latex, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _save_fig(fig, fig_dir, name):
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}.png", dpi=150)
    fig.savefig(fig_dir / f"{name}.pdf")
    plt.close(fig)


def fig_reliability(summary, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    lo = min(C.CONFIDENCES) * 100 - 1
    hi = max(C.CONFIDENCES) * 100 + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Ideal")
    for method in _present(summary):
        sub = summary[summary["method"] == method].sort_values("confidence")
        ax.errorbar(sub["confidence"] * 100, sub["picp_percent_mean"],
                    yerr=sub["picp_percent_std"].fillna(0.0), marker="o", capsize=3,
                    color=COLORS.get(method), label=method)
    ax.set_xlabel("Nominal confidence δ (%)"); ax.set_ylabel("PICP (%)")
    ax.set_title("Reliability: empirical vs nominal coverage")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save_fig(fig, fig_dir, "reliability_picp")


def _grouped_bar(summary, fig_dir, mean_col, std_col, ylabel, title, name, hline=None):
    methods = _present(summary)
    confs = C.CONFIDENCES
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(confs))
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, method in enumerate(methods):
        sub = summary[summary["method"] == method]
        means, stds = [], []
        for c in confs:
            r = sub[sub["confidence"] == round(c, 2)]
            means.append(r[mean_col].iloc[0] if not r.empty else np.nan)
            s = r[std_col].iloc[0] if not r.empty else np.nan
            stds.append(0.0 if pd.isna(s) else s)
        ax.bar(x + i * width, means, width, yerr=stds, capsize=2,
               color=COLORS.get(method), label=method)
    if hline is not None:
        for j, c in enumerate(confs):
            ax.plot([x[j] - 0.1, x[j] + 0.8], [hline[c], hline[c]], "k--", lw=1)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels([f"{int(round(c*100))}%" for c in confs])
    ax.set_xlabel("Nominal confidence δ"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    _save_fig(fig, fig_dir, name)


def fig_picp_box(long_df, fig_dir, conf):
    sub = long_df[long_df["confidence"] == round(conf, 2)]
    methods = [m for m in METHOD_ORDER if m in set(sub["method"])]
    data = [sub[sub["method"] == m]["picp_percent"].to_numpy() for m in methods]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bp = ax.boxplot(data, labels=methods, patch_artist=True, showmeans=True)
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS.get(m, "#cccccc")); patch.set_alpha(0.6)
    for i, m in enumerate(methods, start=1):
        vals = sub[sub["method"] == m]["picp_percent"].to_numpy()
        if len(vals) == 1:
            ax.scatter([i], vals, color=COLORS.get(m), zorder=5, s=40, edgecolor="k")
    ax.axhline(conf * 100, color="k", ls="--", lw=1, label=f"Nominal {int(conf*100)}%")
    ax.set_ylabel("PICP (%)"); ax.set_title(f"PICP across seeds at δ={int(conf*100)}%")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    _save_fig(fig, fig_dir, f"picp_boxplot_{int(conf*100)}")


# --------------------------------------------------------------------------- #
# Artifact writer (works on any subset of methods present in long_df)
# --------------------------------------------------------------------------- #
def _write_artifacts(long_df: pd.DataFrame, out_dir):
    """Write metrics_long/summary + tables + figures for the methods in long_df."""
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    long_df.to_csv(out_dir / "metrics_long.csv", index=False)

    summary = build_summary(long_df)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    point_summary = build_point_summary(long_df)
    point_summary.to_csv(out_dir / "point_metrics_summary.csv", index=False)

    main_ms, main_mean = main_tables(summary)
    save_table(main_ms, tables_dir, "main_table_mean_std",
               "PICP (%) and PINAW by method and confidence (mean ± std over seeds).")
    save_table(main_mean, tables_dir, "main_table_mean",
               "PICP (%) and PINAW by method and confidence (mean over seeds).")
    save_table(full_table(summary), tables_dir, "full_metrics_mean_std",
               "Interval metrics by method and confidence (mean ± std over seeds).")
    save_table(point_table(point_summary), tables_dir, "point_metrics",
               "Point-accuracy metrics by method (mean ± std over seeds).")

    fig_reliability(summary, fig_dir)
    nominal = {c: 100.0 * c for c in C.CONFIDENCES}
    _grouped_bar(summary, fig_dir, "picp_percent_mean", "picp_percent_std",
                 "PICP (%)", "Empirical coverage by method", "picp_by_confidence", hline=nominal)
    _grouped_bar(summary, fig_dir, "pinaw_mean", "pinaw_std",
                 "PINAW", "Normalized interval width by method", "pinaw_by_confidence")
    _grouped_bar(summary, fig_dir, "mpiw_mean", "mpiw_std",
                 "MPIW (months)", "Interval width by method", "mpiw_by_confidence")
    _grouped_bar(summary, fig_dir, "coverage_gap_pp_mean", "coverage_gap_pp_std",
                 "Coverage gap (pp)", "Coverage gap by method", "coverage_gap_by_confidence")
    _grouped_bar(summary, fig_dir, "interval_score_mean", "interval_score_std",
                 "Interval score", "Interval score by method", "interval_score_by_confidence")
    fig_picp_box(long_df, fig_dir, conf=C.PLOT_CONFIDENCE)

    meta = {
        "methods": _present(summary),
        "seeds": sorted(int(s) for s in long_df["seed"].unique()),
        "confidences": C.CONFIDENCES,
        "note": C.SPLIT_NOTE,
    }
    (out_dir / "report_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return summary, main_ms


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_report():
    long_df = build_long()
    if long_df.empty:
        print("No metrics found. Run the training pipeline first.")
        return None

    # 1) Combined report (both methods) at the comparison root.
    _, main_ms = _write_artifacts(long_df, C.OUTPUT_ROOT)
    print(f"Combined report -> {C.OUTPUT_ROOT}")
    print(main_ms.to_string(index=False))

    # 2) A separate, self-contained report per method, inside its own folder
    #    (next to that method's seed_XX folders).
    for method in C.METHODS:
        display = C.METHOD_DISPLAY_NAMES[method]
        sub = long_df[long_df["method"] == display].copy()
        if sub.empty:
            continue
        method_dir = C.OUTPUT_ROOT / display.replace(" ", "_") / "report"
        _write_artifacts(sub, method_dir)
        print(f"  {display} report -> {method_dir}")

    return long_df


if __name__ == "__main__":
    build_report()
