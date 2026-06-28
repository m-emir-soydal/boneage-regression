import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cps")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sweep_test.config as S


plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.edgecolor": "#2F2F2F",
    "axes.linewidth": 0.9,
    "xtick.color": "#2F2F2F",
    "ytick.color": "#2F2F2F",
    "text.color": "#222222",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


METHOD_ORDER = ["SCP", "AS-MCP", "KNN-NCP"]
METHOD_COLORS = {
    # Okabe-Ito inspired, colorblind-safe colors.
    "SCP": "#0072B2",
    "AS-MCP": "#009E73",
    "KNN-NCP": "#D55E00",
}
METHOD_MARKERS = {
    "SCP": "o",
    "AS-MCP": "D",
    "KNN-NCP": "s",
}
METHOD_LINESTYLES = {
    "SCP": "-",
    "AS-MCP": "-.",
    "KNN-NCP": "--",
}


def _save(fig, path_base):
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _style_axis(ax):
    ax.set_facecolor("white")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.75, color="#E6E6E6", alpha=1.0)
    ax.grid(True, axis="x", linestyle="-", linewidth=0.45, color="#F0F0F0", alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3, width=0.8, color="#444444")
    ax.margins(x=0.03)


def _legend_above(ax, ncol=3):
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=True,
        framealpha=0.98,
        edgecolor="#DDDDDD",
        handlelength=2.3,
        columnspacing=1.4,
        borderpad=0.45,
    )


def _percent_text(value):
    percent = float(value) * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return f"{int(round(percent))}%"
    return f"{percent:.1f}%"


def _percent_labels(values):
    return [_percent_text(v) for v in values]


def _set_budget_axis(ax, fractions):
    fractions = sorted(float(v) for v in fractions)
    ticks = [v * 100.0 for v in fractions]
    ax.set_xticks(ticks)
    ax.set_xticklabels(_percent_labels(fractions))
    if ticks:
        ax.set_xlim(min(ticks) - 1.0, max(ticks) + 1.0)


def _confidence_suffix(confidence):
    return str(int(round(confidence * 100)))


def _summary(metrics):
    return metrics.groupby(
        ["method", "confidence", "cal_fraction", "scale_fraction"], dropna=False
    ).agg(
        picp_mean=("picp", "mean"),
        picp_std=("picp", "std"),
        gap_mean=("coverage_gap_pp", "mean"),
        gap_std=("coverage_gap_pp", "std"),
        pinaw_mean=("pinaw", "mean"),
        pinaw_std=("pinaw", "std"),
        interval_score_mean=("interval_score", "mean"),
        interval_score_std=("interval_score", "std"),
        mean_width_months_mean=("mean_width_months", "mean"),
        mean_width_months_std=("mean_width_months", "std"),
    ).reset_index()


def _plot_mean_band(ax, part, y_col, yerr_col, method, y_scale=1.0):
    x = part["cal_fraction"].to_numpy(dtype=float) * 100.0
    y = part[y_col].to_numpy(dtype=float) * y_scale
    yerr = part[yerr_col].fillna(0.0).to_numpy(dtype=float) * y_scale
    color = METHOD_COLORS[method]

    if len(x) > 1 and np.any(yerr > 0):
        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            color=color,
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )

    ax.plot(
        x,
        y,
        marker=METHOD_MARKERS[method],
        linestyle=METHOD_LINESTYLES[method],
        linewidth=2.15,
        markersize=5.8,
        markerfacecolor="white",
        markeredgewidth=1.45,
        label=method,
        color=color,
        zorder=3,
    )


def _line_metric(
    summary,
    confidence,
    y_col,
    yerr_col,
    ylabel,
    fig_path,
    zero_line=False,
    reference=None,
    y_scale=1.0,
):
    part_conf = summary[summary["confidence"].round(6) == round(confidence, 6)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in METHOD_ORDER:
        part = part_conf[part_conf["method"] == method].sort_values("cal_fraction")
        if part.empty:
            continue
        _plot_mean_band(ax, part, y_col, yerr_col, method, y_scale=y_scale)
    if zero_line:
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.05, zorder=2)
    if reference is not None:
        ax.axhline(
            reference,
            color="#333333",
            linestyle="--",
            linewidth=1.05,
            label="Target",
            zorder=2,
        )
    ticks = sorted(part_conf["cal_fraction"].unique())
    _set_budget_axis(ax, ticks)
    ax.set_xlabel("Calibration budget (% of source train.csv)")
    ax.set_ylabel(ylabel)
    _style_axis(ax)
    _legend_above(ax, ncol=4 if reference is not None else 3)
    _save(fig, fig_path)


def _metric_panel(summary, confidence, fig_dir):
    part_conf = summary[summary["confidence"].round(6) == round(confidence, 6)]
    if part_conf.empty:
        print(f"Skipping {int(round(confidence * 100))}% metric panel: no rows found.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 8.2), sharex=True)
    for ax, y_col, yerr_col, ylabel, zero_line in [
        (axes[0], "gap_mean", "gap_std", "Coverage gap (pp)", True),
        (axes[1], "pinaw_mean", "pinaw_std", "PINAW", False),
        (axes[2], "interval_score_mean", "interval_score_std", "Mean interval score", False),
    ]:
        for method in METHOD_ORDER:
            part = part_conf[part_conf["method"] == method].sort_values("cal_fraction")
            if part.empty:
                continue
            _plot_mean_band(ax, part, y_col, yerr_col, method)
        if zero_line:
            ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.05, zorder=2)
        ax.set_ylabel(ylabel)
        _style_axis(ax)
    ticks = sorted(part_conf["cal_fraction"].unique())
    _set_budget_axis(axes[-1], ticks)
    axes[-1].set_xlabel("Calibration budget (% of source train.csv)")
    _legend_above(axes[0], ncol=3)
    suffix = _confidence_suffix(confidence)
    _save(fig, fig_dir / f"gap_pinaw_interval_score_vs_cal_fraction_{suffix}")


def _calibration_curve(metrics, fig_dir, budget_fraction):
    candidates = metrics[metrics["cal_fraction"].round(6) == round(budget_fraction, 6)]
    if candidates.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    confidences = sorted(candidates["confidence"].unique())
    x_ref = np.asarray(confidences) * 100.0
    ax.plot(
        x_ref,
        x_ref,
        color="#222222",
        linestyle="--",
        linewidth=1.45,
        label="Calibration line",
        zorder=2,
    )
    for method in METHOD_ORDER:
        part = candidates[candidates["method"] == method]
        grouped = part.groupby("confidence")["picp"].agg(["mean", "std"]).reset_index()
        if grouped.empty:
            continue
        x = grouped["confidence"].to_numpy(dtype=float) * 100.0
        y = grouped["mean"].to_numpy(dtype=float) * 100.0
        yerr = grouped["std"].fillna(0.0).to_numpy(dtype=float) * 100.0
        if len(x) > 1 and np.any(yerr > 0):
            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=METHOD_COLORS[method],
                alpha=0.13,
                linewidth=0,
                zorder=1,
            )
        ax.plot(
            x,
            y,
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.15,
            markersize=5.8,
            markerfacecolor="white",
            markeredgewidth=1.45,
            label=method,
            color=METHOD_COLORS[method],
            zorder=3,
        )
    ax.set_xlabel("Desired coverage")
    ax.set_ylabel("Empirical PICP (%)")
    ax.set_xticks(x_ref)
    ax.set_xticklabels(_percent_labels(confidences))
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)])
    ax.set_title(f"Calibration curve at cal={_percent_text(budget_fraction)}")
    _style_axis(ax)
    _legend_above(ax, ncol=2)
    suffix = str(int(round(budget_fraction * 1000)))
    _save(fig, fig_dir / f"calibration_curve_cal_{suffix}")


def _runtime_plot(runtime, fig_dir):
    grouped = runtime.groupby(["method", "cal_fraction", "scale_fraction"], dropna=False).agg(
        fit_time_sec_mean=("fit_time_sec", "mean"),
        fit_time_sec_std=("fit_time_sec", "std"),
        predict_ms_mean=("predict_ms_per_sample_conf", "mean"),
        predict_ms_std=("predict_ms_per_sample_conf", "std"),
    ).reset_index()
    for y_col, yerr_col, ylabel, name in [
        ("fit_time_sec_mean", "fit_time_sec_std", "Fit time (sec)", "fit_time_vs_cal_fraction"),
        ("predict_ms_mean", "predict_ms_std", "Prediction time (ms/sample/conf)", "predict_time_vs_cal_fraction"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for method in METHOD_ORDER:
            part = grouped[grouped["method"] == method].sort_values("cal_fraction")
            if part.empty:
                continue
            _plot_mean_band(ax, part, y_col, yerr_col, method)
        ticks = sorted(grouped["cal_fraction"].unique())
        _set_budget_axis(ax, ticks)
        ax.set_xlabel("Calibration budget (% of source train.csv)")
        ax.set_ylabel(ylabel)
        _style_axis(ax)
        _legend_above(ax, ncol=3)
        _save(fig, fig_dir / name)


def build_plots(input_dir, plot_confidences, calibration_curve_budgets):
    input_dir = Path(input_dir)
    fig_dir = input_dir / "figures"
    metrics = pd.read_csv(input_dir / "metrics_long.csv")
    runtime = pd.read_csv(input_dir / "runtime_long.csv")
    summary = _summary(metrics)

    for plot_confidence in plot_confidences:
        _metric_panel(summary, plot_confidence, fig_dir)
        suffix = _confidence_suffix(plot_confidence)
        _line_metric(
            summary,
            plot_confidence,
            "picp_mean",
            "picp_std",
            "Empirical PICP (%)",
            fig_dir / f"picp_vs_cal_fraction_{suffix}",
            reference=plot_confidence * 100.0,
            y_scale=100.0,
        )
        _line_metric(
            summary,
            plot_confidence,
            "interval_score_mean",
            "interval_score_std",
            "Mean interval score",
            fig_dir / f"interval_score_vs_cal_fraction_{suffix}",
        )
        _line_metric(
            summary,
            plot_confidence,
            "mean_width_months_mean",
            "mean_width_months_std",
            "Mean interval width (months)",
            fig_dir / f"mean_width_vs_cal_fraction_{suffix}",
        )
    for budget in calibration_curve_budgets:
        _calibration_curve(metrics, fig_dir, budget)
    _runtime_plot(runtime, fig_dir)
    print(f"Saved sweep figures to {fig_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build plots for sweep_test results.")
    parser.add_argument("--input-dir", type=Path, default=S.OUTPUT_DIR)
    parser.add_argument(
        "--plot-confidence",
        type=float,
        default=None,
        help="Backward-compatible single confidence to plot.",
    )
    parser.add_argument(
        "--plot-confidences",
        type=float,
        nargs="+",
        default=None,
        help="One or more confidence levels to plot.",
    )
    parser.add_argument(
        "--calibration-curve-budgets",
        type=float,
        nargs="+",
        default=[0.125, 0.25],
        help="Calibration fractions for calibration-curve plots.",
    )
    args = parser.parse_args()
    if args.plot_confidences is not None:
        plot_confidences = args.plot_confidences
    elif args.plot_confidence is not None:
        plot_confidences = [args.plot_confidence]
    else:
        plot_confidences = S.PLOT_CONFIDENCES
    build_plots(args.input_dir, plot_confidences, args.calibration_curve_budgets)


if __name__ == "__main__":
    main()
