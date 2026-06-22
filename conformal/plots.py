import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cps_config as C
from cps.io_utils import read_table


METHOD_ORDER = ["SCP", "KNN-NCP", "AS-MCP", "BCP"]
METHOD_COLORS = {
    "SCP": "#4C78A8",
    "KNN-NCP": "#F58518",
    "AS-MCP": "#54A24B",
    "BCP": "#B279A2",
}
METHOD_MARKERS = {
    "SCP": "o",
    "KNN-NCP": "s",
    "AS-MCP": "^",
    "BCP": "D",
}
METHOD_LINESTYLES = {
    "SCP": "-",
    "KNN-NCP": "--",
    "AS-MCP": "-.",
    "BCP": ":",
}
METHOD_OFFSETS = {
    "SCP": -0.9,
    "KNN-NCP": -0.3,
    "AS-MCP": 0.3,
    "BCP": 0.9,
}


def _save(fig, path_base, rect=None):
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(path_base.with_suffix(".png"), dpi=300)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def _confidence_suffix(confidence):
    return str(int(round(confidence * 100)))


def _style_axis(ax):
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.30)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _coverage_ticklabels(confidences):
    return [f"{int(round(c * 100))}%" for c in confidences]


def _calibration_curve(metrics, fig_dir):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    x_ref = np.asarray(C.CONFIDENCES) * 100.0
    ax.plot(
        x_ref,
        x_ref,
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="Calibration line",
        zorder=1,
    )

    for method in METHOD_ORDER:
        part = metrics[metrics["method"] == method]
        grouped = part.groupby("confidence")["picp"].agg(["mean", "std"]).reset_index()
        x = grouped["confidence"].values * 100.0 + METHOD_OFFSETS[method]
        ax.errorbar(
            x,
            grouped["mean"].values * 100.0,
            yerr=grouped["std"].fillna(0.0).values * 100.0,
            marker=METHOD_MARKERS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2,
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.6,
            capsize=4,
            label=method,
            color=METHOD_COLORS[method],
            zorder=3,
        )

    ax.set_xlabel("Desired coverage")
    ax.set_ylabel("Empirical PICP")
    ax.set_xticks(x_ref)
    ax.set_xticklabels(_coverage_ticklabels(C.CONFIDENCES))
    ax.set_xlim(max(0.0, x_ref.min() - 6.0), min(100.0, x_ref.max() + 6.0))
    ax.set_ylim(0.0, 100.0)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)])
    _style_axis(ax)
    ax.legend(loc="upper left", frameon=True)
    _save(fig, fig_dir / "calibration_curve")


def _boxplot_gap_pinaw(metrics, fig_dir):
    table_metrics = metrics[metrics["confidence"].isin(C.TABLE_CONFIDENCES)]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.4), sharex=True)
    offsets = np.linspace(-0.30, 0.30, len(METHOD_ORDER))
    width = 0.14
    group_positions = np.arange(1, len(C.TABLE_CONFIDENCES) + 1)

    for ax, column, ylabel in [
        (axes[0], "coverage_gap_pp", "Coverage gap (pp)"),
        (axes[1], "pinaw", "PINAW"),
    ]:
        for i, center in enumerate(group_positions):
            if i % 2 == 1:
                ax.axvspan(center - 0.5, center + 0.5, color="#F2F2F2", zorder=0)

        for j, method in enumerate(METHOD_ORDER):
            values = []
            positions = []
            for i, confidence in enumerate(C.TABLE_CONFIDENCES):
                part = table_metrics[
                    (table_metrics["method"] == method)
                    & (table_metrics["confidence"].round(6) == round(confidence, 6))
                ]
                values.append(part[column].values)
                positions.append(group_positions[i] + offsets[j])
            bp = ax.boxplot(
                values,
                positions=positions,
                widths=width,
                patch_artist=True,
                boxprops={"linewidth": 1.2},
                whiskerprops={"linewidth": 1.2},
                capprops={"linewidth": 1.2},
                medianprops={"color": "black", "linewidth": 1.3},
                flierprops={
                    "marker": "o",
                    "markersize": 4,
                    "markerfacecolor": "white",
                    "markeredgecolor": METHOD_COLORS[method],
                    "alpha": 0.9,
                },
            )
            for box in bp["boxes"]:
                box.set_facecolor(METHOD_COLORS[method])
                box.set_alpha(0.65)

        ax.set_ylabel(ylabel)
        _style_axis(ax)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_xticks(group_positions)
    axes[1].set_xticklabels(_coverage_ticklabels(C.TABLE_CONFIDENCES))
    axes[1].set_xlabel("Desired coverage")
    axes[0].set_xlim(0.5, len(C.TABLE_CONFIDENCES) + 0.5)

    handles = [
        plt.Line2D([0], [0], color=METHOD_COLORS[m], linewidth=8, alpha=0.65, label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(METHOD_ORDER), frameon=True)
    _save(fig, fig_dir / "boxplot_gap_pinaw", rect=(0.0, 0.0, 1.0, 0.95))


def _width_distribution(intervals, fig_dir, confidence):
    df = intervals[intervals["confidence"].round(6) == round(confidence, 6)]
    values = [df[df["method"] == method]["width"].values for method in METHOD_ORDER]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(values, patch_artist=True)
    ax.set_xticks(range(1, len(METHOD_ORDER) + 1))
    ax.set_xticklabels(METHOD_ORDER)
    for box, method in zip(bp["boxes"], METHOD_ORDER):
        box.set_facecolor(METHOD_COLORS[method])
        box.set_alpha(0.65)
    ax.set_ylabel("Interval width (months)")
    _style_axis(ax)
    _save(fig, fig_dir / f"width_distribution_{_confidence_suffix(confidence)}")


def _subgroup_coverage(intervals, fig_dir, confidence):
    df = intervals[intervals["confidence"].round(6) == round(confidence, 6)].copy()
    age_bins = pd.qcut(df["y_pred"], q=5, labels=False, duplicates="drop")
    if age_bins.isna().all():
        age_bins = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    df["predicted_age_bin"] = age_bins.fillna(0).astype(int)
    df["covered"] = (df["y_true"] >= df["lower"]) & (df["y_true"] <= df["upper"])
    df["pinaw_row"] = df["width"] / (C.Y_MAX - C.Y_MIN)

    grouped = df.groupby(["method", "sex", "predicted_age_bin"]).agg(
        n=("id", "size"),
        PICP=("covered", "mean"),
        PINAW=("pinaw_row", "mean"),
    ).reset_index()
    grouped["coverage_gap_pp"] = 100.0 * (confidence - grouped["PICP"])
    suffix = _confidence_suffix(confidence)
    grouped.to_csv(fig_dir / f"subgroup_coverage_{suffix}.csv", index=False)

    grouped["row"] = grouped.apply(
        lambda r: f"{r['method']} | sex={int(r['sex'])}", axis=1
    )
    rows = [f"{m} | sex={s}" for m in METHOD_ORDER for s in sorted(grouped["sex"].unique())]
    cols = sorted(grouped["predicted_age_bin"].unique())
    heat = grouped.pivot(index="row", columns="predicted_age_bin", values="coverage_gap_pp")
    heat = heat.reindex(index=rows, columns=cols)
    matrix = heat.values.astype(float)
    max_abs = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    max_abs = max(max_abs, 1.0)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"bin {int(c)}" for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_xlabel("Predicted-age quantile bin")
    ax.set_title(f"Subgroup coverage gap at {suffix}%")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Coverage gap (pp)")
    _save(fig, fig_dir / f"subgroup_coverage_{suffix}")


def _width_vs_error(intervals, fig_dir, confidence):
    df = intervals[intervals["confidence"].round(6) == round(confidence, 6)]
    suffix = _confidence_suffix(confidence)
    for method in METHOD_ORDER:
        part = df[df["method"] == method]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(
            part["width"],
            part["abs_error"],
            s=18,
            alpha=0.55,
            color=METHOD_COLORS[method],
            edgecolors="white",
            linewidths=0.3,
        )
        ax.set_xlabel("Interval width (months)")
        ax.set_ylabel("Absolute error (months)")
        ax.set_title(method)
        _style_axis(ax)
        _save(fig, fig_dir / f"width_vs_error_{method}_{suffix}")


def main():
    conformal_dir = C.RUN_OUTPUT_DIR / "conformal"
    fig_dir = conformal_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(conformal_dir / "metrics_long.csv")
    intervals = read_table(conformal_dir / "intervals_long.parquet")
    plot_confidence = C.selected_plot_confidence()

    _calibration_curve(metrics, fig_dir)
    _boxplot_gap_pinaw(metrics, fig_dir)
    _width_distribution(intervals, fig_dir, plot_confidence)
    _subgroup_coverage(intervals, fig_dir, plot_confidence)
    _width_vs_error(intervals, fig_dir, plot_confidence)
    print(f"Saved figures to {fig_dir}")


if __name__ == "__main__":
    main()
