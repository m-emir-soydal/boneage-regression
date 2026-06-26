"""Compare conformal-prediction PICP/PINAW across image backbones.

Reads the per-backbone conformal metrics produced by ``conformal.evaluate``
(``results/cps/<RUN_TAG>/<backbone>/conformal/metrics_long.csv``) for every
backbone in ``cps_config.COMPARE_BACKBONES`` and writes a side-by-side summary
table plus PICP/PINAW comparison plots to ``results/cps/<RUN_TAG>/comparison/``.

Run this after evaluating each backbone:

    # in cps_config.py set BACKBONE = "efficientnet_b3", then
    python cps/run_all.py && python -m conformal.evaluate
    # set BACKBONE = "vit_b_16", then
    python cps/run_all.py && python -m conformal.evaluate
    # finally
    python compare_backbones.py
"""

import sys

import numpy as np
import pandas as pd

import cps_config as C


METRICS = ["picp", "picp_percent", "pinaw", "mpiw", "coverage_gap_pp", "interval_score"]


def _metrics_path(backbone):
    return C.RUN_TAG_DIR / backbone / "conformal" / "metrics_long.csv"


def _load_backbone_metrics():
    frames = []
    missing = []
    for backbone in C.COMPARE_BACKBONES:
        path = _metrics_path(backbone)
        if not path.exists():
            missing.append((backbone, path))
            continue
        df = pd.read_csv(path)
        df.insert(0, "backbone", backbone)
        frames.append(df)

    if missing:
        for backbone, path in missing:
            print(f"  warning: no metrics for backbone '{backbone}' at {path}")
    if not frames:
        raise FileNotFoundError(
            "No backbone metrics found. Run cps/run_all.py and conformal.evaluate "
            "for each backbone in COMPARE_BACKBONES first."
        )
    return pd.concat(frames, ignore_index=True)


def _summarize(long_df):
    group_cols = ["backbone", "method", "confidence"]
    available = [m for m in METRICS if m in long_df.columns]
    summary = (
        long_df.groupby(group_cols)[available]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary


def _wide_pivot(summary, metric):
    mean_col = f"{metric}_mean"
    if mean_col not in summary.columns:
        return None
    wide = summary.pivot_table(
        index=["method", "confidence"],
        columns="backbone",
        values=mean_col,
    ).reset_index()
    wide.columns.name = None
    backbones = [b for b in C.COMPARE_BACKBONES if b in wide.columns]
    if len(backbones) == 2:
        a, b = backbones
        wide[f"delta_{b}_minus_{a}"] = wide[b] - wide[a]
    return wide


def _make_plots(summary, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping comparison plots.")
        return

    methods = sorted(summary["method"].unique())
    backbones = [b for b in C.COMPARE_BACKBONES if b in set(summary["backbone"])]

    for metric, ylabel, add_diagonal in [
        ("picp", "PICP (coverage)", True),
        ("pinaw", "PINAW (normalized width)", False),
    ]:
        mean_col = f"{metric}_mean"
        if mean_col not in summary.columns:
            continue

        n = len(methods)
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False
        )

        for idx, method in enumerate(methods):
            ax = axes[idx // ncols][idx % ncols]
            for backbone in backbones:
                sub = summary[
                    (summary["method"] == method) & (summary["backbone"] == backbone)
                ].sort_values("confidence")
                if sub.empty:
                    continue
                ax.plot(
                    sub["confidence"],
                    sub[mean_col],
                    marker="o",
                    linewidth=2,
                    label=backbone,
                )
            if add_diagonal:
                lo = float(summary["confidence"].min())
                hi = float(summary["confidence"].max())
                ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="ideal")
            ax.set_title(method)
            ax.set_xlabel("Target confidence")
            ax.set_ylabel(ylabel)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=9)

        for j in range(len(methods), nrows * ncols):
            fig.delaxes(axes[j // ncols][j % ncols])

        fig.suptitle(f"{ylabel} by backbone", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plot_path = out_dir / f"compare_{metric}.png"
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
        print(f"Saved {plot_path}")


def main():
    out_dir = C.comparison_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df = _load_backbone_metrics()
    long_path = out_dir / "metrics_long_all_backbones.csv"
    long_df.to_csv(long_path, index=False)

    summary = _summarize(long_df)
    summary_path = out_dir / "metrics_summary_by_backbone.csv"
    summary.to_csv(summary_path, index=False)

    for metric in ["picp", "pinaw"]:
        wide = _wide_pivot(summary, metric)
        if wide is not None:
            wide_path = out_dir / f"compare_{metric}_wide.csv"
            wide.to_csv(wide_path, index=False)
            print(f"Saved {wide_path}")

    _make_plots(summary, out_dir)

    # Console snapshot at the headline table confidences.
    table = summary[summary["confidence"].isin(C.TABLE_CONFIDENCES)]
    cols = [
        c
        for c in ["backbone", "method", "confidence", "picp_mean", "pinaw_mean"]
        if c in table.columns
    ]
    print("\nPICP / PINAW at table confidences:")
    print(table[cols].to_string(index=False))
    print(f"\nSaved comparison artifacts to {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
