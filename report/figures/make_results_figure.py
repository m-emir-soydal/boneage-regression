"""Generate the UQ results comparison figure for the report.

Plots calibration (PICP@90/95 vs nominal) and interval sharpness (MPIW@90/95)
across the reported UQ configurations. Values are taken from the final test-split
metrics under ``UQ/results/`` (n = 1425 test radiographs).

Run with the project environment (has matplotlib):
    ~/miniconda3/envs/rsna-boneage/bin/python report/figures/make_results_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

# Short labels for the reported configurations (test split, n = 1425).
methods = [
    "MC Dropout\n(Uncalibrated)",
    "MC Dropout\n(Calibrated)",
    "UQ-Aware\nMC Dropout",
    "Heteroscedastic",
]

picp90 = [78.74, 92.98, 89.12, 90.11]
picp95 = [83.51, 95.79, 93.47, 94.11]
mpiw90 = [27.73, 54.22, 40.69, 35.35]
mpiw95 = [33.04, 64.61, 48.48, 42.12]

x = np.arange(len(methods))
width = 0.38

fig, (ax_cov, ax_width) = plt.subplots(1, 2, figsize=(12, 4.8))

# --- Panel A: coverage (PICP) vs nominal levels ---
b1 = ax_cov.bar(x - width / 2, picp90, width, label="PICP@90", color="#4C72B0")
b2 = ax_cov.bar(x + width / 2, picp95, width, label="PICP@95", color="#DD8452")
ax_cov.axhline(90, ls="--", lw=1.2, color="#4C72B0", alpha=0.9)
ax_cov.axhline(95, ls="--", lw=1.2, color="#DD8452", alpha=0.9)
ax_cov.text(len(methods) - 0.45, 90.3, "nominal 90%", fontsize=8, color="#4C72B0")
ax_cov.text(len(methods) - 0.45, 95.3, "nominal 95%", fontsize=8, color="#DD8452")
ax_cov.set_ylabel("Coverage / PICP (%)")
ax_cov.set_title("(a) Calibration: empirical coverage vs nominal")
ax_cov.set_ylim(70, 100)
ax_cov.set_xticks(x)
ax_cov.set_xticklabels(methods, fontsize=8)
ax_cov.legend(loc="lower right", fontsize=8)
ax_cov.bar_label(b1, fmt="%.1f", padding=2, fontsize=7)
ax_cov.bar_label(b2, fmt="%.1f", padding=2, fontsize=7)

# --- Panel B: interval width (MPIW) ---
c1 = ax_width.bar(x - width / 2, mpiw90, width, label="MPIW@90", color="#4C72B0")
c2 = ax_width.bar(x + width / 2, mpiw95, width, label="MPIW@95", color="#DD8452")
ax_width.set_ylabel("Mean interval width / MPIW (months)")
ax_width.set_title("(b) Sharpness: prediction interval width")
ax_width.set_ylim(0, 75)
ax_width.set_xticks(x)
ax_width.set_xticklabels(methods, fontsize=8)
ax_width.legend(loc="upper right", fontsize=8)
ax_width.bar_label(c1, fmt="%.1f", padding=2, fontsize=7)
ax_width.bar_label(c2, fmt="%.1f", padding=2, fontsize=7)

for ax in (ax_cov, ax_width):
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)

fig.tight_layout()

pdf_path = OUT_DIR / "uq_results_comparison.pdf"
png_path = OUT_DIR / "uq_results_comparison.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
print(f"Saved: {pdf_path}")
print(f"Saved: {png_path}")
