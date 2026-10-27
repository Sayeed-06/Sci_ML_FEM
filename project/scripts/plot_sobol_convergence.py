#!/usr/bin/env python3
"""Plot Sobol index convergence over increasing base sample sizes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


QOI_LABELS = {
    "mean_abs_w": "Mean |w|",
    "median_abs_w": "Median |w|",
    "p95_abs_w": "95th percentile |w|",
    "max_abs_w": "Maximum |w|",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    qois = [name for name in QOI_LABELS if name in set(frame.qoi)]
    parameters = list(dict.fromkeys(frame.parameter))
    palette = ["#2F6BFF", "#E3A52B", "#C05A9D", "#597A34", "#D46B2E", "#4B5F80"]
    colors = dict(zip(parameters, palette))

    fig, axes = plt.subplots(len(qois), 2, figsize=(15, 3.15 * len(qois) + 1.5), sharex=True)
    for row, qoi in enumerate(qois):
        for column, index_name in enumerate(("S1", "ST")):
            ax = axes[row, column]
            subset = frame[(frame.qoi == qoi) & (frame["index"] == index_name)]
            for parameter in parameters:
                series = subset[subset.parameter == parameter].sort_values("sample_size")
                ax.errorbar(
                    series.sample_size,
                    series.estimate,
                    yerr=series.confidence_half_width,
                    marker="o", markersize=4, linewidth=1.4, capsize=2,
                    color=colors[parameter], label=parameter,
                )
            ax.axhline(0, color="#9AA3B2", linewidth=0.8)
            ax.grid(color="#D9DEE7", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel(QOI_LABELS[qoi])
            if row == 0:
                ax.set_title(
                    "First-order effect, S1" if index_name == "S1" else "Total-order effect, ST",
                    fontweight="bold", fontsize=13,
                )
            if row == len(qois) - 1:
                ax.set_xlabel("Sobol base sample size, N")
                ax.set_xscale("log", base=2)
                ax.set_xticks(sorted(frame.sample_size.unique()))
                ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.93), ncol=6, frameon=False)
    fig.suptitle("Sobol convergence and 95% confidence half-widths", x=0.07, ha="left",
                 fontsize=19, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.09, hspace=0.30, wspace=0.18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
