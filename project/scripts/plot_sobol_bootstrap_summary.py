#!/usr/bin/env python3
"""Plot bootstrap distributions of Sobol S1 and ST estimates."""

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
    final_n = frame["sample_size"].max()
    frame = frame[frame["sample_size"] == final_n]
    qois = [name for name in QOI_LABELS if name in set(frame["qoi"])]
    parameters = list(dict.fromkeys(frame["parameter"]))

    fig, axes = plt.subplots(
        len(qois), 2, figsize=(15, 3.1 * len(qois) + 1.8),
        sharex=True, constrained_layout=False,
    )
    colors = {"S1": "#2F6BFF", "ST": "#E3A52B"}
    for row, qoi in enumerate(qois):
        for column, index_name in enumerate(("S1", "ST")):
            ax = axes[row, column]
            groups = [
                frame[(frame.qoi == qoi) & (frame.parameter == parameter) & (frame["index"] == index_name)]["value"].dropna().values
                for parameter in parameters
            ]
            box = ax.boxplot(
                groups,
                positions=range(len(parameters)),
                widths=0.58,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="#20242B", markersize=5),
                medianprops=dict(color="#20242B", linewidth=1.6),
                whiskerprops=dict(color="#697386"),
                capprops=dict(color="#697386"),
                flierprops=dict(marker=".", markersize=2, alpha=0.25),
            )
            for patch in box["boxes"]:
                patch.set_facecolor(colors[index_name])
                patch.set_alpha(0.72)
                patch.set_edgecolor("#20242B")
            ax.axhline(0, color="#9AA3B2", linewidth=0.8)
            ax.grid(axis="y", color="#D9DEE7", linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel(QOI_LABELS[qoi])
            if row == 0:
                ax.set_title(
                    "First-order effect, S1" if index_name == "S1" else "Total-order effect, ST",
                    fontsize=13, fontweight="bold",
                )
            if row == len(qois) - 1:
                ax.set_xticks(range(len(parameters)), parameters, rotation=30, ha="right")
            else:
                ax.set_xticks(range(len(parameters)), [])

    fig.suptitle(
        f"Sobol estimate variation across bootstrap resamples (N={final_n})",
        x=0.07, ha="left", fontsize=19, fontweight="bold",
    )
    fig.text(
        0.07, 0.94,
        "Box = interquartile range; center line = median; diamond = mean; whiskers = 1.5 x IQR.",
        ha="left", color="#697386", fontsize=10,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.11, hspace=0.30, wspace=0.18)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
