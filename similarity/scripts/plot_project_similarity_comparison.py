#!/usr/bin/env python3
"""Compare project and similarity-model Sobol indices on aligned axes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


QOIS = [
    ("mean_abs_w", "Mean |w|"),
    ("median_abs_w", "Median |w|"),
    ("p95_abs_w", "95th percentile |w|"),
    ("max_abs_w", "Maximum |w|"),
]
PARAMETER_LABELS = ["E", "thickness", "k_winkler", "delta_T", "alpha_T", "pressure_q"]
COLORS = {"Project DeepONet": "#2F6BFF", "Similarity DeepONet": "#E3A52B"}


def load_qoi(directory: Path, qoi: str) -> dict:
    with (directory / f"{qoi}_sobol.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--similarity-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fig, axes = plt.subplots(4, 2, figsize=(15, 15.5), sharex=True, sharey=True)
    y = np.arange(len(PARAMETER_LABELS))
    height = 0.34
    for row, (qoi, qoi_label) in enumerate(QOIS):
        project = load_qoi(args.project_dir, qoi)
        similarity = load_qoi(args.similarity_dir, qoi)
        for column, index_name in enumerate(("S1", "ST")):
            ax = axes[row, column]
            p = np.clip(np.asarray(project[index_name], dtype=float), 0.0, None)
            s = np.clip(np.asarray(similarity[index_name], dtype=float), 0.0, None)
            ax.barh(y - height / 2, p, height, color=COLORS["Project DeepONet"],
                    alpha=0.82, label="Project DeepONet")
            ax.barh(y + height / 2, s, height, color=COLORS["Similarity DeepONet"],
                    alpha=0.82, label="Similarity DeepONet")
            ax.set_yticks(y, PARAMETER_LABELS)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.05)
            ax.grid(axis="x", color="#D9DEE7", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_ylabel(qoi_label, fontweight="bold")
            if row == 0:
                ax.set_title(
                    "First-order effect, S1" if index_name == "S1" else "Total-order effect, ST",
                    fontsize=13, fontweight="bold",
                )
            if row == len(QOIS) - 1:
                ax.set_xlabel("Fraction of output variance")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.935),
               ncol=2, frameon=False)
    fig.suptitle("Sobol sensitivity: project vs similarity representation",
                 x=0.07, ha="left", fontsize=19, fontweight="bold")
    fig.text(
        0.07, 0.947,
        "Same physical input ranges, fixed geometry, base sample N=1024. Negative finite-sample estimates are displayed as zero.",
        ha="left", fontsize=9.5, color="#697386",
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.89, bottom=0.06, hspace=0.25, wspace=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
