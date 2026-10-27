#!/usr/bin/env python3
"""Create a Sobol Variance Budget Map from SALib-style JSON results.

The figure is a coordinated representation rather than a conventional grouped
bar chart:

1. Variance lanes compare S1 (direct/main effect) with ST (total effect).
2. The priority plane separates direct effect from interaction involvement,
   defined as ST - S1.
3. The interaction matrix shows pairwise S2 estimates when they are available.

This script visualises results; it does not estimate Sobol indices.  The input
must come from a valid variance-based Sobol analysis.

Example
-------
python scripts/plot_sobol_variance_map.py \
    --input examples/sobol_results_illustrative.json \
    --output runs/sobol_variance_budget_map.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


BLUE = "#2F6BFF"
GOLD = "#E3A52B"
INK = "#20242B"
MUTED = "#697386"
GRID = "#D9DEE7"
NEGATIVE = "#C84B5A"


def _array(payload: dict[str, Any], key: str, n: int, default: float = 0.0) -> np.ndarray:
    values = payload.get(key, [default] * n)
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n,):
        raise ValueError(f"'{key}' must contain {n} values; got shape {arr.shape}.")
    return arr


def load_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    names = payload.get("names")
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError("'names' must be a non-empty list of parameter names.")

    n = len(names)
    s1 = _array(payload, "S1", n)
    st = _array(payload, "ST", n)
    s1_conf = _array(payload, "S1_conf", n)
    st_conf = _array(payload, "ST_conf", n)

    s2_raw = payload.get("S2")
    s2_conf_raw = payload.get("S2_conf")
    s2 = None if s2_raw is None else np.asarray(s2_raw, dtype=float)
    s2_conf = None if s2_conf_raw is None else np.asarray(s2_conf_raw, dtype=float)
    if s2 is not None and s2.shape != (n, n):
        raise ValueError(f"'S2' must be a {n} x {n} matrix; got shape {s2.shape}.")
    if s2_conf is not None and s2_conf.shape != (n, n):
        raise ValueError(f"'S2_conf' must be a {n} x {n} matrix; got shape {s2_conf.shape}.")

    return {
        "names": names,
        "S1": s1,
        "ST": st,
        "S1_conf": s1_conf,
        "ST_conf": st_conf,
        "S2": s2,
        "S2_conf": s2_conf,
        "metadata": payload.get("metadata", {}),
    }


def _signed_label(value: float) -> str:
    return f"{value:.3f}" if value >= 0 else f"−{abs(value):.3f}*"


def plot_map(results: dict[str, Any], output: Path) -> None:
    names = np.asarray(results["names"])
    s1 = results["S1"]
    st = results["ST"]
    s1_conf = results["S1_conf"]
    st_conf = results["ST_conf"]
    s2 = results["S2"]
    s2_conf = results["S2_conf"]
    metadata = results["metadata"]

    # Sort all factor-level views by total influence.  Keep the S2 matrix in
    # the same order so a reader can move between panels without remapping.
    order = np.argsort(st)[::-1]
    names = names[order]
    s1, st = s1[order], st[order]
    s1_conf, st_conf = s1_conf[order], st_conf[order]
    if s2 is not None:
        s2 = s2[np.ix_(order, order)]
    if s2_conf is not None:
        s2_conf = s2_conf[np.ix_(order, order)]

    interaction_involvement = st - s1
    n = len(names)
    output_name = metadata.get("output", "model output")
    confidence = metadata.get("confidence_level", 0.95)
    confidence_pct = int(round(float(confidence) * 100))
    sample_size = metadata.get("base_sample_size")
    status = metadata.get("status", "estimated")

    fig = plt.figure(figsize=(15.5, max(8.5, 0.62 * n + 5.8)), facecolor="white")
    grid = fig.add_gridspec(
        2, 2, width_ratios=[1.38, 1.0], height_ratios=[1.0, 1.05],
        left=0.08, right=0.97, top=0.86, bottom=0.16, wspace=0.28, hspace=0.38,
    )
    ax_lane = fig.add_subplot(grid[:, 0])
    ax_plane = fig.add_subplot(grid[0, 1])
    ax_matrix = fig.add_subplot(grid[1, 1])

    fig.suptitle("Sobol Variance Budget Map", x=0.08, ha="left", fontsize=22,
                 fontweight="bold", color=INK)
    subtitle = f"Output: {output_name}  |  uncertainty whiskers: {confidence_pct}% confidence half-widths"
    if sample_size is not None:
        subtitle += f"  |  base sample N={sample_size}"
    fig.text(0.08, 0.895, subtitle, ha="left", fontsize=10.5, color=MUTED)
    if status.lower() != "measured":
        fig.text(0.97, 0.91, status.upper(), ha="right", va="center", fontsize=9,
                 color="#7A5A00", bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF4D6", edgecolor="#E8C96B"))

    # Panel A: two nested estimates on a common scale.  We do not turn ST-S1
    # into a stacked variance share because the same interaction is involved in
    # more than one factor's total-order index.
    y = np.arange(n)
    xmin = min(-0.05, float(np.min(np.r_[s1 - s1_conf, st - st_conf])) - 0.03)
    xmax = max(1.0, float(np.max(np.r_[s1 + s1_conf, st + st_conf])) + 0.08)
    ax_lane.axvline(0, color=INK, linewidth=0.8)
    for row, (main, total, main_ci, total_ci) in enumerate(zip(s1, st, s1_conf, st_conf)):
        ax_lane.plot([main, total], [row, row], color=GOLD, linewidth=8,
                     alpha=0.28, solid_capstyle="round", zorder=1)
        ax_lane.errorbar(total, row, xerr=total_ci, fmt="o", color=GOLD,
                         ecolor=GOLD, capsize=3, markersize=7, zorder=3)
        ax_lane.errorbar(main, row, xerr=main_ci, fmt="o", color=BLUE,
                         ecolor=BLUE, capsize=3, markersize=7, zorder=4)
        ax_lane.text(total + 0.022, row, _signed_label(total), va="center",
                     fontsize=9, color=INK)
    ax_lane.set_yticks(y, names)
    ax_lane.invert_yaxis()
    ax_lane.set_xlim(xmin, xmax)
    ax_lane.set_xlabel("Fraction of output variance")
    ax_lane.set_title("A. Direct effect and total influence", loc="left", fontweight="bold", color=INK)
    ax_lane.grid(axis="x", color=GRID, linewidth=0.8)
    ax_lane.spines[["top", "right", "left"]].set_visible(False)
    ax_lane.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE, linewidth=0, label="S1: parameter acting alone"),
            Line2D([0], [0], marker="o", color=GOLD, linewidth=0, label="ST: parameter plus all interactions"),
            Patch(facecolor=GOLD, alpha=0.28, edgecolor="none", label="ST − S1: interaction involvement"),
        ],
        loc="lower right", frameon=False, fontsize=9,
    )

    # Panel B: decision-oriented plane.  The top-right is influential through
    # both a direct response and interactions; the top-left is interaction-led.
    sizes = 90 + 620 * np.clip(np.abs(st), 0, None) / max(float(np.max(np.abs(st))), 1e-12)
    ax_plane.axvline(0, color=GRID, linewidth=0.8)
    ax_plane.axhline(0, color=GRID, linewidth=0.8)
    ax_plane.scatter(s1, interaction_involvement, s=sizes, c=st, cmap="cividis",
                     edgecolor=INK, linewidth=0.7, alpha=0.9)
    for xval, yval, label in zip(s1, interaction_involvement, names):
        ax_plane.annotate(label, (xval, yval), xytext=(5, 5), textcoords="offset points",
                          fontsize=8.5, color=INK)
    ax_plane.set_xlabel("Direct effect, S1")
    ax_plane.set_ylabel("Interaction involvement, ST − S1")
    ax_plane.set_title("B. How each factor matters", loc="left", fontweight="bold", color=INK)
    ax_plane.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax_plane.spines[["top", "right"]].set_visible(False)

    # Panel C: pairwise interactions.  Circle area encodes magnitude; color
    # retains the sign so negative Monte Carlo estimates are not silently hidden.
    ax_matrix.set_title("C. Pairwise interaction estimates, S2", loc="left",
                        fontweight="bold", color=INK)
    if s2 is None:
        ax_matrix.axis("off")
        ax_matrix.text(0.5, 0.55, "S2 was not calculated", ha="center", va="center",
                       fontsize=13, color=MUTED, transform=ax_matrix.transAxes)
        ax_matrix.text(0.5, 0.42, "The S1–ST gap still signals interaction involvement,\n"
                       "but it cannot identify the interacting partner.", ha="center",
                       va="center", fontsize=9.5, color=MUTED, transform=ax_matrix.transAxes)
    else:
        upper_values = [s2[i, j] for i in range(n) for j in range(i + 1, n) if np.isfinite(s2[i, j])]
        limit = max(max((abs(v) for v in upper_values), default=0.0), 0.02)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        cmap = plt.get_cmap("coolwarm")
        ax_matrix.set_xlim(-0.5, n - 0.5)
        ax_matrix.set_ylim(n - 0.5, -0.5)
        ax_matrix.set_xticks(range(n), names, rotation=35, ha="right")
        ax_matrix.set_yticks(range(n), names)
        ax_matrix.set_aspect("equal")
        for i in range(n):
            for j in range(n):
                ax_matrix.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                   facecolor="white", edgecolor=GRID, linewidth=0.7))
                if j <= i or not np.isfinite(s2[i, j]):
                    continue
                value = float(s2[i, j])
                area = 70 + 1150 * min(abs(value) / limit, 1.0)
                ax_matrix.scatter(j, i, s=area, color=cmap(norm(value)),
                                  edgecolor=INK, linewidth=0.5)
                label = f"{value:.2f}"
                if s2_conf is not None and np.isfinite(s2_conf[i, j]) and abs(value) <= s2_conf[i, j]:
                    label += "†"
                ax_matrix.text(j, i, label, ha="center", va="center", fontsize=7.5,
                               color="white" if abs(value) > 0.45 * limit else INK,
                               fontweight="bold")
        ax_matrix.tick_params(length=0, labelsize=8)
        ax_matrix.spines[:].set_visible(False)
        scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(scalar, ax=ax_matrix, fraction=0.046, pad=0.04)
        cbar.set_label("S2 estimate", fontsize=8)

    notes = [
        "Reading rule: S1 is variance explained by one input alone; ST is the variance involving that input in any order.",
        "ST − S1 is interaction involvement, not an additive variance allocation; interaction terms are counted for each participating input.",
        "* Negative estimates are finite-sample noise, not negative physical importance.  † The displayed S2 confidence interval includes zero.",
    ]
    fig.text(0.08, 0.025, "\n".join(notes), ha="left", va="bottom", fontsize=8.5, color=MUTED)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    if output.suffix.lower() != ".svg":
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="SALib-style Sobol result JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output .png or .svg path")
    args = parser.parse_args()
    plot_map(load_results(args.input), args.output)
    print(f"Saved Sobol Variance Budget Map to {args.output}")


if __name__ == "__main__":
    main()
