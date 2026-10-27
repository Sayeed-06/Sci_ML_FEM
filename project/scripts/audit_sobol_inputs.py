#!/usr/bin/env python3
"""Audit ranges and dependence of the six proposed Sobol inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parser import parse_input


NAMES = ("E", "thickness", "k_winkler", "delta_T", "alpha_T", "pressure_q")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "generated_cases")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sobol_input_audit.json")
    args = parser.parse_args()

    rows = []
    for case in sorted(args.data_root.glob("case_*")):
        path = case / "input.in2"
        if not path.exists():
            continue
        values = parse_input(path)
        rows.append([values[name] for name in NAMES])
    matrix = np.asarray(rows, dtype=float)
    if matrix.size == 0:
        raise RuntimeError(f"No input files found below {args.data_root}")

    summary = {}
    for index, name in enumerate(NAMES):
        column = matrix[:, index]
        summary[name] = {
            "min": float(column.min()),
            "q25": float(np.quantile(column, 0.25)),
            "median": float(np.median(column)),
            "mean": float(column.mean()),
            "q75": float(np.quantile(column, 0.75)),
            "max": float(column.max()),
            "std": float(column.std(ddof=1)),
            "unique": int(np.unique(column).size),
        }
    correlation = np.corrcoef(matrix, rowvar=False)
    off_diagonal = correlation[~np.eye(len(NAMES), dtype=bool)]
    result = {
        "cases": int(matrix.shape[0]),
        "parameters": list(NAMES),
        "summary": summary,
        "pearson_correlation": correlation.tolist(),
        "maximum_absolute_pairwise_correlation": float(np.max(np.abs(off_diagonal))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
