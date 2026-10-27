"""
parse_ou2.py
============
Standalone parser for ISLAB2000 .ou2 output files.

The .ou2 file may contain multiple iterations.  We always parse the
**last** iteration block (converged solution).

Each row in the output table has columns:
  node  x  y  w  theta_x  theta_y  sigm_x  sigm_y  sigm_xy  stress  angle

We extract: node_id, x (mm), y (mm), w (mm), sigma_x (MPa), sigma_y (MPa)

Returns
-------
dict with keys:
  node_ids    : np.ndarray int,   shape (N,)
  nodes       : np.ndarray float, shape (N, 2)   [[x, y], ...]  mm
  w           : np.ndarray float, shape (N,)      deflection mm
  sigma_x_bot : np.ndarray float, shape (N,)      bottom-surface σ_x MPa
  sigma_y_bot : np.ndarray float, shape (N,)      bottom-surface σ_y MPa
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict


# Header pattern that marks the start of a node data block
_HEADER_PAT = re.compile(
    r"node\s+x\s+y\s+w\s+theta_x\s+theta_y\s+sigm_x\s+sigm_y", re.IGNORECASE
)

# A data row: integer then 10 floats (possibly in scientific notation)
_ROW_PAT = re.compile(
    r"^\s*(\d+)"             # node id
    r"\s+([-+]?\d[\d.]*(?:[eE][+-]?\d+)?)"  # x
    r"\s+([-+]?\d[\d.]*(?:[eE][+-]?\d+)?)"  # y
    r"\s+([-+]?\d[\d.]*(?:[eE][+-]?\d+)?)"  # w
    r"\s+([-+]?\d[\d.]*(?:[eE][+-]?\d+)?)"  # theta_x
    r"\s+([-+]?\d[\d.]*(?:[eE][+-]?\d+)?)"  # theta_y
    r"\s+([-+]?\d[\d.]*(?:[eE][+-]?\d+)?)"  # sigm_x
    r"\s+([-+]?\d[\d.]*(?:[eE][+-]?\d+)?)"  # sigm_y
    r".*$"                   # rest (sigm_xy, stress, angle — ignored)
)


def parse_ou2(filepath: str | Path) -> Dict[str, np.ndarray]:
    """Parse the bottom-surface nodal results from an ISLAB2000 .ou2 file.

    Handles variable mesh sizes and multiple iteration blocks.
    Always returns results from the **last** completed iteration.
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    # ── Find all header positions ──────────────────────────────────────
    # Each "LAYER N BOTTOM SURFACE" section starts a data block.
    # We collect ALL blocks and return the last one (final iteration).
    header_positions = []
    for i, line in enumerate(lines):
        if _HEADER_PAT.search(line):
            header_positions.append(i)

    if not header_positions:
        raise ValueError(
            f"No node data header found in {filepath}. "
            "Is this a valid ISLAB2000 .ou2 output file?"
        )

    # Use the last header (= last/converged iteration)
    last_header = header_positions[-1]

    # ── Parse rows after the last header ──────────────────────────────
    node_ids = []
    xs, ys, ws = [], [], []
    sigma_xs, sigma_ys = [], []

    for line in lines[last_header + 1:]:
        m = _ROW_PAT.match(line)
        if m:
            node_ids.append(int(m.group(1)))
            xs.append(float(m.group(2)))
            ys.append(float(m.group(3)))
            ws.append(float(m.group(4)))
            sigma_xs.append(float(m.group(7)))
            sigma_ys.append(float(m.group(8)))
        else:
            # Stop when we hit a non-data line (section separator, etc.)
            # but only after we've started collecting data
            if node_ids and line.strip():
                stripped = line.strip()
                # Skip dashes and blank lines; stop at new section keywords
                if stripped.startswith("-") or not stripped:
                    continue
                # Numeric-looking but short lines may be noise; stop on keywords
                if not stripped[0].isdigit() and not stripped[0] in "+-":
                    break

    if not node_ids:
        raise ValueError(f"No nodal data rows found after header in {filepath}")

    return {
        "node_ids":    np.array(node_ids,  dtype=np.int64),
        "nodes":       np.column_stack([xs, ys]).astype(np.float32),
        "w":           np.array(ws,        dtype=np.float32),
        "sigma_x_bot": np.array(sigma_xs,  dtype=np.float32),
        "sigma_y_bot": np.array(sigma_ys,  dtype=np.float32),
    }


if __name__ == "__main__":
    import sys
    out = parse_ou2(sys.argv[1])
    N = len(out["node_ids"])
    print(f"Parsed {N} nodes")
    print(f"  w      range: [{out['w'].min():.4f}, {out['w'].max():.4f}] mm")
    print(f"  sigma_x range: [{out['sigma_x_bot'].min():.4f}, {out['sigma_x_bot'].max():.4f}] MPa")
    print(f"  sigma_y range: [{out['sigma_y_bot'].min():.4f}, {out['sigma_y_bot'].max():.4f}] MPa")
