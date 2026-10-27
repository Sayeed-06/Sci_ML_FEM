"""
parse_output.py
===============
Parses ISLab .ou2 files into structured numpy arrays.

.ou2 structure (key sections):
  ─────────────────────────────────────────────────────────────────
  LAYER 1 BOTTOM SURFACE
     node   x       y       w        theta_x  theta_y  sigm_x  sigm_y  sigm_xy  stress  angle
  ─────────────────────────────────────────────────────────────────
  LAYER 1 TOP SURFACE      ← same columns, signs flipped for bending
  ─────────────────────────────────────────────────────────────────
  REACTIONS section         ← triggered by large reaction values (>1e4)
     node   x       y       w        theta_x  theta_y  reaction  0  0  flag  bc_type

The parser detects section starts using text anchors and switches a state
machine to accumulate rows. Reaction rows are identified by the large
magnitude of column 7 (> REACTION_THRESHOLD).
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, List


# Data line regex: matches lines like
#   "    1      0.00      0.00   54232.489217    0.000003   -0.000016 ..."
_DATA_LINE = re.compile(
    r'^\s{0,8}(\d+)\s+'          # node index
    r'([-+]?[\d.Ee+-]+)\s+'      # x
    r'([-+]?[\d.Ee+-]+)\s+'      # y
    r'([-+]?[\d.Ee+-]+)\s+'      # w
    r'([-+]?[\d.Ee+-]+)\s+'      # theta_x
    r'([-+]?[\d.Ee+-]+)\s+'      # theta_y
    r'([-+]?[\d.Ee+-]+)\s+'      # col7: sigm_x or reaction
    r'([-+]?[\d.Ee+-]+)\s+'      # col8
    r'([-+]?[\d.Ee+-]+)'         # col9
)

# Reactions have huge values (>1e4); normal stresses are ~0.1-1.0
REACTION_THRESHOLD = 1e4


def _parse_section(
    lines: List[str],
    start: int,
    stop: int,
    is_reaction: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Parse a node-data block (bottom surface, top surface, or reactions).

    Returns dict of node index → [x, y, w, theta_x, theta_y, c7, c8, c9].
    """
    rows: Dict[int, List[float]] = {}
    for ln in range(start, stop):
        line = lines[ln]
        m = _DATA_LINE.match(line)
        if not m:
            continue
        node = int(m.group(1))
        vals = [float(m.group(i)) for i in range(2, 10)]
        # sanity check: skip header echo lines that look like node data
        if is_reaction:
            if abs(vals[5]) < REACTION_THRESHOLD:
                continue  # not a real reaction row
        else:
            if abs(vals[5]) > REACTION_THRESHOLD:
                continue  # accidentally hit a reaction row
        rows[node] = vals
    return rows


def parse_output(filepath: str | Path) -> Dict[str, Any]:
    """
    Parse an ISLab .ou2 file.

    Returns
    -------
    dict with keys:
        nodes         : np.ndarray (N, 2)  — [x, y] for each node (mm)
        w             : np.ndarray (N,)    — deflection (micro-mm or file units)
        theta_x       : np.ndarray (N,)   — rotation about x
        theta_y       : np.ndarray (N,)   — rotation about y
        sigma_x_bot   : np.ndarray (N,)   — bottom σ_x (MPa)
        sigma_y_bot   : np.ndarray (N,)   — bottom σ_y (MPa)
        sigma_xy_bot  : np.ndarray (N,)   — bottom τ_xy (MPa)
        sigma_x_top   : np.ndarray (N,)   — top σ_x (MPa)
        sigma_y_top   : np.ndarray (N,)   — top σ_y (MPa)
        sigma_xy_top  : np.ndarray (N,)   — top τ_xy (MPa)
        reaction_z    : np.ndarray (N,)   — Winkler reaction force
        node_ids      : np.ndarray (N,)   — 1-based node numbers
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()

    # ── Locate section boundaries ──────────────────────────────────────────
    bottom_start: int = -1
    top_start: int = -1
    reaction_start: int = -1
    max_min_start: int = -1

    for i, line in enumerate(lines):
        s = line.strip()
        if 'LAYER 1 BOTTOM SURFACE' in s:
            bottom_start = i + 4   # skip header+column header
        elif 'LAYER 1 TOP SURFACE' in s and top_start == -1:
            top_start = i + 4
        elif 'MAXIMUM AND MINIMUM DEFLECTIONS' in s:
            max_min_start = i
        # Reaction section: lines that transition from stress to reaction
        # We detect it as the first line after TOP SURFACE section where
        # column 7 value >> REACTION_THRESHOLD
        # Use heuristic: section starts after last normal stress line

    # Detect reaction_start by scanning top surface block for the first
    # line whose col7 > REACTION_THRESHOLD
    if top_start != -1 and max_min_start != -1:
        for i in range(top_start, max_min_start):
            m = _DATA_LINE.match(lines[i])
            if m and abs(float(m.group(7))) > REACTION_THRESHOLD:
                # This is the first reaction line; the section header
                # is a few lines before it
                reaction_start = i
                break

    assert bottom_start != -1, "Could not find LAYER 1 BOTTOM SURFACE in .ou2"
    assert top_start != -1,    "Could not find LAYER 1 TOP SURFACE in .ou2"
    assert max_min_start != -1,"Could not find MAXIMUM AND MINIMUM DEFLECTIONS"

    # Bottom surface ends at TOP SURFACE header (minus the 4-line gap)
    bottom_stop = top_start - 4
    # Top surface ends at reactions (if found) else at max_min
    top_stop = reaction_start if reaction_start != -1 else max_min_start
    react_stop = max_min_start

    # ── Parse each section ─────────────────────────────────────────────────
    bot_rows = _parse_section(lines, bottom_start, bottom_stop, is_reaction=False)
    top_rows = _parse_section(lines, top_start, top_stop, is_reaction=False)
    react_rows = {}
    if reaction_start != -1:
        react_rows = _parse_section(lines, reaction_start, react_stop, is_reaction=True)

    # ── Build ordered arrays by node ID ───────────────────────────────────
    node_ids = sorted(bot_rows.keys())
    N = len(node_ids)

    nodes       = np.zeros((N, 2), dtype=np.float32)
    w           = np.zeros(N, dtype=np.float32)
    theta_x     = np.zeros(N, dtype=np.float32)
    theta_y     = np.zeros(N, dtype=np.float32)
    sigma_x_bot = np.zeros(N, dtype=np.float32)
    sigma_y_bot = np.zeros(N, dtype=np.float32)
    sigma_xy_bot= np.zeros(N, dtype=np.float32)
    sigma_x_top = np.zeros(N, dtype=np.float32)
    sigma_y_top = np.zeros(N, dtype=np.float32)
    sigma_xy_top= np.zeros(N, dtype=np.float32)
    reaction_z  = np.zeros(N, dtype=np.float32)

    for idx, nid in enumerate(node_ids):
        bvals = bot_rows[nid]
        nodes[idx, 0] = bvals[0]  # x
        nodes[idx, 1] = bvals[1]  # y
        w[idx]            = bvals[2]
        theta_x[idx]      = bvals[3]
        theta_y[idx]      = bvals[4]
        sigma_x_bot[idx]  = bvals[5]
        sigma_y_bot[idx]  = bvals[6]
        sigma_xy_bot[idx] = bvals[7]

        if nid in top_rows:
            tvals = top_rows[nid]
            sigma_x_top[idx]  = tvals[5]
            sigma_y_top[idx]  = tvals[6]
            sigma_xy_top[idx] = tvals[7]

        if nid in react_rows:
            reaction_z[idx] = react_rows[nid][5]

    return {
        'nodes':          nodes,
        'node_ids':       np.array(node_ids, dtype=np.int32),
        'w':              w,
        'theta_x':        theta_x,
        'theta_y':        theta_y,
        'sigma_x_bot':    sigma_x_bot,
        'sigma_y_bot':    sigma_y_bot,
        'sigma_xy_bot':   sigma_xy_bot,
        'sigma_x_top':    sigma_x_top,
        'sigma_y_top':    sigma_y_top,
        'sigma_xy_top':   sigma_xy_top,
        'reaction_z':     reaction_z,
    }


if __name__ == '__main__':
    import sys
    data = parse_output(sys.argv[1])
    print(f"Parsed {len(data['node_ids'])} nodes")
    print(f"  w:      [{data['w'].min():.4f}, {data['w'].max():.4f}]")
    print(f"  σ_x:   [{data['sigma_x_bot'].min():.4f}, {data['sigma_x_bot'].max():.4f}]")
    print(f"  σ_y:   [{data['sigma_y_bot'].min():.4f}, {data['sigma_y_bot'].max():.4f}]")
    print(f"  react: [{data['reaction_z'].min():.2f}, {data['reaction_z'].max():.2f}]")
