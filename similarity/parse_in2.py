"""
parse_in2.py
============
Standalone parser for ISLAB2000 .in2 input files.

.in2 format (section-keyword based, Windows CRLF):
  *GENERAL  → title, symmetry, iteration settings
  *GEOMETRY → mesh node coordinates (variable Nx × Ny)
  *LAYER    → material properties (E, nu, h, alpha_T, gamma)
  *FOUNDATION → Winkler k
  *TEMPERATURE → ΔT (temperature difference)
  *PRESSURE → applied pressure patch

Returns a dict with all parsed values in their native units:
  E          : MPa
  nu         : dimensionless
  thickness  : mm
  alpha_T    : 1/°C
  gamma      : N/m³   (as stored in file; convert via similarity_transform)
  k_winkler  : kPa/mm (as stored in file; convert via similarity_transform)
  delta_T    : °C
  pressure_q : MPa
  x_coords   : np.ndarray (mm)
  y_coords   : np.ndarray (mm)
  n_nodes_x  : int
  n_nodes_y  : int
  n_nodes    : int
  D          : flexural rigidity E·h³/(12(1-ν²)) [MPa·mm³ = N·mm]
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any


def parse_in2(filepath: str | Path) -> Dict[str, Any]:
    """Parse an ISLAB2000 .in2 input file."""
    filepath = Path(filepath)
    text = filepath.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip("\r\n") for ln in text.splitlines()]

    result: Dict[str, Any] = {}

    # ── locate section boundaries ──────────────────────────────────────
    section_map: Dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("*"):
            key = stripped.lstrip("*").split()[0].upper()
            section_map[key] = i

    # ── *GEOMETRY ──────────────────────────────────────────────────────
    gi = section_map["GEOMETRY"]
    # gi+1: "1 1"  (nx_slabs, ny_slabs)
    # gi+2: Nx  (number of x node lines)
    # gi+3: Ny  (number of y node lines)
    # gi+4: x coordinates (space-separated, mm)
    # gi+5: y coordinates (space-separated, mm)
    nx = int(lines[gi + 2].strip())
    ny = int(lines[gi + 3].strip())
    x_coords = np.array([float(v) for v in lines[gi + 4].strip().split()])
    y_coords = np.array([float(v) for v in lines[gi + 5].strip().split()])

    assert len(x_coords) == nx, f"Expected {nx} X-coords, got {len(x_coords)}"
    assert len(y_coords) == ny, f"Expected {ny} Y-coords, got {len(y_coords)}"

    result["x_coords"]  = x_coords
    result["y_coords"]  = y_coords
    result["n_nodes_x"] = nx
    result["n_nodes_y"] = ny
    result["n_nodes"]   = nx * ny

    # ── *LAYER ─────────────────────────────────────────────────────────
    li = section_map["LAYER"]
    # li+1: "1 0"
    # li+2: empty
    # li+3: "<element_type> <E> <nu> <h> <alpha_T> <gamma>"
    layer_vals = lines[li + 3].strip().split()
    result["element_type"] = int(layer_vals[0])
    result["E"]            = float(layer_vals[1])
    result["nu"]           = float(layer_vals[2])
    result["thickness"]    = float(layer_vals[3])
    result["alpha_T"]      = float(layer_vals[4])
    result["gamma"]        = float(layer_vals[5])  # N/m³

    E  = result["E"]
    h  = result["thickness"]
    nu = result["nu"]
    result["D"] = (E * h ** 3) / (12.0 * (1.0 - nu ** 2))

    # ── *FOUNDATION ────────────────────────────────────────────────────
    fi = section_map["FOUNDATION"]
    # fi+1: "51 <k> 1"
    found_vals = lines[fi + 1].strip().split()
    result["k_winkler"] = float(found_vals[1])  # kPa/mm

    # ── *TEMPERATURE ───────────────────────────────────────────────────
    ti = section_map["TEMPERATURE"]
    # ti+1: "1 1 <T_ref> 1"
    temp_vals = lines[ti + 1].strip().split()
    result["delta_T"] = float(temp_vals[2])  # reference temp / ΔT (°C)

    # ── *PRESSURE ──────────────────────────────────────────────────────
    pi = section_map["PRESSURE"]
    # pi+1: "<q_MPa> <x1> <x2> <y1> <y2>"
    pvals = lines[pi + 1].strip().split()
    result["pressure_q"] = float(pvals[0])   # MPa
    result["load_x1"]    = float(pvals[1])   # mm
    result["load_x2"]    = float(pvals[2])
    result["load_y1"]    = float(pvals[3])
    result["load_y2"]    = float(pvals[4])

    return result


if __name__ == "__main__":
    import sys
    data = parse_in2(sys.argv[1])
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"  {k}: {v}")
