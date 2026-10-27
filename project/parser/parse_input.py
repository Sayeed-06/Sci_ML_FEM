"""
parse_input.py
==============
Parses ISLab .in2 files into a structured Python dictionary.

.in2 format (28 lines, Windows CRLF, section-keyword based):

  *GENERAL
  v2.2
  <title>
  <sym_idx> <max_iter> <0>          # symmetry, iterations
  1 1 1 0.1                          # flags
  0

  *GEOMETRY
  1 1                                # nx_slabs, ny_slabs
  <Nx>                               # number of X node lines
  <Ny>                               # number of Y node lines
  <x0 x1 x2 ...>                    # X coordinates (mm)
  <y0 y1 y2 ...>                    # Y coordinates (mm)

  *LAYER
  1 0
  <element_type> <E> <nu> <h> <alpha_T> <gamma>

  *FOUNDATION
  51 <k> 1                           # model=51=Energy Consistent Winkler, k in N/mm³

  *TEMPERATURE
  1 1 <T_ref> 1

  *PRESSURE
  <q_MPa> <x1> <x2> <y1> <y2>      # load patch coordinates (mm)
"""

import re
import numpy as np
from pathlib import Path
from typing import Dict, Any


def parse_input(filepath: str | Path) -> Dict[str, Any]:
    """
    Parse an ISLab .in2 file.

    Returns
    -------
    dict with keys:
        x_coords     : np.ndarray, shape (Nx,), units = mm
        y_coords     : np.ndarray, shape (Ny,), units = mm
        E            : float, elastic modulus (MPa)
        nu           : float, Poisson's ratio
        thickness    : float, plate thickness (mm)
        alpha_T      : float, thermal expansion coefficient (1/°C)
        gamma        : float, unit weight (N/mm³)
        k_winkler    : float, subgrade modulus (N/mm³)
        delta_T      : float, reference temperature (°C)
        pressure_q   : float, applied pressure (MPa)
        load_x1      : float, load patch x-start (mm)
        load_x2      : float, load patch x-end (mm)
        load_y1      : float, load patch y-start (mm)
        load_y2      : float, load patch y-end (mm)
        n_nodes_x    : int
        n_nodes_y    : int
        n_nodes      : int, total = n_nodes_x * n_nodes_y
        D            : float, flexural rigidity = E*h³ / (12*(1-ν²))  [N·mm]
    """
    filepath = Path(filepath)
    text = filepath.read_text(encoding='utf-8', errors='replace')
    lines = [l.rstrip('\r\n') for l in text.splitlines()]

    result: Dict[str, Any] = {}

    # ── locate section boundaries ──────────────────────────────────────────
    section_map: Dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('*'):
            key = stripped.lstrip('*').split()[0].upper()
            section_map[key] = i

    # ── *GEOMETRY ─────────────────────────────────────────────────────────
    gi = section_map['GEOMETRY']
    # line gi+1: "1 1"  (slab counts, always 1×1 in our data)
    # line gi+2: Nx
    # line gi+3: Ny
    # line gi+4: X coordinates (space-separated)
    # line gi+5: Y coordinates
    nx = int(lines[gi + 2].strip())
    ny = int(lines[gi + 3].strip())
    x_coords = np.array([float(v) for v in lines[gi + 4].strip().split()])
    y_coords = np.array([float(v) for v in lines[gi + 5].strip().split()])

    assert len(x_coords) == nx, (
        f"Expected {nx} X-coords, got {len(x_coords)}"
    )
    assert len(y_coords) == ny, (
        f"Expected {ny} Y-coords, got {len(y_coords)}"
    )

    result['x_coords'] = x_coords
    result['y_coords'] = y_coords
    result['n_nodes_x'] = nx
    result['n_nodes_y'] = ny
    result['n_nodes'] = nx * ny

    # ── *LAYER ────────────────────────────────────────────────────────────
    li = section_map['LAYER']
    # line li+1: "1 0"
    # line li+2: <element_type> <E> <nu> <h> <alpha_T> <gamma>
    layer_line = lines[li + 3].strip()
    # handle scientific notation: 3.037E+04
    layer_vals = layer_line.split()
    # element_type, E, nu, h, alpha_T, gamma
    result['element_type'] = int(layer_vals[0])
    result['E'] = float(layer_vals[1])
    result['nu'] = float(layer_vals[2])
    result['thickness'] = float(layer_vals[3])
    result['alpha_T'] = float(layer_vals[4])
    result['gamma'] = float(layer_vals[5])

    # flexural rigidity
    E = result['E']
    h = result['thickness']
    nu = result['nu']
    result['D'] = (E * h ** 3) / (12.0 * (1.0 - nu ** 2))

    # ── *FOUNDATION ───────────────────────────────────────────────────────
    fi = section_map['FOUNDATION']
    # line fi+1: "51 <k> 1"
    found_vals = lines[fi + 1].strip().split()
    result['k_winkler'] = float(found_vals[1])

    # ── *TEMPERATURE ──────────────────────────────────────────────────────
    ti = section_map['TEMPERATURE']
    # line ti+1: "1 1 <T_ref> 1"
    temp_vals = lines[ti + 1].strip().split()
    result['delta_T'] = float(temp_vals[2])

    # ── *PRESSURE ─────────────────────────────────────────────────────────
    pi = section_map['PRESSURE']
    # line pi+1: "<q> <x1> <x2> <y1> <y2>"
    pvals = lines[pi + 1].strip().split()
    result['pressure_q'] = float(pvals[0])
    result['load_x1'] = float(pvals[1])
    result['load_x2'] = float(pvals[2])
    result['load_y1'] = float(pvals[3])
    result['load_y2'] = float(pvals[4])

    return result


if __name__ == '__main__':
    import sys
    data = parse_input(sys.argv[1])
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"  {k}: {v}")
