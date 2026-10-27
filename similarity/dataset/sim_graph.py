"""
sim_graph.py
============
Builds a PyTorch Geometric `Data` object for the Similarity-Based GNN.

Graph topology
--------------
Nodes = all Nx × Ny mesh points (variable count per case, up to thousands)
Edges = horizontal + vertical + diagonal adjacencies in the FEM grid.
        Diagonals capture plate bending coupling (∂²w/∂x∂y terms).

Node features  (4 values per node)
-----------------------------------
0  x_over_l    x coordinate normalised by radius of relative stiffness l
1  y_over_l    y coordinate normalised by l
2  q_local     1 if node is inside the loaded rectangle, else 0
3  bc_flag     1 if node is on the outer perimeter of the domain, else 0

Edge features  (3 values per edge)
-----------------------------------
0  dx_over_l   Δx between endpoints, normalised by l
1  dy_over_l   Δy between endpoints, normalised by l
2  dist_over_l Euclidean distance, normalised by l

Global (graph-level) attributes — used for FiLM conditioning
-------------------------------------------------------------
global_feat : Tensor (3,)  → [l, phi, PQ]  in their raw physical values.
              The model normalises these internally using training statistics.

Targets (3 values per node)
----------------------------
0  w         deflection (mm)          — in absolute physical units
1  sigma_x   bottom-surface σ_x (MPa)
2  sigma_y   bottom-surface σ_y (MPa)

Why x/l, y/l?
--------------
Two cases with the same radius of relative stiffness l have the same
normalised geometry (L/l, W/l) and therefore IDENTICAL non-dimensional
responses.  By expressing positions in similarity coordinates the GNN
sees the same input pattern for mechanically equivalent cases, regardless
of whether they differ in E, h, or k individually.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Grid edge builder
# ---------------------------------------------------------------------------

def _make_grid_edges(
    nx: int,
    ny: int,
    include_diagonals: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build undirected edge index for an nx × ny regular grid.

    Node index convention:  node(ix, iy) = iy * nx + ix
    (X varies fast, Y varies slow — same as ISLAB2000 output order)

    Returns
    -------
    src, dst : 1-D int64 arrays, shape (E,) each (both directions)
    """
    src_list, dst_list = [], []

    def add(a: int, b: int) -> None:
        src_list.extend([a, b])
        dst_list.extend([b, a])

    for iy in range(ny):
        for ix in range(nx):
            n = iy * nx + ix
            if ix + 1 < nx:
                add(n, n + 1)                    # horizontal
            if iy + 1 < ny:
                add(n, n + nx)                   # vertical
            if include_diagonals:
                if ix + 1 < nx and iy + 1 < ny:
                    add(n, n + nx + 1)           # main diagonal
                    add(n + 1, n + nx)           # anti-diagonal

    return (
        np.array(src_list, dtype=np.int64),
        np.array(dst_list, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_sim_graph(
    inp: Dict[str, Any],
    out: Dict[str, Any],
    sim: Dict[str, float],
    include_diagonals: bool = True,
) -> Data:
    """Build a similarity-coordinate PyG Data object.

    Parameters
    ----------
    inp : dict from parse_in2()
    out : dict from parse_ou2()
    sim : dict from similarity_transform.compute_similarity_params()
          Keys: 'l' (mm), 'phi' (dimensionless), 'PQ' (dimensionless)
    include_diagonals : add diagonal edges per FEM cell

    Returns
    -------
    torch_geometric.data.Data
    """
    nx = inp["n_nodes_x"]
    ny = inp["n_nodes_y"]
    N  = nx * ny

    x_coords = inp["x_coords"]  # (nx,)  mm
    y_coords = inp["y_coords"]  # (ny,)  mm
    l        = sim["l"]         # mm  — structural length scale

    # ── Position grid (row-major: X fast, Y slow) ──────────────────────
    xx, yy = np.meshgrid(x_coords, y_coords)   # (ny, nx)
    pos_x  = xx.ravel().astype(np.float32)      # (N,)
    pos_y  = yy.ravel().astype(np.float32)      # (N,)

    # ── Validate node count against .ou2 ───────────────────────────────
    n_out = len(out["node_ids"])
    if n_out != N:
        raise ValueError(
            f"Node count mismatch: .in2 grid {nx}×{ny}={N} "
            f"but .ou2 has {n_out} nodes in {inp.get('filepath', '?')}"
        )

    # ── Sanity: coordinate agreement ───────────────────────────────────
    max_err_x = np.max(np.abs(out["nodes"][:, 0] - pos_x))
    max_err_y = np.max(np.abs(out["nodes"][:, 1] - pos_y))
    if max_err_x > 5.0 or max_err_y > 5.0:
        raise ValueError(
            f"Mesh mismatch! ΔX_max={max_err_x:.2f}mm, ΔY_max={max_err_y:.2f}mm"
        )

    # ── Similarity-coordinate node features ────────────────────────────
    x_over_l = pos_x / l   # dimensionless
    y_over_l = pos_y / l   # dimensionless

    inside_load = (
        (pos_x >= inp["load_x1"]) & (pos_x <= inp["load_x2"]) &
        (pos_y >= inp["load_y1"]) & (pos_y <= inp["load_y2"])
    ).astype(np.float32)

    is_boundary = (
        (pos_x == x_coords[0]) | (pos_x == x_coords[-1]) |
        (pos_y == y_coords[0]) | (pos_y == y_coords[-1])
    ).astype(np.float32)

    node_feat = np.stack([
        x_over_l,     # 0 — x/l
        y_over_l,     # 1 — y/l
        inside_load,  # 2 — load indicator
        is_boundary,  # 3 — boundary flag
    ], axis=1).astype(np.float32)  # (N, 4)

    # ── Similarity-coordinate edge features ────────────────────────────
    src, dst = _make_grid_edges(nx, ny, include_diagonals)

    dx = (pos_x[dst] - pos_x[src]) / l  # normalised by l
    dy = (pos_y[dst] - pos_y[src]) / l
    dist = np.sqrt(dx ** 2 + dy ** 2)

    edge_feat = np.stack([dx, dy, dist], axis=1).astype(np.float32)  # (E, 3)

    # ── Targets (absolute physical units) ──────────────────────────────
    # Note: stored in raw units; normalisation for training is applied
    # separately by the dataset class so we can recover absolute values.
    target = np.stack([
        out["w"],           # mm
        out["sigma_x_bot"], # MPa
        out["sigma_y_bot"], # MPa
    ], axis=1).astype(np.float32)  # (N, 3)

    # ── Assemble PyG Data ──────────────────────────────────────────────
    data = Data(
        x          = torch.from_numpy(node_feat),
        edge_index = torch.from_numpy(np.stack([src, dst], axis=0)),
        edge_attr  = torch.from_numpy(edge_feat),
        y          = torch.from_numpy(target),
        pos        = torch.from_numpy(np.stack([pos_x, pos_y], axis=1)),
    )

    # ── Global graph-level attributes (similarity scalars) ─────────────
    # Shape (1,) so PyG batching concatenates into (B,)
    data.global_feat = torch.tensor(
        [sim["l"], sim["phi"], sim["PQ"]], dtype=torch.float
    ).unsqueeze(0)  # (1, 3)

    # ── Scalar metadata (for reference, not used by model) ─────────────
    data.l      = float(sim["l"])
    data.phi    = float(sim["phi"])
    data.PQ     = float(sim["PQ"])
    data.E      = float(inp["E"])
    data.h      = float(inp["thickness"])
    data.k      = float(inp["k_winkler"])
    data.dT     = float(inp["delta_T"])
    data.P      = float(inp["pressure_q"])
    data.nx     = nx
    data.ny     = ny

    return data


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit("/", 3)[0])
    from parse_in2 import parse_in2
    from parse_ou2 import parse_ou2
    from similarity_transform import compute_similarity_params

    if len(sys.argv) < 3:
        print("Usage: sim_graph.py <input.in2> <input.ou2>")
        sys.exit(1)

    inp = parse_in2(sys.argv[1])
    out = parse_ou2(sys.argv[2])
    sim = compute_similarity_params(
        E=inp["E"], h=inp["thickness"], k_file=inp["k_winkler"],
        mu=inp["nu"], alpha=inp["alpha_T"], gamma_file=inp["gamma"],
        delta_T=inp["delta_T"], pressure_q=inp["pressure_q"],
    )
    g = build_sim_graph(inp, out, sim)
    print(g)
    print(f"  Nodes: {g.num_nodes}  Edges: {g.num_edges}")
    print(f"  x shape: {g.x.shape}")
    print(f"  y shape: {g.y.shape}")
    print(f"  l={sim['l']:.1f}mm  phi={sim['phi']:.4f}  P/Q={sim['PQ']:.2f}")
