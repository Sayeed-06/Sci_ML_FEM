"""
build_graph.py
==============
Converts an (input, output) FEM case pair into a PyTorch Geometric Data object.

Graph topology
--------------
Nodes  = Nx × Ny grid points (Nx = number of X-lines, Ny = Y-lines)
Edges  = horizontal + vertical adjacencies between neighbouring nodes.
         For each 4-node cell we also add both diagonals to capture plate
         bending coupling (∂²w/∂x∂y terms).

Node features (13 values)
------------------------
0  x_norm          x-coordinate normalised to [0,1]
1  y_norm          y-coordinate normalised to [0,1]
2  thickness_norm  plate thickness (normalised)
3  E_norm          elastic modulus (normalised)
4  k_norm          Winkler k (normalised)
5  q_local         1 if node is inside loaded rectangle, 0 otherwise
6  delta_T_norm    temperature delta (normalised)
7  bc_flag         1 if boundary node (edge of domain), 0 interior
8  D_norm          flexural rigidity (normalised)
9  Lx              plate length in x
10 Ly              plate length in y
11 alpha_T         thermal expansion coefficient
12 pressure_q      applied pressure magnitude

Edge features (3 values)
------------------------
0  dx_norm         Δx between nodes (normalised)
1  dy_norm         Δy between nodes (normalised)
2  length_norm     Euclidean distance (normalised)

Target (3 values per node)
--------------------------
0  w               deflection
1  sigma_x         bottom-surface σ_x
2  sigma_y         bottom-surface σ_y

The node ordering matches the .ou2 output: row-major X-varies-fastest,
i.e. node index = iy * Nx + ix  where ix ∈ [0..Nx-1], iy ∈ [0..Ny-1].
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Any, Tuple


def _make_grid_edges(
    nx: int, ny: int, include_diagonals: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build undirected edge index for an nx×ny grid.

    Node index: node(ix, iy) = iy * nx + ix

    Returns
    -------
    src, dst  : 1-D int arrays of shape (E,) each (both directions included)
    """
    src_list, dst_list = [], []

    def add(a: int, b: int) -> None:
        src_list.extend([a, b])
        dst_list.extend([b, a])

    for iy in range(ny):
        for ix in range(nx):
            n = iy * nx + ix
            # horizontal edge
            if ix + 1 < nx:
                add(n, n + 1)
            # vertical edge
            if iy + 1 < ny:
                add(n, n + nx)
            # diagonal edges for plate bending coupling
            if include_diagonals:
                if ix + 1 < nx and iy + 1 < ny:
                    add(n, n + nx + 1)          # bottom-right diagonal
                    add(n + 1, n + nx)           # bottom-left diagonal

    src = np.array(src_list, dtype=np.int64)
    dst = np.array(dst_list, dtype=np.int64)
    return src, dst


def build_graph(
    inp: Dict[str, Any],
    out: Dict[str, Any],
    include_diagonals: bool = True,
    normalise: bool = False,
    norm_stats: Dict[str, Any] | None = None,
) -> Data:
    """
    Convert a parsed (input, output) pair into a PyG Data object.

    Parameters
    ----------
    inp   : output of parse_input()
    out   : output of parse_output()
    include_diagonals : whether to add diagonal edges per cell
    normalise  : if True, apply per-feature normalisation using norm_stats
    norm_stats : dict produced by dataset.normalize.compute_norm_stats()

    Returns
    -------
    torch_geometric.data.Data with attributes:
        x          : (N, 13) node features
        edge_index : (2, E)  edge indices
        edge_attr  : (E, 3)  edge features
        y          : (N, 3)  targets [w, σ_x, σ_y]
        pos        : (N, 2)  raw node positions [x, y] (mm)
        metadata   : dict    scalar case parameters
    """
    nx = inp['n_nodes_x']
    ny = inp['n_nodes_y']
    N = nx * ny

    x_coords = inp['x_coords']  # (nx,)
    y_coords = inp['y_coords']  # (ny,)

    # ── Build position grid (row-major: X varies fastest) ─────────────────
    # Grid: ix in [0..nx-1], iy in [0..ny-1]
    # node(ix, iy) = iy * nx + ix
    xx, yy = np.meshgrid(x_coords, y_coords)  # shape (ny, nx)
    pos_x = xx.ravel()  # (N,)
    pos_y = yy.ravel()  # (N,)

    # ── Sanity: match against .ou2 node coordinates ────────────────────────
    # out['nodes'] is ordered by 1-based node ID from the .ou2 file
    # We verify the total count matches; exact ordering may differ if the
    # .ou2 nodes aren't in row-major order, so we build a lookup.
    n_out = len(out['node_ids'])
    assert n_out == N, (
        f"Node count mismatch: .in2 grid={N}, .ou2 nodes={n_out}. "
        f"Grid {nx}x{ny}={N}. Check parser."
    )
    # ── Match .ou2 nodes to .in2 grid ───────────────────────────────
    # The ISLab .ou2 output file always prints nodes in row-major order
    # (X varies fast, Y varies slowly). This is exactly the same order
    # as our np.meshgrid(x_coords, y_coords).ravel() operation!
    # Therefore, no coordinate mapping or KD-trees are required at all.
    
    # Optional sanity check: Ensure max coordinate mismatch is small
    max_err_x = np.max(np.abs(out['nodes'][:, 0] - pos_x))
    max_err_y = np.max(np.abs(out['nodes'][:, 1] - pos_y))
    if max_err_x > 2.0 or max_err_y > 2.0:
        raise ValueError(f"Mesh grid mismatch! Max error X: {max_err_x}, Y: {max_err_y}")

    # Directly assign the arrays since they are perfectly ordered
    w_ordered       = out['w'].copy()
    sigma_x_ordered = out['sigma_x_bot'].copy()
    sigma_y_ordered = out['sigma_y_bot'].copy()

    # ── Node features ──────────────────────────────────────────────────────
    # Normalise spatial coords to [0,1]
    x_norm = (pos_x - pos_x.min()) / (pos_x.max() - pos_x.min() + 1e-8)
    y_norm = (pos_y - pos_y.min()) / (pos_y.max() - pos_y.min() + 1e-8)

    # Load patch indicator: 1 if node inside pressure rectangle
    inside_load = (
        (pos_x >= inp['load_x1']) & (pos_x <= inp['load_x2']) &
        (pos_y >= inp['load_y1']) & (pos_y <= inp['load_y2'])
    ).astype(np.float32)

    # Boundary flag: nodes on the outer perimeter of the grid
    is_boundary = (
        (pos_x == x_coords[0]) | (pos_x == x_coords[-1]) |
        (pos_y == y_coords[0]) | (pos_y == y_coords[-1])
    ).astype(np.float32)

    # Scalar material properties broadcast to all nodes
    E_arr    = np.full(N, inp['E'],         dtype=np.float32)
    h_arr    = np.full(N, inp['thickness'], dtype=np.float32)
    k_arr    = np.full(N, inp['k_winkler'], dtype=np.float32)
    dT_arr   = np.full(N, inp['delta_T'],   dtype=np.float32)
    D_arr    = np.full(N, inp['D'],         dtype=np.float32)
    Lx_arr   = np.full(N, pos_x.max() - pos_x.min(), dtype=np.float32)
    Ly_arr   = np.full(N, pos_y.max() - pos_y.min(), dtype=np.float32)
    alpha_arr = np.full(N, inp['alpha_T'], dtype=np.float32)
    pressure_arr = np.full(N, inp['pressure_q'], dtype=np.float32)

    node_feat = np.stack([
        x_norm,         # 0
        y_norm,         # 1
        h_arr,          # 2  thickness (mm)
        E_arr,          # 3  elastic modulus (MPa)
        k_arr,          # 4  Winkler k
        inside_load,    # 5  load indicator
        dT_arr,         # 6  temperature
        is_boundary,    # 7  boundary flag
        D_arr,          # 8  flexural rigidity
        Lx_arr,         # 9  absolute domain length X (mm)
        Ly_arr,         # 10 absolute domain length Y (mm)
        alpha_arr,      # 11 thermal expansion coefficient (1/deg C)
        pressure_arr,   # 12 applied pressure magnitude (MPa)
    ], axis=1)  # (N, 13)

    # ── Edges ──────────────────────────────────────────────────────────────
    src, dst = _make_grid_edges(nx, ny, include_diagonals)

    dx = pos_x[dst] - pos_x[src]
    dy = pos_y[dst] - pos_y[src]
    length = np.sqrt(dx ** 2 + dy ** 2)

    # Normalise edge features by domain diagonal
    diag = np.sqrt((pos_x.max() - pos_x.min()) ** 2 +
                   (pos_y.max() - pos_y.min()) ** 2) + 1e-8
    edge_feat = np.stack([
        dx / diag,
        dy / diag,
        length / diag,
    ], axis=1).astype(np.float32)  # (E, 3)

    # ── Optional per-feature normalisation ────────────────────────────────
    if normalise and norm_stats is not None:
        feat_mean = norm_stats['node_feat_mean']   # (9,)
        feat_std  = norm_stats['node_feat_std']    # (9,)
        # Don't normalise binary flags (cols 5, 7)
        cols_to_norm = [2, 3, 4, 6, 8, 9, 10, 11, 12]
        for c in cols_to_norm:
            node_feat[:, c] = (node_feat[:, c] - feat_mean[c]) / (feat_std[c] + 1e-8)

    # ── Targets ───────────────────────────────────────────────────────────
    target = np.stack([w_ordered, sigma_x_ordered, sigma_y_ordered], axis=1)  # (N, 3)

    # ── Assemble PyG Data ─────────────────────────────────────────────────
    data = Data(
        x          = torch.from_numpy(node_feat).float(),
        edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long(),
        edge_attr  = torch.from_numpy(edge_feat).float(),
        y          = torch.from_numpy(target).float(),
        pos        = torch.tensor(np.stack([pos_x, pos_y], axis=1), dtype=torch.float),
    )

    # ── Graph-level scalar attributes (ALL parsed parameters) ────────────
    # Stored as (1,) tensors so PyG batching concatenates them into (B,).
    # Integer metadata stored as plain Python ints (not batched, just metadata).

    # Material / geometry
    data.E            = torch.tensor([inp['E']],            dtype=torch.float)
    data.nu           = torch.tensor([inp['nu']],           dtype=torch.float)
    data.thickness    = torch.tensor([inp['thickness']],    dtype=torch.float)
    data.alpha_T      = torch.tensor([inp['alpha_T']],      dtype=torch.float)
    data.gamma        = torch.tensor([inp['gamma']],        dtype=torch.float)
    data.D            = torch.tensor([inp['D']],            dtype=torch.float)

    # Foundation
    data.k_winkler    = torch.tensor([inp['k_winkler']],    dtype=torch.float)

    # Thermal
    data.delta_T      = torch.tensor([inp['delta_T']],      dtype=torch.float)

    # Loading
    data.pressure     = torch.tensor([inp['pressure_q']],   dtype=torch.float)
    data.load_x1      = torch.tensor([inp['load_x1']],      dtype=torch.float)
    data.load_x2      = torch.tensor([inp['load_x2']],      dtype=torch.float)
    data.load_y1      = torch.tensor([inp['load_y1']],      dtype=torch.float)
    data.load_y2      = torch.tensor([inp['load_y2']],      dtype=torch.float)
    # Convenience: packed load rectangle [x1, x2, y1, y2]
    data.load_rect    = torch.tensor(
        [inp['load_x1'], inp['load_x2'], inp['load_y1'], inp['load_y2']],
        dtype=torch.float,
    )

    # Grid dimensions (int, used by PINN to reconstruct structured indexing)
    data.nx           = inp['n_nodes_x']
    data.ny           = inp['n_nodes_y']
    data.n_nodes_total= inp['n_nodes']
    data.element_type = inp['element_type']

    return data


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 3)[0])
    from parser import parse_input, parse_output

    if len(sys.argv) < 3:
        print("Usage: build_graph.py <input.in2> <input.ou2>")
        sys.exit(1)

    inp = parse_input(sys.argv[1])
    out = parse_output(sys.argv[2])
    g = build_graph(inp, out)
    print(g)
    print(f"  Nodes: {g.num_nodes}, Edges: {g.num_edges}")
    print(f"  x shape: {g.x.shape}")
    print(f"  y shape: {g.y.shape}")
    print(f"  edge_attr shape: {g.edge_attr.shape}")
