"""
sim_gnn.py
==========
Similarity-conditioned GNN with FiLM (Feature-wise Linear Modulation).

Architecture
------------

  SimNodeEncoder  (4 → hidden_dim)          maps [x/l, y/l, q_local, bc_flag]
        ↓
  SimEdgeEncoder  (3 → edge_dim)            maps [dx/l, dy/l, dist/l]
        ↓
  ResGINE + FiLM  × num_layers
  │   GINEConv message-passing (structure + geometry)
  │   FiLMLayer   modulates node embeddings by global (l, φ, P/Q)
  │   Residual + LayerNorm
        ↓
  OutputDecoder   (hidden_dim → 3)          [w, σ_x, σ_y]

Why FiLM?
---------
The global similarity scalars (l, φ, P/Q) do not vary per-node;
they control the MAGNITUDE and SHAPE of the whole-field response:
  - l   determines the spatial decay rate of deflections/stresses
  - φ   controls the curling contribution relative to axle loading
  - P/Q determines the relative importance of loading vs. self-weight
FiLM (Perez et al. 2018) is the principled way to inject these
graph-level conditioning signals into per-node representations at
each message-passing layer, without simply concatenating them (which
wastes capacity and hinders normalisation).

FiLM update at each layer:
  h_i' = γ(global) ⊙ GINE(h_i) + β(global)

The γ and β are affine transformations of (l, φ, P/Q) produced by a
small shared MLP.  This allows the network to adaptively re-scale and
shift its internal representations based on the overall structural
regime of each case.

References
----------
Perez et al. "FiLM: Visual Reasoning with a General Conditioning Layer"
  AAAI 2018  https://arxiv.org/abs/1709.07871
Xu et al. "How Powerful are Graph Neural Networks?" ICLR 2019
Hu et al.  "Strategies for Pre-training Graph Neural Networks" ICLR 2020
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class SimNodeEncoder(nn.Module):
    """Maps (N, 4) → (N, hidden_dim).

    Input dims: [x/l, y/l, q_local, bc_flag]
    Uses LayerNorm + GELU for stable training on the variable-scale
    similarity coordinates.
    """

    def __init__(self, in_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimEdgeEncoder(nn.Module):
    """Maps (E, 3) → (E, edge_dim).

    Input dims: [dx/l, dy/l, dist/l]
    """

    def __init__(self, in_dim: int = 3, edge_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU(),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(edge_attr)


class OutputDecoder(nn.Module):
    """Maps (N, hidden_dim) → (N, 3) = [w, σ_x, σ_y]."""

    def __init__(
        self,
        hidden_dim: int    = 128,
        bottleneck_dim: int = 64,
        out_dim: int       = 3,
        dropout: float     = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# FiLM conditioner
# ---------------------------------------------------------------------------

class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation conditioned on global scalars.

    Computes per-node scale γ and shift β from graph-level features,
    then applies: h' = γ * h + β

    Parameters
    ----------
    global_dim  : dimension of global conditioning vector (default 3: l,φ,P/Q)
    hidden_dim  : dimension of node embeddings to modulate
    """

    def __init__(self, global_dim: int = 3, hidden_dim: int = 128):
        super().__init__()
        # A two-layer MLP that produces 2*hidden_dim values (γ and β)
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.SiLU(),                          # smooth activation
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,          # (N, hidden_dim)
        global_feat: torch.Tensor, # (N, global_dim) — already broadcast to nodes
    ) -> torch.Tensor:
        params = self.mlp(global_feat)           # (N, 2*hidden_dim)
        gamma, beta = params.chunk(2, dim=-1)    # each (N, hidden_dim)
        # Initialise near identity: shift gamma by 1
        return (1.0 + gamma) * h + beta


# ---------------------------------------------------------------------------
# Similarity-conditioned GINE layer
# ---------------------------------------------------------------------------

class SimResGINELayer(nn.Module):
    """One GINE message-passing step with FiLM conditioning and residual.

    Update rule:
        h_mp = GINE(h, edge_index, edge_attr)     # neighbourhood aggregation
        h_fm = FiLM(h_mp, global_feat_per_node)   # global modulation
        h'   = LayerNorm(h_fm + h)                # residual
    """

    def __init__(
        self,
        hidden_dim: int  = 128,
        edge_dim: int    = 64,
        global_dim: int  = 3,
    ):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.conv = GINEConv(mlp, edge_dim=edge_dim)
        self.film = FiLMConditioner(global_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,          # (N, hidden_dim)
        edge_index: torch.Tensor, # (2, E)
        edge_attr: torch.Tensor,  # (E, edge_dim)
        global_per_node: torch.Tensor, # (N, global_dim)
    ) -> torch.Tensor:
        h  = self.conv(x, edge_index, edge_attr)  # GINE aggregation
        h  = self.film(h, global_per_node)         # FiLM modulation
        return self.norm(h + x)                    # residual + LN


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SimGNN(nn.Module):
    """Similarity-conditioned GNN surrogate for concrete pavement FEM.

    Predicts full-field [w(x,y), σ_x(x,y), σ_y(x,y)] for single-layer
    ISLAB2000 Winkler plate cases using similarity coordinates.

    Parameters
    ----------
    node_in_dim  : node feature dimension (default 4: x/l, y/l, q, bc)
    edge_in_dim  : edge feature dimension (default 3: dx/l, dy/l, dist/l)
    global_dim   : global conditioning dimension (default 3: l, φ, P/Q)
    hidden_dim   : width of all hidden layers
    edge_dim     : edge embedding dimension
    num_layers   : number of SimResGINE layers
    out_dim      : output targets per node (default 3: w, σ_x, σ_y)
    dropout      : dropout in decoder
    """

    def __init__(
        self,
        node_in_dim: int  = 4,
        edge_in_dim: int  = 3,
        global_dim: int   = 3,
        hidden_dim: int   = 128,
        edge_dim: int     = 64,
        num_layers: int   = 6,
        out_dim: int      = 3,
        dropout: float    = 0.0,
    ):
        super().__init__()
        self.node_encoder = SimNodeEncoder(node_in_dim, hidden_dim)
        self.edge_encoder = SimEdgeEncoder(edge_in_dim, edge_dim)

        self.layers = nn.ModuleList([
            SimResGINELayer(hidden_dim, edge_dim, global_dim)
            for _ in range(num_layers)
        ])

        self.decoder = OutputDecoder(hidden_dim, hidden_dim // 2, out_dim, dropout)

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Batch or Data with:
            x           (N, 4)   node features
            edge_index  (2, E)   edge connectivity
            edge_attr   (E, 3)   edge features
            global_feat (B, 3)   per-graph similarity scalars
            batch       (N,)     node-to-graph assignment

        Returns
        -------
        out : (N, 3)  [w, σ_x, σ_y] in normalised space during training
                      → denorm by dataset.sim_dataset.denorm_targets() for absolute output
        """
        # Encode
        h = self.node_encoder(data.x)         # (N, hidden)
        e = self.edge_encoder(data.edge_attr) # (E, edge_dim)

        # Broadcast global features to per-node
        # data.batch: (N,) maps each node to its graph index [0..B-1]
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        # global_feat shape: (B, 3) — index by batch to get (N, 3)
        global_per_node = data.global_feat[batch]  # (N, 3)

        # Message passing with FiLM conditioning
        for layer in self.layers:
            h = layer(h, data.edge_index, e, global_per_node)

        return self.decoder(h)  # (N, 3)

    def encode(self, data) -> torch.Tensor:
        """Return node embeddings before the decoder (for analysis/probing)."""
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)
        global_per_node = data.global_feat[batch]
        for layer in self.layers:
            h = layer(h, data.edge_index, e, global_per_node)
        return h


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_sim_gnn(cfg: dict) -> SimGNN:
    """Build SimGNN from a config dict.

    Config keys (all optional, defaults shown):
        node_in_dim  : 4
        edge_in_dim  : 3
        global_dim   : 3
        hidden_dim   : 128
        edge_dim     : 64
        num_layers   : 6
        out_dim      : 3
        dropout      : 0.0
    """
    return SimGNN(
        node_in_dim = cfg.get("node_in_dim",  4),
        edge_in_dim = cfg.get("edge_in_dim",  3),
        global_dim  = cfg.get("global_dim",   3),
        hidden_dim  = cfg.get("hidden_dim", 128),
        edge_dim    = cfg.get("edge_dim",    64),
        num_layers  = cfg.get("num_layers",   6),
        out_dim     = cfg.get("out_dim",      3),
        dropout     = cfg.get("dropout",    0.0),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    N, E = 594, 2800
    g = Data(
        x          = torch.randn(N, 4),
        edge_index = torch.randint(0, N, (2, E)),
        edge_attr  = torch.randn(E, 3),
        y          = torch.randn(N, 3),
        global_feat = torch.randn(1, 3),
    )
    g.batch = torch.zeros(N, dtype=torch.long)

    model = SimGNN()
    out   = model(g)
    print(f"SimGNN output shape: {out.shape}")  # (594, 3)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Test with batch of 2 graphs
    g2 = Data(
        x          = torch.randn(800, 4),
        edge_index = torch.randint(0, 800, (2, 4000)),
        edge_attr  = torch.randn(4000, 3),
        y          = torch.randn(800, 3),
        global_feat = torch.randn(1, 3),
    )
    batch = Batch.from_data_list([g, g2])
    out2  = model(batch)
    print(f"Batched output shape: {out2.shape}")  # (1394, 3)
