"""
gnn.py
======
Baseline GNN model for FEM surrogate modelling using GINE (GIN with Edge
features) convolution layers.

Architecture
------------
NodeEncoder(9 → hidden_dim)
  ↓
GINEConv × num_layers   (each with hidden_dim MLP, edge_dim=edge_hidden_dim)
  with Residual connections + LayerNorm every layer
  ↓
OutputDecoder(hidden_dim → 3)

Why GINE?
---------
1. Expressive: GIN is Weisfeiler-Leman (1-WL) optimal — the most powerful
   message-passing GNN for distinguishing graph structures.
2. Edge-aware: GINE extends GIN to incorporate edge features, critical for
   capturing mesh geometry (element lengths determine stiffness contributions).
3. Residual connections prevent vanishing gradients in 4+ layer stacks.
4. LayerNorm stabilises training on the highly variable FEM output scales.

Reference: Xu et al. "How Powerful are Graph Neural Networks?" (ICLR 2019)
           Hu et al. "Strategies for Pre-training Graph Neural Networks" (ICLR 2020)
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv

from .encoders import NodeEncoder, EdgeEncoder, OutputDecoder


class ResGINELayer(nn.Module):
    """
    One GINE layer with residual connection and LayerNorm.

    GINE update rule:
        h_i' = MLP((1 + ε) * h_i + Σ_{j∈N(i)} ReLU(h_j + e_{ij}))
    where e_{ij} is the edge embedding projected to match node dim.
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        # The MLP inside GINEConv
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.conv = GINEConv(mlp, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        out = self.conv(x, edge_index, edge_attr)
        # Residual: add input before normalisation
        return self.norm(out + x)


class GNNSurrogate(nn.Module):
    """
    Full GNN surrogate model.

    Parameters
    ----------
    node_in_dim  : input node feature dimension (default 9)
    edge_in_dim  : input edge feature dimension (default 3)
    hidden_dim   : width of all hidden layers (default 128)
    edge_hidden  : edge embedding dimension (default 64)
    num_layers   : number of GINE message-passing layers (default 4)
    out_dim      : number of output targets (default 3: w, σ_x, σ_y)
    dropout      : dropout in decoder (default 0.0)

    Notes on depth
    --------------
    4 layers → information radius of 4 hops, matching the influence radius
    of typical Winkler foundation problems (load influence decays quickly).
    For large plates or concentrated loads, increasing to 6-8 may help.
    """

    def __init__(
        self,
        node_in_dim: int = 9,
        edge_in_dim: int = 3,
        hidden_dim: int = 128,
        edge_hidden: int = 64,
        num_layers: int = 4,
        out_dim: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_encoder = NodeEncoder(node_in_dim, hidden_dim)
        self.edge_encoder = EdgeEncoder(edge_in_dim, edge_hidden)

        self.layers = nn.ModuleList([
            ResGINELayer(hidden_dim, edge_hidden)
            for _ in range(num_layers)
        ])

        self.decoder = OutputDecoder(hidden_dim, hidden_dim // 2, out_dim, dropout)

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Batch or Data with fields:
            x          (N, 9)
            edge_index (2, E)
            edge_attr  (E, 3)

        Returns
        -------
        out : (N, 3)  [w, σ_x, σ_y] predictions
        """
        x = self.node_encoder(data.x)               # (N, hidden_dim)
        e = self.edge_encoder(data.edge_attr)        # (E, edge_hidden)

        for layer in self.layers:
            x = layer(x, data.edge_index, e)        # (N, hidden_dim)

        return self.decoder(x)                       # (N, 3)

    def encode(self, data) -> torch.Tensor:
        """Return the final node embeddings before the decoder (for analysis)."""
        x = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)
        for layer in self.layers:
            x = layer(x, data.edge_index, e)
        return x


def build_gnn(cfg: dict) -> GNNSurrogate:
    """
    Construct a GNNSurrogate from a config dict.

    Config keys (all optional, defaults shown):
        node_in_dim  : 9
        edge_in_dim  : 3
        hidden_dim   : 128
        edge_hidden  : 64
        num_layers   : 4
        out_dim      : 3
        dropout      : 0.0
    """
    return GNNSurrogate(
        node_in_dim = cfg.get('node_in_dim', 9),
        edge_in_dim = cfg.get('edge_in_dim', 3),
        hidden_dim  = cfg.get('hidden_dim', 128),
        edge_hidden = cfg.get('edge_hidden', 64),
        num_layers  = cfg.get('num_layers', 4),
        out_dim     = cfg.get('out_dim', 3),
        dropout     = cfg.get('dropout', 0.0),
    )


if __name__ == '__main__':
    # Quick smoke test
    import torch
    from torch_geometric.data import Data

    N, E = 100, 400
    data = Data(
        x          = torch.randn(N, 9),
        edge_index = torch.randint(0, N, (2, E)),
        edge_attr  = torch.randn(E, 3),
        y          = torch.randn(N, 3),
    )

    model = GNNSurrogate()
    out = model(data)
    print(f"GNNSurrogate output shape: {out.shape}")  # (100, 3)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
