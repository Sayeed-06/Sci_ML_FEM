"""
encoders.py
===========
Shared input/output encoder-decoder MLP blocks used by both GNN and
Graph Transformer models.

Design choices
--------------
- NodeEncoder: maps raw node features (dim=9) to a hidden embedding via two
  linear layers with LayerNorm + GELU. GELU is preferred over ReLU for
  smooth gradient flow (empirically better in scientific ML tasks).
- EdgeEncoder: maps raw edge features (dim=3) to the same hidden dim.
- OutputDecoder: maps hidden embedding → 3 output values [w, σ_x, σ_y].
  Two-layer MLP with a bottleneck for regularisation.
"""

import torch
import torch.nn as nn
from typing import Optional


class NodeEncoder(nn.Module):
    """
    Maps (N, in_dim) → (N, hidden_dim) node features.
    Two linear layers + LayerNorm + GELU.
    """

    def __init__(self, in_dim: int = 9, hidden_dim: int = 128):
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


class EdgeEncoder(nn.Module):
    """
    Maps (E, in_dim) → (E, edge_hidden_dim) edge attributes.
    """

    def __init__(self, in_dim: int = 3, edge_hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, edge_hidden_dim),
            nn.LayerNorm(edge_hidden_dim),
            nn.GELU(),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(edge_attr)


class OutputDecoder(nn.Module):
    """
    Maps (N, hidden_dim) → (N, out_dim).
    Two-layer MLP with bottleneck.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        bottleneck_dim: int = 64,
        out_dim: int = 3,
        dropout: float = 0.0,
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
