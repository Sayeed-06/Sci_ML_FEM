"""
graph_transformer.py
====================
Similarity-conditioned Graph Transformer with FiLM.

Architecture
------------
Similar to the GNN, but replaces GINEConv with PyG's TransformerConv.
TransformerConv adds edge-conditioned multi-head attention over the
neighbourhood, which can capture complex spatial relationships in the
FEM grid more effectively than sum-aggregation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

# Reuse encoders and FiLM from sim_gnn
from models.sim_gnn import SimNodeEncoder, SimEdgeEncoder, FiLMConditioner, OutputDecoder


class SimResTransformerLayer(nn.Module):
    """One Graph Transformer step with FiLM conditioning and residual."""

    def __init__(
        self,
        hidden_dim: int  = 128,
        edge_dim: int    = 64,
        global_dim: int  = 3,
        heads: int       = 4,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        head_dim = hidden_dim // heads

        self.conv = TransformerConv(
            in_channels  = hidden_dim,
            out_channels = head_dim,
            heads        = heads,
            edge_dim     = edge_dim,
            concat       = True,
            beta         = True,  # gating over residual connection
        )
        self.film = FiLMConditioner(global_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        global_per_node: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)   # Transformer aggregation
        h = self.film(h, global_per_node)         # FiLM modulation
        return self.norm(h + x)                   # residual + LN


class SimGraphTransformer(nn.Module):
    """Similarity-conditioned Graph Transformer surrogate."""

    def __init__(
        self,
        node_in_dim: int  = 4,
        edge_in_dim: int  = 3,
        global_dim: int   = 3,
        hidden_dim: int   = 128,
        edge_dim: int     = 64,
        num_layers: int   = 6,
        heads: int        = 4,
        out_dim: int      = 3,
        dropout: float    = 0.0,
    ):
        super().__init__()
        self.node_encoder = SimNodeEncoder(node_in_dim, hidden_dim)
        self.edge_encoder = SimEdgeEncoder(edge_in_dim, edge_dim)

        self.layers = nn.ModuleList([
            SimResTransformerLayer(hidden_dim, edge_dim, global_dim, heads)
            for _ in range(num_layers)
        ])

        self.decoder = OutputDecoder(hidden_dim, hidden_dim // 2, out_dim, dropout)

    def forward(self, data) -> torch.Tensor:
        h = self.node_encoder(data.x)
        e = self.edge_encoder(data.edge_attr)

        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                else torch.zeros(data.x.size(0), dtype=torch.long, device=data.x.device)

        global_per_node = data.global_feat[batch]

        for layer in self.layers:
            h = layer(h, data.edge_index, e, global_per_node)

        return self.decoder(h)


def build_sim_graph_transformer(cfg: dict) -> SimGraphTransformer:
    return SimGraphTransformer(
        node_in_dim = cfg.get("node_in_dim",  4),
        edge_in_dim = cfg.get("edge_in_dim",  3),
        global_dim  = cfg.get("global_dim",   3),
        hidden_dim  = cfg.get("hidden_dim", 128),
        edge_dim    = cfg.get("edge_dim",    64),
        num_layers  = cfg.get("num_layers",   6),
        heads       = cfg.get("heads",        4),
        out_dim     = cfg.get("out_dim",      3),
        dropout     = cfg.get("dropout",    0.0),
    )
