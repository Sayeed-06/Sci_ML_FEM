"""
deeponet.py
===========
Similarity-Based Cross-Attention Deep Operator Network (DeepONet).

This model combines the mathematical elegance of DeepONet with the physical
inductive biases of our Similarity coordinates. 

Branch Network (The "Input Function" Encoder)
---------------------------------------------
Reads the structural state (geometry, load, boundary conditions) and the
global similarity parameters (l, phi, P/Q). Instead of collapsing the whole
plate into a single vector, it uses the SimResGINELayer with FiLM to encode
a unique Key and Value vector for EVERY node in the grid.

Trunk Network (The "Spatial Basis" Evaluator)
---------------------------------------------
Evaluates basis functions purely from local spatial coordinates. Uses Random
Fourier Features on (x/l, y/l) to capture high-frequency stress gradients,
plus relative load coordinates.

Cross-Attention Bridge
----------------------
Each Trunk query extracts its precise basis coefficients by attending to the
Branch Keys of its own graph.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

# Import the existing encoders and layers from our similarity GNN
from .sim_gnn import SimNodeEncoder, SimEdgeEncoder, SimResGINELayer


# ── Augmentation-safe load centroid computation ───────────────────────────────

def compute_load_centroid(
    x_norm: torch.Tensor,      # (total_N,)  node x/l
    y_norm: torch.Tensor,      # (total_N,)  node y/l
    q_local: torch.Tensor,     # (total_N,)  1 if node under load
    batch: torch.Tensor,       # (total_N,)  graph index
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute load centroid (mean of loaded-node coords) per graph.

    Augmentation-safe: q_local is flipped by RandomPlateSymmetry together
    with x_norm/y_norm, so the centroid always tracks the flipped load.
    """
    q = q_local.clamp(min=0.0)                         # (total_N,)

    # Per-graph sums
    B = int(batch.max().item()) + 1
    q_sum   = torch.zeros(B, device=x_norm.device).scatter_add(0, batch, q)           # (B,)
    qx_sum  = torch.zeros(B, device=x_norm.device).scatter_add(0, batch, q * x_norm)  # (B,)
    qy_sum  = torch.zeros(B, device=x_norm.device).scatter_add(0, batch, q * y_norm)  # (B,)

    # Centroid per graph; default 0.5 if no loaded nodes
    safe_q = q_sum.clamp(min=1.0)
    cx_graph = qx_sum / safe_q                          # (B,)
    cy_graph = qy_sum / safe_q                          # (B,)

    # Broadcast to nodes
    cx = cx_graph[batch]                                # (total_N,)
    cy = cy_graph[batch]                                # (total_N,)
    return cx, cy


# ── Random Fourier Feature embedding ─────────────────────────────────────────

class FourierEmbedding(nn.Module):
    def __init__(self, d_in: int = 2, n_freqs: int = 64, scale: float = 1.0):
        super().__init__()
        self.n_freqs = n_freqs
        B = torch.randn(d_in, n_freqs) * scale
        self.register_buffer('B_matrix', B)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B_matrix
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


# ── Residual MLP block with dropout ──────────────────────────────────────────

class ResMLPBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ── Trunk Network ────────────────────────────────────────────────────────────

class TrunkNet(nn.Module):
    def __init__(
        self,
        trunk_in: int = 6,           # 4 standard + 2 relative load
        trunk_hidden: int = 256,
        p: int = 128,
        out_dim: int = 3,
        num_layers: int = 6,
        n_freqs: int = 64,
        fourier_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.p        = p
        self.out_dim  = out_dim
        self.n_spatial = 2                            # x/l, y/l get Fourier embedded

        self.fourier  = FourierEmbedding(d_in=2, n_freqs=n_freqs, scale=fourier_scale)
        fourier_out   = self.fourier.out_dim

        n_passthrough = trunk_in - self.n_spatial
        in_dim = fourier_out + n_passthrough

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList([
            ResMLPBlock(trunk_hidden, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.out_head = nn.Linear(trunk_hidden, p * out_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        spatial     = y[:, :self.n_spatial]           # (N, 2)
        passthrough = y[:, self.n_spatial:]           # (N, trunk_in-2)

        spatial_emb = self.fourier(spatial)
        h = torch.cat([spatial_emb, passthrough], dim=-1)
        h = self.input_proj(h)

        for blk in self.res_blocks:
            h = blk(h)

        t = self.out_head(h)                          # (N, p * out_dim)
        return t.view(-1, self.p, self.out_dim)       # (N, p, out_dim)


# ── Branch Network (Cross-Attention) ─────────────────────────────────────────

class SimBranchNet(nn.Module):
    """
    GNN branch that produces per-node key/value pairs for cross-attention.
    Uses FiLM conditioning to inject global similarity scalars!
    """
    def __init__(
        self,
        node_in_dim: int = 4,
        edge_in_dim: int = 3,
        global_dim: int = 3,
        hidden_dim: int = 128,
        edge_hidden: int = 64,
        num_layers: int = 6,
        p: int = 128,
        out_dim: int = 3,
        attn_dim: int | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        self.p        = p
        self.out_dim  = out_dim
        self.n_heads  = n_heads
        self.attn_dim = attn_dim if attn_dim is not None else hidden_dim
        
        assert self.attn_dim % n_heads == 0, "attn_dim must be divisible by n_heads"
        self.d_head = self.attn_dim // n_heads
        self.scale  = math.sqrt(self.d_head)

        # ── Similarity Encoders ──
        self.node_encoder = SimNodeEncoder(node_in_dim, hidden_dim)
        self.edge_encoder = SimEdgeEncoder(edge_in_dim, edge_hidden)

        self.layers = nn.ModuleList([
            SimResGINELayer(hidden_dim, edge_hidden, global_dim) 
            for _ in range(num_layers)
        ])

        # Project node embeddings → keys
        self.W_K = nn.Linear(hidden_dim, self.attn_dim, bias=False)

        # Project node embeddings → values
        assert (p * out_dim) % n_heads == 0, "p*out_dim must be divisible by n_heads"
        self.d_val_head = (p * out_dim) // n_heads
        self.W_V = nn.Linear(hidden_dim, p * out_dim, bias=False)

        self.out_proj = nn.Linear(p * out_dim, p * out_dim)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        global_per_node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e, global_per_node)
        return self.W_K(h), self.W_V(h)

    def cross_attend(
        self,
        q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        batch: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        counts = torch.bincount(batch, minlength=B)
        ptr    = torch.zeros(B + 1, dtype=torch.long, device=batch.device)
        ptr[1:] = counts.cumsum(0)

        results: list[torch.Tensor] = []

        for b in range(B):
            s, e = int(ptr[b].item()), int(ptr[b + 1].item())
            N_b  = e - s

            q_b = q[s:e]
            K_b = K[s:e]
            V_b = V[s:e]

            q_h = q_b.view(N_b, self.n_heads, self.d_head)
            K_h = K_b.view(N_b, self.n_heads, self.d_head)
            V_h = V_b.view(N_b, self.n_heads, self.d_val_head)

            scores = torch.einsum('qhd,khd->hqk', q_h, K_h) / self.scale
            attn   = scores.softmax(dim=-1)

            context_h = torch.einsum('hqk,khd->qhd', attn, V_h)
            context_b = context_h.reshape(N_b, self.p * self.out_dim)

            results.append(context_b)

        context = torch.cat(results, dim=0)
        return self.out_proj(context)


# ── Full DeepONet Surrogate ───────────────────────────────────────────────────

class SimDeepONet(nn.Module):
    """
    Similarity-Based Cross-Attention Deep Operator Network.
    """

    # In similarity dataset: 
    # col 0: x/l, col 1: y/l, col 2: q_local, col 3: bc_flag
    DEFAULT_TRUNK_COLS: list[int] = [0, 1, 2, 3]
    Q_LOCAL_COL: int = 2

    def __init__(
        self,
        node_in_dim: int = 4,
        edge_in_dim: int = 3,
        global_dim: int = 3,
        hidden_dim: int = 128,
        edge_hidden: int = 64,
        num_layers: int = 6,
        trunk_hidden: int = 256,
        trunk_layers: int = 6,
        trunk_cols: list[int] | None = None,
        use_load_coords: bool = True,
        p: int = 128,
        out_dim: int = 3,
        n_freqs: int = 64,
        fourier_scale: float = 1.0,
        trunk_dropout: float = 0.1,
        attn_dim: int | None = None,
        n_heads: int = 4,
    ):
        super().__init__()
        self.p               = p
        self.out_dim         = out_dim
        self.trunk_cols      = trunk_cols if trunk_cols is not None else self.DEFAULT_TRUNK_COLS
        self.use_load_coords = use_load_coords

        _trunk_in = len(self.trunk_cols) + (2 if use_load_coords else 0)
        _attn_dim = attn_dim if attn_dim is not None else hidden_dim

        # ── Branch ──
        self.branch = SimBranchNet(
            node_in_dim = node_in_dim,
            edge_in_dim = edge_in_dim,
            global_dim  = global_dim,
            hidden_dim  = hidden_dim,
            edge_hidden = edge_hidden,
            num_layers  = num_layers,
            p           = p,
            out_dim     = out_dim,
            attn_dim    = _attn_dim,
            n_heads     = n_heads,
        )

        # ── Trunk ──
        self.trunk = TrunkNet(
            trunk_in      = _trunk_in,
            trunk_hidden  = trunk_hidden,
            p             = p,
            out_dim       = out_dim,
            num_layers    = trunk_layers,
            n_freqs       = n_freqs,
            fourier_scale = fourier_scale,
            dropout       = trunk_dropout,
        )

        # ── Trunk query projection ──
        fourier_out = 2 * n_freqs
        passthrough = _trunk_in - 2
        trunk_embed_dim = fourier_out + passthrough
        self.trunk_query_proj = nn.Sequential(
            nn.Linear(trunk_embed_dim, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, _attn_dim),
        )

        B_rff = torch.randn(2, n_freqs) * fourier_scale
        self.register_buffer('_query_B', B_rff)

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def _build_trunk_input(self, data, batch: torch.Tensor) -> torch.Tensor:
        query = data.x[:, self.trunk_cols]

        if self.use_load_coords:
            x_norm  = data.x[:, 0]
            y_norm  = data.x[:, 1]
            q_local = data.x[:, self.Q_LOCAL_COL]
            cx, cy  = compute_load_centroid(x_norm, y_norm, q_local, batch)
            dx = (x_norm - cx).unsqueeze(1)
            dy = (y_norm - cy).unsqueeze(1)
            query = torch.cat([query, dx, dy], dim=1)

        return query    # (total_N, trunk_in)

    def _fourier_embed(self, trunk_input: torch.Tensor) -> torch.Tensor:
        spatial     = trunk_input[:, :2]
        passthrough = trunk_input[:, 2:]
        phases      = 2 * math.pi * spatial @ self._query_B
        fourier     = torch.cat([phases.sin(), phases.cos()], dim=-1)
        return torch.cat([fourier, passthrough], dim=-1)

    def forward(self, data) -> torch.Tensor:
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        B = int(batch.max().item()) + 1

        # Global features per node (for FiLM)
        global_per_node = data.global_feat[batch]

        # Branch
        K, V = self.branch.encode(data.x, data.edge_index, data.edge_attr, global_per_node)

        # Trunk queries
        trunk_input  = self._build_trunk_input(data, batch)
        trunk_embed  = self._fourier_embed(trunk_input)
        q            = self.trunk_query_proj(trunk_embed)

        # Cross-Attention
        b_flat = self.branch.cross_attend(q, K, V, batch, B)
        b = b_flat.view(-1, self.p, self.out_dim)

        # Trunk basis
        t = self.trunk(trunk_input)

        # DeepONet Inner Product
        pred = (b * t).sum(dim=1) + self.bias
        return pred


# ── Builder ───────────────────────────────────────────────────────────────────

def build_sim_deeponet(cfg: dict) -> SimDeepONet:
    """Build SimDeepONet from config dict."""
    trunk_cols = cfg.get('trunk_cols', None)
    if trunk_cols is not None and not isinstance(trunk_cols, list):
        trunk_cols = list(trunk_cols)

    return SimDeepONet(
        node_in_dim     = cfg.get('node_in_dim',  4),
        edge_in_dim     = cfg.get('edge_in_dim',  3),
        global_dim      = cfg.get('global_dim',   3),
        hidden_dim      = cfg.get('hidden_dim',   128),
        edge_hidden     = cfg.get('edge_hidden',  64),
        num_layers      = cfg.get('num_layers',   6),
        trunk_hidden    = cfg.get('trunk_hidden', 256),
        trunk_layers    = cfg.get('trunk_layers', 6),
        trunk_cols      = trunk_cols,
        use_load_coords = cfg.get('use_load_coords', True),
        p               = cfg.get('p',            128),
        out_dim         = cfg.get('out_dim',      3),
        n_freqs         = cfg.get('n_freqs',      64),
        fourier_scale   = cfg.get('fourier_scale', 1.0),
        trunk_dropout   = cfg.get('trunk_dropout', 0.1),
        attn_dim        = cfg.get('attn_dim',     None),
        n_heads         = cfg.get('n_heads',      4),
    )
