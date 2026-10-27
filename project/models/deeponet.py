"""
deeponet.py
===========
GNN-Branch Deep Operator Network (DeepONet) for FEM plate surrogate modelling.

Architecture
------------
Since cases have variable grid resolution (Nx × Ny varies), a standard MLP
branch over a flattened sensor field is not directly applicable. Instead we
use the existing GINEConv message-passing stack as the Branch Network, which
naturally handles variable graph sizes.

                  ┌─────────────────────────────────────────┐
Graph (N nodes)   │  NodeEncoder → GINEConv × L → GlobalPool │ → b ∈ ℝ^(B × p × 3)
                  └─────────────────────────────────────────┘   (Branch)
                                                                      │
                                                              ─── Σ bₖ · tₖ + bias ──→ (N, 3)
                                                                      │
trunk_cols        ┌────────────────────────────────────────┐         │
per node          │  FourierEmbed → Residual MLP → p × 3   │ → t ∈ ℝ^(N × p × 3)
                  └────────────────────────────────────────┘   (Trunk)

Branch Network
--------------
Encoder: NodeEncoder (same as GNNSurrogate)
Message passing: ResGINELayer × num_layers (same as GNNSurrogate)
Pooling: global mean + max → concat → Linear(2*hidden → p*3)
Output: b ∈ ℝ^(B × p × out_dim)

Trunk Network (v3 improvements)
---------------------------------
Input: configurable trunk_cols from data.x — default [0,1,5,7]:
  col 0  x_norm     — normalised x coordinate
  col 1  y_norm     — normalised y coordinate
  col 5  q_local    — load indicator (1 if node is under applied load)
  col 7  bc_flag    — boundary flag

PLUS augmentation-safe load-relative coordinates (use_load_coords=True):
  dx_to_load_center  — (x_norm - load_centroid_x_norm) per node
  dy_to_load_center  — (y_norm - load_centroid_y_norm) per node

  The load centroid is computed from nodes where q_local==1, per graph,
  using the batch vector. This is AUGMENTATION-SAFE because q_local is
  flipped together with x_norm/y_norm by RandomPlateSymmetry.

  WARNING: do NOT use data.load_rect for this — the symmetry transform
  does NOT update load_rect, so it goes stale after augmentation.

Architecture: Fourier embedding → Residual MLP blocks with dropout

Output Assembly
---------------
For each graph i and each of its nodes j:
    pred[j, k] = bias[k] + Σ_{r=0}^{p-1}  b[i, r, k] · t[j, r, k]

where k ∈ {0, 1, 2} = {w, σ_x, σ_y}.

References
----------
Lu et al. "Learning nonlinear operators via DeepONet" (Nature MI, 2021)
Wang et al. "Improved architectures and training algorithms for deep operator
             networks" (J. Sci. Comput., 2022)
Tancik et al. "Fourier Features Let Networks Learn High Frequency Functions"
              (NeurIPS 2020)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool  # kept for optional analysis

from .encoders import NodeEncoder, EdgeEncoder
from .gnn import ResGINELayer


# ── Augmentation-safe load centroid computation ───────────────────────────────

def compute_load_centroid(
    x_norm: torch.Tensor,      # (total_N,)  node x in [0,1]
    y_norm: torch.Tensor,      # (total_N,)  node y in [0,1]
    q_local: torch.Tensor,     # (total_N,)  1 if node under load
    batch: torch.Tensor,       # (total_N,)  graph index
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute load centroid (mean of loaded-node coords) per graph.

    Augmentation-safe: q_local is flipped by RandomPlateSymmetry together
    with x_norm/y_norm, so the centroid always tracks the flipped load.

    Returns
    -------
    cx : (total_N,)  centroid x_norm broadcast to each node
    cy : (total_N,)  centroid y_norm broadcast to each node

    For graphs with no loaded nodes (q_local all 0), centroid defaults to 0.5.
    """
    # Weighted sum of coordinates by q_local, then divide by q_local sum per graph
    q = q_local.clamp(min=0.0)                         # (total_N,)

    # Per-graph sums of q and q*coord
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
    """
    Random Fourier Features for spatial coordinates.

    Maps (N, d_in) → (N, 2*n_freqs) via sin/cos projections.
    Fixed random frequencies capture multi-scale spatial patterns that plain
    coordinate inputs struggle to represent (Tancik et al., 2020).
    """

    def __init__(self, d_in: int = 2, n_freqs: int = 64, scale: float = 1.0):
        super().__init__()
        self.n_freqs = n_freqs
        B = torch.randn(d_in, n_freqs) * scale
        self.register_buffer('B', B)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, d_in) → (N, 2*n_freqs)"""
        proj = 2 * math.pi * x @ self.B          # (N, n_freqs)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)


# ── Residual MLP block with dropout ──────────────────────────────────────────

class ResMLPBlock(nn.Module):
    """Residual block: Linear → LN → GELU → Dropout → Linear → LN → +skip → GELU"""

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


# ── Trunk Network (v3) ────────────────────────────────────────────────────────

class TrunkNet(nn.Module):
    """
    Maps per-node query features to basis function evaluations.

    Input layout (after load-coord augmentation):
        [x_norm, y_norm, q_local, bc_flag, dx_to_load, dy_to_load]
        ← spatial (2) ─→ ← passthrough (2) ─→ ← relative load (2) ─→

    Fourier embedding is applied only to the first 2 (spatial) columns.
    """

    def __init__(
        self,
        trunk_in: int = 6,           # including load-relative coords
        trunk_hidden: int = 256,
        p: int = 256,
        out_dim: int = 3,
        num_layers: int = 6,
        n_freqs: int = 64,
        fourier_scale: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.p        = p
        self.out_dim  = out_dim
        self.n_spatial = 2                            # cols 0,1 get Fourier embedded

        self.fourier  = FourierEmbedding(d_in=2, n_freqs=n_freqs, scale=fourier_scale)
        fourier_out   = self.fourier.out_dim          # 2 * n_freqs

        n_passthrough = trunk_in - self.n_spatial     # remaining cols passed through
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
        """(total_N, trunk_in) → (total_N, p, out_dim)"""
        spatial     = y[:, :self.n_spatial]           # (N, 2)
        passthrough = y[:, self.n_spatial:]           # (N, trunk_in-2)

        spatial_emb = self.fourier(spatial)           # (N, 2*n_freqs)
        h = torch.cat([spatial_emb, passthrough], dim=-1)
        h = self.input_proj(h)

        for blk in self.res_blocks:
            h = blk(h)

        t = self.out_head(h)                          # (N, p * out_dim)
        return t.view(-1, self.p, self.out_dim)       # (N, p, out_dim)


# ── Branch Network (Cross-Attention) ─────────────────────────────────────────

class BranchNet(nn.Module):
    """
    GNN branch that produces per-node key/value pairs for cross-attention.

    In standard DeepONet, the branch compresses the entire input function to a
    SINGLE vector b (via global pooling). This is a hard information bottleneck:
    every query point in a graph gets the SAME coefficients, forcing the trunk
    to reconstruct all spatial variation alone.

    This branch removes that bottleneck entirely. Instead of pooling:

        GNN(x, edge_index) → H ∈ ℝ^(N × hidden)   (per-node embeddings)
        K = W_K(H) ∈ ℝ^(N × d_attn)               (attention keys)
        V = W_V(H) ∈ ℝ^(N × p × out_dim)           (attention values = basis coefficients)

    In DeepONetSurrogate.forward(), the trunk generates a query q_j for each
    query node j. Cross-attention then gives each query node a UNIQUE b_j:

        b_j = softmax(q_j K^T / √d) · V   (attending only within same graph)

    This is theoretically consistent with DeepONet:
    - Branch still encodes the input function (all node features, no spatial info)
    - Trunk still evaluates basis functions at query locations
    - The cross-attention is the "bridge" — trunk queries the branch about what
      the input function looks like FROM this query node's perspective

    Parameters
    ----------
    node_in_dim : raw node feature dimension (default 11)
    edge_in_dim : raw edge feature dimension (default 3)
    hidden_dim  : GNN hidden width (default 128)
    edge_hidden : edge embedding dim (default 64)
    num_layers  : number of GINE layers (default 4)
    p           : number of basis functions (default 128)
    out_dim     : output channels (default 3)
    attn_dim    : attention key/query dimension (default = hidden_dim)
    n_heads     : number of attention heads (default 4)
    """

    def __init__(
        self,
        node_in_dim: int = 11,
        edge_in_dim: int = 3,
        hidden_dim: int = 128,
        edge_hidden: int = 64,
        num_layers: int = 4,
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
        # Per-head key/query dimension
        assert self.attn_dim % n_heads == 0, \
            f"attn_dim ({self.attn_dim}) must be divisible by n_heads ({n_heads})"
        self.d_head = self.attn_dim // n_heads
        self.scale  = math.sqrt(self.d_head)

        self.node_encoder = NodeEncoder(node_in_dim, hidden_dim)
        self.edge_encoder = EdgeEncoder(edge_in_dim, edge_hidden)

        self.layers = nn.ModuleList([
            ResGINELayer(hidden_dim, edge_hidden) for _ in range(num_layers)
        ])

        # Project node embeddings → multi-head keys
        self.W_K = nn.Linear(hidden_dim, self.attn_dim, bias=False)

        # Project node embeddings → values (p * out_dim per head, then concatenate heads)
        # Each head produces (p * out_dim / n_heads) features
        # After concat across heads: p * out_dim total  (matches standard DeepONet b dim)
        assert (p * out_dim) % n_heads == 0, \
            f"p*out_dim ({p * out_dim}) must be divisible by n_heads ({n_heads})"
        self.d_val_head = (p * out_dim) // n_heads
        self.W_V = nn.Linear(hidden_dim, p * out_dim, bias=False)

        # Output projection: blends across heads after attention
        self.out_proj = nn.Linear(p * out_dim, p * out_dim)

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        GNN pass → per-node keys and values.

        Parameters
        ----------
        x          : (total_N, node_in_dim)
        edge_index : (2, E)
        edge_attr  : (E, edge_in_dim)

        Returns
        -------
        K : (total_N, attn_dim)     — attention keys (multi-head)
        V : (total_N, p * out_dim)  — attention values
        """
        h = self.node_encoder(x)
        e = self.edge_encoder(edge_attr)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        return self.W_K(h), self.W_V(h)

    def cross_attend(
        self,
        q: torch.Tensor,    # (total_N, attn_dim) — queries from trunk
        K: torch.Tensor,    # (total_N, attn_dim) — keys from branch
        V: torch.Tensor,    # (total_N, p*out_dim) — values from branch
        batch: torch.Tensor,
        B: int,
    ) -> torch.Tensor:
        """
        Per-graph multi-head cross-attention.

        Each query node j attends to ALL branch nodes in its own graph.
        Returns a unique b_j ∈ ℝ^(p, out_dim) per node — replacing the
        single shared b from standard global-pooling DeepONet.

        Parameters
        ----------
        q     : (total_N, attn_dim)   — query per node
        K     : (total_N, attn_dim)   — key per node
        V     : (total_N, p*out_dim)  — value per node
        batch : (total_N,)
        B     : number of graphs

        Returns
        -------
        context : (total_N, p * out_dim)
        """
        # ── Build ptr (start index of each graph in the batched tensor) ──
        counts = torch.bincount(batch, minlength=B)         # (B,)
        ptr    = torch.zeros(B + 1, dtype=torch.long, device=batch.device)
        ptr[1:] = counts.cumsum(0)                          # (B+1,)

        results: list[torch.Tensor] = []

        for b in range(B):
            s, e = int(ptr[b].item()), int(ptr[b + 1].item())
            N_b  = e - s

            # Slice this graph's q, K, V
            q_b = q[s:e]          # (N_b, attn_dim)
            K_b = K[s:e]          # (N_b, attn_dim)
            V_b = V[s:e]          # (N_b, p*out_dim)

            # ── Reshape for multi-head attention ──────────────────────────
            # (N_b, n_heads, d_head)
            q_h = q_b.view(N_b, self.n_heads, self.d_head)
            K_h = K_b.view(N_b, self.n_heads, self.d_head)
            # (N_b, n_heads, d_val_head)
            V_h = V_b.view(N_b, self.n_heads, self.d_val_head)

            # ── Scaled dot-product per head ───────────────────────────────
            # scores: (n_heads, N_b, N_b)  —  q · K^T / sqrt(d_head)
            scores = torch.einsum('qhd,khd->hqk', q_h, K_h) / self.scale
            attn   = scores.softmax(dim=-1)                 # (n_heads, N_b, N_b)

            # ── Aggregate values ──────────────────────────────────────────
            # context_h: (n_heads, N_b, d_val_head)
            context_h = torch.einsum('hqk,khd->qhd', attn, V_h)
            # Merge heads: (N_b, p*out_dim)
            context_b = context_h.reshape(N_b, self.p * self.out_dim)

            results.append(context_b)

        context = torch.cat(results, dim=0)                 # (total_N, p*out_dim)
        return self.out_proj(context)                       # (total_N, p*out_dim)


# ── GNN-Branch DeepONet (Cross-Attention) ─────────────────────────────────────

class DeepONetSurrogate(nn.Module):
    """
    Cross-Attention GNN-Branch DeepONet.

    Architecture (v5 — cross-attention branch):
    ─────────────────────────────────────────────────────────────────────────
    Branch:
        x, edge_index → GNN → H ∈ ℝ^(N × hidden)
        K = W_K(H) ∈ ℝ^(N × attn_dim)
        V = W_V(H) ∈ ℝ^(N × p × out_dim)

    Trunk:
        trunk_cols + dx_load, dy_load → Fourier embed → Residual MLP
        → t ∈ ℝ^(N × p × out_dim)   (spatial basis functions, unchanged)

    Trunk query projection (new in v5):
        trunk_input → small MLP → q ∈ ℝ^(N × attn_dim)

    Cross-attention (per graph):
        b_j = Σ_i softmax(q_j · K_i^T / √d)_i · V_i   ∈ ℝ^(p × out_dim)

    Output:
        pred_j = Σ_r b_j[r] · t_j[r] + bias    ∈ ℝ^(out_dim)
    ─────────────────────────────────────────────────────────────────────────

    Key difference from v3/v4:
        b_j is NOW UNIQUE PER NODE (not shared across all nodes in a graph).
        Each query node attends to different branch nodes based on its position.
        This eliminates the global-pooling information bottleneck.

    Parameters
    ----------
    node_in_dim     : node feature dim (default 11)
    edge_in_dim     : edge feature dim (default 3)
    hidden_dim      : GNN hidden width (default 128)
    edge_hidden     : edge embedding dim (default 64)
    num_layers      : GNN branch layers (default 4)
    trunk_hidden    : trunk MLP hidden width (default 256)
    trunk_layers    : trunk residual block count (default 6)
    trunk_cols      : data.x columns used as base trunk input (default [0,1,5,7])
    use_load_coords : append dx,dy to load centroid (default True)
    p               : number of basis functions (default 128)
    out_dim         : output channels (default 3)
    n_freqs         : Fourier features for spatial dims (default 64)
    fourier_scale   : Fourier frequency scale (default 1.0)
    trunk_dropout   : dropout in trunk residual blocks (default 0.1)
    attn_dim        : attention key/query dimension (default = hidden_dim)
    n_heads         : number of attention heads (default 4)
    """

    DEFAULT_TRUNK_COLS: list[int] = [0, 1, 5, 7]
    Q_LOCAL_COL: int = 5

    def __init__(
        self,
        node_in_dim: int = 11,
        edge_in_dim: int = 3,
        hidden_dim: int = 128,
        edge_hidden: int = 64,
        num_layers: int = 4,
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

        # ── Branch (produces K, V per node) ───────────────────────────────
        self.branch = BranchNet(
            node_in_dim = node_in_dim,
            edge_in_dim = edge_in_dim,
            hidden_dim  = hidden_dim,
            edge_hidden = edge_hidden,
            num_layers  = num_layers,
            p           = p,
            out_dim     = out_dim,
            attn_dim    = _attn_dim,
            n_heads     = n_heads,
        )

        # ── Trunk (spatial basis functions) ───────────────────────────────
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

        # ── Trunk query projection (generates queries for cross-attention) ─
        # Maps raw trunk features → attention query vector q ∈ ℝ^(attn_dim)
        # Uses the same Fourier-embedded trunk input for consistent spatial encoding
        fourier_out = 2 * n_freqs
        passthrough = _trunk_in - 2          # features after the 2 spatial coords
        trunk_embed_dim = fourier_out + passthrough
        self.trunk_query_proj = nn.Sequential(
            nn.Linear(trunk_embed_dim, trunk_hidden),
            nn.LayerNorm(trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, _attn_dim),
        )

        # Shared Fourier embedding for query projection
        # (same frequencies as trunk but separate buffer — weights not shared)
        B_rff = torch.randn(2, n_freqs) * fourier_scale
        self.register_buffer('_query_B', B_rff)

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def _build_trunk_input(self, data, batch: torch.Tensor) -> torch.Tensor:
        """Assemble per-node trunk feature vector (augmentation-safe)."""
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
        """
        Apply Fourier embedding to the first 2 (spatial) trunk columns,
        then concatenate the remaining passthrough features.
        Returns the same format as TrunkNet's internal embedding.
        """
        spatial     = trunk_input[:, :2]                        # (N, 2)
        passthrough = trunk_input[:, 2:]                        # (N, trunk_in-2)
        phases      = 2 * math.pi * spatial @ self._query_B    # (N, n_freqs)
        fourier     = torch.cat([phases.sin(), phases.cos()], dim=-1)  # (N, 2*n_freqs)
        return torch.cat([fourier, passthrough], dim=-1)        # (N, trunk_embed_dim)

    def forward(self, data) -> torch.Tensor:
        """
        Parameters
        ----------
        data : PyG Batch or Data

        Returns
        -------
        pred : (total_N, out_dim)
        """
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        B = int(batch.max().item()) + 1

        # ── Branch: encode input function → K, V per node ─────────────────
        K, V = self.branch.encode(data.x, data.edge_index, data.edge_attr)

        # ── Trunk: build query features ────────────────────────────────────
        trunk_input  = self._build_trunk_input(data, batch)   # (N, trunk_in)
        trunk_embed  = self._fourier_embed(trunk_input)        # (N, trunk_embed_dim)

        # ── Query projection: trunk features → attention queries ───────────
        q = self.trunk_query_proj(trunk_embed)                 # (N, attn_dim)

        # ── Cross-attention: per-node unique b_j ──────────────────────────
        # Each query node attends over all branch nodes in its own graph
        b_flat = self.branch.cross_attend(q, K, V, batch, B)  # (N, p*out_dim)
        b = b_flat.view(-1, self.p, self.out_dim)              # (N, p, out_dim)

        # ── Trunk: spatial basis functions ────────────────────────────────
        t = self.trunk(trunk_input)                            # (N, p, out_dim)

        # ── DeepONet inner product ─────────────────────────────────────────
        pred = (b * t).sum(dim=1) + self.bias                 # (N, out_dim)
        return pred

    def branch_embed(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (K, V) from branch for analysis."""
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        return self.branch.encode(data.x, data.edge_index, data.edge_attr)

    def trunk_embed(self, data) -> torch.Tensor:
        """Trunk basis t for analysis."""
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
                else torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        return self.trunk(self._build_trunk_input(data, batch))



# ── Builder ───────────────────────────────────────────────────────────────────

def build_deeponet(cfg: dict) -> DeepONetSurrogate:
    """
    Construct a DeepONetSurrogate from a config dict.

    Config keys (all optional):
        node_in_dim     : 11
        edge_in_dim     : 3
        hidden_dim      : 128
        edge_hidden     : 64
        num_layers      : 4        (GNN branch depth)
        trunk_hidden    : 256
        trunk_layers    : 6
        trunk_cols      : [0,1,5,7]
        use_load_coords : true
        p               : 128
        out_dim         : 3
        n_freqs         : 64
        fourier_scale   : 1.0
        trunk_dropout   : 0.1
        attn_dim        : null     (defaults to hidden_dim)
        n_heads         : 4
    """
    trunk_cols = cfg.get('trunk_cols', None)
    if trunk_cols is not None and not isinstance(trunk_cols, list):
        trunk_cols = list(trunk_cols)

    return DeepONetSurrogate(
        node_in_dim     = cfg.get('node_in_dim',      11),
        edge_in_dim     = cfg.get('edge_in_dim',       3),
        hidden_dim      = cfg.get('hidden_dim',       128),
        edge_hidden     = cfg.get('edge_hidden',       64),
        num_layers      = cfg.get('num_layers',         4),
        trunk_hidden    = cfg.get('trunk_hidden',     256),
        trunk_layers    = cfg.get('trunk_layers',       6),
        trunk_cols      = trunk_cols,
        use_load_coords = cfg.get('use_load_coords', True),
        p               = cfg.get('p',               128),
        out_dim         = cfg.get('out_dim',           3),
        n_freqs         = cfg.get('n_freqs',          64),
        fourier_scale   = cfg.get('fourier_scale',   1.0),
        trunk_dropout   = cfg.get('trunk_dropout',   0.1),
        attn_dim        = cfg.get('attn_dim',        None),
        n_heads         = cfg.get('n_heads',            4),
    )


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from torch_geometric.data import Data, Batch

    torch.manual_seed(0)

    def make_fake_graph(n_nodes=100, load_frac=0.2):
        n_edges = n_nodes * 4
        x = torch.randn(n_nodes, 11)
        x[:, 0] = torch.linspace(0, 1, n_nodes)    # x_norm
        x[:, 1] = torch.linspace(0, 1, n_nodes)    # y_norm
        x[:, 5] = (torch.rand(n_nodes) < load_frac).float()  # q_local
        x[:, 7] = (torch.rand(n_nodes) < 0.3).float()        # bc_flag
        return Data(
            x          = x,
            edge_index = torch.randint(0, n_nodes, (2, n_edges)),
            edge_attr  = torch.randn(n_edges, 3),
            y          = torch.randn(n_nodes, 3),
        )

    graphs = [make_fake_graph(80), make_fake_graph(120), make_fake_graph(100)]
    batch  = Batch.from_data_list(graphs)

    model = DeepONetSurrogate()
    out   = model(batch)
    print(f"Output shape: {out.shape}")            # (300, 3)
    assert out.shape == (300, 3), f"Bad shape: {out.shape}"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    loss = out.sum()
    loss.backward()
    branch_grads = [p.grad for p in model.branch.parameters() if p.grad is not None]
    trunk_grads  = [p.grad for p in model.trunk.parameters()  if p.grad is not None]
    qproj_grads  = [p.grad for p in model.trunk_query_proj.parameters() if p.grad is not None]
    print(f"Branch params with grad : {len(branch_grads)}")
    print(f"Trunk  params with grad : {len(trunk_grads)}")
    print(f"QProj  params with grad : {len(qproj_grads)}")
    assert len(branch_grads) > 0, "Branch has no gradients!"
    assert len(trunk_grads)  > 0, "Trunk  has no gradients!"
    assert len(qproj_grads)  > 0, "trunk_query_proj has no gradients!"
    print("Smoke test passed ✓")
