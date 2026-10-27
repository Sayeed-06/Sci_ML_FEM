"""
pinn_loss.py
============
Physics-Informed Neural Network (PINN) loss formulations for the GNN.

Uses a discrete distance-weighted Graph Laplacian approximation to compute spatial
derivatives instead of Autograd. This drastically reduces GPU memory consumption
by bypassing the massive continuous O(N * 2^4) graphs needed for 4th-order PDEs.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

class GraphLaplacian(nn.Module):
    """
    Sparse graph Laplacian operator (distance-weighted).
    (∇²f)_i = Σ_{j∈N(i)} (f_j - f_i) / ‖x_i - x_j‖²
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def _directional_laplacians(
        self,
        f: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ∇²x and ∇²y separately by filtering horizontal/vertical edges."""
        src, dst = edge_index[0], edge_index[1]

        dx = pos[dst, 0] - pos[src, 0]
        dy = pos[dst, 1] - pos[src, 1]
        
        # Horizontal edges: |dy| is zero
        mask_x = torch.abs(dy) < 1e-6
        # Vertical edges: |dx| is zero
        mask_y = torch.abs(dx) < 1e-6

        # Edge weights: inverse squared distance
        dist_sq = (pos[dst] - pos[src]) ** 2
        dist_sq_sum = dist_sq.sum(dim=1).clamp(min=self.epsilon)
        weights = 1.0 / dist_sq_sum

        if f.dim() == 1:
            diff = f[dst] - f[src]
        else:
            diff = f[dst] - f[src]
            weights = weights.unsqueeze(-1)

        # lap_x
        weighted_diff_x = weights * diff * mask_x.float() if f.dim() == 1 else weights * diff * mask_x.unsqueeze(-1).float()
        lap_x = scatter(weighted_diff_x, src, dim=0, dim_size=num_nodes, reduce='sum')

        # lap_y
        weighted_diff_y = weights * diff * mask_y.float() if f.dim() == 1 else weights * diff * mask_y.unsqueeze(-1).float()
        lap_y = scatter(weighted_diff_y, src, dim=0, dim_size=num_nodes, reduce='sum')

        return lap_x, lap_y

    def biharmonic_and_directional(
        self,
        f: torch.Tensor,
        edge_index: torch.Tensor,
        pos: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lap_x, lap_y = self._directional_laplacians(f, edge_index, pos, num_nodes)
        lap1 = lap_x + lap_y
        
        lap2_x, lap2_y = self._directional_laplacians(lap1, edge_index, pos, num_nodes)
        bih_w = lap2_x + lap2_y
        
        return bih_w, lap_x, lap_y


class PINNLoss(nn.Module):
    """
    Computes data MSE + Physics-Informed losses using a discrete Graph Laplacian.
    """
    def __init__(
        self,
        loss_type: str = "pinn_biharmonic",
        lambda_phys: float = 1.0e-6,
        mu: float = 0.15,
        loss_weights: list = [1.0, 1.0, 1.0],
        warmup_epochs: int = 25,
        ramp_epochs: int = 40,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_phys = lambda_phys
        self.mu = mu
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.register_buffer("loss_weights", torch.tensor(loss_weights, dtype=torch.float))
        self.graph_lap = GraphLaplacian()

    def physics_scale(self, epoch: int) -> float:
        """Linear ramp curriculum for physics loss."""
        if epoch < self.warmup_epochs:
            return 0.0
        ramp_progress = (epoch - self.warmup_epochs) / max(self.ramp_epochs, 1)
        return min(1.0, ramp_progress)

    def _weighted_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        se = (pred - target) ** 2
        return (se * self.loss_weights).mean()

    def _pde_loss(self, bih_w: torch.Tensor, w_pred: torch.Tensor, data, num_nodes: int, norm_stats: dict) -> torch.Tensor:
        """
        Compute the biharmonic PDE residual: D ∇⁴w + k w = q
        """
        device = bih_w.device
        E = 3.25e4
        h_slab = 327.12
        D = (E * h_slab**3) / (12 * (1 - self.mu**2))
        k_winkler = 32.68 * 1e-3 # N/mm³

        batch = data.batch if hasattr(data, "batch") and data.batch is not None \
                else torch.zeros(num_nodes, dtype=torch.long, device=device)

        # Reconstruct standard problem constants
        gf_mean = norm_stats["gf_mean"].to(device)
        gf_std  = norm_stats["gf_std"].to(device)
        global_raw = data.global_feat * gf_std + gf_mean # (B, 3) = [l, phi, PQ]

        # We can reconstruct P from PQ ratio
        PQ_nodes = global_raw[:, 2][batch].unsqueeze(1) # P / (gamma * h)
        gamma_h = 23600 * 1e-9 * h_slab # standard gamma
        P_nodes = (PQ_nodes * gamma_h).squeeze(-1)
        
        # data.x[:, 2] is q_local (0 or 1)
        q_applied = data.x[:, 2] * P_nodes 
        
        pde_res = (D * bih_w + k_winkler * w_pred - q_applied)
        
        # Normalize residual by P_nodes so scale is uniform
        pde_res_sq = (pde_res / P_nodes)**2
        
        # MASKING SETUP
        # Boundary nodes are corrupted by incomplete graph stencils.
        is_boundary = data.x[:, 3] > 0.5 # column 3 is is_boundary in similarity logic
        
        # Mask 1-hop boundary for biharmonic (Laplacian applies twice)
        src, dst = data.edge_index
        boundary_neighbors = scatter(is_boundary[src].float(), dst, dim=0, dim_size=num_nodes, reduce='max')
        valid_interior = (~is_boundary) & (boundary_neighbors < 0.5)

        if valid_interior.sum() > 0:
            return pde_res_sq[valid_interior].mean()
        
        return pde_res_sq.mean()

    def forward(
        self,
        pred: torch.Tensor,       # (N, 3) -> [w, sx, sy] in NORMALIZED space
        data,                     # PyG Data batch
        norm_stats: dict,         # to denormalize for physics math
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns (total_loss, metrics_dict)
        """
        # 1. Base Data-Driven Loss (MSE on normalized outputs)
        mse_loss = self._weighted_mse(pred, data.y)
        metrics = {"mse_loss": mse_loss.item()}

        scale = self.physics_scale(epoch)
        metrics["phys_scale"] = scale

        if self.loss_type == "mse" or not torch.is_grad_enabled() or scale == 0.0:
            return mse_loss, metrics

        # 2. Physics Loss requires absolute units and spatial derivatives
        device = pred.device
        y_mean = norm_stats["y_mean"].to(device)
        y_std  = norm_stats["y_std"].to(device)

        # De-normalize predictions to physical units (w in mm)
        # We only need w_pred for biharmonic
        pred_abs = pred * y_std + y_mean
        w_pred  = pred_abs[:, 0]  # (N,)

        num_nodes = w_pred.shape[0]

        # --- Discrete Finite Differences ---
        # Notice we pass physical w (mm) and physical pos (mm). 
        # This naturally outputs ∂²w/∂x² and ∇⁴w with units [mm^-1] and [mm^-3]!
        bih_w, lap_x, lap_y = self.graph_lap.biharmonic_and_directional(
            w_pred, data.edge_index, data.pos, num_nodes
        )

        if self.loss_type == "pinn_biharmonic":
            loss_pde = self._pde_loss(bih_w, w_pred, data, num_nodes, norm_stats)
            
            metrics["loss_pde"] = loss_pde.item()
            total_loss = mse_loss + scale * self.lambda_phys * loss_pde
            return total_loss, metrics

        raise ValueError(f"Unknown loss_type: {self.loss_type}")
