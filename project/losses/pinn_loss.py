"""
pinn_loss.py
============
Physics-Informed total loss for the Plate-on-Winkler-Foundation problem.

    L_total = λ_data * L_MSE + λ_pde * L_PDE + λ_bc * L_BC

PDE Residual
------------
The governing equation is:
    D∇⁴w + kw = q                          (1)

where D = Eh³/[12(1-ν²)] is the flexural rigidity.

Approach: Graph Laplacian approximation of ∇⁴
---------------------------------------------
We approximate the Laplacian at node i using a distance-weighted average:

    (∇²w)_i ≈ Σ_{j∈N(i)} (w_j - w_i) / ‖x_i - x_j‖²  [discrete Laplacian]

This is equivalent to the FDM Laplacian on a regular grid and generalises to
irregular meshes. It is:
  - Differentiable w.r.t. w (enables backpropagation)
  - Consistent (converges to the continuous Laplacian as mesh size → 0)
  - Cheap (O(E) operations, same as one message-passing step)

The biharmonic operator ∇⁴w = ∇²(∇²w) is applied by calling the Laplacian
twice on the predicted deflection field.

Alternative approaches considered
----------------------------------
1. FEM stiffness matrix K·w = f: Requires assembling the 12-DOF plate element
   stiffness matrix per batch — expensive, non-differentiable with standard K.
   Would require building K as sparse tensors per case.

2. Finite difference stencil for irregular mesh: Inconsistent for non-uniform
   spacing; requires special handling at boundaries.

3. Neural FEM (deep Ritz): Minimises the variational form.  Requires integration
   over elements — complex to implement for irregular meshes.

4. Graph Laplacian (CHOSEN): Simple, differentiable, generalises to arbitrary
   mesh topologies, consistent for smooth fields. The second application is
   a clean approximation of ∇⁴ on any mesh. Best balance of accuracy and
   implementation cost.

Boundary Conditions
--------------------
ISLab uses Winkler (elastic) foundation which has NO rigid supports in the
standard configuration. The only BCs are zero shear / moment at the free edge,
which are NATURAL BCs (automatically satisfied by the variational principle).

However, if the problem includes clamped or pinned edges, we add:
  - L_BC = Σ_{i∈boundary} ŵ_i²   (deflection = 0 at supports)

In our dataset the edges appear free (no zero-deflection boundary conditions
in the .ou2 output), so L_BC is included as an optional regulariser and
lambda_bc should be set near 0 unless BCs are confirmed.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from typing import Optional, Dict

from .mse_loss import MSELoss


class GraphLaplacian(nn.Module):
    """
    Sparse graph Laplacian operator (distance-weighted).

    (∇²f)_i = Σ_{j∈N(i)} (f_j - f_i) / ‖x_i - x_j‖²

    Applied twice gives the graph biharmonic operator:
        (∇⁴f)_i = (∇²(∇²f))_i

    Parameters
    ----------
    epsilon : small value added to distances to avoid division by zero
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
        """
        Apply ∇⁴ = ∇²∘∇², but do so using purely directional derivatives.
        Returns: bih_w, lap_x, lap_y
        """
        lap_x, lap_y = self._directional_laplacians(f, edge_index, pos, num_nodes)
        lap1 = lap_x + lap_y
        
        lap2_x, lap2_y = self._directional_laplacians(lap1, edge_index, pos, num_nodes)
        bih_w = lap2_x + lap2_y
        
        return bih_w, lap_x, lap_y


class PINNLoss(nn.Module):
    """
    Total PINN loss for Plate-on-Winkler-Foundation.

    L = λ_data * L_MSE + λ_pde * L_PDE + λ_stress * L_stress + λ_bc * L_BC

    Physics Curriculum
    ------------------
    Training is split into two phases:

    Phase 1  (epoch < warmup_epochs):
        phys_scale = 0  →  pure data loss only
        Lets the model learn basic field reconstruction before physics is added.
        Without this, the PDE residual (≫1000 at epoch 1) dominates and
        prevents the network from fitting the data at all.

    Phase 2  (warmup_epochs ≤ epoch < warmup_epochs + ramp_epochs):
        phys_scale linearly ramps from 0 → 1
        Gradually increases physics pressure without sudden gradient shock.

    Phase 3  (epoch ≥ warmup_epochs + ramp_epochs):
        phys_scale = 1  →  full physics weight

    Parameters
    ----------
    lambda_data    : data MSE weight (default 1.0)
    lambda_pde     : PDE biharmonic residual weight (default 0.01)
    lambda_stress  : Kirchhoff stress consistency weight (default 0.1)
    lambda_bc      : boundary condition weight (default 0.1)
    warmup_epochs  : epochs of pure data training before physics is introduced
    ramp_epochs    : epochs over which physics linearly ramps from 0 → full
    mse_weights    : (3,) per-output weights for data loss
    norm_stats     : normalisation statistics dict
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_pde: float = 0.01,
        lambda_bc: float = 0.1,
        lambda_stress: float = 0.1,
        warmup_epochs: int = 80,
        ramp_epochs: int = 40,
        mse_weights: Optional[torch.Tensor] = None,
        norm_stats: Optional[Dict] = None,
    ):
        super().__init__()
        self.lambda_data   = lambda_data
        self.lambda_pde    = lambda_pde
        self.lambda_bc     = lambda_bc
        self.lambda_stress = lambda_stress
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs   = ramp_epochs

        self.data_loss  = MSELoss(weights=mse_weights)
        self.graph_lap  = GraphLaplacian()
        self.norm_stats = norm_stats

    def _get_physical_w(self, pred_w: torch.Tensor) -> torch.Tensor:
        """
        If training with normalised targets, convert predicted w back to
        physical units for the PDE residual.
        """
        if self.norm_stats is not None:
            w_mean = self.norm_stats['target_mean'][0].to(pred_w.device)
            w_std  = self.norm_stats['target_std'][0].to(pred_w.device)
            return pred_w * w_std + w_mean
        return pred_w

    def physics_scale(self, epoch: int) -> float:
        """
        Compute the physics loss scale factor for a given epoch.

        Returns
        -------
        float in [0, 1]
            0 during warmup, linear ramp during ramp phase, 1 at full strength.

        Timeline
        --------
        epochs 0 … warmup_epochs-1          → 0.0  (data only)
        epochs warmup_epochs … +ramp_epochs  → 0.0 … 1.0  (linear ramp)
        epochs warmup_epochs+ramp_epochs+… → 1.0  (full physics)
        """
        if epoch < self.warmup_epochs:
            return 0.0
        ramp_progress = (epoch - self.warmup_epochs) / max(self.ramp_epochs, 1)
        return min(1.0, ramp_progress)

    def forward(
        self,
        pred: torch.Tensor,           # (N, 3) predicted [w, σ_x, σ_y]
        target: torch.Tensor,         # (N, 3) ground truth
        data,                         # PyG Data/Batch object
        epoch: int = 0,
        # Legacy params kept for backward compatibility (ignored)
        step: int = 0,
        total_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:

        # ── Data Loss ─────────────────────────────────────────────────────
        l_data = self.data_loss(pred, target)

        # ── Physics curriculum scale ───────────────────────────────────────
        # Scale is 0 for first warmup_epochs, then linearly ramps to 1.
        # Avoids catastrophic PDE loss (>1000) destabilising early training.
        scale = self.physics_scale(epoch)

        # Skip expensive Laplacian computation entirely during warmup
        if scale == 0.0:
            zero = torch.tensor(0.0, device=pred.device)
            total = self.lambda_data * l_data
            return {
                'total':      total,
                'data':       l_data,
                'pde':        zero,
                'stress_pde': zero,
                'bc':         zero,
                'phys_scale': zero,
            }

        # ── Compute Physical Gradients (Laplacians) ───────────────────────
        pred_w_norm = pred[:, 0]
        pred_w = self._get_physical_w(pred_w_norm)
        num_nodes = pred_w.shape[0]

        bih_w, lap_x, lap_y = self.graph_lap.biharmonic_and_directional(
            pred_w, data.edge_index, data.pos, num_nodes
        )

        # ── PDE Residual (Biharmonic) ──────────────────────────────────────
        l_pde = self._pde_loss(bih_w, pred_w, data, num_nodes)

        # ── Stress Physics Loss (Kirchhoff) ────────────────────────────────
        l_stress = self._stress_loss(pred, data, lap_x, lap_y, num_nodes)

        # ── BC Loss ────────────────────────────────────────────────────────
        l_bc = self._bc_loss(pred, data)

        total = (self.lambda_data   * l_data
                 + scale * self.lambda_pde    * l_pde
                 + scale * self.lambda_stress * l_stress
                 +         self.lambda_bc     * l_bc)

        return {
            'total':      total,
            'data':       l_data,
            'pde':        l_pde,
            'stress_pde': l_stress,
            'bc':         l_bc,
            'phys_scale': torch.tensor(scale),
        }

    def _pde_loss(self, bih_w: torch.Tensor, pred_w: torch.Tensor, data, num_nodes: int) -> torch.Tensor:
        # Get per-node physical parameters
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
            D_per_node  = data.D[batch].squeeze(-1)           # (N,)
            k_per_node  = data.k_winkler[batch].squeeze(-1)   # (N,)
            q_per_node  = data.x[:, 5] * data.pressure[batch].squeeze(-1)
        else:
            D_per_node  = data.D.expand(num_nodes)
            k_per_node  = data.k_winkler.expand(num_nodes)
            q_per_node  = data.x[:, 5] * data.pressure.expand(num_nodes)

        # PDE residual at each node
        residual = D_per_node * bih_w + k_per_node * pred_w - q_per_node  # (N,)
        res_sq = residual ** 2

        # ── INTERIOR MASKING ──
        # The Graph Laplacian is a centered-difference stencil. At the physical boundaries,
        # nodes are missing neighbors, meaning the stencil mathematically collapses and evaluates
        # to a 1st-derivative term instead of a 2nd-derivative term.
        # Since the biharmonic operator applies the Laplacian twice, this error corrupts both
        # the boundary nodes AND the nodes exactly 1-hop away from the boundary.
        # If we do not mask these out, the PINN will try to minimize a massive, broken PDE residual
        # at the edges, destroying the true data predictions.
        
        # bc_flag is 1 for boundary nodes
        is_boundary = data.x[:, 7] > 0.5
        
        # We need to find nodes that are 1-hop from the boundary.
        # We can do this by scattering the boundary flag across the edges.
        src, dst = data.edge_index
        boundary_neighbors = scatter(is_boundary[src].float(), dst, dim=0, dim_size=num_nodes, reduce='max')
        
        # Valid interior nodes are those that are NOT boundaries, and do NOT touch a boundary
        valid_interior = (~is_boundary) & (boundary_neighbors < 0.5)
        
        if valid_interior.sum() == 0:
            return res_sq.mean()  # Fallback if graph is too small

        return res_sq[valid_interior].mean()

    def _stress_loss(self, pred: torch.Tensor, data, lap_x: torch.Tensor, lap_y: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Enforce Kirchhoff Plate Theory theoretical stress vs predicted stress.
        """
        # Un-normalize predicted stress
        if self.norm_stats is not None:
            sx_mean = self.norm_stats['target_mean'][1].to(pred.device)
            sx_std  = self.norm_stats['target_std'][1].to(pred.device)
            sy_mean = self.norm_stats['target_mean'][2].to(pred.device)
            sy_std  = self.norm_stats['target_std'][2].to(pred.device)
            
            pred_sx = pred[:, 1] * sx_std + sx_mean
            pred_sy = pred[:, 2] * sy_std + sy_mean
        else:
            pred_sx = pred[:, 1]
            pred_sy = pred[:, 2]

        # Get per-node parameters
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
            E  = data.E[batch].squeeze(-1)           # (N,)
            h  = data.thickness[batch].squeeze(-1)   # (N,)
            nu = data.nu[batch].squeeze(-1)          # (N,)
        else:
            E  = data.E.expand(num_nodes)
            h  = data.thickness.expand(num_nodes)
            nu = data.nu.expand(num_nodes)

        # Theoretical stress at BOTTOM surface (tension positive for downward deflection)
        # sigma_x = - (E * h) / (2 * (1 - nu^2)) * (d2w/dx2 + nu * d2w/dy2)
        coef = - (E * h) / (2.0 * (1.0 - nu**2))
        
        theo_sx = coef * (lap_x + nu * lap_y)
        theo_sy = coef * (lap_y + nu * lap_x)

        # MSE
        res_total = (pred_sx - theo_sx)**2 + (pred_sy - theo_sy)**2
        
        # MASKING: Unlike the biharmonic operator, lap_x and lap_y only apply the laplacian ONCE.
        # This means only the EXACT boundary nodes are corrupted by the missing-neighbor stencil.
        # Nodes 1-hop away are safe to use for stress physics!
        is_boundary = data.x[:, 7] > 0.5
        valid_stress = ~is_boundary

        if valid_stress.sum() == 0:
            return res_total.mean()
            
        return res_total[valid_stress].mean()

    def _bc_loss(self, pred: torch.Tensor, data) -> torch.Tensor:
        """
        Boundary condition loss.

        For our Winkler foundation slab, the edges are FREE (no rigid supports).
        If any node is flagged as a clamped/pinned support (bc_flag > 0.5),
        we penalise non-zero predicted deflection there.

        bc_flag is stored in node feature column 7.
        """
        bc_flag = data.x[:, 7]  # (N,), 1 = boundary
        pred_w  = pred[:, 0]    # (N,)

        bc_mask = bc_flag > 0.5
        if bc_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=False)

        # For Winkler: boundary nodes are still free to deflect (no zero BC).
        # This loss applies only if the problem has hard supports.
        # We include it for completeness; it has near-zero effect when
        # lambda_bc is small and there are no clamped edges.
        bc_penalty = (pred_w[bc_mask]) ** 2
        return bc_penalty.mean()
