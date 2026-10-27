"""
mse_loss.py
===========
Standard MSE loss for data-driven training.

L = (1/N) Σ_i Σ_k (ŷ_{ik} - y_{ik})²

where k ∈ {w, σ_x, σ_y}.

The loss is computed on ALL nodes. If normalised targets are used (recommended),
all three outputs are automatically on similar scales and equal weighting is
appropriate. If un-normalised, the deflection w (~54000) dominates σ (~0.1)
by a factor of ~500,000 — so always use target normalisation.
"""

import torch
import torch.nn as nn
from typing import Optional


class MSELoss(nn.Module):
    """
    Weighted mean-squared error across all target dimensions.

    Parameters
    ----------
    weights : optional (3,) tensor of per-output weights
              Defaults to [1, 1, 1] (equal weighting).
              Example: [1.0, 5.0, 5.0] to up-weight stresses vs deflection
              when operating in un-normalised space.
    reduction : 'mean' or 'sum'
    """

    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        if weights is None:
            weights = torch.ones(3)
        self.register_buffer('weights', weights.float())
        assert reduction in ('mean', 'sum')
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : (N, 3) predictions
        target : (N, 3) ground-truth values

        Returns
        -------
        scalar loss
        """
        # Per-output squared error
        sq_err = (pred - target) ** 2  # (N, 3)
        # Weight by output importance
        weighted = sq_err * self.weights.unsqueeze(0)  # (N, 3)

        if self.reduction == 'mean':
            return weighted.mean()
        return weighted.sum()
