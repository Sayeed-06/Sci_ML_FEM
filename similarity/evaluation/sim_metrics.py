"""
sim_metrics.py
==============
Evaluation metrics in absolute physical units for the Similarity-Based GNN.

All metrics are reported in natural problem units:
  w       -> mm   (deflection)
  sigma_x -> MPa  (normal stress in X)
  sigma_y -> MPa  (normal stress in Y)

Metrics: RMSE, MAE, MaxAE, R2, NRMSE%
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict
from torch_geometric.loader import DataLoader


def rmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    return torch.sqrt(((pred - true) ** 2).mean()).item()


def mae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (pred - true).abs().mean().item()


def max_ae(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (pred - true).abs().max().item()


def r2(pred: torch.Tensor, true: torch.Tensor) -> float:
    ss_res = ((pred - true) ** 2).sum().item()
    ss_tot = ((true - true.mean()) ** 2).sum().item()
    return 1.0 - ss_res / (ss_tot + 1e-12)


def nrmse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """RMSE normalised by range of true values (percentage)."""
    rng = (true.max() - true.min()).item()
    return rmse(pred, true) / (rng + 1e-12) * 100.0


def evaluate(model, loader: DataLoader, norm_stats: Dict,
             device: str = "cpu") -> Dict[str, float]:
    """Evaluate model; return metrics in absolute physical units.

    Returns keys: rmse_w, mae_w, maxae_w, r2_w, nrmse_w (and sx, sy variants)
    """
    model.eval()
    dev    = torch.device(device)
    y_mean = norm_stats["y_mean"].to(dev)
    y_std  = norm_stats["y_std"].to(dev)
    all_pred, all_true = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(dev)
            pred_abs = model(data) * y_std + y_mean
            true_abs = data.y     * y_std + y_mean
            all_pred.append(pred_abs.cpu())
            all_true.append(true_abs.cpu())

    P = torch.cat(all_pred, 0)
    T = torch.cat(all_true, 0)

    out = {}
    for i, nm in enumerate(["w", "sx", "sy"]):
        p, t = P[:, i], T[:, i]
        out[f"rmse_{nm}"]  = rmse(p, t)
        out[f"mae_{nm}"]   = mae(p, t)
        out[f"maxae_{nm}"] = max_ae(p, t)
        out[f"r2_{nm}"]    = r2(p, t)
        out[f"nrmse_{nm}"] = nrmse(p, t)
    return out


def print_metrics_table(metrics: Dict[str, float],
                        title: str = "Metrics (absolute units)") -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    hdr = f"{'Field':<12} {'RMSE':>10} {'MAE':>10} {'Max AE':>10} {'R2':>8} {'NRMSE%':>8}"
    print(hdr)
    print("-" * 62)
    rows = [
        ("w (mm)",    "w"),
        ("sx (MPa)",  "sx"),
        ("sy (MPa)",  "sy"),
    ]
    for label, nm in rows:
        print(
            f"{label:<12} "
            f"{metrics[f'rmse_{nm}']:>10.5f} "
            f"{metrics[f'mae_{nm}']:>10.5f} "
            f"{metrics[f'maxae_{nm}']:>10.5f} "
            f"{metrics[f'r2_{nm}']:>8.5f} "
            f"{metrics[f'nrmse_{nm}']:>8.3f}"
        )
    print(f"{'='*60}\n")


def predict_case(model, data, norm_stats: Dict,
                 device: str = "cpu") -> Dict[str, np.ndarray]:
    """Inference on one Data object; returns absolute-unit numpy arrays.

    Returns
    -------
    dict: pos (N,2) mm, w (N,) mm, sigma_x (N,) MPa, sigma_y (N,) MPa
    """
    model.eval()
    dev    = torch.device(device)
    y_mean = norm_stats["y_mean"].to(dev)
    y_std  = norm_stats["y_std"].to(dev)
    data   = data.to(dev)

    if not hasattr(data, "batch") or data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=dev)

    with torch.no_grad():
        pred_abs = (model(data) * y_std + y_mean).cpu().numpy()

    return {
        "pos":     data.pos.cpu().numpy(),
        "w":       pred_abs[:, 0],
        "sigma_x": pred_abs[:, 1],
        "sigma_y": pred_abs[:, 2],
        "l":       getattr(data, "l", None),
        "phi":     getattr(data, "phi", None),
        "PQ":      getattr(data, "PQ", None),
    }
