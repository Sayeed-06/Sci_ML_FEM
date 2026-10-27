"""
metrics.py
==========
Evaluation metrics for FEM surrogate model comparison.

Computed per-output (w, σ_x, σ_y) and globally (mean over outputs).

Metrics
-------
MAE          : Mean Absolute Error  = (1/N) Σ |ŷ - y|
RMSE         : Root Mean Squared Error = √((1/N) Σ (ŷ-y)²)
RelErr       : Relative L2 Error = ‖ŷ - y‖₂ / (‖y‖₂ + ε)
MaxNodeErr   : Maximum absolute error across all nodes and outputs
R2           : Coefficient of determination = 1 - SS_res/SS_tot
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


OUTPUT_NAMES = ['w', 'sigma_x', 'sigma_y']


@torch.no_grad()
def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    norm_stats: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Compute all metrics on a batch of predictions.

    Parameters
    ----------
    pred       : (N, 3) predicted values (may be normalised)
    target     : (N, 3) ground truth (same scale as pred)
    norm_stats : if provided, de-normalise before computing metrics
                 so reported values are in physical units

    Returns
    -------
    dict of metric_name → float
    Example keys: 'MAE_w', 'RMSE_w', 'RelErr_w', 'R2_w', ...
                  'MAE_mean', 'RMSE_mean', 'RelErr_mean', 'MaxNodeErr'
    """
    if norm_stats is not None:
        tgt_mean = norm_stats['target_mean'].to(pred.device)
        tgt_std  = norm_stats['target_std'].to(pred.device)
        pred   = pred   * tgt_std + tgt_mean
        target = target * tgt_std + tgt_mean

    results: Dict[str, float] = {}

    mae_list, rmse_list, rel_list, r2_list = [], [], [], []

    for k, name in enumerate(OUTPUT_NAMES):
        p = pred[:, k]
        t = target[:, k]

        mae  = (p - t).abs().mean().item()
        rmse = ((p - t) ** 2).mean().sqrt().item()
        rel  = ((p - t).norm() / (t.norm() + 1e-10)).item()

        ss_res = ((t - p) ** 2).sum().item()
        ss_tot = ((t - t.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)

        results[f'MAE_{name}']    = mae
        results[f'RMSE_{name}']   = rmse
        results[f'RelErr_{name}'] = rel
        results[f'R2_{name}']     = r2

        mae_list.append(mae)
        rmse_list.append(rmse)
        rel_list.append(rel)
        r2_list.append(r2)

    results['MAE_mean']    = np.mean(mae_list)
    results['RMSE_mean']   = np.mean(rmse_list)
    results['RelErr_mean'] = np.mean(rel_list)
    results['R2_mean']     = np.mean(r2_list)
    results['MaxNodeErr']  = (pred - target).abs().max().item()

    return results


@torch.no_grad()
def evaluate_dataset(
    model,
    loader,
    device: str = 'cpu',
    norm_stats: Optional[Dict] = None,
    loss_fn=None,
) -> Tuple[Dict[str, float], float]:
    """
    Run model on all batches in loader and aggregate metrics.

    Returns
    -------
    (metrics_dict, avg_loss)
    """
    model.eval()
    model.to(device)

    all_pred   = []
    all_target = []
    total_loss = 0.0
    n_batches  = 0

    for data in loader:
        data = data.to(device)
        pred = model(data)
        all_pred.append(pred.cpu())
        all_target.append(data.y.cpu())

        if loss_fn is not None:
            try:
                loss = loss_fn(pred, data.y, data)
            except TypeError:
                loss = loss_fn(pred, data.y)
                
            if isinstance(loss, dict):
                loss = loss['total']
            total_loss += loss.item()
            n_batches  += 1

    all_pred   = torch.cat(all_pred,   dim=0)
    all_target = torch.cat(all_target, dim=0)

    metrics = compute_metrics(all_pred, all_target, norm_stats)
    avg_loss = total_loss / max(n_batches, 1)

    return metrics, avg_loss


def print_metrics_table(
    results: Dict[str, Dict[str, float]],
    model_names: Optional[List[str]] = None,
) -> None:
    """
    Print a formatted comparison table.

    Parameters
    ----------
    results    : {model_name: metrics_dict}
    model_names: ordered list of model names to display
    """
    if model_names is None:
        model_names = list(results.keys())

    header_metrics = [
        'MAE_w', 'RMSE_w', 'RelErr_w',
        'MAE_sigma_x', 'RMSE_sigma_x', 'RelErr_sigma_x',
        'MAE_sigma_y', 'RMSE_sigma_y', 'RelErr_sigma_y',
        'RelErr_mean', 'MaxNodeErr', 'R2_mean',
    ]

    col_w = max(len(m) for m in model_names) + 2
    hdr_w = max(len(m) for m in header_metrics) + 2

    # Header row
    header = f"{'Metric':<{hdr_w}}" + "".join(f"{n:>{col_w}}" for n in model_names)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for metric in header_metrics:
        row = f"{metric:<{hdr_w}}"
        for mn in model_names:
            val = results.get(mn, {}).get(metric, float('nan'))
            row += f"{val:>{col_w}.6f}"
        print(row)

    print("=" * len(header) + "\n")
