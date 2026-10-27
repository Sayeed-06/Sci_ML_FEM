"""
normalize.py
============
Compute and apply z-score normalisation statistics for the FEM dataset.

Statistics are computed on the TRAINING set only, then applied to val/test.

Usage
-----
    stats = compute_norm_stats(train_data_list)
    train_data_list = apply_norm(train_data_list, stats)
    val_data_list   = apply_norm(val_data_list, stats)
    save_norm_stats(stats, 'configs/norm_stats.pt')
    # later:
    stats = load_norm_stats('configs/norm_stats.pt')
"""

from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from torch_geometric.data import Data


# Which node feature columns to normalise
# (cols 0,1 are already in [0,1]; cols 5,7 are binary flags)
NODE_COLS_TO_NORM = [2, 3, 4, 6, 8, 9, 10, 11, 12]  # + alpha_T, pressure_q


def compute_norm_stats(
    data_list: List[Data],
) -> Dict[str, torch.Tensor]:
    """
    Compute per-feature mean and std from a list of PyG Data objects.

    Returns
    -------
    dict with keys:
        node_feat_mean  : (9,) tensor
        node_feat_std   : (9,) tensor
        target_mean     : (3,) tensor  — [w, σ_x, σ_y]
        target_std      : (3,) tensor
    """
    # Stack all node features across all graphs
    all_x = torch.cat([d.x for d in data_list], dim=0)  # (total_nodes, 9)
    all_y = torch.cat([d.y for d in data_list], dim=0)  # (total_nodes, 3)

    feat_mean = all_x.mean(dim=0)  # (9,)
    feat_std  = all_x.std(dim=0)   # (9,)
    # Avoid division by zero for constant columns
    feat_std  = feat_std.clamp(min=1e-8)

    tgt_mean = all_y.mean(dim=0)   # (3,)
    tgt_std  = all_y.std(dim=0)
    tgt_std  = tgt_std.clamp(min=1e-8)

    return {
        'node_feat_mean': feat_mean,
        'node_feat_std':  feat_std,
        'target_mean':    tgt_mean,
        'target_std':     tgt_std,
    }


def apply_norm(
    data_list: List[Data],
    stats: Dict[str, torch.Tensor],
    normalise_targets: bool = True,
) -> List[Data]:
    """
    Apply z-score normalisation in-place to each graph in data_list.

    Only normalises the continuous node features (NODE_COLS_TO_NORM).
    Spatial coordinates (already in [0,1]) and binary flags are kept as-is.
    """
    feat_mean = stats['node_feat_mean']  # (9,)
    feat_std  = stats['node_feat_std']   # (9,)
    tgt_mean  = stats['target_mean']     # (3,)
    tgt_std   = stats['target_std']      # (3,)

    for data in data_list:
        x = data.x.clone()
        # Ensure we only normalize columns that exist in the loaded stats and node features
        valid_cols = [c for c in NODE_COLS_TO_NORM if c < len(feat_mean) and c < x.shape[1]]
        for c in valid_cols:
            x[:, c] = (x[:, c] - feat_mean[c]) / feat_std[c]
        data.x = x

        if normalise_targets:
            data.y = (data.y - tgt_mean) / tgt_std

    return data_list


def denorm_targets(
    y_norm: torch.Tensor,
    stats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Inverse transform normalised targets back to physical units."""
    return y_norm * stats['target_std'] + stats['target_mean']


def save_norm_stats(
    stats: Dict[str, torch.Tensor],
    path: str | Path,
) -> None:
    torch.save(stats, path)


def load_norm_stats(path: str | Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, weights_only=True)
