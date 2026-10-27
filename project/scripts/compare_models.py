#!/usr/bin/env python3
"""
compare_models.py
=================
Load all four trained model checkpoints and produce a comparison table.

Usage
-----
    python scripts/compare_models.py

Assumes all four runs have been completed and checkpoints exist in:
    runs/gnn_mse/best.pt
    runs/gnn_pinn/best.pt
    runs/transformer_mse/best.pt
    runs/transformer_pinn/best.pt
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.loader import DataLoader

from dataset.fem_dataset import FEMDataset
from dataset.normalize import apply_norm, load_norm_stats
from models.gnn import build_gnn
from models.graph_transformer import build_graph_transformer
from evaluation.metrics import evaluate_dataset, print_metrics_table


# ── Configuration ─────────────────────────────────────────────────────────

RUNS = {
    'GNN + MSE':         ('runs/gnn_mse',         'gnn'),
    'GNN + PINN':        ('runs/gnn_pinn',         'gnn'),
    'Transformer + MSE': ('runs/transformer_mse',  'transformer'),
    'Transformer + PINN':('runs/transformer_pinn', 'transformer'),
}

DATA_ROOT  = 'generated_cases'
BATCH_SIZE = 4


def load_model(run_dir: str, model_type: str, device: str):
    """Load model from best checkpoint."""
    ckpt_path = Path(run_dir) / 'best.pt'
    if not ckpt_path.exists():
        print(f"  [WARN] No checkpoint at {ckpt_path}, skipping.")
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt['cfg']

    if model_type == 'gnn':
        model = build_gnn(cfg.get('model', {}))
    else:
        model = build_graph_transformer(cfg.get('model', {}))

    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, cfg


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load dataset ───────────────────────────────────────────────────────
    dataset = FEMDataset(root=DATA_ROOT)
    _, _, test_ds = dataset.split(seed=42)
    print(f"Test cases: {len(test_ds)}")

    all_results: dict = {}

    for model_name, (run_dir, mtype) in RUNS.items():
        print(f"\n── {model_name} ────────────────────────────────")

        model, cfg = load_model(run_dir, mtype, device)
        if model is None:
            continue

        # Load norm stats for this run
        norm_path = Path(run_dir) / 'norm_stats.pt'
        norm_stats = None
        if norm_path.exists():
            norm_stats = load_norm_stats(norm_path)
            test_list = apply_norm(list(test_ds), norm_stats, normalise_targets=True)
        else:
            test_list = list(test_ds)

        test_loader = DataLoader(test_list, batch_size=BATCH_SIZE, shuffle=False)

        metrics, avg_loss = evaluate_dataset(
            model, test_loader,
            device=device,
            norm_stats=norm_stats,
        )
        metrics['test_loss'] = avg_loss
        all_results[model_name] = metrics

        print(f"  RelErr_mean: {metrics['RelErr_mean']:.4f}")
        print(f"  RMSE_w:     {metrics['RMSE_w']:.4f}")
        print(f"  RMSE_σx:    {metrics['RMSE_sigma_x']:.4f}")
        print(f"  RMSE_σy:    {metrics['RMSE_sigma_y']:.4f}")

    # ── Print comparison table ─────────────────────────────────────────────
    if all_results:
        print_metrics_table(all_results)

    # ── Save results ───────────────────────────────────────────────────────
    Path('runs').mkdir(exist_ok=True)
    torch.save(all_results, 'runs/comparison_results.pt')
    print("Saved to runs/comparison_results.pt")


if __name__ == '__main__':
    main()
