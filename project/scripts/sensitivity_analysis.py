#!/usr/bin/env python3
"""
sensitivity_analysis.py
=======================
Perform a One-Factor-At-A-Time (OFAT) sensitivity analysis on the surrogate models.
Generates plots showing the effect of sweeping parameters (E, thickness, k_winkler, delta_T)
on the maximum predicted deflection.

Usage:
    python scripts/sensitivity_analysis.py
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.fem_dataset import FEMDataset
from dataset.normalize import apply_norm, load_norm_stats, denorm_targets
from models.gnn import build_gnn
from models.graph_transformer import build_graph_transformer
from models.deeponet import build_deeponet

# ── Configuration ─────────────────────────────────────────────────────────
RUNS = {
    'GNN MSE':          ('runs/gnn_mse',          'gnn'),
    'GNN PINN':         ('runs/gnn_pinn',         'gnn'),
    'Transformer MSE':  ('runs/transformer_mse',  'transformer'),
    'Transformer PINN': ('runs/transformer_pinn', 'transformer'),
    'DeepONet MSE':     ('runs/deeponet_mse',     'deeponet'),
    'DeepONet PINN':    ('runs/deeponet_pinn',    'deeponet'),
}

DATA_ROOT  = 'generated_cases'

# Define parameter sweeps
# Format: param_name: (col_idx, sweep_values, label)
SWEEPS = {
    'E':         (3, np.linspace(27600, 41400, 200), 'Elastic Modulus E (MPa)'),
    'thickness': (2, np.linspace(150, 380, 200),     'Thickness (mm)'),
    'k_winkler': (4, np.linspace(0.027, 0.3, 200),    'Winkler k (MPa/mm)'),
    'delta_T':   (6, np.linspace(0, 50, 200),        'Delta T (°C)'),
}

def load_model(run_dir: str, model_type: str, device: str):
    """Load model from best checkpoint."""
    ckpt_path = Path(run_dir) / 'best.pt'
    if not ckpt_path.exists():
        print(f"  [WARN] No checkpoint at {ckpt_path}, skipping.")
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt['cfg']
    
    model_cfg = cfg.get('model', {})
    model_state = ckpt['model_state']
    
    # Auto-infer architectural parameters from state_dict to ensure load succeeds
    layer_idx = [int(k.split('.')[1]) for k in model_state.keys() if k.startswith('layers.')]
    if layer_idx: model_cfg['num_layers'] = max(layer_idx) + 1
    branch_idx = [int(k.split('.')[2]) for k in model_state.keys() if k.startswith('branch.layers.')]
    if branch_idx: model_cfg['num_layers'] = max(branch_idx) + 1
    trunk_idx = [int(k.split('.')[2]) for k in model_state.keys() if k.startswith('trunk.res_blocks.')]
    if trunk_idx: model_cfg['trunk_layers'] = max(trunk_idx) + 1
    
    if 'node_encoder.net.0.weight' in model_state:
        model_cfg['node_in_dim'] = model_state['node_encoder.net.0.weight'].shape[1]
        model_cfg['hidden_dim'] = model_state['node_encoder.net.0.weight'].shape[0]
    elif 'branch.node_encoder.net.0.weight' in model_state:
        model_cfg['node_in_dim'] = model_state['branch.node_encoder.net.0.weight'].shape[1]
        model_cfg['hidden_dim'] = model_state['branch.node_encoder.net.0.weight'].shape[0]

    if 'edge_encoder.net.0.weight' in model_state:
        model_cfg['edge_hidden'] = model_state['edge_encoder.net.0.weight'].shape[0]
    elif 'branch.edge_encoder.net.0.weight' in model_state:
        model_cfg['edge_hidden'] = model_state['branch.edge_encoder.net.0.weight'].shape[0]

    if 'trunk.input_proj.0.weight' in model_state:
        model_cfg['trunk_hidden'] = model_state['trunk.input_proj.0.weight'].shape[0]
    if 'trunk.out_head.weight' in model_state:
        model_cfg['p'] = model_state['trunk.out_head.weight'].shape[0] // model_cfg.get('out_dim', 3)

    if model_type == 'gnn':
        model = build_gnn(model_cfg)
    elif model_type == 'transformer':
        model = build_graph_transformer(model_cfg)
    elif model_type == 'deeponet':
        model = build_deeponet(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model, cfg

def evaluate_sweep(model, base_data, param_col, sweep_values, norm_stats, device):
    results = []
    for val in sweep_values:
        # Clone base graph
        data = base_data.clone()
        
        # Modify the parameter across all nodes (unnormalized scale)
        data.x[:, param_col] = float(val)
        
        # Apply normalization to match model's expected input
        if norm_stats is not None:
            # apply_norm modifies data.x and (optionally) data.y in-place
            _ = apply_norm([data], norm_stats, normalise_targets=False)
            
        data = data.to(device)
        
        # Dynamically truncate node features to match the model's expected input dimension
        # (Needed because the dataset might now generate 11 features but older models expect 9)
        if hasattr(model, 'node_encoder'):
            expected_dim = model.node_encoder.net[0].in_features
        elif hasattr(model, 'branch') and hasattr(model.branch, 'node_encoder'):
            expected_dim = model.branch.node_encoder.net[0].in_features
        else:
            expected_dim = data.x.shape[1]
            
        if data.x.shape[1] > expected_dim:
            data.x = data.x[:, :expected_dim]
        
        with torch.no_grad():
            pred = model(data)
            
            # Denormalize output
            if norm_stats is not None:
                pred = denorm_targets(pred, norm_stats)
                
            # Compute max absolute deflection (w is at index 0)
            max_w = pred[:, 0].abs().max().item()
            results.append(max_w)
            
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load dataset ───────────────────────────────────────────────────────
    dataset = FEMDataset(root=DATA_ROOT)
    _, _, test_ds = dataset.split(seed=42)
    
    if len(test_ds) == 0:
        print("Test dataset is empty! Cannot perform sensitivity analysis.")
        return
        
    base_data = test_ds[0]
    print(f"Using base graph from test dataset with {base_data.num_nodes} nodes.")

    # ── Collect results ────────────────────────────────────────────────────
    # structure: all_results[param_name][model_name] = [max_w_val1, max_w_val2, ...]
    all_results = {param: {} for param in SWEEPS.keys()}
    
    for model_name, (run_dir, mtype) in RUNS.items():
        print(f"\nEvaluating {model_name}...")
        model, cfg = load_model(run_dir, mtype, device)
        if model is None:
            continue
            
        norm_path = Path(run_dir) / 'norm_stats.pt'
        norm_stats = None
        if norm_path.exists():
            norm_stats = load_norm_stats(norm_path)
            
        for param_name, (col_idx, sweep_vals, _) in SWEEPS.items():
            max_w_list = evaluate_sweep(model, base_data, col_idx, sweep_vals, norm_stats, device)
            all_results[param_name][model_name] = max_w_list

    # ── Plotting ───────────────────────────────────────────────────────────
    Path('runs').mkdir(exist_ok=True)
    
    for model_name in RUNS.keys():
        # Check if this model was evaluated for any parameter
        if not any(model_name in all_results[p] for p in SWEEPS.keys()):
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (param_name, (col_idx, sweep_vals, xlabel)) in enumerate(SWEEPS.items()):
            ax = axes[idx]
            
            if model_name in all_results[param_name]:
                y_vals = all_results[param_name][model_name]
                ax.plot(sweep_vals, y_vals, color='tab:blue', linewidth=2, label=model_name)
                
            ax.set_title(f"Sensitivity to {param_name}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Max Absolute Deflection |w| (mm)")
            ax.grid(True, linestyle='--', alpha=0.7)
            
        fig.suptitle(f"Sensitivity Analysis: {model_name}", fontsize=16)
        plt.tight_layout()
        
        safe_name = model_name.replace(' ', '_').lower()
        out_path = Path('runs') / f'sensitivity_{safe_name}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved sensitivity plot for {model_name} to {out_path}")

if __name__ == '__main__':
    main()
