#!/usr/bin/env python3
"""
sim_sensitivity_analysis.py
===========================
Perform a One-Factor-At-A-Time (OFAT) sensitivity analysis on the Similarity models.
Generates plots showing the effect of sweeping parameters (E, thickness, k_winkler, delta_T)
on the maximum predicted deflection.

Usage:
    python scripts/sim_sensitivity_analysis.py
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from parse_in2 import parse_in2
from parse_ou2 import parse_ou2
from similarity_transform import compute_similarity_params
from dataset.sim_graph import build_sim_graph
from dataset.sim_dataset import load_norm_stats
from models.sim_gnn import build_sim_gnn
from models.graph_transformer import build_sim_graph_transformer
from models.deeponet import build_sim_deeponet
from evaluation.sim_metrics import predict_case

# ── Configuration ─────────────────────────────────────────────────────────
RUNS = {
    'SimGNN v1':                 ('runs/sim_gnn_v1',              'sim_gnn'),
    'SimGNN Deep Augmented':     ('runs/sim_gnn_deep_augmented',  'sim_gnn'),
    'Sim Transformer v1':        ('runs/graph_transformer_v1',    'graph_transformer'),
    'Sim Transformer v2':        ('runs/graph_transformer_v2',    'graph_transformer'),
    'Sim Transformer v3':        ('runs/graph_transformer_v3',    'graph_transformer'),
    'Sim DeepONet v1':           ('runs/sim_deeponet_v1',         'sim_deeponet'),
}

# Update DATA_ROOT based on where your test cases are located
# Defaulted to a typical HPC location or a local fallback.
DATA_ROOT  = os.environ.get("DATA_ROOT", '/scratch/$USER/generated_cases_20k')

# Define parameter sweeps
# Format: param_name: (sweep_values, label)
SWEEPS = {
    'E':         (np.linspace(27600, 41400, 200), 'Elastic Modulus E (MPa)'),
    'thickness': (np.linspace(150, 380, 200),     'Thickness (mm)'),
    'k_winkler': (np.linspace(0.027, 0.3, 200),    'Winkler k (MPa/mm)'),
    'delta_T':   (np.linspace(0, 50, 200),        'Delta T (°C)'),
}

def load_model(run_dir: str, model_type: str, device: str):
    """Load model from best checkpoint."""
    ckpt_path = Path(run_dir) / 'best.pt'
    if not ckpt_path.exists():
        print(f"  [WARN] No checkpoint at {ckpt_path}, skipping.")
        return None, None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt.get('cfg', {})
    model_cfg = cfg.get('model', {})
    
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    # Auto-infer architectural parameters from state_dict to ensure load succeeds
    layer_idx = [int(k.split('.')[1]) for k in state.keys() if k.startswith('layers.')]
    if layer_idx: model_cfg['num_layers'] = max(layer_idx) + 1
    branch_idx = [int(k.split('.')[2]) for k in state.keys() if k.startswith('branch.layers.')]
    if branch_idx: model_cfg['num_layers'] = max(branch_idx) + 1
    trunk_idx = [int(k.split('.')[2]) for k in state.keys() if k.startswith('trunk.res_blocks.')]
    if trunk_idx: model_cfg['trunk_layers'] = max(trunk_idx) + 1
    
    if 'node_encoder.net.0.weight' in state:
        model_cfg['node_in_dim'] = state['node_encoder.net.0.weight'].shape[1]
        model_cfg['hidden_dim'] = state['node_encoder.net.0.weight'].shape[0]
    elif 'branch.node_encoder.net.0.weight' in state:
        model_cfg['node_in_dim'] = state['branch.node_encoder.net.0.weight'].shape[1]
        model_cfg['hidden_dim'] = state['branch.node_encoder.net.0.weight'].shape[0]

    if 'edge_encoder.net.0.weight' in state:
        # Some models use edge_hidden, some use edge_dim in kwargs
        model_cfg['edge_hidden'] = state['edge_encoder.net.0.weight'].shape[0]
        model_cfg['edge_dim'] = state['edge_encoder.net.0.weight'].shape[0]
    elif 'branch.edge_encoder.net.0.weight' in state:
        model_cfg['edge_hidden'] = state['branch.edge_encoder.net.0.weight'].shape[0]
        model_cfg['edge_dim'] = state['branch.edge_encoder.net.0.weight'].shape[0]

    if 'trunk.input_proj.0.weight' in state:
        model_cfg['trunk_hidden'] = state['trunk.input_proj.0.weight'].shape[0]
    if 'trunk.out_head.weight' in state:
        # Assuming out_dim=3. If p*out_dim is shape, p = shape[0] // 3
        model_cfg['p'] = state['trunk.out_head.weight'].shape[0] // model_cfg.get('out_dim', 3)
    
    if model_type == 'sim_gnn':
        model = build_sim_gnn(model_cfg)
    elif model_type == 'graph_transformer':
        model = build_sim_graph_transformer(model_cfg)
    elif model_type == 'sim_deeponet':
        model = build_sim_deeponet(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, cfg

def evaluate_sweep(model, inp_base, out_base, param_name, sweep_values, norm_stats, device):
    results = []
    
    # Extract norm stats for global features
    gf_mean = norm_stats["gf_mean"].unsqueeze(0).to(device)  # (1, 3)
    gf_std  = norm_stats["gf_std"].unsqueeze(0).to(device)   # (1, 3)
    
    for val in sweep_values:
        # Clone input dictionary
        inp = inp_base.copy()
        
        # Override the parameter being swept
        if param_name == 'E':
            inp['E'] = float(val)
        elif param_name == 'thickness':
            inp['thickness'] = float(val)
        elif param_name == 'k_winkler':
            inp['k_winkler'] = float(val)
        elif param_name == 'delta_T':
            inp['delta_T'] = float(val)
            
        # Recompute similarity parameters
        sim = compute_similarity_params(
            E=inp["E"], h=inp["thickness"], k_file=inp["k_winkler"],
            mu=inp["nu"], alpha=inp["alpha_T"], gamma_file=inp["gamma"],
            delta_T=inp["delta_T"], pressure_q=inp["pressure_q"],
        )
        
        # Build graph with new similarity parameters
        # include_diagonals=True is standard for sim models
        g = build_sim_graph(inp, out_base, sim, include_diagonals=True)
        
        # Normalize global features
        g.global_feat = g.global_feat.to(device)
        g.global_feat = (g.global_feat - gf_mean) / gf_std
        
        with torch.no_grad():
            pred = predict_case(model, g, norm_stats, device)
            # Compute max absolute deflection
            # predict_case denormalizes output, so w is in physical units (mm)
            max_w = np.max(np.abs(pred['w']))
            results.append(max_w)
            
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── Load a base case ───────────────────────────────────────────────────
    data_root = Path(os.path.expandvars(str(DATA_ROOT)))
    
    case_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and p.name.lower().startswith("case_")]) if data_root.exists() else []
    
    if not case_dirs:
        print(f"No cases found in {data_root}. Trying fallbacks...")
        fallbacks = [Path('generated_cases'), Path('../generated_cases')]
        for fallback in fallbacks:
            if fallback.exists():
                print(f"Trying fallback: {fallback}")
                case_dirs = sorted([p for p in fallback.iterdir() if p.is_dir() and p.name.lower().startswith("case_")])
                if case_dirs:
                    data_root = fallback
                    break
                    
        if not case_dirs:
            print("Could not find any cases to run sensitivity analysis. Please set the correct DATA_ROOT.")
            return
        
    base_case_dir = case_dirs[0]
    print(f"Using base case: {base_case_dir}")
    
    inp_base = parse_in2(base_case_dir / "input.in2")
    out_base = parse_ou2(base_case_dir / "input.ou2")

    # ── Collect results ────────────────────────────────────────────────────
    # structure: all_results[param_name][model_name] = [max_w_val1, max_w_val2, ...]
    all_results = {param: {} for param in SWEEPS.keys()}
    
    for model_name, (run_dir, mtype) in RUNS.items():
        print(f"\nEvaluating {model_name}...")
        
        # Expand user/vars in run_dir just in case
        run_dir_path = os.path.expandvars(run_dir)
        model, cfg = load_model(run_dir_path, mtype, device)
        if model is None:
            continue
            
        norm_path = Path(run_dir_path) / 'norm_stats.pt'
        if not norm_path.exists():
            print(f"  [WARN] No norm_stats.pt found at {norm_path}, skipping.")
            continue
            
        norm_stats = load_norm_stats(norm_path)
            
        for param_name, (sweep_vals, _) in SWEEPS.items():
            max_w_list = evaluate_sweep(model, inp_base, out_base, param_name, sweep_vals, norm_stats, device)
            all_results[param_name][model_name] = max_w_list

    # ── Plotting ───────────────────────────────────────────────────────────
    Path('runs').mkdir(exist_ok=True)
    
    for model_name in RUNS.keys():
        # Check if this model was evaluated for any parameter
        if not any(model_name in all_results[p] for p in SWEEPS.keys()):
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (param_name, (sweep_vals, xlabel)) in enumerate(SWEEPS.items()):
            ax = axes[idx]
            
            if model_name in all_results[param_name]:
                y_vals = all_results[param_name][model_name]
                ax.plot(sweep_vals, y_vals, color='tab:blue', linewidth=2, label=model_name)
                
            ax.set_title(f"Sensitivity to {param_name}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Max Absolute Deflection |w| (mm)")
            ax.grid(True, linestyle='--', alpha=0.7)
            
        fig.suptitle(f"Similarity Model Sensitivity Analysis: {model_name}", fontsize=16)
        plt.tight_layout()
        
        safe_name = model_name.replace(' ', '_').lower()
        out_path = Path('runs') / f'sim_sensitivity_{safe_name}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved sensitivity plot for {model_name} to {out_path}")

if __name__ == '__main__':
    main()
