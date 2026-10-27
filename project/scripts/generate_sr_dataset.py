#!/usr/bin/env python3
"""
generate_sr_dataset.py
======================
Generates a synthetic dataset of low-resolution (LR) and high-resolution (HR)
meshes/graphs for training a Super-Resolution model.

It uses the DeepONet PINN to predict the output variables (w, sigma_x, sigma_y)
for various sampled input parameters at both LR and HR grids.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.fem_dataset import FEMDataset
from dataset.build_graph import build_graph
from dataset.normalize import apply_norm, load_norm_stats, denorm_targets
from scripts.sensitivity_analysis import load_model, SWEEPS

def generate_mock_inp_out(nx, ny, base_data, E, thickness, k_winkler, delta_T):
    # Extract domain size from base_data (node_feat[9] is Lx, node_feat[10] is Ly)
    Lx = float(base_data.x[0, 9].item())
    Ly = float(base_data.x[0, 10].item())
    nu = float(base_data.nu[0].item())
    pressure = float(base_data.pressure[0].item())
    alpha_T = float(base_data.alpha_T[0].item())
    gamma = float(base_data.gamma[0].item())
    
    # load rect
    lx1 = float(base_data.load_rect[0].item())
    lx2 = float(base_data.load_rect[1].item())
    ly1 = float(base_data.load_rect[2].item())
    ly2 = float(base_data.load_rect[3].item())

    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)
    N = nx * ny
    D = (E * thickness**3) / (12 * (1 - nu**2))
    
    inp = {
        'n_nodes_x': nx,
        'n_nodes_y': ny,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'load_x1': lx1, 'load_x2': lx2,
        'load_y1': ly1, 'load_y2': ly2,
        'E': E,
        'thickness': thickness,
        'k_winkler': k_winkler,
        'delta_T': delta_T,
        'D': D,
        'pressure_q': pressure,
        'nu': nu,
        'alpha_T': alpha_T,
        'gamma': gamma,
        'n_nodes': N,
        'element_type': 1
    }
    
    xx, yy = np.meshgrid(x_coords, y_coords)
    nodes = np.column_stack((xx.ravel(), yy.ravel()))
    out = {
        'node_ids': np.arange(1, N + 1),
        'nodes': nodes,
        'w': np.zeros(N),
        'sigma_x_bot': np.zeros(N),
        'sigma_y_bot': np.zeros(N)
    }
    return inp, out

def generate_sr_dataset(num_samples=100, lr_nx_range=(8, 24), lr_ny_range=(8, 24), scale_factor=4, out_dir='dataset_sr'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load base dataset to get constants
    dataset = FEMDataset(root='generated_cases')
    _, _, test_ds = dataset.split(seed=42)
    if len(test_ds) == 0:
        raise ValueError("Dataset is empty. Cannot extract base parameters.")
    base_data = test_ds[0]

    # Load DeepONet PINN model
    run_dir = 'runs/deeponet_pinn'
    print(f"Loading model from {run_dir}...")
    model, cfg = load_model(run_dir, 'deeponet', device)
    if model is None:
        raise RuntimeError("Could not load DeepONet PINN model. Please check runs/deeponet_pinn.")

    norm_path = Path(run_dir) / 'norm_stats.pt'
    norm_stats = None
    if norm_path.exists():
        norm_stats = load_norm_stats(norm_path)

    # Setup directories
    lr_dir = Path(out_dir) / 'LR'
    hr_dir = Path(out_dir) / 'HR'
    lr_dir.mkdir(parents=True, exist_ok=True)
    hr_dir.mkdir(parents=True, exist_ok=True)

    # Ranges from sensitivity analysis
    E_min, E_max = SWEEPS['E'][1][0], SWEEPS['E'][1][-1]
    t_min, t_max = SWEEPS['thickness'][1][0], SWEEPS['thickness'][1][-1]
    k_min, k_max = SWEEPS['k_winkler'][1][0], SWEEPS['k_winkler'][1][-1]
    dt_min, dt_max = SWEEPS['delta_T'][1][0], SWEEPS['delta_T'][1][-1]

    for i in tqdm(range(num_samples), desc="Generating pairs"):
        E = np.random.uniform(E_min, E_max)
        thickness = np.random.uniform(t_min, t_max)
        k_winkler = np.random.uniform(k_min, k_max)
        delta_T = np.random.uniform(dt_min, dt_max)
        
        # Randomly choose shape for this specific graph
        curr_lr_nx = np.random.randint(lr_nx_range[0], lr_nx_range[1] + 1)
        curr_lr_ny = np.random.randint(lr_ny_range[0], lr_ny_range[1] + 1)

        def get_prediction(nx, ny):
            inp, out = generate_mock_inp_out(nx, ny, base_data, E, thickness, k_winkler, delta_T)
            data = build_graph(inp, out, include_diagonals=True, normalise=False)
            
            if norm_stats is not None:
                _ = apply_norm([data], norm_stats, normalise_targets=False)
                
            data = data.to(device)
            if hasattr(model, 'branch') and hasattr(model.branch, 'node_encoder'):
                expected_dim = model.branch.node_encoder.net[0].in_features
                if data.x.shape[1] > expected_dim:
                    data.x = data.x[:, :expected_dim]
                    
            with torch.no_grad():
                pred = model(data)
                if norm_stats is not None:
                    pred = denorm_targets(pred, norm_stats)
                    
            # Shape (N, 3). Reshape to (ny, nx, 3) -> (3, ny, nx)
            pred_img = pred.view(ny, nx, 3).permute(2, 0, 1).cpu()
            return pred_img

        lr_img = get_prediction(curr_lr_nx, curr_lr_ny)
        hr_img = get_prediction(curr_lr_nx * scale_factor, curr_lr_ny * scale_factor)

        torch.save(lr_img, lr_dir / f"sample_{i:04d}.pt")
        torch.save(hr_img, hr_dir / f"sample_{i:04d}.pt")
        
    print(f"Dataset generated at '{out_dir}' with varying shapes.")

if __name__ == '__main__':
    # Default 100 samples, LR shapes between 8x8 and 24x24 (upscaled 4x)
    generate_sr_dataset(num_samples=100, lr_nx_range=(8, 24), lr_ny_range=(8, 24), scale_factor=4)
