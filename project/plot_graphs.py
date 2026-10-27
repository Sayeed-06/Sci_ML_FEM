import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the 'project' directory to path to load models and evaluation scripts
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from dataset.fem_dataset import FEMDataset
from dataset.normalize import load_norm_stats
from models.gnn import build_gnn
from models.graph_transformer import build_graph_transformer
from models.deeponet import build_deeponet

import yaml

def load_model(run_dir, device="cpu"):
    ckpt_path = Path(run_dir) / 'best.pt'
    if not ckpt_path.exists():
        print(f"Warning: Checkpoint not found at {ckpt_path}")
        return None, None
        
    run_name = Path(run_dir).name
    config_path = Path(project_dir) / "configs" / f"{run_name}.yaml"
    model_cfg = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get('model', {})
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state']
    
    # Infer core architecture params from state_dict to avoid mismatches
    if 'node_encoder.net.0.weight' in state_dict:
        model_cfg['node_in_dim'] = state_dict['node_encoder.net.0.weight'].shape[1]
        model_cfg['hidden_dim'] = state_dict['node_encoder.net.0.weight'].shape[0]
    if 'edge_encoder.net.0.weight' in state_dict:
        model_cfg['edge_in_dim'] = state_dict['edge_encoder.net.0.weight'].shape[1]
        model_cfg['edge_hidden'] = state_dict['edge_encoder.net.0.weight'].shape[0]
    
    if "gnn" in str(run_dir).lower() or "transformer" in str(run_dir).lower():
        layer_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('layers.')]
        if layer_indices:
            model_cfg['num_layers'] = max(layer_indices) + 1
        
        if "transformer" in str(run_dir).lower():
            if 'layers.0.attn.in_proj_weight' in state_dict:
                model_cfg['attn_dim'] = state_dict['layers.0.attn.in_proj_weight'].shape[1]
                model_cfg['n_heads'] = model_cfg['attn_dim'] // 32
    elif "deeponet" in str(run_dir).lower():
        # Wait, for deeponet the node encoder is in 'branch'
        pass
        
    if "deeponet" in str(run_dir).lower():
        if 'branch.node_encoder.net.0.weight' in state_dict:
            model_cfg['node_in_dim'] = state_dict['branch.node_encoder.net.0.weight'].shape[1]
            model_cfg['hidden_dim'] = state_dict['branch.node_encoder.net.0.weight'].shape[0]
        if 'branch.edge_encoder.net.0.weight' in state_dict:
            model_cfg['edge_in_dim'] = state_dict['branch.edge_encoder.net.0.weight'].shape[1]
            model_cfg['edge_hidden'] = state_dict['branch.edge_encoder.net.0.weight'].shape[0]
        layer_indices = [int(k.split('.')[2]) for k in state_dict.keys() if k.startswith('branch.layers.')]
        if layer_indices:
            model_cfg['num_layers'] = max(layer_indices) + 1
        if 'trunk.input_proj.0.weight' in state_dict:
            model_cfg['trunk_in'] = state_dict['trunk.input_proj.0.weight'].shape[1]
            model_cfg['trunk_hidden'] = state_dict['trunk.input_proj.0.weight'].shape[0]
        trunk_layers = [int(k.split('.')[2]) for k in state_dict.keys() if k.startswith('trunk.res_blocks.')]
        if trunk_layers:
            model_cfg['trunk_layers'] = max(trunk_layers) + 1
    
    if "gnn" in str(run_dir).lower():
        model = build_gnn(model_cfg)
    elif "transformer" in str(run_dir).lower():
        model = build_graph_transformer(model_cfg)
    elif "deeponet" in str(run_dir).lower():
        model = build_deeponet(model_cfg)
    else:
        return None, None
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    norm_path = Path(run_dir) / 'norm_stats.pt'
    norm_stats = load_norm_stats(norm_path) if norm_path.exists() else None
    
    return model, norm_stats

def predict_case(model, data, norm_stats, device="cpu"):
    """Run inference for a single case."""
    model.eval()
    # Normalize features
    x = data.x.clone()
    feat_mean = norm_stats['node_feat_mean'].to(device)
    feat_std = norm_stats['node_feat_std'].to(device)
    
    # Slice x to match the model's expected input dimension (older models used 9 features, newer use 11)
    expected_dim = len(feat_mean)
    x = x[:, :expected_dim]
    
    NODE_COLS_TO_NORM = [2, 3, 4, 6, 8, 9, 10]
    valid_cols = [c for c in NODE_COLS_TO_NORM if c < expected_dim]
    for c in valid_cols:
        x[:, c] = (x[:, c] - feat_mean[c]) / (feat_std[c] + 1e-8)
    
    g = data.clone()
    g.x = x
    # we need batch size of 1 for deeponet and transformers to work properly if they rely on batch
    if not hasattr(g, 'batch') or g.batch is None:
        g.batch = torch.zeros(g.x.shape[0], dtype=torch.long)
        
    g = g.to(device)
    
    with torch.no_grad():
        pred = model(g)
        
    # De-normalize targets
    tgt_mean = norm_stats['target_mean'].to(device)
    tgt_std = norm_stats['target_std'].to(device)
    pred = pred * tgt_std + tgt_mean
    
    # Target 0 is w
    return {"w": pred[:, 0].cpu().numpy()}

TARGET_CASE = "case_00001_EValue28229_Alpha7.93e-06_Kval0.174_pressure0.80"

def plot_first_case(dataset_root, checkpoint_dirs=None):
    print("Loading dataset...")
    dataset = FEMDataset(root=dataset_root)

    data = None
    for i in range(len(dataset)):
        d = dataset[i]
        if getattr(d, 'case_name', None) == TARGET_CASE:
            data = d
            break

    if data is None:
        print(f"Case {TARGET_CASE} not found in dataset.")
        return

    print(f"Plotting case {TARGET_CASE}...")

    models = []
    if checkpoint_dirs:
        for cdir in checkpoint_dirs:
            ckpt_path = os.path.join(cdir, "best.pt")
            norm_path = os.path.join(cdir, "norm_stats.pt")
            if os.path.exists(ckpt_path) and os.path.exists(norm_path):
                print(f"Loading eval model from {cdir}...")
                m, ns = load_model(cdir, device="cpu")
                models.append((Path(cdir).name, m, ns))
            else:
                print(f"Checkpoint or norm_stats not found in {cdir}")
                models.append((Path(cdir).name, None, None))

    nx_val = int(data.nx)
    ny_val = int(data.ny)
    
    if hasattr(data, 'pos'):
        x_coords = data.pos[:, 0].numpy().reshape((ny_val, nx_val))[0, :]
        y_coords = data.pos[:, 1].numpy().reshape((ny_val, nx_val))[:, 0]
        dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
        dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0
        extent = [x_coords[0] - dx/2, x_coords[-1] + dx/2, y_coords[0] - dy/2, y_coords[-1] + dy/2]
    else:
        extent = [0, nx_val, 0, ny_val]
        
    true_w = data.y[:, 0].numpy().reshape((ny_val, nx_val))

    num_models = len(models)
    
    fig = plt.figure(figsize=(24, 16))
    from matplotlib.gridspec import GridSpec
    # Massively increased hspace to prevent overlapping of labels and colorbars
    gs = GridSpec(3, max(2, num_models), figure=fig, hspace=1.0, wspace=0.6)
    
    # 1. Grid/Nodes
    ax_grid = fig.add_subplot(gs[0, 0])
    if hasattr(data, 'pos'):
        ax_grid.scatter(data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), s=1, c='gray')
        ax_grid.set_aspect('equal', 'box')
        
    if hasattr(data, 'load_x1'):
        import matplotlib.patches as patches
        x1 = data.load_x1.item()
        x2 = data.load_x2.item()
        y1 = data.load_y1.item()
        y2 = data.load_y2.item()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
        ax_grid.add_patch(rect)
        # Also outline it
        rect_edge = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
        ax_grid.add_patch(rect_edge)

    ax_grid.set_title("Grid / Nodes")
    ax_grid.set_xlim(extent[0], extent[1])
    ax_grid.set_ylim(extent[2], extent[3])
    
    # 2. True Deflection
    ax_true = fig.add_subplot(gs[0, 1])
    im1 = ax_true.imshow(true_w, cmap='viridis', origin='lower', extent=extent)
    ax_true.set_title("True Deflection (w)")
    fig.colorbar(im1, ax=ax_true, fraction=0.046, pad=0.04)
    
    all_axes = [ax_grid, ax_true]

    # Publication-ready model names mapping
    name_map = {
        "gnn_mse": "GNN\n(MSE)",
        "gnn_pinn": "GNN\n(PINN)",
        "transformer_mse": "Graph Transformer\n(MSE)",
        "transformer_pinn": "Graph Transformer\n(PINN)",
        "deeponet_mse": "DeepONet\n(MSE)",
        "deeponet_pinn": "DeepONet\n(PINN)"
    }

    for m_idx, (raw_model_name, model, norm_stats) in enumerate(models):
        model_name = name_map.get(raw_model_name, raw_model_name)
        
        ax_pred = fig.add_subplot(gs[1, m_idx])
        ax_err  = fig.add_subplot(gs[2, m_idx])
        all_axes.extend([ax_pred, ax_err])
        
        pred_w = None
        if model is not None and norm_stats is not None:
            try:
                result = predict_case(model, data, norm_stats, "cpu")
                pred_w = np.array(result["w"])
            except Exception as e:
                print(f"[{model_name}] Inference failed: {e}")
        
        if pred_w is not None:
            pred_w_2d = pred_w.reshape((ny_val, nx_val))
            im2 = ax_pred.imshow(pred_w_2d, cmap='viridis', origin='lower', extent=extent)
            ax_pred.set_title(f"Pred (w)\n{model_name}")
            fig.colorbar(im2, ax=ax_pred, fraction=0.046, pad=0.04)
            
            err_w = np.abs(true_w - pred_w_2d)
            im3 = ax_err.imshow(err_w, cmap='magma', origin='lower', extent=extent)
            ax_err.set_title(f"Abs Error\nMax: {err_w.max():.4f}\n({model_name})")
            fig.colorbar(im3, ax=ax_err, fraction=0.046, pad=0.04)
        else:
            ax_pred.text(0.5, 0.5, "No Prediction", ha='center', va='center')
            ax_pred.set_title(f"Pred (w)\n{model_name}")
            ax_err.text(0.5, 0.5, "No Error", ha='center', va='center')
            ax_err.set_title(f"Abs Error\n({model_name})")

    # Formatting for all axes
    for ax in all_axes:
        ax.set_xlabel('Slab X (mm)', fontsize=8)
        ax.set_ylabel('Slab Y (mm)', fontsize=8)
        if hasattr(data, 'pos'):
            ax.set_xticks(np.linspace(extent[0], extent[1], nx_val + 1), minor=True)
            ax.set_yticks(np.linspace(extent[2], extent[3], ny_val + 1), minor=True)

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=1.0, wspace=0.6)
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    print("Saved plot to comparison_plot.png")

if __name__ == "__main__":
    target_dir = r"generated_cases"
    
    chkpts = [
        str(project_dir / "runs" / "gnn_mse"),
        str(project_dir / "runs" / "gnn_pinn"),
        str(project_dir / "runs" / "transformer_mse"),
        str(project_dir / "runs" / "transformer_pinn"),
        str(project_dir / "runs" / "deeponet_mse"),
        str(project_dir / "runs" / "deeponet_pinn")
    ]
    
    plot_first_case(target_dir, checkpoint_dirs=chkpts)
