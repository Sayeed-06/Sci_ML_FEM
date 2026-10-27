import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

# Add the 'similarity' directory to path to load models and dataset scripts
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from dataset.sim_dataset import SimDataset, apply_norm, load_norm_stats
from models.sim_gnn import build_sim_gnn
from models.graph_transformer import build_sim_graph_transformer
from models.deeponet import build_sim_deeponet
from evaluation.sim_metrics import predict_case


BUILDERS = {
    "sim_gnn": build_sim_gnn,
    "graph_transformer": build_sim_graph_transformer,
    "deeponet": build_sim_deeponet,
}

ARCH_LABELS = {
    "sim_gnn": "SimGNN",
    "graph_transformer": "Graph Transformer",
    "deeponet": "DeepONet",
}


def load_model(run_dir, device="cpu"):
    run_dir = Path(run_dir)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        print(f"Warning: Checkpoint not found at {ckpt_path}")
        return None, None

    config_path = run_dir / "config_resolved.yaml"
    model_cfg = {}
    loss_type = "mse"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get("model", {})
        loss_type = cfg.get("training", {}).get("loss_type", "mse")

    model_type = model_cfg.get("type", "sim_gnn")
    builder = BUILDERS.get(model_type, build_sim_gnn)
    model = builder(model_cfg)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    norm_path = run_dir / "norm_stats.pt"
    norm_stats = load_norm_stats(norm_path) if norm_path.exists() else None

    label = f"{ARCH_LABELS.get(model_type, model_type)}\n({'PINN' if 'pinn' in loss_type else 'MSE'}, {run_dir.name.split('_')[-1]})"
    return (model, norm_stats), label


TARGET_CASE = "case_00001_EValue28229_Alpha7.93e-06_Kval0.174_pressure0.80"

def plot_first_case(dataset_root, run_dirs):
    print("Loading dataset...")
    dataset = SimDataset(root=dataset_root)

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
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        ckpt_path = run_dir / "best.pt"
        norm_path = run_dir / "norm_stats.pt"
        if ckpt_path.exists() and norm_path.exists():
            print(f"Loading eval model from {run_dir}...")
            loaded, label = load_model(run_dir, device="cpu")
            models.append((label, loaded[0], loaded[1]))
        else:
            print(f"Checkpoint or norm_stats not found in {run_dir}")
            models.append((run_dir.name, None, None))

    nx_val = int(data.nx)
    ny_val = int(data.ny)

    x_coords = data.pos[:, 0].numpy().reshape((ny_val, nx_val))[0, :]
    y_coords = data.pos[:, 1].numpy().reshape((ny_val, nx_val))[:, 0]
    dx = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
    dy = y_coords[1] - y_coords[0] if len(y_coords) > 1 else 1.0
    extent = [x_coords[0] - dx / 2, x_coords[-1] + dx / 2, y_coords[0] - dy / 2, y_coords[-1] + dy / 2]

    true_w = data.y[:, 0].numpy().reshape((ny_val, nx_val))

    num_models = len(models)

    fig = plt.figure(figsize=(4 * max(2, num_models), 16))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(3, max(2, num_models), figure=fig, hspace=1.0, wspace=0.6)

    # 1. Grid/Nodes
    ax_grid = fig.add_subplot(gs[0, 0])
    ax_grid.scatter(data.pos[:, 0].numpy(), data.pos[:, 1].numpy(), s=1, c='gray')
    ax_grid.set_aspect('equal', 'box')

    load_mask = data.x[:, 2].numpy() > 0.5
    if load_mask.any():
        ax_grid.scatter(
            data.pos[load_mask, 0].numpy(), data.pos[load_mask, 1].numpy(),
            s=4, c='red', alpha=0.6, label='Loaded nodes'
        )

    ax_grid.set_title("Grid / Nodes")
    ax_grid.set_xlim(extent[0], extent[1])
    ax_grid.set_ylim(extent[2], extent[3])

    # 2. True Deflection
    ax_true = fig.add_subplot(gs[0, 1])
    im1 = ax_true.imshow(true_w, cmap='viridis', origin='lower', extent=extent)
    ax_true.set_title("True Deflection (w)")
    fig.colorbar(im1, ax=ax_true, fraction=0.046, pad=0.04)

    all_axes = [ax_grid, ax_true]

    for m_idx, (model_name, model, norm_stats) in enumerate(models):
        ax_pred = fig.add_subplot(gs[1, m_idx])
        ax_err = fig.add_subplot(gs[2, m_idx])
        all_axes.extend([ax_pred, ax_err])

        pred_w = None
        if model is not None and norm_stats is not None:
            try:
                normed = apply_norm([data], norm_stats)[0]
                normed.batch = torch.zeros(normed.x.shape[0], dtype=torch.long)
                result = predict_case(model, normed, norm_stats, "cpu")
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
        ax.set_xticks(np.linspace(extent[0], extent[1], nx_val + 1), minor=True)
        ax.set_yticks(np.linspace(extent[2], extent[3], ny_val + 1), minor=True)

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=1.0, wspace=0.6)
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
    print("Saved plot to comparison_plot.png")


if __name__ == "__main__":
    target_dir = r"generated_cases"

    runs_dir = project_dir / "runs"
    run_dirs = sorted(
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "best.pt").exists()
    )

    plot_first_case(target_dir, run_dirs=run_dirs)
