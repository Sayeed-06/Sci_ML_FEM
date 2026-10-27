import sys
import json
from pathlib import Path

project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from torch_geometric.loader import DataLoader
from dataset.sim_dataset import SimDataset, apply_norm, load_norm_stats
from evaluation.sim_metrics import evaluate, print_metrics_table
from plot_graphs import load_model

runs = ["runs/sim_deeponet_pinn", "runs/sim_transformer_pinn_v1"]

print("Loading dataset...")
ds = SimDataset(root="generated_cases")
_, _, test_list = ds.split(train_frac=0.70, val_frac=0.15, seed=42)
print(f"Test cases: {len(test_list)}")

for run_dir in runs:
    run_dir = Path(run_dir)
    print(f"\nEvaluating {run_dir}...")
    loaded, label = load_model(run_dir, device="cpu")
    model, norm_stats = loaded
    if model is None or norm_stats is None:
        print(f"Skipping {run_dir}: missing model/norm_stats")
        continue

    test_norm = apply_norm(test_list, norm_stats)
    test_loader = DataLoader(test_norm, batch_size=8, shuffle=False)

    metrics = evaluate(model, test_loader, norm_stats, device="cpu")
    print_metrics_table(metrics, title=str(run_dir))

    out_path = run_dir / "test_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved -> {out_path}")
