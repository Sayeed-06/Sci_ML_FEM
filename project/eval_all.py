import torch
import sys
from pathlib import Path

project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from plot_graphs import load_model
from dataset.fem_dataset import FEMDataset
from dataset.normalize import load_norm_stats, apply_norm
from torch_geometric.loader import DataLoader
from evaluation.metrics import evaluate_dataset

def main():
    device = 'cpu'
    
    # Load test dataset
    dataset = FEMDataset(root='generated_cases')
    _, _, test_ds = dataset.split(seed=42)
    print(f"Test cases: {len(test_ds)}")
    
    runs = [
        'runs/gnn_mse',
        'runs/gnn_pinn',
        'runs/transformer_mse',
        'runs/transformer_pinn',
        'runs/deeponet_mse',
        'runs/deeponet_pinn'
    ]
    
    all_results = {}
    
    for run_dir in runs:
        if not Path(run_dir).exists():
            continue
            
        print(f"Evaluating {run_dir}...")
        model, norm_stats = load_model(run_dir, device)
        if model is None:
            continue
            
        if norm_stats is not None:
            test_list = apply_norm([data.clone() for data in test_ds], norm_stats, normalise_targets=True)
            expected_dim = len(norm_stats['node_feat_mean'])
        else:
            test_list = [data.clone() for data in test_ds]
            # fallback if norm_stats missing, assume from shape
            expected_dim = test_list[0].x.shape[1]
            
        for d in test_list:
            d.x = d.x[:, :expected_dim]
            
        # evaluate_dataset
        test_loader = DataLoader(test_list, batch_size=4, shuffle=False)
        metrics, avg_loss = evaluate_dataset(model, test_loader, device=device, norm_stats=norm_stats)
        
        all_results[Path(run_dir).name] = metrics
        
    print("\n\n--- RESULTS ---")
    import json
    with open('metrics_output.json', 'w') as f:
        json.dump(all_results, f, indent=2)
        
if __name__ == '__main__':
    main()
