import torch
from torch_geometric.loader import DataLoader
from dataset.fem_dataset import FEMDataset
from losses.pinn_loss import PINNLoss

def test_pde_interior_vs_boundary():
    dataset = FEMDataset(root='generated_cases', split='val')
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    batch = next(iter(loader))
    loss_fn = PINNLoss(norm_stats=dataset.norm_stats)
    
    # Ground truth w (normalized)
    target_w_norm = batch.y[:, 0:1] 
    dummy_pred = batch.y.clone()
    
    # ── Re-implement _pde_loss locally to split by boundary ──
    pred_w = loss_fn._get_physical_w(dummy_pred[:, 0])
    num_nodes = pred_w.shape[0]
    
    D_per_node  = batch.D[batch.batch].squeeze(-1)
    k_per_node  = batch.k_winkler[batch.batch].squeeze(-1)
    q_per_node  = batch.x[:, 5] * batch.pressure[batch.batch].squeeze(-1)
    
    bih_w = loss_fn.graph_lap.biharmonic(pred_w, batch.edge_index, batch.pos, num_nodes)
    residual = D_per_node * bih_w + k_per_node * pred_w - q_per_node
    
    res_sq = residual ** 2
    
    # bc_flag is in col 7
    is_boundary = batch.x[:, 7] > 0.5
    is_interior = ~is_boundary
    
    loss_interior = res_sq[is_interior].mean().item()
    loss_boundary = res_sq[is_boundary].mean().item()
    loss_total = res_sq.mean().item()
    
    print(f"Total PDE MSE:    {loss_total:.4f}")
    print(f"Interior PDE MSE: {loss_interior:.4f}")
    print(f"Boundary PDE MSE: {loss_boundary:.4f}")

if __name__ == '__main__':
    test_pde_interior_vs_boundary()
