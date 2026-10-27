import torch
from torch_geometric.loader import DataLoader
from dataset.fem_dataset import FEMDataset
from losses.pinn_loss import PINNLoss
from models.encoders import build_graph_encoders

def test_ground_truth_pde():
    dataset = FEMDataset(root='generated_cases', split='val')
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    batch = next(iter(loader))
    
    # Instantiate PINN Loss
    loss_fn = PINNLoss(norm_stats=dataset.norm_stats)
    
    # Extract ground truth w (target is normalized, so we need to un-normalize or pass normalized)
    # The PINNLoss expects normalized predictions in pred[:, 0] and un-normalizes them internally.
    target_w_norm = batch.y[:, 0:1] # shape (N, 1)
    
    # We will pass the normalized targets as if they were predictions!
    dummy_pred = batch.y.clone()
    
    # Compute PDE loss using the ground truth targets
    pde_loss_val = loss_fn._pde_loss(dummy_pred, batch).item()
    
    print(f"PDE Loss of Ground Truth labels (with Diagonals): {pde_loss_val:.4f}")

if __name__ == '__main__':
    test_ground_truth_pde()
