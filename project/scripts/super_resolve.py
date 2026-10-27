#!/usr/bin/env python3
"""
super_resolve.py
================
Runs inference using the trained ESRGAN model on a low-resolution mesh tensor
and generates visualizations comparing LR and SR outputs.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.esrgan import build_esrgan

def super_resolve(lr_path, model_path='runs/esrgan/best.pt', out_dir='runs/esrgan_inference', device='cpu'):
    print(f"Running inference on device: {device}")
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Load LR tensor
    lr_tensor = torch.load(lr_path, map_location=device, weights_only=True)
    if lr_tensor.dim() == 3:
        lr_tensor = lr_tensor.unsqueeze(0) # (1, 3, H, W)
    
    print(f"Loaded LR tensor with shape: {lr_tensor.shape}")
    
    # Load Model
    model = build_esrgan(in_channels=3, out_channels=3).to(device)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"Loaded trained weights from {model_path}")
    else:
        print(f"[WARN] No trained weights found at {model_path}. Using initialized weights!")
        
    model.eval()
    with torch.no_grad():
        sr_tensor = model(lr_tensor)
        
    print(f"Generated SR tensor with shape: {sr_tensor.shape}")
    
    # Visualization
    lr_np = lr_tensor.squeeze(0).cpu().numpy()
    sr_np = sr_tensor.squeeze(0).cpu().numpy()
    
    var_names = ['w (Deflection)', 'sigma_x', 'sigma_y']
    
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for i in range(3):
        # LR Plot
        im_lr = axes[i, 0].imshow(lr_np[i], cmap='viridis')
        axes[i, 0].set_title(f"LR {var_names[i]} ({lr_np.shape[1]}x{lr_np.shape[2]})")
        plt.colorbar(im_lr, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # SR Plot
        im_sr = axes[i, 1].imshow(sr_np[i], cmap='viridis')
        axes[i, 1].set_title(f"SR {var_names[i]} ({sr_np.shape[1]}x{sr_np.shape[2]})")
        plt.colorbar(im_sr, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
    plt.tight_layout()
    out_file = Path(out_dir) / f"{Path(lr_path).stem}_comparison.png"
    plt.savefig(out_file, dpi=300)
    print(f"Saved visualization to {out_file}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='dataset_sr/LR/sample_0000.pt', help='Path to LR .pt file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Could not find {args.input}. Please pass the path to an LR .pt file.")
        sys.exit(1)
        
    super_resolve(args.input, device=args.device)
