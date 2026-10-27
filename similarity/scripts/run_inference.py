#!/usr/bin/env python3
"""
run_inference.py
================
Inference script: load a trained SimGNN and predict w, sigma_x, sigma_y
for one or more cases, outputting absolute physical values.

Outputs per case
----------------
  - Console: table of l, phi, P/Q + field statistics
  - CSV file: node-level [x, y, w_pred, sx_pred, sy_pred]  (absolute units)
  - (Optional) numpy .npz archive

Usage
-----
# Predict a single case directory:
python scripts/run_inference.py \\
    --checkpoint runs/sim_gnn_v1/best.pt \\
    --norm_stats  runs/sim_gnn_v1/norm_stats.pt \\
    --case_dir    /path/to/Case_XXXX \\
    --output_dir  predictions/

# Predict all test cases from a data directory:
python scripts/run_inference.py \\
    --checkpoint runs/sim_gnn_v1/best.pt \\
    --norm_stats  runs/sim_gnn_v1/norm_stats.pt \\
    --data_root   /path/to/cases \\
    --output_dir  predictions/
"""

from __future__ import annotations

import sys
import csv
import argparse
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from models.graph_transformer import build_sim_graph_transformer
from parse_in2 import parse_in2
from parse_ou2 import parse_ou2
from similarity_transform import compute_similarity_params
from dataset.sim_graph import build_sim_graph
from dataset.sim_dataset import load_norm_stats
from models.sim_gnn import build_sim_gnn
from evaluation.sim_metrics import predict_case


def load_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    # Auto-detect model type and hidden_dim if not in cfg
    model_cfg = cfg.get("model", {})
    if "type" not in model_cfg:
        if "transformer" in checkpoint_path.lower():
            model_cfg["type"] = "graph_transformer"
        elif "deeponet" in checkpoint_path.lower():
            model_cfg["type"] = "sim_deeponet"
        else:
            model_cfg["type"] = "sim_gnn"
            
    if "hidden_dim" not in model_cfg:
        state = ckpt["model_state"]
        # Look at the first linear layer weight in node_encoder to find hidden_dim
        for key, tensor in state.items():
            if key == "node_encoder.net.0.weight" or key == "module.node_encoder.net.0.weight":
                model_cfg["hidden_dim"] = tensor.shape[0]
                break
            elif "branch.node_encoder.net.0.weight" in key:
                model_cfg["hidden_dim"] = tensor.shape[0]
                break
                
    if "num_layers" not in model_cfg:
        state = ckpt["model_state"]
        max_layer = -1
        for key in state.keys():
            # For GNN / Transformer: layers.0...
            # For DeepONet: branch.layers.0...
            if "layers." in key:
                parts = key.split(".")
                idx = parts.index("layers") + 1
                if idx < len(parts) and parts[idx].isdigit():
                    max_layer = max(max_layer, int(parts[idx]))
        if max_layer >= 0:
            model_cfg["num_layers"] = max_layer + 1

    model_type = model_cfg.get("type", "sim_gnn")
    if model_type == "graph_transformer":
        model = build_sim_graph_transformer(model_cfg).to(device)
    elif "deeponet" in model_type:
        from models.deeponet import build_sim_deeponet
        model = build_sim_deeponet(model_cfg).to(device)
    else:
        model = build_sim_gnn(model_cfg).to(device)
        
    # Handle DataParallel-saved state dicts
    state = ckpt["model_state"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}  (epoch={ckpt.get('epoch','?')})")
    return model


def infer_case(case_dir: Path, model, norm_stats: dict, device: str) -> dict:
    """Parse one case directory and run inference. Returns result dict."""
    inp = parse_in2(case_dir / "input.in2")
    out = parse_ou2(case_dir / "input.ou2")
    sim = compute_similarity_params(
        E=inp["E"], h=inp["thickness"], k_file=inp["k_winkler"],
        mu=inp["nu"], alpha=inp["alpha_T"], gamma_file=inp["gamma"],
        delta_T=inp["delta_T"], pressure_q=inp["pressure_q"],
    )
    g = build_sim_graph(inp, out, sim, include_diagonals=True)

    # Normalise global features using training stats
    # gf_mean/gf_std are shape (3,); global_feat is (1,3) — broadcast OK
    gf_mean = norm_stats["gf_mean"].unsqueeze(0)  # (1, 3)
    gf_std  = norm_stats["gf_std"].unsqueeze(0)   # (1, 3)
    g.global_feat = (g.global_feat - gf_mean) / gf_std

    result = predict_case(model, g, norm_stats, device)
    result["case"] = case_dir.name
    result["true_w"]  = out["w"]
    result["true_sx"] = out["sigma_x_bot"]
    result["true_sy"] = out["sigma_y_bot"]
    return result


def save_csv(result: dict, out_path: Path) -> None:
    """Save node-level predictions to CSV."""
    pos = result["pos"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x_mm", "y_mm",
                         "w_pred_mm",    "w_true_mm",
                         "sx_pred_MPa",  "sx_true_MPa",
                         "sy_pred_MPa",  "sy_true_MPa"])
        for i in range(len(pos)):
            writer.writerow([
                pos[i, 0], pos[i, 1],
                result["w"][i],       result["true_w"][i],
                result["sigma_x"][i], result["true_sx"][i],
                result["sigma_y"][i], result["true_sy"][i],
            ])


def print_case_summary(result: dict) -> None:
    print(f"\n── {result['case']} ──")
    print(f"  l={result['l']:.1f}mm  phi={result['phi']:.4f}  P/Q={result['PQ']:.3f}")
    for field, pred, true in [
        ("w (mm)",    result["w"],       result["true_w"]),
        ("σx (MPa)", result["sigma_x"], result["true_sx"]),
        ("σy (MPa)", result["sigma_y"], result["true_sy"]),
    ]:
        p, t = np.array(pred), np.array(true)
        rmse = np.sqrt(((p - t)**2).mean())
        maxe = np.abs(p - t).max()
        r2   = 1 - ((p-t)**2).sum() / (((t-t.mean())**2).sum()+1e-12)
        print(f"  {field:<10} RMSE={rmse:.5f}  MaxAE={maxe:.5f}  R²={r2:.5f}")


def main():
    parser = argparse.ArgumentParser(description="SimGNN inference in physical units")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--norm_stats",  required=True)
    parser.add_argument("--case_dir",    default=None)
    parser.add_argument("--data_root",   default=None)
    parser.add_argument("--output_dir",  default="predictions")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--save_npz",    action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm_stats = load_norm_stats(args.norm_stats)
    model      = load_model(args.checkpoint, args.device)

    case_dirs = []
    if args.case_dir:
        case_dirs = [Path(args.case_dir)]
    elif args.data_root:
        case_dirs = sorted([
            p for p in Path(args.data_root).iterdir()
            if p.is_dir() and p.name.startswith("Case_")
        ])
    else:
        parser.error("Provide --case_dir or --data_root")

    print(f"\nRunning inference on {len(case_dirs)} case(s)...")

    for case_dir in case_dirs:
        try:
            result = infer_case(case_dir, model, norm_stats, args.device)
            print_case_summary(result)
            save_csv(result, out_dir / f"{result['case']}_predictions.csv")
            if args.save_npz:
                np.savez_compressed(
                    out_dir / f"{result['case']}.npz",
                    pos=result["pos"],
                    w_pred=result["w"],       w_true=result["true_w"],
                    sx_pred=result["sigma_x"], sx_true=result["true_sx"],
                    sy_pred=result["sigma_y"], sy_true=result["true_sy"],
                )
        except Exception as e:
            print(f"[ERROR] {case_dir.name}: {e}")

    print(f"\nPredictions saved → {out_dir}/")


if __name__ == "__main__":
    main()
