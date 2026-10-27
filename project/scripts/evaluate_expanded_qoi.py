#!/usr/bin/env python3
"""Evaluate the expanded DeepONet on Sobol quantities of interest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.features import ensure_expanded_dataset
from dataset.fem_dataset import FEMDataset
from dataset.normalize import apply_norm, denorm_targets, load_norm_stats
from models.deeponet import build_deeponet


QOIS = ("mean_abs_w", "median_abs_w", "p95_abs_w", "max_abs_w")


def summarize(values: torch.Tensor) -> dict[str, float]:
    absolute = values.abs()
    return {
        "mean_abs_w": float(absolute.mean()),
        "median_abs_w": float(torch.quantile(absolute, 0.50)),
        "p95_abs_w": float(torch.quantile(absolute, 0.95)),
        "max_abs_w": float(absolute.max()),
    }


def metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((observed - observed.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "R2": float(1.0 - ss_res / (ss_tot + 1e-12)),
        "relative_L2": float(np.linalg.norm(error) / (np.linalg.norm(observed) + 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deeponet_mse_expanded.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/deeponet_mse_expanded/best.pt")
    parser.add_argument("--norm-stats", type=Path, default=ROOT / "runs/deeponet_mse_expanded/norm_stats.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/deeponet_mse_expanded/qoi_test_metrics.json")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    seed = int(cfg["training"].get("seed", 42))
    dataset = FEMDataset(root=cfg["data"]["root"], include_diagonals=cfg["data"].get("include_diagonals", True))
    max_cases = cfg["data"].get("max_cases")
    if max_cases and int(max_cases) < len(dataset):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(dataset), size=int(max_cases), replace=False))
        dataset = dataset[keep.tolist()]
    _, _, test_dataset = dataset.split(
        train_frac=cfg["data"].get("train_frac", 0.70),
        val_frac=cfg["data"].get("val_frac", 0.15),
        seed=seed,
    )
    graphs = ensure_expanded_dataset(test_dataset)
    norm_stats = load_norm_stats(args.norm_stats)
    graphs = apply_norm(graphs, norm_stats, normalise_targets=True)
    loader = DataLoader(graphs, batch_size=int(cfg["training"].get("batch_size", 4)), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_stats = {key: value.to(device) for key, value in norm_stats.items()}
    model = build_deeponet(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    predicted = {name: [] for name in QOIS}
    observed = {name: [] for name in QOIS}
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            pred = denorm_targets(model(batch), device_stats)
            target = denorm_targets(batch.y, device_stats)
            for graph_index in range(batch.num_graphs):
                left = int(batch.ptr[graph_index])
                right = int(batch.ptr[graph_index + 1])
                pred_summary = summarize(pred[left:right, 0])
                target_summary = summarize(target[left:right, 0])
                for name in QOIS:
                    predicted[name].append(pred_summary[name])
                    observed[name].append(target_summary[name])

    result = {
        "model": "deeponet_mse_expanded",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "test_cases": len(graphs),
        "qoi_metrics": {
            name: metrics(np.asarray(predicted[name]), np.asarray(observed[name]))
            for name in QOIS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
