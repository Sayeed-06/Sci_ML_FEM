#!/usr/bin/env python3
"""Evaluate similarity DeepONet accuracy for the four Sobol deflection QoIs."""

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

from dataset.sim_dataset import AugmentedSimDataset, SimDataset, load_norm_stats
from models.deeponet import build_sim_deeponet


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
    parser.add_argument("--config", type=Path, default=ROOT / "runs/sim_deeponet_v1/config_resolved.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/sim_deeponet_v1/best.pt")
    parser.add_argument("--norm-stats", type=Path, default=ROOT / "runs/sim_deeponet_v1/norm_stats.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/sim_deeponet_v1/qoi_test_metrics.json")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    data_cfg = cfg["data"]
    seed = int(cfg["training"].get("seed", 42))
    dataset = SimDataset(
        root=ROOT / data_cfg["root"], cache_dir=data_cfg.get("cache_dir"),
        include_diagonals=data_cfg.get("include_diagonals", True),
        force_rebuild=False, max_cases=data_cfg.get("max_cases"),
    )
    _, _, test_graphs = dataset.split(
        train_frac=data_cfg.get("train_frac", 0.70),
        val_frac=data_cfg.get("val_frac", 0.15), seed=seed,
    )
    norm_stats = load_norm_stats(args.norm_stats)
    test_dataset = AugmentedSimDataset(test_graphs, norm_stats, is_training=False)
    loader = DataLoader(
        test_dataset, batch_size=int(cfg["training"].get("batch_size", 8)), shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_mean = norm_stats["y_mean"].to(device)
    y_std = norm_stats["y_std"].to(device)
    model = build_sim_deeponet(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    predicted = {name: [] for name in QOIS}
    observed = {name: [] for name in QOIS}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = batch.to(device)
            pred = model(batch) * y_std + y_mean
            target = batch.y * y_std + y_mean
            for graph_index in range(batch.num_graphs):
                left = int(batch.ptr[graph_index])
                right = int(batch.ptr[graph_index + 1])
                pred_summary = summarize(pred[left:right, 0])
                target_summary = summarize(target[left:right, 0])
                for name in QOIS:
                    predicted[name].append(pred_summary[name])
                    observed[name].append(target_summary[name])
            if batch_index == 0 or (batch_index + 1) % 50 == 0:
                print(f"  evaluated {min((batch_index + 1) * loader.batch_size, len(test_graphs))}/{len(test_graphs)}", flush=True)

    result = {
        "model": "sim_deeponet_v1",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "test_cases": len(test_graphs),
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
