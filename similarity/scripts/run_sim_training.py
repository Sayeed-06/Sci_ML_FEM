#!/usr/bin/env python3
"""
run_sim_training.py
===================
Main training entry point for the Similarity-Based GNN surrogate.

Usage
-----
# Local quick test (5 epochs, first 50 cases):
python scripts/run_sim_training.py \\
    --config configs/sim_gnn.yaml \\
    --data.root /path/to/cases \\
    --training.epochs 5 \\
    --data.max_cases 50

# Full HPC training:
python scripts/run_sim_training.py \\
    --config configs/sim_gnn.yaml \\
    --data.root /scratch/user/generated_cases_20k \\
    --output.run_dir /scratch/user/runs/sim_gnn_v1

# Resume interrupted training:
python scripts/run_sim_training.py \\
    --config configs/sim_gnn.yaml \\
    --training.resume true

Config key overrides
--------------------
Any config key can be overridden with --section.key value, e.g.:
  --training.batch_size 4
  --model.num_layers 8
  --training.device cuda
"""

from __future__ import annotations

import sys
import os
import json
import logging
import argparse
from pathlib import Path

# ── Make sure similarity/ root is on the path ──────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import yaml
import torch
from torch_geometric.loader import DataLoader

from dataset.sim_dataset import (
    SimDataset,
    compute_norm_stats,
    apply_norm,
    save_norm_stats,
    AugmentedSimDataset,
)
from models.sim_gnn import build_sim_gnn
from models.graph_transformer import build_sim_graph_transformer
from models.deeponet import build_sim_deeponet
from training.sim_trainer import SimTrainer
from evaluation.sim_metrics import evaluate, print_metrics_table

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading + CLI override
# ---------------------------------------------------------------------------

def _deep_set(d: dict, keys: list, value):
    """Set d[key[0]][key[1]]... = value, creating sub-dicts as needed."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    # Type coercion
    prev = d.get(keys[-1])
    if isinstance(prev, bool):
        value = value.lower() in ("true", "1", "yes")
    elif isinstance(prev, int) and prev is not None:
        try: value = int(value)
        except: pass
    elif isinstance(prev, float):
        try: value = float(value)
        except: pass
    elif value == "null":
        value = None
    d[keys[-1]] = value


def load_config(config_path: str, overrides: list[str]) -> dict:
    """Load YAML config and apply CLI overrides of the form --section.key value."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    for i in range(0, len(overrides), 2):
        key_str = overrides[i].lstrip("-")
        val_str = overrides[i + 1] if i + 1 < len(overrides) else ""
        keys = key_str.split(".")
        _deep_set(cfg, keys, val_str)

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Parse CLI ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train Similarity-Based GNN surrogate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)

    # ── Setup ──────────────────────────────────────────────────────────
    out_dir = Path(cfg["output"]["run_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "train.log"),
        ],
    )
    log.info(f"Config:\n{json.dumps(cfg, indent=2, default=str)}")

    # Save resolved config
    with open(out_dir / "config_resolved.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Reproducibility
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Dataset ────────────────────────────────────────────────────────
    data_cfg = cfg["data"]
    log.info(f"Loading dataset from: {data_cfg['root']}")

    ds = SimDataset(
        root              = data_cfg["root"],
        cache_dir         = data_cfg.get("cache_dir"),
        include_diagonals = data_cfg.get("include_diagonals", True),
        force_rebuild     = data_cfg.get("force_rebuild", False),
        max_cases         = data_cfg.get("max_cases"),
    )

    if len(ds) == 0:
        raise RuntimeError(f"No cases loaded from {data_cfg['root']}")

    train_list, val_list, test_list = ds.split(
        train_frac = data_cfg.get("train_frac", 0.70),
        val_frac   = data_cfg.get("val_frac",   0.15),
        seed       = seed,
    )
    log.info(f"Split: train={len(train_list)}  val={len(val_list)}  test={len(test_list)}")

    # ── Normalisation ──────────────────────────────────────────────────
    norm_stats = compute_norm_stats(train_list)
    save_norm_stats(norm_stats, out_dir / "norm_stats.pt")

    log.info("Target normalisation stats (training set):")
    labels = ["w(mm)", "sigma_x(MPa)", "sigma_y(MPa)"]
    for i, lbl in enumerate(labels):
        log.info(f"  {lbl}: mean={norm_stats['y_mean'][i]:.4f}  "
                 f"std={norm_stats['y_std'][i]:.4f}")

    # ── Augmentation & Normalisation ───────────────────────────────────
    # We now use AugmentedSimDataset which performs on-the-fly augmentation and normalisation
    aug_cfg = cfg["training"].get("augmentation", {})
    
    train_norm = AugmentedSimDataset(train_list, norm_stats, augment_cfg=aug_cfg, is_training=True)
    val_norm   = AugmentedSimDataset(val_list,   norm_stats, augment_cfg=None,    is_training=False)
    test_norm  = AugmentedSimDataset(test_list,  norm_stats, augment_cfg=None,    is_training=False)

    # ── DataLoaders ────────────────────────────────────────────────────
    batch_size  = cfg["training"].get("batch_size", 8)
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

    train_loader = DataLoader(train_norm, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_norm,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_norm,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    # ── Model ──────────────────────────────────────────────────────────
    log.info("Building model...")
    model_type = cfg.get("model", {}).get("type", "sim_gnn")
    if model_type == "graph_transformer":
        model = build_sim_graph_transformer(cfg.get("model", {}))
    elif model_type == "deeponet":
        model = build_sim_deeponet(cfg.get("model", {}))
    else:
        model = build_sim_gnn(cfg.get("model", {}))
    log.info(f"Model: {model.__class__.__name__}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"SimGNN: {n_params:,} trainable parameters")

    # ── Training ───────────────────────────────────────────────────────
    train_cfg = {**cfg["training"], "run_name": cfg.get("run_name", "sim_gnn")}
    # merge loss weights into training cfg
    train_cfg["loss_weights"] = cfg.get("loss", {}).get("weights", [1.0, 1.0, 1.0])

    trainer = SimTrainer(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = train_cfg,
        norm_stats   = norm_stats,
        out_dir      = out_dir,
    )

    history = trainer.train()

    # ── Test evaluation (best checkpoint) ──────────────────────────────
    log.info("\n[Test] Loading best checkpoint for final evaluation...")
    trainer.load_best()

    test_metrics = evaluate(
        model      = trainer.model,
        loader     = test_loader,
        norm_stats = norm_stats,
        device     = str(trainer.device),
    )
    print_metrics_table(test_metrics, title="Test Set — Absolute Physical Units")

    # Save test metrics as JSON
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    log.info(f"Test metrics saved → {out_dir}/test_metrics.json")


if __name__ == "__main__":
    main()
