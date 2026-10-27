#!/usr/bin/env python3
"""
run_training.py
===============
CLI entry point to train any of the four FEM surrogate models.

Usage
-----
    python scripts/run_training.py --config configs/gnn_mse.yaml
    python scripts/run_training.py --config configs/gnn_pinn.yaml
    python scripts/run_training.py --config configs/transformer_mse.yaml
    python scripts/run_training.py --config configs/transformer_pinn.yaml

    # Override any config key from command line:
    python scripts/run_training.py --config configs/gnn_mse.yaml \
        --training.epochs 100 --training.device cuda
"""

import sys
import argparse
import random
import yaml
import numpy as np
import torch
from pathlib import Path

# Make sure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_geometric.loader import DataLoader

from dataset.fem_dataset import FEMDataset, AugmentedListDataset
from dataset.features import ensure_expanded_dataset
from dataset.transforms import RandomPlateSymmetry, RandomLoadJitter
from torch_geometric.transforms import Compose
from dataset.normalize import (
    compute_norm_stats, apply_norm, save_norm_stats
)
from models.gnn import build_gnn
from models.graph_transformer import build_graph_transformer
from models.deeponet import build_deeponet
from losses.mse_loss import MSELoss
from losses.pinn_loss import PINNLoss
from training.trainer import Trainer
from evaluation.metrics import print_metrics_table


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively update nested dicts."""
    for k, v in overrides.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: str, overrides: list[str]) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Normalize overrides: handle both '--key.sub=value' and '--key.sub value' forms.
    # parse_known_args returns space-separated pairs as two separate list items,
    # so we combine consecutive '--flag value' tokens into '--flag=value' first.
    normalized: list[str] = []
    it = iter(overrides)
    for token in it:
        if token.startswith('--') and '=' not in token:
            # Peek at the next token; if it doesn't look like a flag, treat as value
            try:
                nxt = next(it)
                if nxt.startswith('--'):
                    # nxt is another flag — current token is a boolean flag; skip
                    normalized.append(token + '=True')
                    normalized.append(nxt)
                else:
                    normalized.append(f'{token}={nxt}')
            except StopIteration:
                # Trailing flag with no value — treat as boolean True
                normalized.append(token + '=True')
        else:
            normalized.append(token)

    # Apply --key.subkey=value overrides
    for override in normalized:
        if '=' not in override:
            continue  # safety: skip anything still unparseable
        k, v = override.split('=', 1)
        keys = k.lstrip('--').split('.')
        # Try to parse v as number, else keep as string
        try:
            v = float(v) if '.' in v else int(v)
        except ValueError:
            pass

        d = cfg
        for ki in keys[:-1]:
            d = d.setdefault(ki, {})
        d[keys[-1]] = v

    return cfg


def build_model(cfg: dict):
    model_cfg = cfg['model']
    mtype = model_cfg.get('type', 'gnn').lower()
    if mtype == 'gnn':
        return build_gnn(model_cfg)
    elif mtype in ('transformer', 'graph_transformer'):
        return build_graph_transformer(model_cfg)
    elif mtype == 'deeponet':
        return build_deeponet(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {mtype}")


def build_loss(cfg: dict, norm_stats=None):
    loss_cfg = cfg.get('loss', {})
    ltype = loss_cfg.get('type', 'mse').lower()
    weights = loss_cfg.get('weights', None)
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float)

    if ltype == 'mse':
        return MSELoss(weights=weights), False
    elif ltype == 'pinn':
        loss = PINNLoss(
            lambda_data    = loss_cfg.get('lambda_data',    1.0),
            lambda_pde     = loss_cfg.get('lambda_pde',     1.0e-4),
            lambda_bc      = loss_cfg.get('lambda_bc',      0.05),
            lambda_stress  = loss_cfg.get('lambda_stress',  1.0e-4),
            warmup_epochs  = loss_cfg.get('warmup_epochs',  80),
            ramp_epochs    = loss_cfg.get('ramp_epochs',    40),
            mse_weights    = weights,
            norm_stats     = norm_stats,
        )
        return loss, True
    else:
        raise ValueError(f"Unknown loss type: {ltype}")


def load_warm_start(model, checkpoint_path: str | Path) -> None:
    """Load compatible weights and expand a legacy node-encoder input layer."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = ckpt["model_state"]
    target = model.state_dict()
    loaded, expanded, skipped = 0, 0, []

    for key, value in source.items():
        if key not in target:
            skipped.append(key)
            continue
        if value.shape == target[key].shape:
            target[key] = value
            loaded += 1
            continue
        if (
            key.endswith("node_encoder.net.0.weight")
            and value.ndim == 2
            and target[key].ndim == 2
            and value.shape[0] == target[key].shape[0]
            and value.shape[1] < target[key].shape[1]
        ):
            merged = target[key].clone()
            merged[:, : value.shape[1]] = value
            target[key] = merged
            expanded += 1
            continue
        skipped.append(key)

    model.load_state_dict(target)
    print(
        f"Warm start: {loaded} tensors loaded, {expanded} input layer expanded, "
        f"{len(skipped)} incompatible tensors skipped."
    )


def main():
    parser = argparse.ArgumentParser(
        description='Train FEM surrogate model'
    )
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--resume', action='store_true', help='Resume from latest epoch checkpoint if available')
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    if 'training' not in cfg:
        cfg['training'] = {}
    cfg['training']['resume'] = args.resume
    set_seed(cfg['training'].get('seed', 42))

    print(f"\n{'='*60}")
    print(f"  Run: {cfg.get('run_name', 'unnamed')}")
    print(f"{'='*60}")

    # ── Dataset ────────────────────────────────────────────────────────────
    data_cfg = cfg['data']
    root     = data_cfg.get('root', 'generated_cases')

    print(f"\nLoading dataset from: {root}")
    dataset = FEMDataset(
        root=root,
        include_diagonals=data_cfg.get('include_diagonals', True),
    )
    print(f"Total cases: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError(
            "No cases found! Check that generated_cases/ contains Case_* dirs."
        )

    max_cases = data_cfg.get('max_cases')
    if max_cases is not None and int(max_cases) < len(dataset):
        rng = np.random.default_rng(cfg['training'].get('seed', 42))
        keep = np.sort(rng.choice(len(dataset), size=int(max_cases), replace=False))
        dataset = dataset[keep.tolist()]
        print(f"Using reproducible training subset: {len(dataset)} cases")

    train_ds, val_ds, test_ds = dataset.split(
        train_frac = data_cfg.get('train_frac', 0.70),
        val_frac   = data_cfg.get('val_frac',   0.15),
        seed       = cfg['training'].get('seed', 42),
    )
    print(f"Split: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

    if data_cfg.get('expanded_features', False):
        print("Expanding cached graphs: + alpha_T and pressure magnitude (11 -> 13 features)")
        train_ds = ensure_expanded_dataset(train_ds)
        val_ds = ensure_expanded_dataset(val_ds)
        test_ds = ensure_expanded_dataset(test_ds)

    # ── Normalisation ──────────────────────────────────────────────────────
    norm_stats = None
    if data_cfg.get('normalise_targets', True):
        norm_stats = compute_norm_stats(list(train_ds))
        train_list = apply_norm(list(train_ds), norm_stats, normalise_targets=True)
        val_list   = apply_norm(list(val_ds),   norm_stats, normalise_targets=True)
        test_list  = apply_norm(list(test_ds),  norm_stats, normalise_targets=True)

        out_dir = Path(cfg['output']['run_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        save_norm_stats(norm_stats, out_dir / 'norm_stats.pt')
        print(f"Norm stats saved → {out_dir / 'norm_stats.pt'}")

        from torch_geometric.data import Dataset
        # Wrap back as lists (DataLoader accepts lists of Data objects)
        train_ds_list = train_list
        val_ds_list   = val_list
        test_ds_list  = test_list
    else:
        train_ds_list = list(train_ds)
        val_ds_list   = list(val_ds)
        test_ds_list  = list(test_ds)

    # ── Augmentation ───────────────────────────────────────────────────────
    transforms_list = []
    if data_cfg.get('augment_symmetry', True):
        transforms_list.append(RandomPlateSymmetry(p=0.5))
    if data_cfg.get('augment_jitter', True):
        transforms_list.append(RandomLoadJitter(std=0.02, p=0.5))
        
    transform = Compose(transforms_list) if transforms_list else None
    if transform:
        print(f"Applying data augmentation: {transform}")
        
    train_ds_list = AugmentedListDataset(train_ds_list, transform=transform)

    # ── DataLoaders ────────────────────────────────────────────────────────
    batch_size = cfg['training'].get('batch_size', 4)
    train_loader = DataLoader(train_ds_list, batch_size=batch_size,
                              shuffle=True)
    val_loader   = DataLoader(val_ds_list,   batch_size=batch_size,
                              shuffle=False)
    test_loader  = DataLoader(test_ds_list,  batch_size=batch_size,
                              shuffle=False)

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(cfg)
    warm_start = cfg['training'].get('warm_start')
    if warm_start:
        load_warm_start(model, warm_start)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {cfg['model']['type'].upper()}  |  Params: {n_params:,}")

    # ── Loss ───────────────────────────────────────────────────────────────
    loss_fn, is_pinn = build_loss(cfg, norm_stats)
    print(f"Loss:  {'PINN' if is_pinn else 'MSE'}")

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer_cfg = {
        **cfg['training'],
        'run_name': cfg.get('run_name', 'unnamed'),
        'is_pinn': is_pinn,
        'full_cfg': cfg,
    }

    trainer = Trainer(
        model        = model,
        loss_fn      = loss_fn,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = trainer_cfg,
        norm_stats   = norm_stats,
        out_dir      = cfg['output']['run_dir'],
    )

    # ── Train ──────────────────────────────────────────────────────────────
    history = trainer.train()

    # ── Test evaluation ───────────────────────────────────────────────────
    if len(test_ds_list) > 0:
        print("\nEvaluating on test set...")
        test_metrics = trainer.evaluate_test(test_loader)
        print("\nTest Metrics:")
        for k, v in sorted(test_metrics.items()):
            print(f"  {k:<25s}: {v:.6f}")
    else:
        print("\n[WARN] No test samples.")

    print(f"\nResults saved to: {cfg['output']['run_dir']}")


if __name__ == '__main__':
    main()
