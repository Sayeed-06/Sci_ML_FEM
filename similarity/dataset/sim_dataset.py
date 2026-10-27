"""
sim_dataset.py
==============
PyTorch Geometric dataset for the Similarity-Based GNN.

Features
--------
- Scans a root directory for Case_* subdirectories containing
  input.in2 and input.ou2 files.
- Computes similarity parameters (l, phi, PQ) per case.
- Builds and caches graph objects as .pt files for fast re-loading
  (critical for 20k datasets on HPC where parsing all .ou2 files
  on every run is expensive).
- Provides train/val/test splits with reproducible seeds.
- Computes per-target normalisation statistics from the training split.

Usage
-----
    ds = SimDataset(root='/path/to/generated_cases', cache_dir='/path/to/cache')
    train_ds, val_ds, test_ds = ds.split(train_frac=0.7, val_frac=0.15, seed=42)

    # Compute normalisation stats from training data
    norm_stats = compute_norm_stats(train_ds)

    # Apply normalisation (in-place on copies)
    train_norm = apply_norm(train_ds, norm_stats)
    val_norm   = apply_norm(val_ds,   norm_stats)
    test_norm  = apply_norm(test_ds,  norm_stats)
"""

from __future__ import annotations

import os
import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
from torch_geometric.data import Data

# local imports (relative to similarity/ root)
import sys
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from parse_in2 import parse_in2
from parse_ou2 import parse_ou2
from similarity_transform import compute_similarity_params
from dataset.sim_graph import build_sim_graph

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SimDataset:
    """ISLAB2000 case dataset with similarity-coordinate graph representation.

    Parameters
    ----------
    root             : directory containing Case_* subdirectories
    cache_dir        : where to write .pt graph cache files (default: root/cache_sim)
    include_diagonals: add diagonal edges in graph (default True)
    force_rebuild    : ignore cache and re-parse all files (default False)
    max_cases        : limit dataset size (useful for quick tests)
    """

    def __init__(
        self,
        root: str | Path,
        cache_dir: Optional[str | Path] = None,
        include_diagonals: bool = True,
        force_rebuild: bool = False,
        max_cases: Optional[int] = None,
    ):
        self.root              = Path(root)
        self.cache_dir         = Path(cache_dir) if cache_dir else self.root / "cache_sim"
        self.include_diagonals = include_diagonals
        self.force_rebuild     = force_rebuild

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._graphs: List[Data] = []
        self._load(max_cases)

    # ── Loading & caching ──────────────────────────────────────────────

    def _case_dirs(self) -> List[Path]:
        return sorted([
            p for p in self.root.iterdir()
            if p.is_dir() and p.name.lower().startswith("case_")
        ])

    def _cache_path(self, case_dir: Path) -> Path:
        return self.cache_dir / f"{case_dir.name}.pt"

    def _load(self, max_cases: Optional[int]) -> None:
        case_dirs = self._case_dirs()
        if max_cases is not None:
            case_dirs = case_dirs[:max_cases]

        n_total   = len(case_dirs)
        n_cached  = 0
        n_built   = 0
        n_errors  = 0

        log.info(f"Found {n_total} cases in {self.root}")

        for i, case_dir in enumerate(case_dirs):
            cache_pt = self._cache_path(case_dir)
            in2_path = case_dir / "input.in2"
            ou2_path = case_dir / "input.ou2"

            if not in2_path.exists() or not ou2_path.exists():
                log.warning(f"Missing files in {case_dir.name}, skipping.")
                n_errors += 1
                continue

            if not self.force_rebuild and cache_pt.exists():
                try:
                    g = torch.load(cache_pt, weights_only=False)
                    self._graphs.append(g)
                    n_cached += 1
                    continue
                except Exception as e:
                    log.warning(f"Cache corrupt for {case_dir.name}: {e}. Rebuilding.")

            try:
                inp = parse_in2(in2_path)
                out = parse_ou2(ou2_path)
                sim = compute_similarity_params(
                    E=inp["E"], h=inp["thickness"], k_file=inp["k_winkler"],
                    mu=inp["nu"], alpha=inp["alpha_T"], gamma_file=inp["gamma"],
                    delta_T=inp["delta_T"], pressure_q=inp["pressure_q"],
                )
                g = build_sim_graph(inp, out, sim, self.include_diagonals)
                g.case_name = case_dir.name  # store for debugging

                torch.save(g, cache_pt)
                self._graphs.append(g)
                n_built += 1

            except Exception as e:
                log.error(f"Error processing {case_dir.name}: {e}")
                n_errors += 1

            if (i + 1) % 500 == 0:
                log.info(f"  Processed {i+1}/{n_total} cases...")

        log.info(
            f"Dataset loaded: {len(self._graphs)} graphs "
            f"({n_cached} cached, {n_built} built, {n_errors} errors)"
        )
        print(
            f"[SimDataset] {len(self._graphs)} cases loaded "
            f"({n_cached} from cache, {n_built} parsed, {n_errors} skipped)"
        )

    # ── Split ──────────────────────────────────────────────────────────

    def split(
        self,
        train_frac: float = 0.70,
        val_frac: float   = 0.15,
        seed: int         = 42,
    ) -> Tuple[List[Data], List[Data], List[Data]]:
        """Reproducible random train/val/test split."""
        rng = random.Random(seed)
        indices = list(range(len(self._graphs)))
        rng.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        train_idx = indices[:n_train]
        val_idx   = indices[n_train : n_train + n_val]
        test_idx  = indices[n_train + n_val :]

        return (
            [self._graphs[i] for i in train_idx],
            [self._graphs[i] for i in val_idx],
            [self._graphs[i] for i in test_idx],
        )

    def __len__(self) -> int:
        return len(self._graphs)

    def __getitem__(self, idx: int) -> Data:
        return self._graphs[idx]


# ---------------------------------------------------------------------------
# Normalisation utilities
# ---------------------------------------------------------------------------

def compute_norm_stats(graphs: List[Data]) -> Dict[str, torch.Tensor]:
    """Compute mean and std for targets and global features from a list of graphs.

    Targets: y = [w, sigma_x, sigma_y] per node
    Global:  global_feat = [l, phi, PQ] per graph
    """
    # ── Targets ────────────────────────────────────────────────────────
    all_y = torch.cat([g.y for g in graphs], dim=0)  # (N_total, 3)
    y_mean = all_y.mean(dim=0)
    y_std  = all_y.std(dim=0).clamp(min=1e-6)

    # ── Global features ────────────────────────────────────────────────
    all_gf = torch.cat([g.global_feat for g in graphs], dim=0)  # (N_graphs, 3)
    gf_mean = all_gf.mean(dim=0)
    gf_std  = all_gf.std(dim=0).clamp(min=1e-6)

    return {
        "y_mean": y_mean,
        "y_std":  y_std,
        "gf_mean": gf_mean,
        "gf_std":  gf_std,
    }


def apply_norm(
    graphs: List[Data],
    norm_stats: Dict[str, torch.Tensor],
    normalise_targets: bool = True,
) -> List[Data]:
    """Return new list of Data objects with normalised y and global_feat.

    The original .y values (absolute physical units) are stored as .y_raw
    so they can be recovered at evaluation time.
    """
    normed = []
    y_mean  = norm_stats["y_mean"]
    y_std   = norm_stats["y_std"]
    gf_mean = norm_stats["gf_mean"]
    gf_std  = norm_stats["gf_std"]

    for g in graphs:
        g2 = g.clone()

        # Normalise global features (l, phi, PQ) — always
        g2.global_feat = (g.global_feat - gf_mean) / gf_std

        # Normalise targets
        if normalise_targets:
            g2.y_raw = g.y.clone()                        # keep absolute values
            g2.y     = (g.y - y_mean) / y_std

        normed.append(g2)
    return normed


def denorm_targets(
    y_normed: torch.Tensor,
    norm_stats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Convert normalised predictions back to absolute physical units."""
    y_mean = norm_stats["y_mean"].to(y_normed.device)
    y_std  = norm_stats["y_std"].to(y_normed.device)
    return y_normed * y_std + y_mean


def save_norm_stats(norm_stats: Dict, path: str | Path) -> None:
    torch.save(norm_stats, path)
    print(f"Norm stats saved → {path}")


def load_norm_stats(path: str | Path) -> Dict:
    return torch.load(path, weights_only=False)


class AugmentedSimDataset(torch.utils.data.Dataset):
    """Wrapper to apply dynamic physical augmentation and on-the-fly normalisation."""
    def __init__(
        self,
        graphs: List[Data],
        norm_stats: Dict[str, torch.Tensor],
        augment_cfg: Optional[Dict[str, Any]] = None,
        is_training: bool = True
    ):
        self.graphs = graphs
        self.norm_stats = norm_stats
        self.augment_cfg = augment_cfg or {}
        self.is_training = is_training
        
        self.p_edge_drop = self.augment_cfg.get("p_edge_drop", 0.0)
        self.load_scale_range = self.augment_cfg.get("load_scale_range", [1.0, 1.0])

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        # 1. Clone the graph to avoid modifying the cached data in memory
        g = self.graphs[idx].clone()

        # 2. Apply Augmentations (only during training)
        if self.is_training:
            # 2a. Edge Dropout
            if self.p_edge_drop > 0.0 and g.edge_index.size(1) > 0:
                mask = torch.rand(g.edge_index.size(1)) >= self.p_edge_drop
                g.edge_index = g.edge_index[:, mask]
                if g.edge_attr is not None:
                    g.edge_attr = g.edge_attr[mask]

            # 2b. Linear Load Scaling (physics-consistent!)
            if self.load_scale_range[0] != 1.0 or self.load_scale_range[1] != 1.0:
                alpha = random.uniform(self.load_scale_range[0], self.load_scale_range[1])
                # Scale load indicator (q_local is feature index 2 in g.x)
                g.x[:, 2] = g.x[:, 2] * alpha
                # Scale PQ global feature (index 2 in g.global_feat)
                g.global_feat[0, 2] = g.global_feat[0, 2] * alpha
                # Scale all physical targets (w, sigma_x, sigma_y)
                g.y = g.y * alpha

        # 3. Apply Normalisation on-the-fly
        # We save raw targets for evaluation later
        g.y_raw = g.y.clone()
        
        # Normalise targets
        y_mean = self.norm_stats["y_mean"].to(g.y.device)
        y_std  = self.norm_stats["y_std"].to(g.y.device)
        g.y = (g.y - y_mean) / y_std

        # Normalise global features
        gf_mean = self.norm_stats["gf_mean"].to(g.global_feat.device)
        gf_std  = self.norm_stats["gf_std"].to(g.global_feat.device)
        g.global_feat = (g.global_feat - gf_mean) / gf_std

        return g


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ds   = SimDataset(root=root, max_cases=10)
    print(f"Dataset: {len(ds)} graphs")
    g = ds[0]
    print(g)
    print(f"  global_feat: {g.global_feat}")
    print(f"  l={g.l:.1f}  phi={g.phi:.4f}  PQ={g.PQ:.2f}")
