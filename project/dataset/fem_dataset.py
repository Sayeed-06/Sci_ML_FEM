"""
fem_dataset.py
==============
PyTorch Geometric InMemoryDataset for the ISLab FEM cases.

Usage
-----
    dataset = FEMDataset(root='generated_cases')
    train_ds, val_ds, test_ds = dataset.split(0.7, 0.15, seed=42)
"""

from __future__ import annotations
import os
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from torch_geometric.data import Data, InMemoryDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from parser import parse_input, parse_output
from dataset.build_graph import build_graph


class AugmentedListDataset(torch.utils.data.Dataset):
    """
    A simple wrapper that applies PyG transforms dynamically to a list of Data objects.
    Used to apply on-the-fly augmentation after the dataset has been parsed and
    statically normalized in memory.
    """
    def __init__(self, data_list: List[Data], transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        # Clone to prevent the transform from permanently mutating the cached list
        data = self.data_list[idx].clone()
        if self.transform is not None:
            data = self.transform(data)
        return data


class FEMDataset(InMemoryDataset):
    """
    Scans `root/` for directories matching `Case_*`, parses `input.in2`
    and `input.ou2` from each, builds a graph, and stores everything in
    memory via InMemoryDataset.

    Parameters
    ----------
    root        : path to the `generated_cases/` directory
    transform   : optional PyG transform
    pre_transform: applied once before caching
    pre_filter  : optional filter function
    include_diagonals : pass to build_graph (default True)
    """

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        include_diagonals: bool = True,
    ):
        self.include_diagonals = include_diagonals
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            weights_only=False)

    @property
    def raw_file_names(self) -> List[str]:
        # Each case dir has input.in2 and input.ou2
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ['fem_dataset.pt']

    def _case_dirs(self) -> List[Path]:
        root = Path(self.root)
        dirs = []
        if not root.exists():
            return dirs
        
        # Look at every folder inside generated_cases/
        for d in root.iterdir():
            if d.is_dir():
                # Only include it if it actually contains the two required files!
                if (d / 'input.in2').exists() and (d / 'input.ou2').exists():
                    dirs.append(d)
                    
        return sorted(dirs)

    def download(self) -> None:
        pass  # data is already on disk

    def process(self) -> None:
        case_dirs = self._case_dirs()
        data_list: List[Data] = []

        for case_dir in case_dirs:
            in2_path = case_dir / 'input.in2'
            ou2_path = case_dir / 'input.ou2'

            if not in2_path.exists() or not ou2_path.exists():
                print(f"  [WARN] Missing files in {case_dir.name}, skipping.")
                continue

            try:
                inp = parse_input(in2_path)
                out = parse_output(ou2_path)
                g = build_graph(inp, out,
                                include_diagonals=self.include_diagonals)
                g.case_name = case_dir.name
                data_list.append(g)
                print(f"  [OK] {case_dir.name}: {g.num_nodes} nodes, "
                      f"{g.num_edges} edges")
            except Exception as e:
                print(f"  [ERR] {case_dir.name}: {e}")
                raise  # fail fast during development

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if len(data_list) == 0:
            raise RuntimeError(
                f"No FEM cases found! Please check that your '{self.root}' directory "
                "contains subdirectories, and that those subdirectories contain "
                "'input.in2' and 'input.ou2'."
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"\nDataset saved: {len(data_list)} cases → {self.processed_paths[0]}")

    def split(
        self,
        train_frac: float = 0.70,
        val_frac: float = 0.15,
        seed: int = 42,
    ) -> Tuple['FEMDataset', 'FEMDataset', 'FEMDataset']:
        """
        Random split into train / val / test subsets.

        Returns three index-based dataset views.
        """
        n = len(self)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n).tolist()

        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)

        train_idx = idx[:n_train]
        val_idx   = idx[n_train:n_train + n_val]
        test_idx  = idx[n_train + n_val:]

        return (self[train_idx], self[val_idx], self[test_idx])


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'generated_cases'
    ds = FEMDataset(root=root)
    print(f"\nTotal cases: {len(ds)}")
    if len(ds) > 0:
        sample = ds[0]
        print(f"Sample graph: {sample}")
        train, val, test = ds.split()
        print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
