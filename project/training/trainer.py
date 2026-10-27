"""
trainer.py
==========
Unified training loop for all four FEM surrogate models.

Supports:
  - MSE-only training (Models 1 & 3)
  - PINN training (Models 2 & 4)
  - Checkpointing (best val loss + periodic)
  - Early stopping
  - TensorBoard logging (optional)
  - Gradient clipping
"""

from __future__ import annotations
import os
import time
import math
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from evaluation.metrics import evaluate_dataset, print_metrics_table


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter   = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Unified trainer for GNN/GraphTransformer × MSE/PINN.

    Parameters
    ----------
    model        : nn.Module (GNNSurrogate or GraphTransformerSurrogate)
    loss_fn      : MSELoss or PINNLoss instance
    train_loader : PyG DataLoader
    val_loader   : PyG DataLoader
    cfg          : training config dict (see default_cfg())
    norm_stats   : normalisation stats (for metric de-normalisation)
    out_dir      : where to save checkpoints and logs
    """

    def __init__(
        self,
        model,
        loss_fn,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg: Dict[str, Any],
        norm_stats: Optional[Dict] = None,
        out_dir: str | Path = 'runs/unnamed',
    ):
        self.model        = model
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.norm_stats   = norm_stats
        self.out_dir      = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect CUDA if config says 'cpu' but GPU is available
        device_str = cfg.get('device', 'cpu')
        if device_str == 'cpu' and torch.cuda.is_available():
            print("  [Auto-Detect] CUDA is available! Switching device to 'cuda'.")
            device_str = 'cuda'
            
        self.device = torch.device(device_str)
        self.model.to(self.device)
        
        # Loss functions (MSELoss, PINNLoss) are nn.Modules containing buffers (e.g. weights)
        if hasattr(self.loss_fn, 'to'):
            self.loss_fn.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=cfg.get('lr', 1e-3),
            weight_decay=cfg.get('weight_decay', 1e-4),
        )

        self.epochs     = cfg.get('epochs', 200)
        self.grad_clip  = cfg.get('grad_clip', 1.0)
        self.patience   = cfg.get('patience', 30)
        self.checkpoint_every = cfg.get('checkpoint_every', 50)
        self.is_pinn    = cfg.get('is_pinn', False)

        # LR scheduler: cosine annealing
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=cfg.get('lr_min', 1e-6),
        )

        self.early_stop = EarlyStopping(patience=self.patience)

        # History
        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [],
            'lr': [],
        }
        if self.is_pinn:
            self.history.update({'pde_loss': [], 'stress_pde_loss': [], 'bc_loss': [], 'data_loss': []})

        self.best_val_loss = float('inf')
        self.global_step   = 0

        # Total steps (for PINN curriculum)
        steps_per_epoch = len(train_loader)
        self.total_steps = self.epochs * steps_per_epoch

    # ── Training loop ──────────────────────────────────────────────────────

    def train(self) -> Dict[str, List[float]]:
        """Run full training. Returns history dict."""
        print(f"\n{'='*60}")
        print(f" Training: {self.cfg.get('run_name', 'unnamed')}")
        print(f" Device: {self.device}")
        print(f" Epochs: {self.epochs}  |  Params: {self._count_params():,}")
        print(f"{'='*60}\n")

        start_epoch = 1
        if self.cfg.get('resume', False):
            start_epoch = self._resume_latest() + 1

        for epoch in range(start_epoch, self.epochs + 1):
            t0 = time.time()
            train_loss, train_sub = self._train_epoch(epoch)
            val_loss,   val_sub   = self._val_epoch(epoch)

            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(lr)

            if self.is_pinn and train_sub:
                self.history['data_loss'].append(train_sub.get('data', 0.0))
                self.history['pde_loss'].append(train_sub.get('pde', 0.0))
                self.history['stress_pde_loss'].append(train_sub.get('stress_pde', 0.0))
                self.history['bc_loss'].append(train_sub.get('bc', 0.0))

            elapsed = time.time() - t0

            if epoch % 10 == 0 or epoch == 1:
                msg = (f"Epoch {epoch:>4d}/{self.epochs} | "
                       f"Train: {train_loss:.6f} | "
                       f"Val: {val_loss:.6f} | "
                       f"LR: {lr:.2e} | "
                       f"t: {elapsed:.1f}s")
                if self.is_pinn and train_sub:
                    msg += (f" | data: {train_sub.get('data', 0):.4f} "
                            f"pde: {train_sub.get('pde', 0):.4f} "
                            f"stress: {train_sub.get('stress_pde', 0):.4f} "
                            f"bc: {train_sub.get('bc', 0):.4f}")
                print(msg)

            # Use pure data loss for early stopping and best checkpointing if PINN
            track_loss = val_loss
            if self.is_pinn and val_sub and 'data' in val_sub:
                track_loss = val_sub['data']

            # Checkpoint best model
            if track_loss < self.best_val_loss:
                self.best_val_loss = track_loss
                self._save_checkpoint('best.pt', epoch, track_loss)

            # Periodic checkpoint
            if epoch % self.checkpoint_every == 0:
                self._save_checkpoint(f'epoch_{epoch:04d}.pt', epoch, val_loss)

            # Early stopping
            if self.early_stop.step(track_loss):
                print(f"\n[Early stopping] No improvement for {self.patience} epochs.")
                break

        print(f"\n[Done] Best val loss: {self.best_val_loss:.6f}")
        self._save_history()
        return self.history

    def _train_epoch(self, epoch: int):
        """One training epoch. Returns (avg_loss, sub_losses_dict)."""
        self.model.train()
        total_loss = 0.0
        sub_totals: Dict[str, float] = {}
        n = 0

        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(data)

            if self.is_pinn:
                loss_dict = self.loss_fn(
                    pred, data.y, data,
                    epoch=epoch,
                    step=self.global_step,
                    total_steps=self.total_steps,
                )
                loss = loss_dict['total']
                for k, v in loss_dict.items():
                    if k != 'total':
                        sub_totals[k] = sub_totals.get(k, 0.0) + v.item()
            else:
                loss = self.loss_fn(pred, data.y)

            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )

            self.optimizer.step()
            total_loss += loss.item()
            self.global_step += 1
            n += 1

        avg_loss = total_loss / max(n, 1)
        avg_sub  = {k: v / max(n, 1) for k, v in sub_totals.items()}
        return avg_loss, avg_sub

    def _val_epoch(self, epoch: int = 0):
        """One validation epoch. Returns (avg_loss, sub_losses_dict)."""
        self.model.eval()
        total_loss = 0.0
        sub_totals: Dict[str, float] = {}
        n = 0

        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                pred = self.model(data)

                if self.is_pinn:
                    loss_dict = self.loss_fn(
                        pred, data.y, data,
                        epoch=epoch,
                        step=self.global_step,
                        total_steps=self.total_steps,
                    )
                    loss = loss_dict['total']
                    for k, v in loss_dict.items():
                        if k != 'total':
                            sub_totals[k] = sub_totals.get(k, 0.0) + v.item()
                else:
                    loss = self.loss_fn(pred, data.y)

                total_loss += loss.item()
                n += 1

        avg_loss = total_loss / max(n, 1)
        avg_sub  = {k: v / max(n, 1) for k, v in sub_totals.items()}
        return avg_loss, avg_sub

    # ── Utilities ──────────────────────────────────────────────────────────

    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> None:
        path = self.out_dir / filename
        torch.save({
            'epoch':       epoch,
            'val_loss':    val_loss,
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'sched_state': self.scheduler.state_dict(),
            'cfg':         self.cfg.get('full_cfg', self.cfg),
            'global_step': self.global_step,
            'history':     self.history,
        }, path)

    def _resume_latest(self) -> int:
        """Resume from the latest epoch checkpoint."""
        ckpts = list(self.out_dir.glob("epoch_*.pt"))
        if not ckpts:
            print("  [Resume] No checkpoints found to resume from.")
            return 0
        latest_ckpt = max(ckpts, key=lambda p: int(p.stem.split('_')[1]))
        ckpt = torch.load(latest_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])
        self.scheduler.load_state_dict(ckpt['sched_state'])
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        self.global_step = ckpt.get('global_step', 0)
        self.history = ckpt.get('history', self.history)
        
        print(f"  [Resume] Successfully resumed from {latest_ckpt.name} (epoch {ckpt['epoch']})")
        return ckpt['epoch']

    def load_best(self) -> None:
        """Load the best checkpoint into self.model."""
        ckpt = torch.load(self.out_dir / 'best.pt', map_location=self.device,
                          weights_only=False)
        self.model.load_state_dict(ckpt['model_state'])
        print(f"Loaded best checkpoint (epoch={ckpt['epoch']}, "
              f"val_loss={ckpt['val_loss']:.6f})")

    def _save_history(self) -> None:
        torch.save(self.history, self.out_dir / 'history.pt')

    def _count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def evaluate_test(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Load best model and evaluate on test set.
        Returns metrics dict.
        """
        self.load_best()
        metrics, avg_loss = evaluate_dataset(
            self.model, test_loader,
            device=str(self.device),
            norm_stats=self.norm_stats,
            loss_fn=self.loss_fn,
        )
        metrics['test_loss'] = avg_loss
        return metrics
