"""
sim_trainer.py
==============
HPC-ready training loop for the Similarity-Based GNN surrogate.

Features
--------
- Single-GPU and multi-GPU (DataParallel) support
- AdamW + CosineAnnealingWarmRestarts scheduler
- Gradient clipping
- Best-checkpoint + periodic checkpoint saving
- Early stopping
- Logging to both console and a CSV file (for HPC where TensorBoard may
  not be available)
- Reports metrics in ABSOLUTE physical units at validation time
  (not normalised) so you can directly compare MPa values

Absolute-unit metrics at validation
-------------------------------------
All RMSE and R² values reported during training are computed after
de-normalising predictions back to mm (w) and MPa (σ_x, σ_y).
This ensures the logged numbers are directly physically interpretable
and comparable to ISLAB2000 reference values.
"""

from __future__ import annotations

import os
import csv
import time
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

from training.pinn_loss import PINNLoss

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 40, min_delta: float = 1e-7):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_loss   = float("inf")
        self.counter     = 0
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SimTrainer:
    """Training loop for SimGNN.

    Parameters
    ----------
    model        : SimGNN instance
    train_loader : PyG DataLoader
    val_loader   : PyG DataLoader
    cfg          : dict from YAML config (training section)
    norm_stats   : dict from sim_dataset.compute_norm_stats()
                   REQUIRED for absolute-unit validation metrics
    out_dir      : directory for checkpoints and logs
    """

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg: Dict[str, Any],
        norm_stats: Dict,
        out_dir: str | Path,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.norm_stats   = norm_stats
        self.out_dir      = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Device setup ──────────────────────────────────────────────
        device_str = cfg.get("device", "cpu")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # PyG DataParallel requires DataListLoader instead of DataLoader.
        # To keep the training loop simple and compatible with DDP later,
        # we stick to single-GPU (cuda:0) here.
        if torch.cuda.device_count() > 1 and "cuda" in device_str:
            log.warning(
                f"Found {torch.cuda.device_count()} GPUs, but PyG multi-GPU "
                "requires DataListLoader or DDP. Using single GPU (cuda:0)."
            )

        self.model.to(self.device)

        # ── Optimiser ─────────────────────────────────────────────────
        lr = cfg.get("lr", 1e-3)
        wd = cfg.get("weight_decay", 1e-4)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

        # ── Scheduler: CosineAnnealingLR ─────────────────────
        lr_min = cfg.get("lr_min", 1e-6)
        self.epochs = cfg.get("epochs", 500)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=lr_min
        )

        # ── Loss ──────────────────────────────────────────────────────
        self.loss_type = cfg.get("loss_type", "mse")
        self.lambda_phys = cfg.get("lambda_phys", 0.1)
        weights = cfg.get("loss_weights", [1.0, 1.0, 1.0])
        
        self.criterion = PINNLoss(
            loss_type=self.loss_type,
            lambda_phys=self.lambda_phys,
            loss_weights=weights
        ).to(self.device)

        # ── Training settings ─────────────────────────────────────────
        self.grad_clip        = cfg.get("grad_clip", 1.0)
        self.checkpoint_every = cfg.get("checkpoint_every", 50)
        self.early_stop       = EarlyStopping(
            patience  = cfg.get("patience", 40),
            min_delta = cfg.get("min_delta", 1e-7),
        )

        # ── History ───────────────────────────────────────────────────
        self.history: Dict[str, List] = {
            "epoch": [], "train_loss": [], "val_loss": [],
            "val_rmse_w": [], "val_rmse_sx": [], "val_rmse_sy": [],
            "val_r2_w":   [], "val_r2_sx":   [], "val_r2_sy":   [],
            "lr": [],
        }
        self.best_val_loss = float("inf")

        # CSV log (HPC-friendly alternative to TensorBoard)
        self._csv_path = self.out_dir / "training_log.csv"
        self._csv_file = None
        self._csv_writer = None

    # ── Main training loop ─────────────────────────────────────────────

    def train(self) -> Dict:
        """Run training. Returns history dict."""
        self._open_csv()
        print(f"\n{'='*65}")
        print(f" SimGNN Training — {self.cfg.get('run_name', 'unnamed')}")
        print(f" Device: {self.device}  |  Epochs: {self.epochs}")
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f" Parameters: {n_params:,}")
        print(f"{'='*65}\n")

        start_epoch = 1
        if self.cfg.get("resume", False):
            start_epoch = self._resume() + 1

        for epoch in range(start_epoch, self.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics = self._val_epoch(epoch)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr)
            for k in ["val_rmse_w", "val_rmse_sx", "val_rmse_sy",
                       "val_r2_w",   "val_r2_sx",   "val_r2_sy"]:
                self.history[k].append(val_metrics.get(k, float("nan")))

            # Console print
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"Ep {epoch:>4d}/{self.epochs} | "
                    f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                    f"RMSE w={val_metrics['val_rmse_w']:.4f}mm "
                    f"σx={val_metrics['val_rmse_sx']:.4f}MPa "
                    f"σy={val_metrics['val_rmse_sy']:.4f}MPa | "
                    f"R² w={val_metrics['val_r2_w']:.4f} | "
                    f"LR={lr:.2e} | {elapsed:.1f}s"
                )

            # CSV log
            self._csv_writer.writerow({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "lr": lr, **val_metrics,
            })
            self._csv_file.flush()

            # Best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best.pt", epoch, val_loss)

            # Periodic checkpoint
            if epoch % self.checkpoint_every == 0:
                self._save_checkpoint(f"epoch_{epoch:04d}.pt", epoch, val_loss)

            # Early stopping
            if self.early_stop.step(val_loss):
                print(f"\n[Early Stop] No improvement for {self.cfg.get('patience',40)} epochs.")
                break

        self._close_csv()
        torch.save(self.history, self.out_dir / "history.pt")
        print(f"\n[Done] Best val loss: {self.best_val_loss:.6f}")
        print(f"       Checkpoints → {self.out_dir}")
        return self.history

    # ── One epoch ──────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int = 0) -> float:
        self.model.train()
        total, n = 0.0, 0
        for data in self.train_loader:
            data = data.to(self.device)
            if self.loss_type.startswith("pinn"):
                data.x.requires_grad_(True)
                
            self.optimizer.zero_grad()
            pred = self.model(data)
            
            loss, metrics = self.criterion(pred, data, self.norm_stats, epoch=epoch)
            loss.backward()
            
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def _val_epoch(self, epoch: int = 0):
        """Validation: MSE loss + absolute-unit RMSE/R²."""
        self.model.eval()
        all_pred, all_true = [], []
        total, n = 0.0, 0

        y_mean = self.norm_stats["y_mean"].to(self.device)
        y_std  = self.norm_stats["y_std"].to(self.device)

        with torch.no_grad():
            for data in self.val_loader:
                data = data.to(self.device)
                if self.loss_type.startswith("pinn"):
                    data.x.requires_grad_(True)
                
                pred = self.model(data)
                loss, metrics = self.criterion(pred, data, self.norm_stats, epoch=epoch)
                total += loss.item()
                n += 1

                # De-normalise for absolute-unit metrics
                pred_abs = pred * y_std + y_mean
                true_abs = data.y * y_std + y_mean
                all_pred.append(pred_abs.cpu())
                all_true.append(true_abs.cpu())

        all_pred = torch.cat(all_pred, dim=0)  # (N_total, 3)
        all_true = torch.cat(all_true, dim=0)

        metrics = _compute_field_metrics(all_pred, all_true)
        return total / max(n, 1), metrics

    # ── Utilities ──────────────────────────────────────────────────────

    def load_best(self) -> None:
        ckpt = torch.load(self.out_dir / "best.pt",
                          map_location=self.device, weights_only=False)
        # Handle DataParallel wrapper (legacy, but safe to keep just in case)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint  (epoch={ckpt['epoch']}, "
              f"val_loss={ckpt['val_loss']:.6f})")

    def evaluate(self, loader: DataLoader) -> Dict:
        """Evaluate on any DataLoader; return absolute-unit metrics."""
        self.model.eval()
        all_pred, all_true = [], []
        y_mean = self.norm_stats["y_mean"].to(self.device)
        y_std  = self.norm_stats["y_std"].to(self.device)

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                pred = self.model(data) * y_std + y_mean
                true = data.y          * y_std + y_mean
                all_pred.append(pred.cpu())
                all_true.append(true.cpu())

        return _compute_field_metrics(
            torch.cat(all_pred, 0),
            torch.cat(all_true, 0),
        )

    def _save_checkpoint(self, name: str, epoch: int, val_loss: float) -> None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save({
            "epoch":       epoch,
            "val_loss":    val_loss,
            "model_state": model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "sched_state": self.scheduler.state_dict(),
            "cfg":         self.cfg,
            "history":     self.history,
        }, self.out_dir / name)

    def _resume(self) -> int:
        ckpts = sorted(self.out_dir.glob("epoch_*.pt"))
        if not ckpts:
            return 0
        ckpt = torch.load(ckpts[-1], map_location=self.device, weights_only=False)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self.scheduler.load_state_dict(ckpt["sched_state"])
        self.best_val_loss = ckpt.get("val_loss", float("inf"))
        self.history = ckpt.get("history", self.history)
        epoch = ckpt["epoch"]
        print(f"[Resume] Resumed from epoch {epoch}")
        return epoch

    def _open_csv(self) -> None:
        fields = ["epoch", "train_loss", "val_loss", "lr",
                  "val_rmse_w", "val_rmse_sx", "val_rmse_sy",
                  "val_r2_w",   "val_r2_sx",   "val_r2_sy"]
        self._csv_file   = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fields)
        self._csv_writer.writeheader()

    def _close_csv(self) -> None:
        if self._csv_file:
            self._csv_file.close()


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_field_metrics(
    pred: torch.Tensor,  # (N, 3)  absolute units
    true: torch.Tensor,  # (N, 3)
) -> Dict[str, float]:
    """RMSE and R² per output field, computed in absolute physical units."""
    names = ["w", "sx", "sy"]
    out   = {}
    for i, nm in enumerate(names):
        p = pred[:, i]
        t = true[:, i]
        rmse = torch.sqrt(((p - t) ** 2).mean()).item()
        ss_res = ((p - t) ** 2).sum().item()
        ss_tot = ((t - t.mean()) ** 2).sum().item()
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        out[f"val_rmse_{nm}"] = rmse
        out[f"val_r2_{nm}"]   = r2
    return out
