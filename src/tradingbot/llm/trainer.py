"""
CandleStick Transformer — Training Loop
Multi-task training with AdamW, cosine scheduling, mixed precision.
"""

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader

from .config import (
    ModelConfig,
    TrainingConfig,
    TokenizerConfig,
    get_model_config,
    get_tokenizer_config,
    get_training_config,
)
from .data.dataset import CandleDataset, create_train_val_test_split
from .heads import TradingHeads, TradingLoss
from .model import CandleTransformer
from .tokenizer import CandleStickTokenizer


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            scale = self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = max(self.min_lr / self.base_lrs[0], 0.5 * (1.0 + math.cos(math.pi * progress)))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class CandleTrainer:
    """
    Trainer for the CandleStick Transformer.

    Handles:
      - Multi-task loss computation
      - AdamW optimization with weight decay
      - Cosine warmup learning rate scheduling
      - Mixed precision training (FP16)
      - Gradient clipping
      - Checkpointing
      - Validation
      - Logging
    """

    def __init__(
        self,
        model: CandleTransformer,
        heads: TradingHeads,
        tokenizer: CandleStickTokenizer,
        config: TrainingConfig = None,
        device: str = None,
    ):
        self.model = model
        self.heads = heads
        self.tokenizer = tokenizer
        self.config = config or get_training_config()

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.heads = self.heads.to(self.device)

        # Loss function
        self.loss_fn = TradingLoss(
            alpha=self.config.ALPHA_CANDLE,
            beta=self.config.BETA_SIGNAL,
            gamma=self.config.GAMMA_CONFIDENCE,
        ).to(self.device)

        # Mixed precision
        self.use_fp16 = self.config.USE_FP16 and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.use_fp16)

        # Logging
        self.train_history = []
        self.val_history = []
        self.global_step = 0

    def _setup_optimizer(self, total_steps: int):
        """Setup optimizer and scheduler."""
        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        for name, param in self.heads.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(
            param_groups,
            lr=self.config.LEARNING_RATE,
            betas=(self.config.BETA1, self.config.BETA2),
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=self.config.WARMUP_STEPS,
            total_steps=total_steps,
            min_lr=self.config.MIN_LR,
        )

    def _compute_confidence_target(self, candles: np.ndarray, target_idx: int) -> float:
        """Compute confidence target based on realized PnL."""
        if target_idx >= len(candles) - 1:
            return 0.0

        entry = candles[target_idx, 3]  # close price as entry
        # Look ahead up to 5 candles for exit
        exit_idx = min(target_idx + 5, len(candles) - 1)
        exit_price = candles[exit_idx, 3]

        if entry == 0:
            return 0.0

        pnl_pct = abs(exit_price - entry) / entry
        return min(1.0, pnl_pct / 0.05)  # 5% move = max confidence

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.heads.train()

        epoch_losses = {
            "total": 0.0,
            "candle": 0.0,
            "signal": 0.0,
            "confidence": 0.0,
        }
        n_batches = 0

        for batch in dataloader:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_candle = batch["target_candle"].to(self.device)
            target_signal = batch["target_signal"].to(self.device)
            target_confidence = batch["target_confidence"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            with autocast(self.device.type, enabled=self.use_fp16):
                hidden = self.model(input_ids, attention_mask)
                predictions = self.heads(hidden)

                targets = {
                    "next_candle": target_candle,
                    "trade_signal": target_signal,
                    "confidence": target_confidence,
                }
                losses = self.loss_fn(predictions, targets)

            # Backward pass
            self.scaler.scale(losses["total_loss"]).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.heads.parameters()),
                self.config.GRAD_CLIP,
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Accumulate losses
            epoch_losses["total"] += losses["total_loss"].item()
            epoch_losses["candle"] += losses["candle_loss"].item()
            epoch_losses["signal"] += losses["signal_loss"].item()
            epoch_losses["confidence"] += losses["confidence_loss"].item()
            n_batches += 1
            self.global_step += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= max(1, n_batches)

        return epoch_losses

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        self.heads.eval()

        val_losses = {"total": 0.0, "candle": 0.0, "signal": 0.0, "confidence": 0.0}
        correct_signals = 0
        total_signals = 0
        n_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_candle = batch["target_candle"].to(self.device)
            target_signal = batch["target_signal"].to(self.device)
            target_confidence = batch["target_confidence"].to(self.device)

            with autocast(self.device.type, enabled=self.use_fp16):
                hidden = self.model(input_ids, attention_mask)
                predictions = self.heads(hidden)

                targets = {
                    "next_candle": target_candle,
                    "trade_signal": target_signal,
                    "confidence": target_confidence,
                }
                losses = self.loss_fn(predictions, targets)

            val_losses["total"] += losses["total_loss"].item()
            val_losses["candle"] += losses["candle_loss"].item()
            val_losses["signal"] += losses["signal_loss"].item()
            val_losses["confidence"] += losses["confidence_loss"].item()

            # Signal accuracy
            pred_signals = predictions["trade_signal"].argmax(dim=1)
            correct_signals += (pred_signals == target_signal).sum().item()
            total_signals += target_signal.size(0)

            n_batches += 1

        for key in val_losses:
            val_losses[key] /= max(1, n_batches)

        val_losses["signal_accuracy"] = correct_signals / max(1, total_signals)
        return val_losses

    def save_checkpoint(self, path: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "heads_state_dict": self.heads.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "global_step": self.global_step,
            "config": {
                "model": self.model.config.__dict__,
                "training": self.config.__dict__,
            },
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.heads.load_state_dict(checkpoint["heads_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint.get("epoch", 0)

    def fit(
        self,
        train_candles: np.ndarray,
        val_candles: np.ndarray,
        timeframe: str = "1h",
    ) -> Dict[str, list]:
        """
        Full training loop.

        Args:
            train_candles: Training OHLCV data (N, 6)
            val_candles: Validation OHLCV data (M, 6)
            timeframe: Candle timeframe

        Returns:
            Training history dict
        """
        # Create datasets
        train_dataset = CandleDataset(
            train_candles, self.tokenizer, self.config, timeframe, augment=True
        )
        val_dataset = CandleDataset(
            val_candles, self.tokenizer, self.config, timeframe, augment=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        # Setup optimizer
        total_steps = len(train_loader) * self.config.EPOCHS
        self._setup_optimizer(total_steps)

        # Training loop
        best_val_loss = float("inf")
        checkpoint_dir = Path(self.config.CHECKPOINT_DIR)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training on {self.device}")
        print(f"  Model parameters: {self.model.count_parameters():,}")
        print(f"  Train samples: {len(train_dataset):,}")
        print(f"  Val samples: {len(val_dataset):,}")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  Epochs: {self.config.EPOCHS}")
        print(f"  Total steps: {total_steps:,}")
        print()

        for epoch in range(1, self.config.EPOCHS + 1):
            t0 = time.time()

            # Train
            train_losses = self.train_epoch(train_loader)
            self.train_history.append(train_losses)

            # Validate
            val_losses = self.validate(val_loader)
            self.val_history.append(val_losses)

            elapsed = time.time() - t0
            lr = self.scheduler.get_lr()

            # Log
            print(
                f"Epoch {epoch:3d}/{self.config.EPOCHS} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
                f"Signal Acc: {val_losses['signal_accuracy']:.3f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                self.save_checkpoint(
                    str(checkpoint_dir / "best_model.pt"),
                    epoch,
                    best_val_loss,
                )

            # Periodic checkpoint
            if epoch % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(
                    str(checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"),
                    epoch,
                    val_losses["total"],
                )

        # Save final model
        self.save_checkpoint(
            str(checkpoint_dir / "final_model.pt"),
            self.config.EPOCHS,
            val_losses["total"],
        )

        # Save training history
        history = {
            "train": self.train_history,
            "val": self.val_history,
        }
        with open(checkpoint_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
        return history
