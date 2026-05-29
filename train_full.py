"""
CandleStick Transformer — Full Training with 1 Year Multi-Timeframe Data
Anti-overfitting: label smoothing, early stopping, dropout=0.2, weight_decay=0.05
"""

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")

from tradingbot.llm.config import ModelConfig, TokenizerConfig, TrainingConfig
from tradingbot.llm.tokenizer import CandleStickTokenizer
from tradingbot.llm.model import CandleTransformer
from tradingbot.llm.heads import TradingHeads
from tradingbot.llm.trainer import CandleTrainer
from tradingbot.llm.data.dataset import CandleDataset, create_train_val_test_split
from torch.utils.data import DataLoader


DATA_DIR = "data/btc_multi_tf"
CHECKPOINT_DIR = "checkpoints/full_1y"
LOG_FILE = f"{CHECKPOINT_DIR}/training_log.txt"

# Timeframe configs: (file, interval_name, epochs, batch, lr)
SCHEDULE = [
    ("btc_1d.npy",   "1d",  50, 8,  3e-4),   # Daily first (smallest, fastest)
    ("btc_4h.npy",   "4h",  40, 8,  2e-4),   # 4-hour
    ("btc_1h.npy",   "1h",  30, 8,  1.5e-4), # 1-hour
    ("btc_15m.npy",  "15m", 25, 4,  1e-4),   # 15-min
    ("btc_5m.npy",   "5m",  20, 4,  8e-5),   # 5-min
    ("btc_1m.npy",   "1m",  15, 4,  5e-5),   # 1-min (most data, careful)
]


def log(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def load_data(filename: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, filename)
    candles = np.load(path)
    log(f"  Loaded {filename}: {len(candles)} candles, "
        f"price ${candles[:, 3].min():.0f}-${candles[:, 3].max():.0f}")
    return candles


def main():
    start_time = time.time()
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  CandleStick Transformer — Full 1-Year Training")
    log("=" * 60)
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Data: {DATA_DIR}")
    log("")

    # Model config — full 46M param model
    tok_cfg = TokenizerConfig()
    model_cfg = ModelConfig(
        VOCAB_SIZE=tok_cfg.vocab_size,
        D_MODEL=512,
        N_HEADS=8,
        N_LAYERS=12,
        D_FF=2048,
        DROPOUT=0.2,
        MAX_SEQ_LEN=512,
        LOCAL_WINDOW=10,
        MAX_RELATIVE_POSITION=128,
    )

    tokenizer = CandleStickTokenizer(tok_cfg)
    model = CandleTransformer(model_cfg)
    heads = TradingHeads(model_cfg)

    log(f"Vocab size: {tok_cfg.vocab_size}")
    log(f"Model: {model_cfg.N_LAYERS}L / {model_cfg.N_HEADS}H / {model_cfg.D_MODEL}d / {model_cfg.D_FF}ff")
    log(f"Parameters: {model.count_parameters():,}")
    log(f"Dropout: {model_cfg.DROPOUT}")
    log(f"Weight decay: 0.05")
    log(f"Label smoothing: 0.1")
    log(f"Early stopping: patience=15")
    log("")

    best_checkpoint = None
    all_results = []

    for file_name, tf_name, epochs, batch_size, lr in SCHEDULE:
        elapsed_hours = (time.time() - start_time) / 3600
        log(f"\n{'='*50}")
        log(f"  Training on {tf_name} data | Elapsed: {elapsed_hours:.1f}h")
        log(f"{'='*50}")

        # Load data
        candles = load_data(file_name)

        # Split
        train, val, test = create_train_val_test_split(candles, 0.15, 0.1)
        log(f"  Split: train={len(train)} val={len(val)} test={len(test)}")

        # Training config
        cfg = TrainingConfig()
        cfg.EPOCHS = epochs
        cfg.BATCH_SIZE = batch_size
        cfg.LEARNING_RATE = lr
        cfg.WARMUP_STEPS = min(300, len(train) // batch_size * 3)
        cfg.WINDOW_SIZE = 32
        cfg.CHECKPOINT_DIR = f"{CHECKPOINT_DIR}/{tf_name}"
        cfg.USE_FP16 = False  # CPU
        cfg.SAVE_EVERY = max(1, epochs // 5)
        cfg.GRAD_CLIP = 1.0

        # Fresh model for each timeframe (transfer learning from previous best)
        model = CandleTransformer(model_cfg)
        heads = TradingHeads(model_cfg)

        if best_checkpoint and os.path.exists(best_checkpoint):
            try:
                ckpt = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                heads.load_state_dict(ckpt["heads_state_dict"])
                log(f"  Resumed from {best_checkpoint}")
            except Exception as e:
                log(f"  Resume failed: {e}, starting fresh")

        trainer = CandleTrainer(model, heads, tokenizer, config=cfg, device="cpu")
        history = trainer.fit(train, val, timeframe=tf_name)

        # Stats
        if history["train"]:
            best_train = min(h["total"] for h in history["train"])
            best_val = min(h["total"] for h in history["val"]) if history["val"] else 0
            best_acc = max(h.get("signal_accuracy", 0) for h in history["val"])
            log(f"  {tf_name} done. Train={best_train:.4f} Val={best_val:.4f} Acc={best_acc:.3f}")

        ckpt_path = f"{cfg.CHECKPOINT_DIR}/best_model.pt"
        if os.path.exists(ckpt_path):
            best_checkpoint = ckpt_path

        result = {
            "timeframe": tf_name,
            "candles": len(candles),
            "epochs": len(history["train"]),
            "best_train": best_train,
            "best_val": best_val,
            "best_acc": best_acc,
            "elapsed_hours": round((time.time() - start_time) / 3600, 2),
        }
        all_results.append(result)

        with open(f"{CHECKPOINT_DIR}/training_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

        del model, heads, trainer
        gc.collect()

    # Final summary
    total_time = (time.time() - start_time) / 3600
    log(f"\n{'='*60}")
    log(f"  TRAINING COMPLETE — {total_time:.1f} hours")
    log(f"  Best checkpoint: {best_checkpoint}")
    log(f"{'='*60}")

    for r in all_results:
        log(f"  {r['timeframe']:>4s}: {r['candles']:>6d} candles, "
            f"{r['epochs']} epochs, val={r['best_val']:.4f}, acc={r['best_acc']:.3f}")


if __name__ == "__main__":
    main()
