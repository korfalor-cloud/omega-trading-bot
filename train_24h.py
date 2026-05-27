"""
CandleStick Transformer — 24-Hour Continuous Training
Runs non-stop for 24 hours, cycling through BTC data.
Larger model than previous run, periodic re-fetch of fresh data.
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
from tradingbot.llm.data.fetcher import BinanceFetcher
from tradingbot.llm.data.dataset import create_train_val_test_split, CandleDataset
from torch.utils.data import DataLoader


RUN_SECONDS = 24 * 60 * 60  # 24 hours
CHECKPOINT_DIR = "checkpoints/candle_llm_24h"
LOG_FILE = "checkpoints/candle_llm_24h/training_log.txt"


def log(msg: str):
    """Write to both stdout and log file."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def fetch_data() -> np.ndarray:
    """Fetch fresh BTC data from Binance."""
    fetcher = BinanceFetcher()
    # Fetch last 90 days of 1h candles (up to 2160 candles)
    candles = fetcher.fetch_all(symbol="BTCUSDT", interval="1h", days=90)
    return candles


def build_model(tok_cfg: TokenizerConfig):
    """Build medium-sized model for extended training."""
    model_cfg = ModelConfig(
        VOCAB_SIZE=tok_cfg.vocab_size,
        D_MODEL=192,       # medium embedding
        N_HEADS=4,         # 4 heads
        N_LAYERS=5,        # 5 transformer layers
        D_FF=384,          # medium FFN
        DROPOUT=0.1,
        MAX_SEQ_LEN=512,
        LOCAL_WINDOW=8,
        MAX_RELATIVE_POSITION=48,
    )
    return model_cfg


def train_round(
    candles: np.ndarray,
    tokenizer: CandleStickTokenizer,
    model_cfg: ModelConfig,
    round_num: int,
    epochs: int,
    batch_size: int,
    lr: float,
    start_time: float,
    resume_checkpoint: str = None,
) -> str:
    """Run one training round. Returns path to best checkpoint."""
    log(f"--- Round {round_num} ---")
    log(f"Data: {len(candles)} candles")

    # Split
    train, val, test = create_train_val_test_split(candles, 0.15, 0.1)
    log(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    # Build fresh model for each round (clean gradients)
    model = CandleTransformer(model_cfg)
    heads = TradingHeads(model_cfg)
    log(f"Parameters: {model.count_parameters():,}")

    # Config
    cfg = TrainingConfig()
    cfg.EPOCHS = epochs
    cfg.BATCH_SIZE = batch_size
    cfg.LEARNING_RATE = lr
    cfg.WARMUP_STEPS = min(200, len(train) // batch_size * 2)
    cfg.WINDOW_SIZE = 32
    cfg.CHECKPOINT_DIR = f"{CHECKPOINT_DIR}/round_{round_num}"
    cfg.USE_FP16 = False
    cfg.SAVE_EVERY = max(1, epochs // 5)
    cfg.GRAD_CLIP = 1.0

    # Resume from previous round's best if available
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        try:
            ckpt = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            heads.load_state_dict(ckpt["heads_state_dict"])
            log(f"Resumed from {resume_checkpoint}")
        except Exception as e:
            log(f"Resume failed: {e}, starting fresh")

    trainer = CandleTrainer(model, heads, tokenizer, config=cfg, device="cpu")
    history = trainer.fit(train, val, timeframe="1h")

    # Stats
    if history["train"]:
        best_train = min(h["total"] for h in history["train"])
        best_val = min(h["total"] for h in history["val"]) if history["val"] else 0
        log(f"Round {round_num} done. Best train={best_train:.4f} val={best_val:.4f}")

    best_ckpt = f"{cfg.CHECKPOINT_DIR}/best_model.pt"

    # Cleanup
    del model, heads, trainer
    gc.collect()

    return best_ckpt


def main():
    start_time = time.time()
    end_time = start_time + RUN_SECONDS

    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  CandleStick Transformer — 24-Hour Training")
    log("=" * 60)
    log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"End target: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Device: CPU (2 cores)")
    log("")

    tok_cfg = TokenizerConfig()
    model_cfg = build_model(tok_cfg)
    tokenizer = CandleStickTokenizer(tok_cfg)

    log(f"Vocab size: {tok_cfg.vocab_size}")
    log(f"Model: {model_cfg.N_LAYERS}L / {model_cfg.N_HEADS}H / {model_cfg.D_MODEL}d / {model_cfg.D_FF}ff")
    tmp_model = CandleTransformer(model_cfg)
    log(f"Parameters: {tmp_model.count_parameters():,}")
    del tmp_model
    gc.collect()

    # Training schedule: multiple rounds with increasing data/epochs
    round_num = 0
    best_checkpoint = None
    all_results = []

    while time.time() < end_time:
        round_num += 1
        elapsed_hours = (time.time() - start_time) / 3600
        remaining_hours = (end_time - time.time()) / 3600

        log(f"\n{'='*50}")
        log(f"Round {round_num} | Elapsed: {elapsed_hours:.1f}h | Remaining: {remaining_hours:.1f}h")
        log(f"{'='*50}")

        # Fetch fresh data each round
        try:
            log("Fetching fresh BTC/USDT data from Binance...")
            t0 = time.time()
            candles = fetch_data()
            log(f"Fetched {len(candles)} candles in {time.time()-t0:.1f}s")
            log(f"Price range: ${candles[:, 3].min():.0f} - ${candles[:, 3].max():.0f}")
        except Exception as e:
            log(f"Fetch failed: {e}")
            log("Generating synthetic data...")
            candles = generate_synthetic(2000)

        # Adjust training params based on remaining time
        if remaining_hours > 16:
            epochs, batch_size, lr = 40, 8, 5e-4
        elif remaining_hours > 8:
            epochs, batch_size, lr = 30, 8, 3e-4
        elif remaining_hours > 4:
            epochs, batch_size, lr = 20, 8, 2e-4
        elif remaining_hours > 1:
            epochs, batch_size, lr = 15, 4, 1e-4
        else:
            epochs, batch_size, lr = 10, 4, 5e-5

        log(f"Config: {epochs} epochs, batch={batch_size}, lr={lr}")

        try:
            best_ckpt = train_round(
                candles, tokenizer, model_cfg, round_num,
                epochs, batch_size, lr, start_time,
                resume_checkpoint=best_checkpoint,
            )
            best_checkpoint = best_ckpt

            # Run inference test
            log("\nRunning inference test...")
            test_inference(best_checkpoint, candles)

            result = {
                "round": round_num,
                "elapsed_hours": round((time.time() - start_time) / 3600, 2),
                "candles": len(candles),
                "checkpoint": best_ckpt,
            }
            all_results.append(result)

            # Save summary
            with open(f"{CHECKPOINT_DIR}/rounds_summary.json", "w") as f:
                json.dump(all_results, f, indent=2)

        except Exception as e:
            log(f"Round {round_num} failed: {e}")
            import traceback
            traceback.print_exc()

        gc.collect()

    # Final summary
    total_time = (time.time() - start_time) / 3600
    log(f"\n{'='*60}")
    log(f"  TRAINING COMPLETE — {total_time:.1f} hours, {round_num} rounds")
    log(f"  Best checkpoint: {best_checkpoint}")
    log(f"{'='*60}")


def test_inference(checkpoint_path: str, candles: np.ndarray):
    """Quick inference test after training."""
    from tradingbot.llm.inference import CandlePredictor, format_prediction

    try:
        predictor = CandlePredictor(checkpoint_path, device="cpu")
        test_candles = candles[-32:]
        pred = predictor.predict(test_candles, timeframe="1h")
        log(f"Inference: {pred['signal']} (conf={pred['confidence']:.2%})")
        log(f"  BUY={pred['signal_probs']['BUY']:.2%} SELL={pred['signal_probs']['SELL']:.2%} HOLD={pred['signal_probs']['HOLD']:.2%}")
        log(f"  Next candle: O={pred['next_candle']['open']:.0f} C={pred['next_candle']['close']:.0f} H={pred['next_candle']['high']:.0f} L={pred['next_candle']['low']:.0f}")
    except Exception as e:
        log(f"Inference test failed: {e}")


def generate_synthetic(n: int) -> np.ndarray:
    np.random.seed(int(time.time()))
    price = 70000.0
    candles = []
    for i in range(n):
        o = price
        c = o + np.random.normal(0, 0.015) * price
        h = max(o, c) * (1 + abs(np.random.normal(0, 0.005)))
        l = min(o, c) * (1 - abs(np.random.normal(0, 0.005)))
        v = abs(np.random.normal(500, 100))
        candles.append([o, h, l, c, v, float(i * 3600000)])
        price = c
    return np.array(candles, dtype=np.float64)


if __name__ == "__main__":
    main()
