"""
Train the CandleStick Transformer on real BTC data from Binance.
CPU-optimized configuration.
"""

import sys
import time
import numpy as np

sys.path.insert(0, "src")

from tradingbot.llm.config import ModelConfig, TrainingConfig, TokenizerConfig
from tradingbot.llm.tokenizer import CandleStickTokenizer
from tradingbot.llm.model import CandleTransformer
from tradingbot.llm.heads import TradingHeads
from tradingbot.llm.trainer import CandleTrainer
from tradingbot.llm.data.fetcher import BinanceFetcher
from tradingbot.llm.data.dataset import create_train_val_test_split


def main():
    print("=" * 60)
    print("  CandleStick Transformer — Training")
    print("=" * 60)

    # ── Fetch Data ──
    print("\n[1/5] Fetching BTC/USDT 1h candles from Binance...")
    t0 = time.time()
    fetcher = BinanceFetcher()

    try:
        candles = fetcher.fetch_all(symbol="BTCUSDT", interval="1h", days=60)
        print(f"  Fetched {len(candles)} candles in {time.time() - t0:.1f}s")
        print(f"  Date range: {candles[0, 5]:.0f} to {candles[-1, 5]:.0f}")
        print(f"  Price range: ${candles[:, 3].min():.2f} - ${candles[:, 3].max():.2f}")
    except Exception as e:
        print(f"  Binance fetch failed: {e}")
        print("  Generating synthetic data instead...")
        candles = generate_synthetic_data(2000)
        print(f"  Generated {len(candles)} synthetic candles")

    # ── Split Data ──
    print("\n[2/5] Splitting data...")
    train, val, test = create_train_val_test_split(candles, val_split=0.15, test_split=0.1)
    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # ── Model Setup (CPU-optimized) ──
    print("\n[3/5] Building model...")
    tok_cfg = TokenizerConfig()

    # Small model for CPU training (fits in ~2GB RAM)
    model_cfg = ModelConfig(
        VOCAB_SIZE=tok_cfg.vocab_size,
        D_MODEL=128,       # small embedding
        N_HEADS=2,         # 2 attention heads
        N_LAYERS=3,        # 3 transformer layers
        D_FF=256,          # small FFN
        DROPOUT=0.1,
        MAX_SEQ_LEN=512,
        LOCAL_WINDOW=8,
        MAX_RELATIVE_POSITION=32,
    )

    tokenizer = CandleStickTokenizer(tok_cfg)
    model = CandleTransformer(model_cfg)
    heads = TradingHeads(model_cfg)

    print(f"  Vocab size: {tok_cfg.vocab_size}")
    print(f"  Model: {model_cfg.N_LAYERS}L / {model_cfg.N_HEADS}H / {model_cfg.D_MODEL}d")
    print(f"  Parameters: {model.count_parameters():,}")

    # ── Training Config ──
    train_cfg = TrainingConfig()
    train_cfg.EPOCHS = 30
    train_cfg.BATCH_SIZE = 8
    train_cfg.LEARNING_RATE = 3e-4
    train_cfg.WARMUP_STEPS = 100
    train_cfg.WINDOW_SIZE = 30
    train_cfg.CHECKPOINT_DIR = "checkpoints/candle_llm"
    train_cfg.USE_FP16 = False  # CPU doesn't support fp16
    train_cfg.SAVE_EVERY = 10
    train_cfg.GRAD_CLIP = 1.0

    # ── Train ──
    print("\n[4/5] Training...")
    print(f"  Epochs: {train_cfg.EPOCHS}")
    print(f"  Batch size: {train_cfg.BATCH_SIZE}")
    print(f"  Learning rate: {train_cfg.LEARNING_RATE}")
    print(f"  Window size: {train_cfg.WINDOW_SIZE}")
    print()

    trainer = CandleTrainer(model, heads, tokenizer, config=train_cfg, device="cpu")
    history = trainer.fit(train, val, timeframe="1h")

    # ── Results ──
    print("\n[5/5] Training complete!")
    final_train = history["train"][-1]["total"]
    final_val = history["val"][-1]["total"] if history["val"] else 0
    print(f"  Final train loss: {final_train:.4f}")
    print(f"  Final val loss:   {final_val:.4f}")

    best_train = min(h["total"] for h in history["train"])
    print(f"  Best train loss:  {best_train:.4f}")

    if history["val"]:
        best_val = min(h["total"] for h in history["val"])
        print(f"  Best val loss:    {best_val:.4f}")
        best_acc = max(h.get("signal_accuracy", 0) for h in history["val"])
        print(f"  Best val signal accuracy: {best_acc:.3f}")

    print(f"\n  Checkpoints saved to: {train_cfg.CHECKPOINT_DIR}/")


def generate_synthetic_data(n: int) -> np.ndarray:
    """Generate realistic synthetic candle data."""
    np.random.seed(42)
    price = 40000.0  # BTC-like starting price
    candles = []
    for i in range(n):
        o = price
        # Random walk with mean reversion
        drift = np.random.normal(0, 0.015) * price
        c = o + drift
        h = max(o, c) * (1 + abs(np.random.normal(0, 0.005)))
        l = min(o, c) * (1 - abs(np.random.normal(0, 0.005)))
        v = abs(np.random.normal(500, 100))
        candles.append([o, h, l, c, v, float(i * 3600000)])
        price = c
    return np.array(candles, dtype=np.float64)


if __name__ == "__main__":
    main()
