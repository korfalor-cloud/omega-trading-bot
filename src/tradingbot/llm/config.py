"""
CandleStick Transformer — Configuration
All hyperparameters in one place.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TokenizerConfig:
    # Price binning
    PRICE_BIN_COUNT: int = 1024          # bins per price channel (high/low/close)
    PRICE_CHANGE_RANGE: float = 0.10    # ±10% max price change from open
    VOLUME_BIN_COUNT: int = 256          # logarithmic volume bins

    # Pattern vocabulary
    PATTERN_COUNT: int = 128             # candlestick pattern token slots
    PATTERN_OFFSET: int = 0              # start index for pattern tokens

    # Timeframe tokens
    TIMEFRAME_COUNT: int = 16
    TIMEFRAME_OFFSET: int = 128

    # Indicator tokens
    INDICATOR_BIN_COUNT: int = 256
    INDICATOR_OFFSET: int = 144

    # Price bin tokens
    PRICE_OFFSET: int = 400

    # Special tokens
    PAD_TOKEN: int = 0
    BOS_TOKEN: int = 1
    EOS_TOKEN: int = 2
    MASK_TOKEN: int = 3
    SEP_TOKEN: int = 4
    SPECIAL_COUNT: int = 16

    @property
    def vocab_size(self) -> int:
        return (
            self.SPECIAL_COUNT
            + self.TIMEFRAME_COUNT
            + self.PATTERN_COUNT
            + self.INDICATOR_BIN_COUNT
            + self.PRICE_BIN_COUNT * 3  # high, low, close relative to open
            + self.VOLUME_BIN_COUNT
        )

    # Tokens per single candle
    TOKENS_PER_CANDLE: int = 8  # timeframe + close_bin + high_bin + low_bin + vol_bin + pattern + 2 indicator


@dataclass
class ModelConfig:
    # Transformer architecture
    VOCAB_SIZE: int = 0                  # filled from TokenizerConfig
    D_MODEL: int = 512                   # embedding dimension
    N_HEADS: int = 8                     # attention heads
    N_LAYERS: int = 12                   # transformer blocks
    D_FF: int = 2048                     # feed-forward hidden dim
    DROPOUT: float = 0.1
    MAX_SEQ_LEN: int = 512              # max tokens in context

    # Multi-scale attention
    LOCAL_WINDOW: int = 10               # local attention window size

    # Relative position encoding
    MAX_RELATIVE_POSITION: int = 128

    # Output heads
    SIGNAL_CLASSES: int = 3              # BUY, SELL, HOLD
    CONFIDENCE_DIM: int = 1              # scalar confidence


@dataclass
class TrainingConfig:
    # Optimization
    LEARNING_RATE: float = 3e-4
    WEIGHT_DECAY: float = 0.01
    BETA1: float = 0.9
    BETA2: float = 0.95
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    GRAD_ACCUM_STEPS: int = 1
    GRAD_CLIP: float = 1.0

    # Scheduler
    WARMUP_STEPS: int = 1000
    MIN_LR: float = 1e-5

    # Multi-task loss weights
    ALPHA_CANDLE: float = 1.0            # next candle prediction
    BETA_SIGNAL: float = 2.0             # trade signal
    GAMMA_CONFIDENCE: float = 0.5        # confidence score

    # Data
    WINDOW_SIZE: int = 64                # candles per input sequence
    STRIDE: int = 1                      # sliding window stride
    VAL_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.1

    # Checkpointing
    CHECKPOINT_DIR: str = "checkpoints"
    SAVE_EVERY: int = 5                  # epochs
    LOG_EVERY: int = 100                 # steps

    # Mixed precision
    USE_FP16: bool = True


@dataclass
class DataConfig:
    # Binance
    BINANCE_BASE_URL: str = "https://api.binance.com/api/v3"
    BINANCE_DEFAULT_SYMBOL: str = "BTCUSDT"
    BINANCE_DEFAULT_INTERVAL: str = "1h"
    BINANCE_MAX_LIMIT: int = 1000

    # Yahoo Finance
    YAHOO_DEFAULT_TICKER: str = "BTC-USD"
    YAHOO_DEFAULT_INTERVAL: str = "1h"

    # Supported timeframes
    TIMEFRAMES: List[str] = field(
        default_factory=lambda: [
            "1m", "3m", "5m", "15m", "30m",
            "1h", "2h", "4h", "6h", "8h", "12h",
            "1d", "3d", "1w", "1M",
        ]
    )


@dataclass
class TradingConfig:
    # Inference
    MIN_CONFIDENCE: float = 0.6          # minimum confidence to act
    MIN_SIGNAL_PROB: float = 0.55        # minimum signal probability
    CONTEXT_CANDLES: int = 64            # candles to feed model

    # Risk
    MAX_POSITION_SIZE: float = 0.1       # 10% of portfolio per trade
    STOP_LOSS_PCT: float = 0.02          # 2% stop loss
    TAKE_PROFIT_PCT: float = 0.04        # 4% take profit


def get_tokenizer_config() -> TokenizerConfig:
    return TokenizerConfig()


def get_model_config(tok_config: TokenizerConfig = None) -> ModelConfig:
    if tok_config is None:
        tok_config = get_tokenizer_config()
    cfg = ModelConfig()
    cfg.VOCAB_SIZE = tok_config.vocab_size
    return cfg


def get_training_config() -> TrainingConfig:
    return TrainingConfig()


def get_data_config() -> DataConfig:
    return DataConfig()


def get_trading_config() -> TradingConfig:
    return TradingConfig()
