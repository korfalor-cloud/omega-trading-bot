"""
CandleStick Transformer — Custom Trading LLM

A from-scratch transformer language model that tokenizes candlestick (OHLCV)
data and learns to predict next candles, trade signals, and confidence scores.

Architecture:
  - Hybrid tokenizer: OHLCV bins + pattern tokens + indicator tokens
  - 12-layer transformer with multi-scale attention
  - Multi-task output heads: next candle, signal, confidence

Usage:
    from tradingbot.llm import (
        CandleStickTokenizer,
        CandleTransformer,
        TradingHeads,
        CandleTrainer,
        CandlePredictor,
    )
"""

from .config import (
    DataConfig,
    ModelConfig,
    TokenizerConfig,
    TradingConfig,
    TrainingConfig,
    get_data_config,
    get_model_config,
    get_tokenizer_config,
    get_trading_config,
    get_training_config,
)
from .heads import ConfidenceHead, NextCandleHead, TradeSignalHead, TradingHeads, TradingLoss
from .inference import CandlePredictor, format_prediction
from .model import CandleTransformer
from .tokenizer import CandleStickTokenizer
from .trainer import CandleTrainer

__all__ = [
    # Config
    "TokenizerConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "TradingConfig",
    "get_tokenizer_config",
    "get_model_config",
    "get_training_config",
    "get_data_config",
    "get_trading_config",
    # Model
    "CandleTransformer",
    "CandleStickTokenizer",
    # Heads
    "TradingHeads",
    "TradingLoss",
    "NextCandleHead",
    "TradeSignalHead",
    "ConfidenceHead",
    # Training
    "CandleTrainer",
    # Inference
    "CandlePredictor",
    "format_prediction",
]
