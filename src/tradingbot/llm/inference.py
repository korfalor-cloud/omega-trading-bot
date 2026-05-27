"""
CandleStick Transformer — Inference Pipeline
Load a trained model and generate trading signals from live candlestick data.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import (
    ModelConfig,
    TradingConfig,
    get_model_config,
    get_tokenizer_config,
    get_trading_config,
)
from .data.fetcher import BinanceFetcher, YahooFetcher
from .heads import TradingHeads
from .model import CandleTransformer
from .tokenizer import CandleStickTokenizer


class CandlePredictor:
    """
    Inference engine for the CandleStick Transformer.

    Loads a trained model and generates:
      - Next candle prediction (OHLCV bins)
      - Trade signal (BUY/SELL/HOLD)
      - Confidence score (0-1)
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
        trading_config: TradingConfig = None,
    ):
        self.trading_config = trading_config or get_trading_config()

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Reconstruct model from saved config
        model_cfg_dict = checkpoint["config"]["model"]
        self.model_config = ModelConfig(**model_cfg_dict)

        # Initialize tokenizer
        self.tokenizer = CandleStickTokenizer(get_tokenizer_config())

        # Initialize model and heads
        self.model = CandleTransformer(self.model_config).to(self.device)
        self.heads = TradingHeads(self.model_config).to(self.device)

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.heads.load_state_dict(checkpoint["heads_state_dict"])

        self.model.eval()
        self.heads.eval()

    @torch.no_grad()
    def predict(
        self,
        candles: np.ndarray,
        timeframe: str = "1h",
    ) -> Dict[str, any]:
        """
        Generate predictions from a sequence of candles.

        Args:
            candles: OHLCV array of shape (N, 6)
            timeframe: Candle timeframe

        Returns:
            Dict with:
              - "signal": "BUY" / "SELL" / "HOLD"
              - "signal_probs": {"BUY": float, "SELL": float, "HOLD": float}
              - "confidence": float (0-1)
              - "next_candle": {"close": float, "high": float, "low": float}
        """
        # Tokenize
        tokens = self.tokenizer.tokenize_sequence(
            candles,
            timeframe=timeframe,
            add_bos=True,
            add_eos=False,
        )

        # Pad/truncate to model max length
        max_len = self.model_config.MAX_SEQ_LEN
        tokens = self.tokenizer.pad_sequence(tokens, max_len)

        # Create tensors
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            [[1] * min(len(tokens), max_len) + [0] * max(0, max_len - len(tokens))],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = attention_mask[:, :max_len]

        # Forward pass
        hidden = self.model(input_ids, attention_mask)
        predictions = self.heads(hidden)

        # Decode signal
        signal_probs = torch.softmax(predictions["trade_signal"], dim=-1)[0]
        signal_idx = signal_probs.argmax().item()
        signal_names = ["BUY", "SELL", "HOLD"]
        signal = signal_names[signal_idx]

        # Decode confidence
        confidence = predictions["confidence"][0, 0].item()

        # Decode next candle prediction
        next_candle_logits = predictions["next_candle"][0]  # (tpc, vocab)
        next_candle_tokens = next_candle_logits.argmax(dim=-1).tolist()

        # Decode price bins back to prices
        # Tokens: [tf, close_bin, high_bin, low_bin, vol_bin, pattern, ind1, ind2]
        open_price = candles[-1, 3]  # use last close as reference
        if len(next_candle_tokens) >= 4:
            close, high, low = self.tokenizer.decode_price_bins(
                next_candle_tokens[1],  # close bin
                next_candle_tokens[2],  # high bin
                next_candle_tokens[3],  # low bin
                open_price,
            )
        else:
            close, high, low = open_price, open_price, open_price

        return {
            "signal": signal,
            "signal_probs": {
                "BUY": signal_probs[0].item(),
                "SELL": signal_probs[1].item(),
                "HOLD": signal_probs[2].item(),
            },
            "confidence": confidence,
            "next_candle": {
                "open": open_price,
                "close": close,
                "high": high,
                "low": low,
            },
        }

    def should_trade(self, prediction: Dict) -> Tuple[bool, str]:
        """
        Determine if a trade should be executed based on prediction.

        Returns:
            (should_trade, reason)
        """
        signal = prediction["signal"]
        confidence = prediction["confidence"]
        probs = prediction["signal_probs"]

        # Must not be HOLD
        if signal == "HOLD":
            return False, "Signal is HOLD"

        # Confidence threshold
        if confidence < self.trading_config.MIN_CONFIDENCE:
            return False, f"Confidence {confidence:.2f} below threshold {self.trading_config.MIN_CONFIDENCE}"

        # Signal probability threshold
        max_prob = max(probs.values())
        if max_prob < self.trading_config.MIN_SIGNAL_PROB:
            return False, f"Signal probability {max_prob:.2f} below threshold {self.trading_config.MIN_SIGNAL_PROB}"

        return True, f"Signal: {signal}, Confidence: {confidence:.2f}, Prob: {max_prob:.2f}"

    def predict_from_exchange(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        source: str = "binance",
    ) -> Dict:
        """
        Fetch latest candles from exchange and generate prediction.

        Args:
            symbol: Trading pair
            interval: Candle interval
            source: "binance" or "yahoo"
        """
        # Fetch data
        if source == "binance":
            fetcher = BinanceFetcher()
            candles = fetcher.fetch_klines(
                symbol=symbol,
                interval=interval,
                limit=self.trading_config.CONTEXT_CANDLES + 10,
            )
        elif source == "yahoo":
            fetcher = YahooFetcher()
            candles = fetcher.fetch(ticker=symbol, interval=interval, period="5d")
        else:
            raise ValueError(f"Unknown source: {source}")

        if len(candles) < self.trading_config.CONTEXT_CANDLES:
            raise ValueError(
                f"Not enough candles: got {len(candles)}, "
                f"need {self.trading_config.CONTEXT_CANDLES}"
            )

        # Take last N candles as context
        context = candles[-self.trading_config.CONTEXT_CANDLES:]

        # Predict
        prediction = self.predict(context, timeframe=interval)

        # Add trade recommendation
        should, reason = self.should_trade(prediction)
        prediction["should_trade"] = should
        prediction["reason"] = reason

        return prediction


def format_prediction(prediction: Dict) -> str:
    """Format prediction as a readable string."""
    lines = [
        f"Signal:      {prediction['signal']}",
        f"Confidence:  {prediction['confidence']:.2%}",
        f"Probabilities:",
        f"  BUY:       {prediction['signal_probs']['BUY']:.2%}",
        f"  SELL:      {prediction['signal_probs']['SELL']:.2%}",
        f"  HOLD:      {prediction['signal_probs']['HOLD']:.2%}",
        f"Next Candle Prediction:",
        f"  Open:      {prediction['next_candle']['open']:.2f}",
        f"  Close:     {prediction['next_candle']['close']:.2f}",
        f"  High:      {prediction['next_candle']['high']:.2f}",
        f"  Low:       {prediction['next_candle']['low']:.2f}",
    ]
    if "should_trade" in prediction:
        lines.append(f"Trade:       {'YES' if prediction['should_trade'] else 'NO'}")
        lines.append(f"Reason:      {prediction['reason']}")
    return "\n".join(lines)
