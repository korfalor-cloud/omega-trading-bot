"""
CandleStick Transformer — Hybrid Candlestick Tokenizer
Converts OHLCV candle data into a sequence of discrete tokens.

Token vocabulary layout:
  [0-15]      Special tokens (PAD, BOS, EOS, MASK, SEP, ...)
  [16-79]     Pattern tokens (64 patterns)
  [80-95]     Timeframe tokens (16 timeframes)
  [96-351]    Indicator tokens (256 bins)
  [351-3495]  Price/volume bin tokens (1024*3 price + 256 volume)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import TokenizerConfig, get_tokenizer_config
from .patterns import PatternID, detect_pattern_for_candle


class CandleStickTokenizer:
    """
    Tokenizes OHLCV candlestick data into discrete token sequences.

    Each candle produces a fixed-length token subsequence:
      [timeframe_tok, close_bin, high_bin, low_bin, vol_bin, pattern_tok, ind_tok_1, ind_tok_2]

    A sequence of N candles becomes N * TOKENS_PER_CANDLE tokens.
    """

    def __init__(self, config: TokenizerConfig = None):
        self.config = config or get_tokenizer_config()

        # Timeframe string → token ID mapping
        self.timeframe_map: Dict[str, int] = {
            "1m": self.config.TIMEFRAME_OFFSET,
            "3m": self.config.TIMEFRAME_OFFSET + 1,
            "5m": self.config.TIMEFRAME_OFFSET + 2,
            "15m": self.config.TIMEFRAME_OFFSET + 3,
            "30m": self.config.TIMEFRAME_OFFSET + 4,
            "1h": self.config.TIMEFRAME_OFFSET + 5,
            "2h": self.config.TIMEFRAME_OFFSET + 6,
            "4h": self.config.TIMEFRAME_OFFSET + 7,
            "6h": self.config.TIMEFRAME_OFFSET + 8,
            "8h": self.config.TIMEFRAME_OFFSET + 9,
            "12h": self.config.TIMEFRAME_OFFSET + 10,
            "1d": self.config.TIMEFRAME_OFFSET + 11,
            "3d": self.config.TIMEFRAME_OFFSET + 12,
            "1w": self.config.TIMEFRAME_OFFSET + 13,
            "1M": self.config.TIMEFRAME_OFFSET + 14,
        }

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def tokens_per_candle(self) -> int:
        return self.config.TOKENS_PER_CANDLE

    @property
    def pad_token(self) -> int:
        return self.config.PAD_TOKEN

    @property
    def bos_token(self) -> int:
        return self.config.BOS_TOKEN

    @property
    def eos_token(self) -> int:
        return self.config.EOS_TOKEN

    def _quantize_price_change(self, change_pct: float, n_bins: int) -> int:
        """
        Quantize a percentage price change into a bin index.
        change_pct is in range [-PRICE_CHANGE_RANGE, +PRICE_CHANGE_RANGE].
        Returns bin index in [0, n_bins-1].
        """
        rng = self.config.PRICE_CHANGE_RANGE
        # Clamp to range
        clamped = max(-rng, min(rng, change_pct))
        # Normalize to [0, 1]
        normalized = (clamped + rng) / (2.0 * rng)
        # Quantize to bin
        bin_idx = int(normalized * (n_bins - 1))
        return max(0, min(n_bins - 1, bin_idx))

    def _quantize_volume(self, volume: float, ref_volume: float) -> int:
        """
        Quantize volume into logarithmic bins relative to a reference volume.
        """
        n_bins = self.config.VOLUME_BIN_COUNT
        if ref_volume <= 0 or volume <= 0:
            return 0

        ratio = volume / ref_volume
        # Log scale: map ratio [0.1, 10] to bins
        log_ratio = math.log10(max(0.1, min(10.0, ratio)))
        # log10(0.1) = -1, log10(10) = 1, so normalize from [-1, 1] to [0, 1]
        normalized = (log_ratio + 1.0) / 2.0
        bin_idx = int(normalized * (n_bins - 1))
        return max(0, min(n_bins - 1, bin_idx))

    def _compute_indicators(self, candles: np.ndarray, idx: int) -> Tuple[int, int]:
        """
        Compute simple technical indicators for candle at index idx.
        Returns two indicator token IDs.

        Indicators computed:
          1. RSI-like momentum (14-period)
          2. Bollinger Band position
        """
        n_bins = self.config.INDICATOR_BIN_COUNT
        offset = self.config.INDICATOR_OFFSET

        # Need at least 14 candles for RSI
        lookback = min(14, idx + 1)
        start = max(0, idx - lookback + 1)
        window = candles[start:idx + 1]

        # RSI-like: ratio of up moves to total moves
        closes = window[:, 3]
        if len(closes) < 2:
            rsi_bin = n_bins // 2  # neutral
        else:
            diffs = np.diff(closes)
            up_moves = np.sum(diffs[diffs > 0])
            down_moves = -np.sum(diffs[diffs < 0])
            total = up_moves + down_moves
            if total == 0:
                rsi_val = 0.5
            else:
                rsi_val = up_moves / total  # 0 to 1
            rsi_bin = int(rsi_val * (n_bins - 1))
            rsi_bin = max(0, min(n_bins - 1, rsi_bin))

        # Bollinger position: where is current close relative to 20-period BB
        bb_lookback = min(20, idx + 1)
        bb_start = max(0, idx - bb_lookback + 1)
        bb_closes = candles[bb_start:idx + 1, 3]
        if len(bb_closes) < 2:
            bb_bin = n_bins // 2
        else:
            mean = np.mean(bb_closes)
            std = np.std(bb_closes)
            if std == 0:
                bb_bin = n_bins // 2
            else:
                # Position within [-2σ, +2σ]
                z_score = (closes[-1] - mean) / (2.0 * std)
                z_score = max(-1.0, min(1.0, z_score))
                bb_bin = int((z_score + 1.0) / 2.0 * (n_bins - 1))
                bb_bin = max(0, min(n_bins - 1, bb_bin))

        return offset + rsi_bin, offset + bb_bin

    def tokenize_candle(
        self,
        candle: np.ndarray,
        all_candles: np.ndarray,
        candle_idx: int,
        timeframe: str = "1h",
        ref_volume: float = None,
    ) -> List[int]:
        """
        Tokenize a single candle into a fixed-length token sequence.

        Args:
            candle: OHLCV array [open, high, low, close, volume, timestamp]
            all_candles: Full candle history for computing indicators/patterns
            candle_idx: Index of this candle in all_candles
            timeframe: String like "1h", "1d", etc.
            ref_volume: Reference volume for normalization (mean volume)

        Returns:
            List of token IDs (length = tokens_per_candle)
        """
        o, h, l, c, v = candle[0], candle[1], candle[2], candle[3], candle[4]

        # Timeframe token
        tf_tok = self.timeframe_map.get(timeframe, self.config.TIMEFRAME_OFFSET)

        # Price change bins (relative to open)
        if o == 0:
            close_change = 0.0
            high_change = 0.0
            low_change = 0.0
        else:
            close_change = (c - o) / o
            high_change = (h - o) / o
            low_change = (l - o) / o

        close_bin = self.config.PRICE_OFFSET + self._quantize_price_change(
            close_change, self.config.PRICE_BIN_COUNT
        )
        high_bin = self.config.PRICE_OFFSET + self.config.PRICE_BIN_COUNT + self._quantize_price_change(
            high_change, self.config.PRICE_BIN_COUNT
        )
        low_bin = self.config.PRICE_OFFSET + 2 * self.config.PRICE_BIN_COUNT + self._quantize_price_change(
            low_change, self.config.PRICE_BIN_COUNT
        )

        # Volume bin
        if ref_volume is None:
            ref_volume = v if v > 0 else 1.0
        vol_bin = self.config.SPECIAL_COUNT + self.config.TIMEFRAME_COUNT + self.config.PATTERN_COUNT + self._quantize_volume(v, ref_volume)

        # Pattern token
        history = all_candles[:candle_idx + 1]
        pattern_id = detect_pattern_for_candle(history)
        pattern_tok = self.config.PATTERN_OFFSET + pattern_id

        # Indicator tokens
        ind1, ind2 = self._compute_indicators(all_candles, candle_idx)

        return [tf_tok, close_bin, high_bin, low_bin, vol_bin, pattern_tok, ind1, ind2]

    def tokenize_sequence(
        self,
        candles: np.ndarray,
        timeframe: str = "1h",
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """
        Tokenize a sequence of candles.

        Args:
            candles: Array of shape (N, 6) — OHLCV + timestamp
            timeframe: Candle timeframe string
            add_bos: Prepend BOS token
            add_eos: Append EOS token

        Returns:
            Flat list of token IDs
        """
        if len(candles) == 0:
            return [self.bos_token] if add_bos else []

        # Reference volume = mean volume of sequence
        ref_vol = float(np.mean(candles[:, 4])) if np.any(candles[:, 4] > 0) else 1.0

        tokens = []
        if add_bos:
            tokens.append(self.bos_token)

        for i in range(len(candles)):
            candle_tokens = self.tokenize_candle(
                candles[i], candles, i, timeframe, ref_vol
            )
            tokens.extend(candle_tokens)

        if add_eos:
            tokens.append(self.eos_token)

        return tokens

    def pad_sequence(self, tokens: List[int], max_len: int) -> List[int]:
        """Pad or truncate token sequence to max_len."""
        if len(tokens) >= max_len:
            return tokens[:max_len]
        return tokens + [self.pad_token] * (max_len - len(tokens))

    def decode_price_bins(
        self, close_bin: int, high_bin: int, low_bin: int, open_price: float
    ) -> Tuple[float, float, float]:
        """
        Decode price bin tokens back to approximate price values.

        Returns: (close, high, low) prices
        """
        rng = self.config.PRICE_CHANGE_RANGE
        n_bins = self.config.PRICE_BIN_COUNT

        def bin_to_pct(bin_id: int, offset: int) -> float:
            raw = bin_id - offset
            normalized = raw / (n_bins - 1)
            return normalized * (2.0 * rng) - rng

        close_pct = bin_to_pct(close_bin, self.config.PRICE_OFFSET)
        high_pct = bin_to_pct(high_bin, self.config.PRICE_OFFSET + n_bins)
        low_pct = bin_to_pct(low_bin, self.config.PRICE_OFFSET + 2 * n_bins)

        close = open_price * (1.0 + close_pct)
        high = open_price * (1.0 + high_pct)
        low = open_price * (1.0 + low_pct)

        return close, high, low

    def decode_volume_bin(self, vol_bin: int, ref_volume: float) -> float:
        """Decode volume bin back to approximate volume."""
        n_bins = self.config.VOLUME_BIN_COUNT
        offset = self.config.SPECIAL_COUNT + self.config.TIMEFRAME_COUNT + self.config.PATTERN_COUNT
        raw = vol_bin - offset
        normalized = raw / (n_bins - 1)
        log_ratio = normalized * 2.0 - 1.0  # back to [-1, 1]
        ratio = 10.0 ** log_ratio
        return ref_volume * ratio
