"""
CandleStick Transformer — PyTorch Dataset
Creates sliding window training samples from candlestick data.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import TrainingConfig, get_training_config
from ..tokenizer import CandleStickTokenizer


def compute_targets(
    candles: np.ndarray,
    target_idx: int,
) -> Tuple[List[int], int, float]:
    """
    Compute training targets for a given candle index.

    Args:
        candles: Full OHLCV array (N, 6)
        target_idx: Index of the target candle (the one after the input window)

    Returns:
        (target_candle_tokens, trade_signal, confidence)
        - target_candle_tokens: token IDs for the target candle
        - trade_signal: 0=BUY, 1=SELL, 2=HOLD
        - confidence: 0.0 to 1.0 based on realized price move
    """
    if target_idx >= len(candles):
        # Pad with zeros if out of range
        return [0] * 8, 2, 0.0

    candle = candles[target_idx]
    o, c = candle[0], candle[3]

    # Trade signal based on price movement
    if o == 0:
        pct_change = 0.0
    else:
        pct_change = (c - o) / o

    # Signal: BUY if price went up significantly, SELL if down, HOLD otherwise
    threshold = 0.005  # 0.5% threshold
    if pct_change > threshold:
        signal = 0  # BUY
    elif pct_change < -threshold:
        signal = 1  # SELL
    else:
        signal = 2  # HOLD

    # Confidence based on magnitude of move (normalized)
    confidence = min(1.0, abs(pct_change) / 0.05)  # 5% move = max confidence

    return [], signal, confidence


class CandleDataset(Dataset):
    """
    PyTorch Dataset for candlestick sequences.

    Creates sliding windows of candle data with corresponding targets:
      - Input: window_size candles (tokenized)
      - Target: next candle tokens, trade signal, confidence score
    """

    def __init__(
        self,
        candles: np.ndarray,
        tokenizer: CandleStickTokenizer,
        config: TrainingConfig = None,
        timeframe: str = "1h",
        augment: bool = False,
    ):
        """
        Args:
            candles: OHLCV array of shape (N, 6)
            tokenizer: CandleStickTokenizer instance
            config: Training configuration
            timeframe: Candle timeframe string
            augment: Whether to apply data augmentation
        """
        self.candles = candles
        self.tokenizer = tokenizer
        self.config = config or get_training_config()
        self.timeframe = timeframe
        self.augment = augment

        self.tokens_per_candle = tokenizer.tokens_per_candle
        self.stride = self.config.STRIDE

        # Cap window size so total tokens fit in model's max sequence length (512)
        # Total tokens = window_size * tokens_per_candle + 1 (BOS) + 1 (EOS)
        max_model_seq = 512
        max_window = (max_model_seq - 2) // self.tokens_per_candle
        self.window_size = min(self.config.WINDOW_SIZE, max_window)
        self.max_seq_len = self.tokens_per_candle * self.window_size + 2  # +BOS +EOS

        # Compute valid sample indices
        self.n_samples = max(0, len(candles) - self.window_size - 1)
        self.indices = list(range(0, self.n_samples, self.stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training sample.

        Returns:
            Dict with:
              - "input_ids": (max_seq_len,) — tokenized input sequence
              - "attention_mask": (max_seq_len,) — 1 for real tokens, 0 for padding
              - "target_candle": (tokens_per_candle,) — target candle token IDs
              - "target_signal": () — scalar trade signal (0/1/2)
              - "target_confidence": (1,) — confidence score
        """
        start = self.indices[idx]
        end = start + self.window_size

        # Input candles
        input_candles = self.candles[start:end]

        # Target candle (the one after the window)
        target_candle_idx = end
        target_candle = self.candles[target_candle_idx] if target_candle_idx < len(self.candles) else None

        # Data augmentation
        if self.augment:
            input_candles = self._augment(input_candles.copy())

        # Tokenize input sequence
        tokens = self.tokenizer.tokenize_sequence(
            input_candles,
            timeframe=self.timeframe,
            add_bos=True,
            add_eos=False,
        )

        # Pad to max length
        input_ids = self.tokenizer.pad_sequence(tokens, self.max_seq_len)
        attention_mask = [1] * min(len(tokens), self.max_seq_len) + [0] * max(0, self.max_seq_len - len(tokens))
        attention_mask = attention_mask[:self.max_seq_len]

        # Compute target candle tokens
        if target_candle is not None:
            all_candles_up_to_target = self.candles[:target_candle_idx + 1]
            ref_vol = float(np.mean(all_candles_up_to_target[:, 4])) if np.any(all_candles_up_to_target[:, 4] > 0) else 1.0
            target_tokens = self.tokenizer.tokenize_candle(
                target_candle,
                all_candles_up_to_target,
                target_candle_idx,
                self.timeframe,
                ref_vol,
            )
        else:
            target_tokens = [0] * self.tokens_per_candle

        # Compute signal and confidence
        if target_candle is not None:
            o, c = target_candle[0], target_candle[3]
            pct_change = (c - o) / o if o != 0 else 0.0

            threshold = 0.005
            if pct_change > threshold:
                signal = 0  # BUY
            elif pct_change < -threshold:
                signal = 1  # SELL
            else:
                signal = 2  # HOLD

            confidence = min(1.0, abs(pct_change) / 0.05)
        else:
            signal = 2
            confidence = 0.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "target_candle": torch.tensor(target_tokens, dtype=torch.long),
            "target_signal": torch.tensor(signal, dtype=torch.long),
            "target_confidence": torch.tensor([confidence], dtype=torch.float32),
        }

    def _augment(self, candles: np.ndarray) -> np.ndarray:
        """Apply data augmentation: noise, scaling, time warping."""
        # 1. Gaussian noise (0.15% of price)
        noise_scale = 0.0015
        for i in range(4):
            base = candles[:, i]
            noise = np.random.normal(0, noise_scale * np.abs(base))
            candles[:, i] = base + noise

        # 2. Random price scaling (0.98-1.02)
        scale = np.random.uniform(0.98, 1.02)
        candles[:, :4] *= scale

        # 3. Random volume scaling (0.8-1.2)
        vol_scale = np.random.uniform(0.8, 1.2)
        candles[:, 4] *= vol_scale

        # 4. Random candle dropout (zero out 1-3 random candles)
        n_drop = np.random.randint(0, min(4, len(candles)))
        if n_drop > 0:
            drop_idx = np.random.choice(len(candles), n_drop, replace=False)
            for idx in drop_idx:
                mid = candles[idx, 0]  # use open as center
                candles[idx, 0] = mid  # open stays
                candles[idx, 3] = mid  # close = open (doji)
                candles[idx, 1] = mid * 1.001  # tiny high
                candles[idx, 2] = mid * 0.999  # tiny low

        # Ensure OHLC consistency
        candles[:, 1] = np.maximum(candles[:, 1], np.maximum(candles[:, 0], candles[:, 3]))
        candles[:, 2] = np.minimum(candles[:, 2], np.minimum(candles[:, 0], candles[:, 3]))

        # Ensure no negative prices
        candles[:, :5] = np.maximum(candles[:, :5], 0.0)

        return candles


def create_train_val_test_split(
    candles: np.ndarray,
    val_split: float = 0.15,
    test_split: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split candle data into train/val/test sets (chronologically).

    Returns:
        (train_candles, val_candles, test_candles)
    """
    n = len(candles)
    test_start = int(n * (1.0 - test_split))
    val_start = int(n * (1.0 - test_split - val_split))

    train = candles[:val_start]
    val = candles[val_start:test_start]
    test = candles[test_start:]

    return train, val, test
