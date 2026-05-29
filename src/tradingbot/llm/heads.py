"""
CandleStick Transformer — Output Heads
Three specialized heads for multi-task learning:
  1. Next Candle Predictor — generative, predicts next candle's tokens
  2. Trade Signal Classifier — BUY / SELL / HOLD
  3. Confidence Scorer — 0.0 to 1.0 confidence
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class NextCandleHead(nn.Module):
    """
    Predicts the next candle's token distribution.
    Takes the last hidden state and projects to vocabulary space
    for each token position in the candle.
    """

    def __init__(self, d_model: int, vocab_size: int, tokens_per_candle: int = 8):
        super().__init__()
        self.tokens_per_candle = tokens_per_candle

        # Shared transformer for candle token generation
        self.candle_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Output projection for each candle token
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Learned candle position embeddings
        self.candle_pos = nn.Parameter(
            torch.zeros(1, tokens_per_candle, d_model)
        )
        nn.init.trunc_normal_(self.candle_pos, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model) from the transformer

        Returns:
            (batch, tokens_per_candle, vocab_size) — logits for next candle tokens
        """
        B = hidden_states.size(0)

        # Use last few hidden states as memory
        memory = hidden_states[:, -16:, :]  # last 16 tokens as context

        # Create candle token queries
        queries = self.candle_pos.expand(B, -1, -1)  # (B, tokens_per_candle, D)

        # Decode next candle tokens
        decoded = self.candle_transformer(queries, memory)  # (B, tpc, D)

        # Project to vocabulary
        logits = self.output_proj(decoded)  # (B, tpc, vocab_size)
        return logits


class TradeSignalHead(nn.Module):
    """
    Classifies the next trading action: BUY, SELL, or HOLD.
    Uses pooled representation from the last N hidden states.
    """

    def __init__(self, d_model: int, n_classes: int = 3):
        super().__init__()
        self.pool_size = 8  # pool last 8 token representations

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, n_classes),
        )

        # Attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            (batch, n_classes) — logits for BUY/SELL/HOLD
        """
        # Take last pool_size tokens
        x = hidden_states[:, -self.pool_size:, :]  # (B, pool, D)

        # Attention-weighted pooling
        attn_weights = self.attn_pool(x)  # (B, pool, 1)
        pooled = (x * attn_weights).sum(dim=1)  # (B, D)

        logits = self.classifier(pooled)  # (B, n_classes)
        return logits


class ConfidenceHead(nn.Module):
    """
    Predicts confidence score (0.0 to 1.0) for the trading signal.
    Trained against realized PnL of the signal.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.pool_size = 8

        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),  # output in [0, 1]
        )

        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            (batch, 1) — confidence scores in [0, 1]
        """
        x = hidden_states[:, -self.pool_size:, :]
        attn_weights = self.attn_pool(x)
        pooled = (x * attn_weights).sum(dim=1)

        confidence = self.scorer(pooled)  # (B, 1)
        return confidence


class TradingHeads(nn.Module):
    """
    Combined output heads for the CandleStick Transformer.
    Returns all three outputs in a single forward pass.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.next_candle = NextCandleHead(
            config.D_MODEL, config.VOCAB_SIZE, tokens_per_candle=8
        )
        self.trade_signal = TradeSignalHead(config.D_MODEL, config.SIGNAL_CLASSES)
        self.confidence = ConfidenceHead(config.D_MODEL)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, d_model) from CandleTransformer

        Returns:
            Dict with keys:
              - "next_candle": (batch, tokens_per_candle, vocab_size)
              - "trade_signal": (batch, n_classes)
              - "confidence": (batch, 1)
        """
        return {
            "next_candle": self.next_candle(hidden_states),
            "trade_signal": self.trade_signal(hidden_states),
            "confidence": self.confidence(hidden_states),
        }


class TradingLoss(nn.Module):
    """
    Multi-task loss combining:
      - Cross-entropy for next candle prediction
      - Cross-entropy for trade signal
      - MSE for confidence score
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 2.0,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.candle_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.signal_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.confidence_loss_fn = nn.MSELoss()

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: output from TradingHeads.forward()
            targets: dict with keys:
              - "next_candle": (batch, tokens_per_candle) — target token IDs
              - "trade_signal": (batch,) — target class (0=BUY, 1=SELL, 2=HOLD)
              - "confidence": (batch, 1) — target confidence [0, 1]

        Returns:
            Dict with "total_loss", "candle_loss", "signal_loss", "confidence_loss"
        """
        # Next candle loss — reshape for cross entropy
        candle_logits = predictions["next_candle"]  # (B, tpc, vocab)
        candle_targets = targets["next_candle"]  # (B, tpc)
        candle_loss = self.candle_loss_fn(
            candle_logits.reshape(-1, candle_logits.size(-1)),
            candle_targets.reshape(-1),
        )

        # Signal loss
        signal_logits = predictions["trade_signal"]  # (B, n_classes)
        signal_targets = targets["trade_signal"]  # (B,)
        signal_loss = self.signal_loss_fn(signal_logits, signal_targets)

        # Confidence loss
        conf_pred = predictions["confidence"]  # (B, 1)
        conf_target = targets["confidence"]  # (B, 1)
        confidence_loss = self.confidence_loss_fn(conf_pred, conf_target)

        total = (
            self.alpha * candle_loss
            + self.beta * signal_loss
            + self.gamma * confidence_loss
        )

        return {
            "total_loss": total,
            "candle_loss": candle_loss,
            "signal_loss": signal_loss,
            "confidence_loss": confidence_loss,
        }
