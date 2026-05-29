"""Transformer Model — attention-based price prediction.

Implements:
- Multi-head self-attention
- Positional encoding
- Price sequence prediction
- Feature importance via attention weights
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class SimpleAttention:
    """Simplified multi-head attention (numpy only)."""

    def __init__(self, d_model: int = 64, n_heads: int = 4):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Initialize weights
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass. x: (seq_len, d_model)."""
        seq_len = x.shape[0]

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head
        Q = Q.reshape(seq_len, self.n_heads, self.d_k)
        K = K.reshape(seq_len, self.n_heads, self.d_k)
        V = V.reshape(seq_len, self.n_heads, self.d_k)

        # Attention scores
        scores = np.einsum("ihd,jhd->ijh", Q, K) / np.sqrt(self.d_k)

        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        scores = scores + mask[:, :, np.newaxis]

        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        attn_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-10)

        # Apply attention
        out = np.einsum("ijh,jhd->ihd", attn_weights, V)
        out = out.reshape(seq_len, self.d_model)

        return out @ self.W_o


class TransformerPredictor:
    """Transformer-based price predictor."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.d_model = config.get("d_model", 64)
        self.n_heads = config.get("n_heads", 4)
        self.n_layers = config.get("n_layers", 2)
        self.seq_len = config.get("seq_len", 60)

        self.attention_layers = [SimpleAttention(self.d_model, self.n_heads) for _ in range(self.n_layers)]
        self.output_weight = np.random.randn(self.d_model, 1) * 0.01
        self.trained = False

    def encode(self, prices: np.ndarray) -> np.ndarray:
        """Encode price sequence into features."""
        n = len(prices)
        features = np.zeros((n, self.d_model))

        # Price features
        features[:, 0] = prices / prices[0] if prices[0] != 0 else 0
        if n > 1:
            features[1:, 1] = np.diff(prices) / prices[:-1]

        # Technical features
        for period in [5, 10, 20]:
            if n >= period:
                sma = np.convolve(prices, np.ones(period) / period, mode="valid")
                features[period - 1:, 2 + period // 10] = prices[period - 1:] / sma - 1

        # Positional encoding
        pos = np.arange(n)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        features[:, 3::2] = np.sin(pos * div_term[:features[:, 3::2].shape[1]])
        features[:, 4::2] = np.cos(pos * div_term[:features[:, 4::2].shape[1]])

        return features

    def predict(self, prices: np.ndarray) -> float:
        """Predict next price direction."""
        if len(prices) < self.seq_len:
            return 0.0

        x = self.encode(prices[-self.seq_len:])

        # Forward through attention layers
        for layer in self.attention_layers:
            x = layer.forward(x)

        # Output prediction
        output = x[-1] @ self.output_weight
        return float(np.tanh(output))

    def train_step(self, prices: np.ndarray, target: float, lr: float = 0.001) -> float:
        """Single training step."""
        pred = self.predict(prices)
        loss = (pred - target) ** 2

        # Simple gradient descent on output layer
        grad = 2 * (pred - target)
        x = self.encode(prices[-self.seq_len:])
        for layer in self.attention_layers:
            x = layer.forward(x)

        self.output_weight -= lr * grad * x[-1:, :].T
        self.trained = True
        return loss

    def get_attention_weights(self, prices: np.ndarray) -> np.ndarray:
        """Get attention weights for interpretability."""
        x = self.encode(prices[-self.seq_len:])
        # Simplified — return last layer attention
        Q = x @ self.attention_layers[-1].W_q
        K = x @ self.attention_layers[-1].W_k
        scores = Q @ K.T / np.sqrt(self.d_model)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-10)
