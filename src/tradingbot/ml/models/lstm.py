"""LSTM Model — recurrent neural network for sequence prediction.

Implements:
- Simple LSTM cell (numpy only)
- Sequence prediction
- Time series forecasting
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class LSTMCell:
    """Simple LSTM cell implementation."""

    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size

        scale = np.sqrt(2.0 / (input_size + hidden_size))

        # Gate weights
        self.W_f = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b_f = np.zeros(hidden_size)
        self.W_i = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b_i = np.zeros(hidden_size)
        self.W_c = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b_c = np.zeros(hidden_size)
        self.W_o = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b_o = np.zeros(hidden_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        return np.tanh(np.clip(x, -500, 500))

    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> tuple:
        """Forward pass. Returns (h, c)."""
        concat = np.concatenate([x, h_prev])

        f = self.sigmoid(concat @ self.W_f + self.b_f)  # Forget gate
        i = self.sigmoid(concat @ self.W_i + self.b_i)  # Input gate
        c_tilde = self.tanh(concat @ self.W_c + self.b_c)  # Candidate
        o = self.sigmoid(concat @ self.W_o + self.b_o)  # Output gate

        c = f * c_prev + i * c_tilde
        h = o * self.tanh(c)

        return h, c


class LSTMPredictor:
    """LSTM-based price predictor."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.input_size = config.get("input_size", 5)
        self.hidden_size = config.get("hidden_size", 32)
        self.seq_len = config.get("seq_len", 30)

        self.cell = LSTMCell(self.input_size, self.hidden_size)
        self.output_weight = np.random.randn(self.hidden_size, 1) * 0.01
        self.trained = False

    def extract_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract features from prices."""
        n = len(prices)
        features = np.zeros((n, self.input_size))

        features[:, 0] = prices / prices[0] if prices[0] != 0 else 0
        if n > 1:
            features[1:, 1] = np.diff(prices) / prices[:-1]

        # Moving averages
        for i in range(5, n):
            features[i, 2] = np.mean(prices[i - 5:i]) / prices[i] if prices[i] > 0 else 0
        for i in range(10, n):
            features[i, 3] = np.mean(prices[i - 10:i]) / prices[i] if prices[i] > 0 else 0
        for i in range(20, n):
            features[i, 4] = np.mean(prices[i - 20:i]) / prices[i] if prices[i] > 0 else 0

        return features

    def predict(self, prices: np.ndarray) -> float:
        """Predict next price direction."""
        if len(prices) < self.seq_len:
            return 0.0

        features = self.extract_features(prices[-self.seq_len:])

        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)

        for i in range(self.seq_len):
            h, c = self.cell.forward(features[i], h, c)

        output = h @ self.output_weight
        return float(np.tanh(output))

    def train_step(self, prices: np.ndarray, target: float, lr: float = 0.001) -> float:
        """Single training step."""
        pred = self.predict(prices)
        loss = (pred - target) ** 2

        # Simplified gradient update
        features = self.extract_features(prices[-self.seq_len:])
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        for i in range(self.seq_len):
            h, c = self.cell.forward(features[i], h, c)

        grad = 2 * (pred - target)
        self.output_weight -= lr * grad * h[:, np.newaxis]
        self.trained = True
        return loss
