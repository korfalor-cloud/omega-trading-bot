"""Gradient Boost Strategy — ML-based signal generation.

Uses a gradient boosting model (XGBoost/LightGBM) trained on
technical features to predict price direction.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...features.technical import TechnicalIndicators, compute_features

logger = logging.getLogger(__name__)


class GradientBoostStrategy(Strategy):
    """ML strategy using gradient boosting on technical features.

    Trains on historical bars and predicts next-bar direction.
    Features: RSI, MACD, BB position, ATR, ADX, volume ratio, price momentum.
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        self._model = None
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = 200  # Need enough data to train
        self._retrain_interval = 500  # Retrain every N bars
        self._bars_since_train = 0
        self._feature_names = [
            "rsi_14", "macd", "macd_hist", "adx_14",
            "atr_14", "bb_upper", "bb_lower", "volatility",
            "momentum_10", "roc_10", "signal_strength",
        ]
        self._lookforward = 5  # Predict 5-bar forward return

    def _build_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix and labels from buffer."""
        features = compute_features(self._bar_buffer)
        closes = np.array([b.close for b in self._bar_buffer])

        # Feature matrix
        X_parts = []
        for name in self._feature_names:
            if name in features:
                X_parts.append(features[name])
            else:
                X_parts.append([0.0] * len(self._bar_buffer))
        X = np.column_stack(X_parts)

        # Labels: forward return sign (1 = up, 0 = down)
        y = np.zeros(len(closes))
        for i in range(len(closes) - self._lookforward):
            fwd_ret = (closes[i + self._lookforward] - closes[i]) / closes[i]
            y[i] = 1 if fwd_ret > 0 else 0

        # Remove NaN rows and last lookforward rows
        valid = ~np.isnan(X).any(axis=1)
        valid[-self._lookforward:] = False
        return X[valid], y[valid]

    def _train_model(self) -> bool:
        """Train the gradient boosting model."""
        try:
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            logger.warning("sklearn not available, falling back to simple signal")
            return False

        X, y = self._build_training_data()
        if len(X) < 100:
            return False

        self._model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self._model.fit(X, y)
        self._bars_since_train = 0
        logger.info(f"ML model trained on {len(X)} samples, accuracy: {self._model.score(X, y):.3f}")
        return True

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) > 2000:
            self._bar_buffer = self._bar_buffer[-1500:]

        if len(self._bar_buffer) < self._min_bars:
            return None

        self._bars_since_train += 1

        # Train or retrain
        if self._model is None or self._bars_since_train >= self._retrain_interval:
            if not self._train_model():
                return None

        # Predict
        features = compute_features(self._bar_buffer[-100:])
        X_pred = np.array([[
            features.get(name, [0.0] * 100)[-1]
            for name in self._feature_names
        ]])

        if np.isnan(X_pred).any():
            return None

        proba = self._model.predict_proba(X_pred)[0]
        pred_class = np.argmax(proba)
        confidence = max(proba)

        # Only trade with high confidence
        if confidence < 0.6:
            return None

        ti = TechnicalIndicators(self._bar_buffer[-50:])
        atr = ti.atr(14)[-1]
        if atr is None or atr != atr:
            atr = bar.close * 0.02

        atr_mult = self.genome.stop_loss_param

        if pred_class == 1:  # Bullish
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=confidence,
                confidence=confidence,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=bar.close - atr_mult * atr,
                take_profit=bar.close + atr_mult * atr * self.genome.take_profit_ratio,
            )
        else:  # Bearish
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=confidence,
                confidence=confidence,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=bar.close + atr_mult * atr,
                take_profit=bar.close - atr_mult * atr * self.genome.take_profit_ratio,
            )

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
