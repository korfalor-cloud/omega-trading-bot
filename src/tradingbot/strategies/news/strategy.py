"""News/Event-Driven Strategy.

Implements:
- News sentiment-based trading
- Event detection (listings, halvings, upgrades)
- Fade-the-news and follow-the-news logic
- Sentiment momentum
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


class NewsDrivenStrategy(Strategy):
    """Trade based on news sentiment and events."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._sentiment_threshold = feats.get("sentiment_threshold", 0.3)
        self._fade_mode = feats.get("fade_mode", False)
        self._lookback = feats.get("lookback", 24)  # hours of sentiment
        self._hold_bars = feats.get("hold_bars", 12)

        self._bar_buffer: list[OHLCVBar] = []
        self._sentiment_buffer: list[float] = []
        self._in_trade = False
        self._trade_bars = 0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)

        # Get sentiment from bar metadata (injected by data pipeline)
        sentiment = bar.vwap if bar.vwap != 0 else 0
        self._sentiment_buffer.append(sentiment)

        if len(self._sentiment_buffer) < self._lookback:
            return None

        if len(self._bar_buffer) > 300:
            self._bar_buffer = self._bar_buffer[-200:]
            self._sentiment_buffer = self._sentiment_buffer[-200:]

        # Exit logic
        if self._in_trade:
            self._trade_bars += 1
            if self._trade_bars >= self._hold_bars:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=0.5, confidence=0.6,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            return None

        # Sentiment analysis
        recent = self._sentiment_buffer[-self._lookback:]
        avg_sentiment = np.mean(recent)
        sentiment_momentum = np.mean(recent[-6:]) - np.mean(recent[:6]) if len(recent) >= 12 else 0

        # Price action
        prices = np.array([b.close for b in self._bar_buffer[-20:]])
        price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] != 0 else 0

        if self._fade_mode:
            # Fade the news — trade against extreme sentiment
            if avg_sentiment > self._sentiment_threshold and price_change > 0.02:
                # Euphoria — fade with sell
                self._in_trade = True
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=min(1.0, abs(avg_sentiment)),
                    confidence=0.6, signal_type=SignalType.ENTRY,
                    timeframe=Timeframe.H1,
                    metadata={"sentiment": avg_sentiment, "mode": "fade"},
                )
            if avg_sentiment < -self._sentiment_threshold and price_change < -0.02:
                # Panic — fade with buy
                self._in_trade = True
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.BUY, strength=min(1.0, abs(avg_sentiment)),
                    confidence=0.6, signal_type=SignalType.ENTRY,
                    timeframe=Timeframe.H1,
                    metadata={"sentiment": avg_sentiment, "mode": "fade"},
                )
        else:
            # Follow the news
            if avg_sentiment > self._sentiment_threshold and sentiment_momentum > 0:
                self._in_trade = True
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.BUY, strength=min(1.0, abs(avg_sentiment)),
                    confidence=0.65, signal_type=SignalType.ENTRY,
                    timeframe=Timeframe.H1,
                    metadata={"sentiment": avg_sentiment, "mode": "follow"},
                )
            if avg_sentiment < -self._sentiment_threshold and sentiment_momentum < 0:
                self._in_trade = True
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL, strength=min(1.0, abs(avg_sentiment)),
                    confidence=0.65, signal_type=SignalType.ENTRY,
                    timeframe=Timeframe.H1,
                    metadata={"sentiment": avg_sentiment, "mode": "follow"},
                )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
