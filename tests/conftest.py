"""Shared fixtures for the test suite."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from tradingbot.core.enums import Side, Timeframe
from tradingbot.core.types import Fill, OHLCVBar, Position, Signal, StrategyGenome
from tradingbot.genome.strategy_genome import create_random_genome


@pytest.fixture
def sample_bars() -> list[OHLCVBar]:
    """Generate 500 realistic-looking OHLCV bars for testing."""
    bars = []
    price = 50000.0
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rng = np.random.RandomState(42)

    for i in range(500):
        change = rng.normal(0.0002, 0.015)
        price *= (1 + change)
        high = price * (1 + abs(rng.normal(0, 0.005)))
        low = price * (1 - abs(rng.normal(0, 0.005)))
        open_price = price / (1 + change)

        bars.append(OHLCVBar(
            timestamp=ts + timedelta(hours=i),
            symbol="BTC/USDT",
            timeframe=Timeframe.H1,
            open=open_price,
            high=high,
            low=low,
            close=price,
            volume=rng.uniform(100, 1000),
            exchange="binance",
        ))
    return bars


@pytest.fixture
def trending_bars() -> list[OHLCVBar]:
    """Generate bars with a clear uptrend."""
    bars = []
    price = 50000.0
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for i in range(300):
        # Strong upward bias
        change = 0.003 + random.gauss(0, 0.005)
        price *= (1 + change)
        high = price * 1.003
        low = price * 0.997
        open_price = price * 0.999

        bars.append(OHLCVBar(
            timestamp=ts + timedelta(hours=i),
            symbol="BTC/USDT",
            timeframe=Timeframe.H1,
            open=open_price,
            high=high,
            low=low,
            close=price,
            volume=random.uniform(100, 500),
            exchange="binance",
        ))
    return bars


@pytest.fixture
def mean_reverting_bars() -> list[OHLCVBar]:
    """Generate bars that oscillate around a mean."""
    bars = []
    mean_price = 50000.0
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

    for i in range(300):
        # Mean reverting: pull back toward mean
        deviation = mean_price * 0.03 * np.sin(i * 0.1)
        price = mean_price + deviation + random.gauss(0, 200)
        high = price * 1.002
        low = price * 0.998
        open_price = price + random.gauss(0, 50)

        bars.append(OHLCVBar(
            timestamp=ts + timedelta(hours=i),
            symbol="BTC/USDT",
            timeframe=Timeframe.H1,
            open=open_price,
            high=high,
            low=low,
            close=price,
            volume=random.uniform(100, 500),
            exchange="binance",
        ))
    return bars


@pytest.fixture
def random_genome() -> StrategyGenome:
    """Create a random strategy genome."""
    return create_random_genome("test_strategy")


@pytest.fixture
def sample_fill() -> Fill:
    """Create a sample fill."""
    return Fill(
        order_id="order-123",
        symbol="BTC/USDT",
        side=Side.BUY,
        price=50000.0,
        quantity=0.1,
        commission=5.0,
        exchange="binance",
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_position() -> Position:
    """Create a sample position."""
    return Position(
        symbol="BTC/USDT",
        strategy_id="strat-1",
        side=Side.BUY,
        quantity=0.1,
        avg_entry_price=50000.0,
        current_price=51000.0,
        exchange="binance",
    )
