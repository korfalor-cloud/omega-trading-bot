"""Tests for core types, enums, and events."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from tradingbot.core.enums import (
    NodeType,
    OrderState,
    OrderType,
    Side,
    SignalType,
    StrategyStatus,
    Timeframe,
)
from tradingbot.core.events import Event, EventBus
from tradingbot.core.types import (
    Fill,
    OHLCVBar,
    Order,
    OrderBookLevel,
    OrderBookSnapshot,
    Position,
    RiskCheck,
    Signal,
    StrategyGenome,
)


class TestEnums:
    def test_timeframe_seconds(self):
        assert Timeframe.M1.seconds == 60
        assert Timeframe.H1.seconds == 3600
        assert Timeframe.D1.seconds == 86400
        assert Timeframe.W1.seconds == 604800

    def test_side_values(self):
        assert Side.BUY.value == "buy"
        assert Side.SELL.value == "sell"

    def test_order_type_values(self):
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP_MARKET.value == "stop_market"

    def test_node_type_enum(self):
        assert NodeType.RSI.value == "rsi"
        assert NodeType.AND.value == "and"
        assert NodeType.CLOSE.value == "close"


class TestOHLCVBar:
    def test_creation(self, sample_bars):
        bar = sample_bars[0]
        assert bar.symbol == "BTC/USDT"
        assert bar.timeframe == Timeframe.H1
        assert bar.open > 0
        assert bar.high >= bar.low

    def test_properties(self, sample_bars):
        bar = sample_bars[0]
        assert bar.mid == (bar.high + bar.low) / 2
        assert bar.range == bar.high - bar.low
        assert bar.body == abs(bar.close - bar.open)
        assert isinstance(bar.is_bullish, bool)

    def test_immutability(self, sample_bars):
        bar = sample_bars[0]
        with pytest.raises(AttributeError):
            bar.close = 99999


class TestOrderBook:
    def test_snapshot_properties(self):
        book = OrderBookSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            exchange="binance",
            bids=[OrderBookLevel(50000, 1.0), OrderBookLevel(49999, 2.0)],
            asks=[OrderBookLevel(50001, 1.5), OrderBookLevel(50002, 3.0)],
        )
        assert book.best_bid == 50000
        assert book.best_ask == 50001
        assert book.mid_price == 50000.5
        assert book.spread == 1.0
        assert abs(book.spread_bps - 0.2) < 0.01
        assert book.bid_depth == 3.0
        assert book.ask_depth == 4.5
        assert book.imbalance < 0  # More asks than bids

    def test_empty_book(self):
        book = OrderBookSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            exchange="binance",
            bids=[],
            asks=[],
        )
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None


class TestSignal:
    def test_creation(self):
        sig = Signal(
            strategy_id="strat-1",
            symbol="BTC/USDT",
            side=Side.BUY,
            strength=0.8,
            confidence=0.7,
        )
        assert sig.strategy_id == "strat-1"
        assert sig.side == Side.BUY
        assert 0 <= sig.strength <= 1
        assert 0 <= sig.confidence <= 1
        assert sig.id  # Auto-generated


class TestOrder:
    def test_properties(self):
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            quantity=1.0,
            price=50000.0,
        )
        assert order.remaining_quantity == 1.0
        assert order.is_active
        assert order.notional_value == 50000.0

    def test_partial_fill(self):
        order = Order(
            symbol="BTC/USDT",
            side=Side.BUY,
            quantity=1.0,
            price=50000.0,
            filled_quantity=0.5,
        )
        assert order.remaining_quantity == 0.5


class TestPosition:
    def test_pnl_long(self, sample_position):
        sample_position.update_price(52000.0)
        assert sample_position.unrealized_pnl == 200.0  # (52000 - 50000) * 0.1
        assert sample_position.pnl_pct > 0

    def test_pnl_short(self):
        pos = Position(
            symbol="BTC/USDT",
            strategy_id="strat-1",
            side=Side.SELL,
            quantity=0.1,
            avg_entry_price=50000.0,
        )
        pos.update_price(48000.0)
        assert pos.unrealized_pnl == 200.0  # (50000 - 48000) * 0.1
        assert pos.pnl_pct > 0


class TestStrategyGenome:
    def test_creation(self, random_genome):
        assert random_genome.id
        assert random_genome.name == "test_strategy"
        assert random_genome.signal_tree
        assert random_genome.generation == 0
        assert random_genome.status == StrategyStatus.DORMANT.value

    def test_fitness_fields(self, random_genome):
        assert random_genome.fitness == 0.0
        assert random_genome.sharpe == 0.0
        assert random_genome.max_drawdown == 0.0


class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        bus = EventBus()
        received = []

        async def handler(payload):
            received.append(payload)

        bus.subscribe(Event.BAR_CLOSED, handler)
        bus_task = asyncio.create_task(bus.run())

        await bus.publish(Event.BAR_CLOSED, "test_data")
        await asyncio.sleep(0.1)

        assert len(received) == 1
        assert received[0] == "test_data"

        await bus.stop()
        bus_task.cancel()

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        bus = EventBus()
        results = {"a": 0, "b": 0}

        async def handler_a(payload):
            results["a"] += 1

        async def handler_b(payload):
            results["b"] += 1

        bus.subscribe(Event.BAR_CLOSED, handler_a)
        bus.subscribe(Event.BAR_CLOSED, handler_b)
        bus_task = asyncio.create_task(bus.run())

        await bus.publish(Event.BAR_CLOSED, None)
        await asyncio.sleep(0.1)

        assert results["a"] == 1
        assert results["b"] == 1

        await bus.stop()
        bus_task.cancel()

    @pytest.mark.asyncio
    async def test_stats(self):
        bus = EventBus()
        bus_task = asyncio.create_task(bus.run())

        await bus.publish(Event.BAR_CLOSED, None)
        await bus.publish(Event.BAR_CLOSED, None)
        await bus.publish(Event.TICK_RECEIVED, None)
        await asyncio.sleep(0.1)

        assert bus.stats[Event.BAR_CLOSED] == 2
        assert bus.stats[Event.TICK_RECEIVED] == 1

        await bus.stop()
        bus_task.cancel()
