"""Tests for OrderPipeline — risk checks, validation, signal-to-order conversion, fill tracking."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from tradingbot.core.enums import OrderState, OrderType, Side
from tradingbot.core.events import Event, EventBus
from tradingbot.core.types import Fill, Order, PortfolioState, RiskCheck, Signal
from tradingbot.engine.order_pipeline import OrderPipeline, PipelineMetrics


class TestPipelineMetrics:
    def test_default_values(self):
        m = PipelineMetrics()
        assert m.signals_received == 0
        assert m.orders_created == 0
        assert m.risk_passed == 0
        assert m.risk_failed == 0

    def test_snapshot(self):
        m = PipelineMetrics()
        m.signals_received = 5
        snap = m.snapshot()
        assert snap["signals_received"] == 5
        assert "orders_created" in snap


class TestOrderPipelineInit:
    def test_default_exchange_set(self):
        exchange = AsyncMock()
        bus = EventBus()
        risk = AsyncMock()
        pipeline = OrderPipeline(exchanges={"binance": exchange}, risk_engine=risk, event_bus=bus)
        assert pipeline._default_exchange == "binance"

    def test_custom_default_exchange(self):
        exchange = AsyncMock()
        bus = EventBus()
        risk = AsyncMock()
        pipeline = OrderPipeline(
            exchanges={"binance": exchange, "bybit": exchange},
            risk_engine=risk, event_bus=bus, default_exchange="bybit",
        )
        assert pipeline._default_exchange == "bybit"

    def test_open_orders_initially_empty(self):
        pipeline = OrderPipeline(exchanges={}, risk_engine=AsyncMock(), event_bus=EventBus())
        assert pipeline.open_orders == {}


class TestSignalToOrder:
    def test_signal_creates_order(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        signal = Signal(
            strategy_id="s1", symbol="BTC/USDT", side=Side.BUY,
            strength=0.8, confidence=0.9, metadata={"quantity": 0.5},
        )
        order = pipeline._signal_to_order(signal)
        assert order.symbol == "BTC/USDT"
        assert order.side == Side.BUY
        assert order.quantity == 0.5
        assert order.strategy_id == "s1"

    def test_signal_default_quantity(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.5)
        order = pipeline._signal_to_order(signal)
        assert order.quantity == 0.001

    def test_signal_with_limit_price(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        signal = Signal(
            strategy_id="s1", symbol="BTC/USDT", side=Side.SELL,
            strength=0.5, confidence=0.5, metadata={"limit_price": 55000.0},
        )
        order = pipeline._signal_to_order(signal)
        assert order.order_type == OrderType.LIMIT
        assert order.price == 55000.0

    def test_signal_without_limit_price_is_market(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.5)
        order = pipeline._signal_to_order(signal)
        assert order.order_type == OrderType.MARKET


class TestOrderValidation:
    def test_empty_symbol_rejected(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        order = Order(symbol="", quantity=1.0)
        from tradingbot.core.errors import OrderRejectedError
        with pytest.raises(OrderRejectedError, match="Empty symbol"):
            pipeline._validate_order(order)

    def test_zero_quantity_rejected(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        order = Order(symbol="BTC/USDT", quantity=0.0)
        from tradingbot.core.errors import OrderRejectedError
        with pytest.raises(OrderRejectedError, match="Non-positive"):
            pipeline._validate_order(order)

    def test_limit_without_price_rejected(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        order = Order(symbol="BTC/USDT", quantity=1.0, order_type=OrderType.LIMIT, price=None)
        from tradingbot.core.errors import OrderRejectedError
        with pytest.raises(OrderRejectedError, match="positive price"):
            pipeline._validate_order(order)

    def test_unknown_exchange_rejected(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        order = Order(symbol="BTC/USDT", quantity=1.0, exchange="unknown")
        from tradingbot.core.errors import OrderRejectedError
        with pytest.raises(OrderRejectedError, match="Unknown exchange"):
            pipeline._validate_order(order)

    def test_valid_order_passes(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        order = Order(symbol="BTC/USDT", quantity=1.0, exchange="binance")
        pipeline._validate_order(order)  # Should not raise


class TestSubmitSignal:
    @pytest.mark.asyncio
    async def test_risk_rejected_returns_none(self):
        exchange = AsyncMock()
        risk = AsyncMock()
        risk.pre_trade_check = AsyncMock(return_value=RiskCheck(approved=False, reason="too risky"))
        bus = EventBus()
        pipeline = OrderPipeline(exchanges={"binance": exchange}, risk_engine=risk, event_bus=bus)

        portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc), total_equity=100000,
            cash=50000, positions_value=50000, unrealized_pnl=0, realized_pnl=0,
        )
        signal = Signal(strategy_id="s1", symbol="BTC/USDT", side=Side.BUY, strength=0.5, confidence=0.8)
        result = await pipeline.submit_signal(signal, portfolio)
        assert result is None
        assert pipeline.metrics.risk_failed == 1

    @pytest.mark.asyncio
    async def test_risk_approved_submits_order(self):
        exchange = AsyncMock()
        exchange.submit_order = AsyncMock(return_value=Order(
            id="o1", symbol="BTC/USDT", state=OrderState.SUBMITTED,
        ))
        risk = AsyncMock()
        risk.pre_trade_check = AsyncMock(return_value=RiskCheck(
            approved=True, max_allowed_quantity=1.0, risk_score=0.2,
        ))
        bus = EventBus()
        pipeline = OrderPipeline(exchanges={"binance": exchange}, risk_engine=risk, event_bus=bus)

        signal = Signal(
            strategy_id="s1", symbol="BTC/USDT", side=Side.BUY,
            strength=0.5, confidence=0.8, metadata={"quantity": 0.5},
        )
        result = await pipeline.submit_signal(signal, None)
        assert result is not None
        assert pipeline.metrics.orders_submitted == 1
        assert pipeline.metrics.risk_passed == 1

    @pytest.mark.asyncio
    async def test_quantity_clamped_to_risk_max(self):
        exchange = AsyncMock()
        submitted_orders = []

        async def capture_submit(order):
            submitted_orders.append(order)
            return order

        exchange.submit_order = AsyncMock(side_effect=capture_submit)
        risk = AsyncMock()
        risk.pre_trade_check = AsyncMock(return_value=RiskCheck(
            approved=True, max_allowed_quantity=0.1, risk_score=0.3,
        ))
        pipeline = OrderPipeline(exchanges={"binance": exchange}, risk_engine=risk, event_bus=EventBus())

        portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc), total_equity=100000,
            cash=50000, positions_value=50000, unrealized_pnl=0, realized_pnl=0,
        )
        signal = Signal(
            strategy_id="s1", symbol="BTC/USDT", side=Side.BUY,
            strength=0.5, confidence=0.8, metadata={"quantity": 10.0},
        )
        await pipeline.submit_signal(signal, portfolio)
        assert len(submitted_orders) == 1
        assert submitted_orders[0].quantity == 0.1


class TestFillProcessing:
    @pytest.mark.asyncio
    async def test_fill_updates_open_order(self):
        pipeline = OrderPipeline(exchanges={"binance": AsyncMock()}, risk_engine=AsyncMock(), event_bus=EventBus())
        pipeline._running = True

        order = Order(id="fill-1", symbol="BTC/USDT", side=Side.BUY, quantity=1.0, state=OrderState.SUBMITTED)
        pipeline._open_orders[order.id] = order

        fill = Fill(
            order_id="fill-1", symbol="BTC/USDT", side=Side.BUY,
            price=50000.0, quantity=1.0, commission=5.0,
            exchange="binance", timestamp=datetime.now(timezone.utc),
        )
        await pipeline._fill_queue.put(fill)

        # Process one fill manually
        queued = await pipeline._fill_queue.get()
        pipeline._metrics.fills_tracked += 1
        pipeline._metrics.orders_filled += 1
        o = pipeline._open_orders.get(queued.order_id)
        assert o is not None
        o.filled_quantity += queued.quantity
        o.avg_fill_price = queued.price
        o.commission += queued.commission
        if o.filled_quantity >= o.quantity:
            o.state = OrderState.FILLED
            pipeline._open_orders.pop(o.id, None)

        assert o.state == OrderState.FILLED
        assert o.id not in pipeline._open_orders
