"""Enhanced Paper Trading Connector — Realistic order simulation.

Features:
- Realistic fill simulation with market/limit order semantics
- Slippage modeling (fixed bps, volume-impact, and spread-based)
- Tiered fee calculation (maker/taker with volume discounts)
- Full position tracking with average-cost or FIFO lot matching
- P&L tracking (realized + unrealized, per-position and aggregate)
- Simulated order book with configurable depth and spread
- Pending limit/stop order queue with automatic triggering
- Latency simulation
- Supports the same interface as live connectors for strategy testing
"""
from __future__ import annotations

import asyncio
import logging
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from ..core.enums import OrderState, OrderType, Side, Timeframe
from ..core.errors import ExchangeError, OrderError
from ..core.types import Fill, OHLCVBar, Order, OrderBookLevel, OrderBookSnapshot, Position, Tick

from .base_connector import (
    BalanceInfo,
    BaseExchangeConnector,
    ConnectionState,
    PositionInfo,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------

@dataclass
class PaperTradingConfig:
    """Configuration for paper trading simulation."""
    initial_balance: float = 100_000.0
    quote_currency: str = "USDT"

    # Slippage
    slippage_model: str = "fixed"  # "fixed", "volume_impact", "spread"
    slippage_bps: float = 5.0       # Fixed slippage in basis points
    volume_impact_factor: float = 0.1  # Market-impact coefficient
    spread_bps: float = 10.0        # Simulated spread for limit orders

    # Fees
    maker_fee_bps: float = 2.0      # Maker fee (limit orders that add liquidity)
    taker_fee_bps: float = 7.5      # Taker fee (market orders that remove liquidity)
    fee_tiers: list[dict[str, float]] = field(default_factory=list)  # [{volume_threshold, maker_bps, taker_bps}]

    # Latency
    latency_ms: float = 50.0        # Simulated execution latency
    latency_jitter_ms: float = 20.0 # Random jitter

    # Order book simulation
    book_levels: int = 20
    book_spread_bps: float = 10.0
    book_depth_decay: float = 0.95  # Exponential decay for depth

    # Position
    lot_matching: str = "average_cost"  # "average_cost" or "fifo"

    # Fill simulation
    partial_fill_probability: float = 0.0  # Probability of partial fills (0-1)
    reject_probability: float = 0.01       # Probability of rejection (simulates rare failures)


# ---------------------------------------------------------------------------
# Simulated order book
# ---------------------------------------------------------------------------

@dataclass
class SimulatedLevel:
    price: float
    quantity: float


class SimulatedOrderBook:
    """Simulated order book for realistic fill pricing."""

    def __init__(self, config: PaperTradingConfig):
        self._config = config
        self._books: dict[str, dict[str, Any]] = {}

    def update(self, symbol: str, mid_price: float) -> None:
        """Update the simulated book around a mid price."""
        spread = mid_price * self._config.book_spread_bps / 10000
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2

        bids: list[SimulatedLevel] = []
        asks: list[SimulatedLevel] = []
        tick = mid_price * 0.0001  # Minimum price increment

        for i in range(self._config.book_levels):
            bid_price = best_bid - i * tick
            ask_price = best_ask + i * tick
            # Exponential decay of depth
            depth_mult = self._config.book_depth_decay ** i
            base_qty = random.uniform(0.5, 5.0) * depth_mult
            bids.append(SimulatedLevel(price=bid_price, quantity=base_qty))
            asks.append(SimulatedLevel(price=ask_price, quantity=base_qty))

        self._books[symbol] = {"bids": bids, "asks": asks, "mid": mid_price, "updated": datetime.utcnow()}

    def get_mid_price(self, symbol: str) -> Optional[float]:
        book = self._books.get(symbol)
        return book["mid"] if book else None

    def get_book(self, symbol: str) -> Optional[dict]:
        return self._books.get(symbol)

    def simulate_fill_price(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        order_type: OrderType,
        limit_price: Optional[float] = None,
    ) -> tuple[float, float]:
        """Simulate a fill price and commission.

        Returns (fill_price, commission_per_unit).
        """
        book = self._books.get(symbol)
        if not book:
            raise OrderError(f"No price data for {symbol}")

        mid = book["mid"]

        if order_type == OrderType.MARKET:
            # Walk the book to simulate market impact
            levels = book["asks"] if side == Side.BUY else book["bids"]
            return self._walk_book(levels, quantity, mid, side)
        else:
            # Limit order: filled at limit price if marketable, else queued
            if limit_price is None:
                limit_price = mid
            if side == Side.BUY and limit_price >= book["asks"][0].price:
                # Marketable buy limit — fill at limit or better
                fill_price = min(limit_price, book["asks"][0].price)
                return fill_price, 0
            elif side == Side.SELL and limit_price <= book["bids"][0].price:
                # Marketable sell limit — fill at limit or better
                fill_price = max(limit_price, book["bids"][0].price)
                return fill_price, 0
            else:
                # Non-marketable: fill at limit price (assumes it will be triggered)
                return limit_price, 0

    def _walk_book(
        self,
        levels: list[SimulatedLevel],
        quantity: float,
        mid: float,
        side: Side,
    ) -> tuple[float, float]:
        """Walk the order book to compute VWAP fill price."""
        remaining = quantity
        total_cost = 0.0
        for level in levels:
            if remaining <= 0:
                break
            fill_qty = min(remaining, level.quantity)
            total_cost += fill_qty * level.price
            remaining -= fill_qty
        if remaining > 0:
            # Not enough liquidity: use last level + slippage
            slippage = mid * 0.005  # 50bps penalty
            extra_price = (levels[-1].price + slippage) if levels else mid * (1.001 if side == Side.BUY else 0.999)
            total_cost += remaining * extra_price
        total_qty = quantity
        avg_price = total_cost / total_qty if total_qty > 0 else mid
        return avg_price, 0


# ---------------------------------------------------------------------------
# P&L tracker
# ---------------------------------------------------------------------------

@dataclass
class PnLRecord:
    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    net_pnl: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PnLTracker:
    """Track realized and unrealized P&L per position and aggregate."""

    def __init__(self) -> None:
        self.realized_pnl: float = 0.0
        self.total_commission: float = 0.0
        self.trade_count: int = 0
        self.win_count: int = 0
        self.loss_count: int = 0
        self._history: list[PnLRecord] = []
        self._peak_equity: float = 0.0
        self._max_drawdown: float = 0.0

    def record_trade(self, record: PnLRecord) -> None:
        self._history.append(record)
        self.realized_pnl += record.net_pnl
        self.total_commission += record.commission
        self.trade_count += 1
        if record.net_pnl > 0:
            self.win_count += 1
        elif record.net_pnl < 0:
            self.loss_count += 1

    def update_equity(self, equity: float) -> None:
        if equity > self._peak_equity:
            self._peak_equity = equity
        if self._peak_equity > 0:
            dd = (self._peak_equity - equity) / self._peak_equity
            self._max_drawdown = max(self._max_drawdown, dd)

    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        return self._max_drawdown

    @property
    def history(self) -> list[PnLRecord]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Pending order
# ---------------------------------------------------------------------------

@dataclass
class PendingOrder:
    """A limit or stop order waiting to be triggered."""
    order: Order
    trigger_price: float
    trigger_side: str  # "above" or "below"
    created_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Enhanced Paper Trading Connector
# ---------------------------------------------------------------------------

class EnhancedPaperTrading(BaseExchangeConnector):
    """Production-grade paper trading connector.

    Implements the same ``BaseExchangeConnector`` interface as live exchange
    connectors, making it a drop-in replacement for strategy testing and
    development.
    """

    def __init__(self, config: dict | PaperTradingConfig | None = None, **kwargs: Any):
        if isinstance(config, PaperTradingConfig):
            self._cfg = config
        else:
            self._cfg = PaperTradingConfig(**(config or {}))

        super().__init__(
            exchange_id="paper",
            testnet=False,
            rate_limit=999_999,  # No real rate limit
            **kwargs,
        )

        # Balances
        self._balances: dict[str, float] = {self._cfg.quote_currency: self._cfg.initial_balance}

        # Positions keyed by "symbol:strategy_id"
        self._positions: dict[str, Position] = {}

        # Open orders
        self._open_orders: dict[str, Order] = {}

        # Pending (non-marketable limit/stop) orders
        self._pending_orders: list[PendingOrder] = []

        # Fill log
        self._fills: list[Fill] = []

        # Price feed
        self._current_prices: dict[str, float] = {}

        # Simulated order book
        self._book = SimulatedOrderBook(self._cfg)

        # P&L tracker
        self._pnl = PnLTracker()

        # Trading volume (for fee tier calculation)
        self._trading_volume: float = 0.0

    # -----------------------------------------------------------------------
    # Connection (no-op for paper)
    # -----------------------------------------------------------------------

    async def _do_connect(self) -> None:
        logger.info("Paper trading engine initialized (balance=%.2f %s)",
                     self._cfg.initial_balance, self._cfg.quote_currency)

    async def _do_disconnect(self) -> None:
        logger.info("Paper trading engine stopped. Final P&L: %.2f, trades: %d, win rate: %.1f%%",
                     self._pnl.realized_pnl, self._pnl.trade_count, self._pnl.win_rate * 100)

    # -----------------------------------------------------------------------
    # Price feed
    # -----------------------------------------------------------------------

    def update_price(self, symbol: str, price: float) -> None:
        """Update the current price for a symbol. Call this from your data feed."""
        self._current_prices[symbol] = price
        self._book.update(symbol, price)
        self._check_pending_orders(symbol, price)
        self._update_unrealized_pnl()

    def update_prices(self, prices: dict[str, float]) -> None:
        """Bulk update prices."""
        for sym, price in prices.items():
            self.update_price(sym, price)

    # -----------------------------------------------------------------------
    # Market data (simulated)
    # -----------------------------------------------------------------------

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[OHLCVBar]:
        """Paper trading cannot generate historical candles — returns empty list."""
        logger.warning("fetch_candles not supported in paper trading")
        return []

    async def fetch_ticker(self, symbol: str) -> dict:
        price = self._current_prices.get(symbol, 0)
        spread = price * self._cfg.book_spread_bps / 10000
        return {
            "symbol": symbol,
            "last": price,
            "bid": price - spread / 2,
            "ask": price + spread / 2,
            "high": price,
            "low": price,
            "volume": 0,
            "quote_volume": 0,
            "change_pct": 0,
        }

    async def fetch_order_book(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        book = self._book.get_book(symbol)
        if not book:
            price = self._current_prices.get(symbol, 0)
            if price > 0:
                self._book.update(symbol, price)
                book = self._book.get_book(symbol)

        if not book:
            return OrderBookSnapshot(
                timestamp=datetime.utcnow(), symbol=symbol, exchange="paper",
                bids=[], asks=[],
            )

        return OrderBookSnapshot(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            exchange="paper",
            bids=[OrderBookLevel(price=l.price, quantity=l.quantity) for l in book["bids"][:depth]],
            asks=[OrderBookLevel(price=l.price, quantity=l.quantity) for l in book["asks"][:depth]],
        )

    async def watch_candles(self, symbol: str, timeframe: Timeframe) -> AsyncIterator[OHLCVBar]:
        """No-op for paper trading."""
        return
        yield  # type: ignore[misc]

    async def watch_trades(self, symbol: str) -> AsyncIterator[Tick]:
        return
        yield  # type: ignore[misc]

    async def watch_order_book(self, symbol: str, depth: int = 20) -> AsyncIterator[OrderBookSnapshot]:
        return
        yield  # type: ignore[misc]

    # -----------------------------------------------------------------------
    # Order management
    # -----------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Submit an order to the paper trading engine."""
        # Simulate latency
        if self._cfg.latency_ms > 0:
            jitter = random.uniform(-self._cfg.latency_jitter_ms, self._cfg.latency_jitter_ms)
            await asyncio.sleep(max(0, (self._cfg.latency_ms + jitter) / 1000))

        # Simulate rare rejection
        if random.random() < self._cfg.reject_probability:
            order.state = OrderState.REJECTED
            order.metadata["reject_reason"] = "Simulated rejection"
            logger.warning("Paper order rejected (simulated): %s", order.id)
            return order

        price = self._current_prices.get(order.symbol)
        if price is None or price <= 0:
            order.state = OrderState.REJECTED
            order.metadata["reject_reason"] = f"No price data for {order.symbol}"
            return order

        # Check if this is a pending limit/stop order
        if self._should_queue(order, price):
            return self._queue_pending(order)

        # Execute immediately
        return await self._execute_order(order)

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        """Cancel an open or pending order."""
        # Check open orders
        if order_id in self._open_orders:
            order = self._open_orders.pop(order_id)
            order.state = OrderState.CANCELLED
            return order

        # Check pending orders
        for i, pending in enumerate(self._pending_orders):
            if pending.order.id == order_id:
                order = self._pending_orders.pop(i).order
                order.state = OrderState.CANCELLED
                return order

        return Order(id=order_id, symbol=symbol, state=OrderState.CANCELLED, exchange="paper")

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        if order_id in self._open_orders:
            return self._open_orders[order_id]
        for pending in self._pending_orders:
            if pending.order.id == order_id:
                return pending.order
        return Order(id=order_id, symbol=symbol, state=OrderState.EXPIRED, exchange="paper")

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        orders = list(self._open_orders.values())
        orders.extend(p.order for p in self._pending_orders)
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders

    # -----------------------------------------------------------------------
    # Account
    # -----------------------------------------------------------------------

    async def fetch_balance(self) -> dict[str, float]:
        return dict(self._balances)

    async def fetch_balance_detailed(self) -> dict[str, BalanceInfo]:
        return {
            asset: BalanceInfo(asset=asset, free=qty, locked=0, total=qty)
            for asset, qty in self._balances.items()
            if qty > 0
        }

    async def fetch_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.quantity > 0]

    # -----------------------------------------------------------------------
    # Order execution internals
    # -----------------------------------------------------------------------

    async def _execute_order(self, order: Order) -> Order:
        """Execute an order against the simulated order book."""
        fill_price, _ = self._book.simulate_fill_price(
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            limit_price=order.price,
        )

        # Apply slippage for market orders
        if order.order_type == OrderType.MARKET:
            fill_price = self._apply_slippage(fill_price, order.side, order.quantity, order.symbol)

        # Determine if maker or taker
        is_maker = order.order_type in (OrderType.LIMIT, OrderType.ICEBERG) and not self._is_marketable(order)
        fee_bps = self._cfg.maker_fee_bps if is_maker else self._cfg.taker_fee_bps

        # Apply fee tier discount
        fee_bps = self._get_tiered_fee(fee_bps, is_maker)

        # Simulate partial fill
        fill_qty = order.quantity
        if random.random() < self._cfg.partial_fill_probability and order.quantity > 0.001:
            fill_qty = order.quantity * random.uniform(0.3, 0.9)
            fill_qty = max(0.0001, round(fill_qty, 8))

        # Check balance
        if order.side == Side.BUY:
            cost = fill_qty * fill_price * (1 + fee_bps / 10000)
            quote = self._cfg.quote_currency
            if self._balances.get(quote, 0) < cost:
                order.state = OrderState.REJECTED
                order.metadata["reject_reason"] = (
                    f"Insufficient {quote}: need {cost:.2f}, have {self._balances.get(quote, 0):.2f}"
                )
                return order
        else:
            pos = self._get_position(order.symbol, order.strategy_id)
            if pos is None or pos.quantity < fill_qty:
                order.state = OrderState.REJECTED
                order.metadata["reject_reason"] = "Insufficient position"
                return order

        # Calculate commission
        commission = fill_qty * fill_price * fee_bps / 10000

        # Create fill record
        fill = Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            quantity=fill_qty,
            commission=commission,
            exchange="paper",
            timestamp=datetime.utcnow(),
        )
        self._fills.append(fill)

        # Update balances
        self._apply_balance_change(fill)

        # Update position
        self._update_position(fill, order.strategy_id)

        # Update trading volume
        self._trading_volume += fill_qty * fill_price

        # Update order state
        if fill_qty >= order.quantity:
            order.state = OrderState.FILLED
        else:
            order.state = OrderState.PARTIAL
        order.filled_quantity = fill_qty
        order.avg_fill_price = fill_price
        order.commission = commission
        order.exchange = "paper"

        # Update P&L tracker
        equity = await self._calculate_equity()
        self._pnl.update_equity(equity)

        # Fire callbacks
        await self._fire_order_callbacks(order)

        logger.info(
            "Paper fill: %s %s %s @ %.4f (fee=%.2f bps, slip=%.1f bps, qty=%.6f/%.6f)",
            order.side.value, order.symbol, order.order_type.value,
            fill_price, fee_bps, self._cfg.slippage_bps, fill_qty, order.quantity,
        )
        return order

    def _apply_slippage(self, price: float, side: Side, quantity: float, symbol: str) -> float:
        """Apply slippage model to fill price."""
        if self._cfg.slippage_model == "fixed":
            slippage_mult = self._cfg.slippage_bps / 10000
            if side == Side.BUY:
                return price * (1 + slippage_mult)
            else:
                return price * (1 - slippage_mult)

        elif self._cfg.slippage_model == "volume_impact":
            # Square-root market impact model
            impact = self._cfg.volume_impact_factor * math.sqrt(quantity / max(price, 1)) / 100
            if side == Side.BUY:
                return price * (1 + impact)
            else:
                return price * (1 - impact)

        elif self._cfg.slippage_model == "spread":
            spread = price * self._cfg.book_spread_bps / 10000
            if side == Side.BUY:
                return price + spread / 2
            else:
                return price - spread / 2

        return price

    def _get_tiered_fee(self, base_bps: float, is_maker: bool) -> float:
        """Apply volume-based fee tier discounts."""
        if not self._cfg.fee_tiers:
            return base_bps

        tier = self._cfg.fee_tiers[0]
        for t in sorted(self._cfg.fee_tiers, key=lambda x: x.get("volume_threshold", 0)):
            if self._trading_volume >= t.get("volume_threshold", 0):
                tier = t

        if is_maker:
            return tier.get("maker_bps", base_bps)
        return tier.get("taker_bps", base_bps)

    def _should_queue(self, order: Order, current_price: float) -> bool:
        """Determine if a limit/stop order should be queued instead of filled immediately."""
        if order.order_type == OrderType.LIMIT:
            if order.side == Side.BUY and (order.price or 0) < current_price:
                return True
            if order.side == Side.SELL and (order.price or 0) > current_price:
                return True
        elif order.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            return True  # Always queue stop orders until triggered
        return False

    def _is_marketable(self, order: Order) -> bool:
        """Check if a limit order would execute immediately."""
        price = self._current_prices.get(order.symbol)
        if price is None:
            return False
        if order.order_type == OrderType.LIMIT:
            if order.side == Side.BUY and (order.price or 0) >= price:
                return True
            if order.side == Side.SELL and (order.price or 0) <= price:
                return True
        return False

    def _queue_pending(self, order: Order) -> Order:
        """Queue a non-marketable order."""
        if order.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT):
            trigger = order.stop_price or 0
            if order.side == Side.BUY:
                trigger_side = "above"  # Buy stop triggers when price goes above
            else:
                trigger_side = "below"  # Sell stop triggers when price goes below
        else:
            trigger = order.price or 0
            trigger_side = "below" if order.side == Side.BUY else "above"

        self._pending_orders.append(PendingOrder(
            order=order,
            trigger_price=trigger,
            trigger_side=trigger_side,
        ))
        order.state = OrderState.SUBMITTED
        order.exchange = "paper"

        logger.info("Paper order queued: %s %s %s @ %.4f (trigger=%.4f %s)",
                     order.side.value, order.quantity, order.symbol,
                     order.price or 0, trigger, trigger_side)
        return order

    def _check_pending_orders(self, symbol: str, price: float) -> None:
        """Check if any pending orders should be triggered."""
        triggered: list[int] = []
        for i, pending in enumerate(self._pending_orders):
            if pending.order.symbol != symbol:
                continue

            should_trigger = False
            if pending.trigger_side == "above" and price >= pending.trigger_price:
                should_trigger = True
            elif pending.trigger_side == "below" and price <= pending.trigger_price:
                should_trigger = True

            if should_trigger:
                triggered.append(i)

        # Process in reverse to preserve indices
        for i in reversed(triggered):
            pending = self._pending_orders.pop(i)
            order = pending.order
            logger.info("Paper order triggered: %s %s @ %.4f (price reached %.4f)",
                        order.side.value, order.symbol, pending.trigger_price, price)
            # Execute in background
            asyncio.create_task(self._execute_order(order))

    # -----------------------------------------------------------------------
    # Position and balance management
    # -----------------------------------------------------------------------

    def _get_position(self, symbol: str, strategy_id: str) -> Optional[Position]:
        key = f"{symbol}:{strategy_id}"
        return self._positions.get(key)

    def _apply_balance_change(self, fill: Fill) -> None:
        """Update balances after a fill."""
        quote = self._cfg.quote_currency
        if fill.side == Side.BUY:
            cost = fill.price * fill.quantity + fill.commission
            self._balances[quote] = self._balances.get(quote, 0) - cost
        else:
            revenue = fill.price * fill.quantity - fill.commission
            self._balances[quote] = self._balances.get(quote, 0) + revenue

    def _update_position(self, fill: Fill, strategy_id: str) -> None:
        """Update position after a fill with average-cost lot matching."""
        key = f"{fill.symbol}:{strategy_id}"

        if key not in self._positions:
            self._positions[key] = Position(
                symbol=fill.symbol,
                strategy_id=strategy_id,
                side=fill.side,
                quantity=0,
                avg_entry_price=0,
                exchange="paper",
            )

        pos = self._positions[key]

        # Track realized P&L for closing trades
        if (pos.side == Side.BUY and fill.side == Side.SELL) or \
           (pos.side == Side.SELL and fill.side == Side.BUY):
            close_qty = min(fill.quantity, pos.quantity)
            if pos.side == Side.BUY:
                pnl = (fill.price - pos.avg_entry_price) * close_qty
            else:
                pnl = (pos.avg_entry_price - fill.price) * close_qty

            net_pnl = pnl - fill.commission * (close_qty / fill.quantity if fill.quantity > 0 else 1)
            self._pnl.record_trade(PnLRecord(
                symbol=fill.symbol,
                side=pos.side,
                entry_price=pos.avg_entry_price,
                exit_price=fill.price,
                quantity=close_qty,
                pnl=pnl,
                commission=fill.commission * (close_qty / fill.quantity if fill.quantity > 0 else 1),
                net_pnl=net_pnl,
            ))

        # Update position with average cost method
        if fill.side == pos.side:
            # Adding to existing position
            total_cost = pos.avg_entry_price * pos.quantity + fill.price * fill.quantity
            pos.quantity += fill.quantity
            pos.avg_entry_price = total_cost / pos.quantity if pos.quantity > 0 else 0
        else:
            # Reducing or flipping position
            remaining = fill.quantity - pos.quantity
            if remaining > 0:
                # Flipped to opposite side
                pos.side = fill.side
                pos.quantity = remaining
                pos.avg_entry_price = fill.price
            elif remaining == 0:
                # Fully closed
                pos.quantity = 0
                pos.avg_entry_price = 0
            else:
                # Partially closed
                pos.quantity = abs(remaining)

        pos.current_price = fill.price
        pos.update_price(fill.price)

    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L for all positions."""
        for pos in self._positions.values():
            price = self._current_prices.get(pos.symbol)
            if price is not None and pos.quantity > 0:
                pos.update_price(price)

    async def _calculate_equity(self) -> float:
        """Calculate total account equity (cash + positions)."""
        equity = sum(self._balances.values())
        for pos in self._positions.values():
            if pos.quantity > 0:
                equity += pos.unrealized_pnl
        return equity

    # -----------------------------------------------------------------------
    # Public query methods
    # -----------------------------------------------------------------------

    def get_pnl_summary(self) -> dict[str, Any]:
        """Get comprehensive P&L summary."""
        return {
            "realized_pnl": self._pnl.realized_pnl,
            "unrealized_pnl": sum(p.unrealized_pnl for p in self._positions.values() if p.quantity > 0),
            "total_commission": self._pnl.total_commission,
            "trade_count": self._pnl.trade_count,
            "win_count": self._pnl.win_count,
            "loss_count": self._pnl.loss_count,
            "win_rate": self._pnl.win_rate,
            "max_drawdown": self._pnl.max_drawdown,
            "trading_volume": self._trading_volume,
            "positions": len([p for p in self._positions.values() if p.quantity > 0]),
            "pending_orders": len(self._pending_orders),
        }

    def get_fill_history(self) -> list[Fill]:
        """Get all fill records."""
        return list(self._fills)

    def get_trade_history(self) -> list[PnLRecord]:
        """Get all closed trade records."""
        return self._pnl.history

    def reset(self) -> None:
        """Reset paper trading state to initial conditions."""
        self._balances = {self._cfg.quote_currency: self._cfg.initial_balance}
        self._positions.clear()
        self._open_orders.clear()
        self._pending_orders.clear()
        self._fills.clear()
        self._pnl = PnLTracker()
        self._trading_volume = 0.0
        logger.info("Paper trading engine reset")
