"""Base Exchange Connector — Abstract base class for all exchange connectors.

Provides:
- Standard interface for all exchanges (extends core ExchangeAdapter)
- Connection state management with auto-reconnection
- Callback registration for market data and user data events
- Rate limiting with token bucket algorithm
- Retry logic with exponential backoff
- Shared dataclasses for balances and positions
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, AsyncIterator, Callable, Coroutine, Optional

from ..core.enums import OrderState, OrderType, Side, Timeframe
from ..core.errors import ExchangeError, OrderError
from ..core.interfaces import ExchangeAdapter
from ..core.types import Fill, OHLCVBar, Order, OrderBookLevel, OrderBookSnapshot, Position, Tick

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and dataclasses
# ---------------------------------------------------------------------------

class ConnectionState(Enum):
    """Connection lifecycle states."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    CLOSED = auto()


@dataclass
class BalanceInfo:
    """Detailed balance information for an asset."""
    asset: str
    free: float = 0.0
    locked: float = 0.0
    total: float = 0.0
    usd_value: float = 0.0

    @property
    def available(self) -> float:
        return self.free


@dataclass
class PositionInfo:
    """Extended position information."""
    symbol: str
    side: Side = Side.BUY
    quantity: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: float = 1.0
    margin_type: str = "cross"
    liquidation_price: float = 0.0
    mark_price: float = 0.0
    exchange: str = ""
    opened_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def notional_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        mult = 1 if self.side == Side.BUY else -1
        return mult * (self.current_price - self.avg_entry_price) / self.avg_entry_price


@dataclass
class RateLimitRule:
    """Rate limit configuration."""
    max_requests: int = 1200
    window_seconds: float = 60.0
    weight_per_request: int = 1


# ---------------------------------------------------------------------------
# Callback type aliases
# ---------------------------------------------------------------------------
CandleCallback = Callable[[OHLCVBar], Coroutine[Any, Any, None]]
TradeCallback = Callable[[Tick], Coroutine[Any, Any, None]]
OrderBookCallback = Callable[[OrderBookSnapshot], Coroutine[Any, Any, None]]
OrderCallback = Callable[[Order], Coroutine[Any, Any, None]]
BalanceCallback = Callable[[dict[str, BalanceInfo]], Coroutine[Any, Any, None]]
ErrorCallback = Callable[[Exception], Coroutine[Any, Any, None]]


# ---------------------------------------------------------------------------
# Rate limiter (token bucket)
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """Async-safe token bucket rate limiter."""

    def __init__(self, max_tokens: int, refill_rate: float):
        self.max_tokens = max_tokens
        self.tokens = float(max_tokens)
        self.refill_rate = refill_rate  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Wait until enough tokens are available, then consume them."""
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                # Calculate wait time for enough tokens
                deficit = tokens - self.tokens
                wait_time = deficit / self.refill_rate
            await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self._last_refill = now


# ---------------------------------------------------------------------------
# Base connector
# ---------------------------------------------------------------------------

class BaseExchangeConnector(ExchangeAdapter):
    """Abstract base class for all exchange connectors.

    Extends core ``ExchangeAdapter`` with:
    - Connection state machine
    - Auto-reconnection with exponential backoff
    - Token-bucket rate limiting
    - Event callback registration
    - Retry logic for transient failures
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        testnet: bool = True,
        rate_limit: int = 1200,
        rate_limit_window: float = 60.0,
        max_reconnect_attempts: int = 10,
        reconnect_base_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
        request_timeout: float = 30.0,
    ):
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._testnet = testnet
        self._request_timeout = request_timeout

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = max_reconnect_attempts
        self._reconnect_base_delay = reconnect_base_delay
        self._max_reconnect_delay = max_reconnect_delay

        # Rate limiter
        self._rate_limiter = TokenBucketRateLimiter(
            max_tokens=rate_limit,
            refill_rate=rate_limit / rate_limit_window,
        )

        # Callbacks
        self._on_candle: list[CandleCallback] = []
        self._on_trade: list[TradeCallback] = []
        self._on_orderbook: list[OrderBookCallback] = []
        self._on_order_update: list[OrderCallback] = []
        self._on_balance_update: list[BalanceCallback] = []
        self._on_error: list[ErrorCallback] = []

        # WebSocket management
        self._ws_tasks: dict[str, asyncio.Task] = {}
        self._ws_stop_events: dict[str, asyncio.Event] = {}

        # HTTP session
        self._session: Any = None

        # Internal state
        self._listen_key: str = ""
        self._listen_key_task: Optional[asyncio.Task] = None

    # ---- Properties --------------------------------------------------------

    @property
    def name(self) -> str:
        return self._exchange_id

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    @property
    def connection_state(self) -> ConnectionState:
        return self._state

    @property
    def is_testnet(self) -> bool:
        return self._testnet

    # ---- Callback registration ---------------------------------------------

    def on_candle(self, callback: CandleCallback) -> None:
        self._on_candle.append(callback)

    def on_trade(self, callback: TradeCallback) -> None:
        self._on_trade.append(callback)

    def on_orderbook(self, callback: OrderBookCallback) -> None:
        self._on_orderbook.append(callback)

    def on_order_update(self, callback: OrderCallback) -> None:
        self._on_order_update.append(callback)

    def on_balance_update(self, callback: BalanceCallback) -> None:
        self._on_balance_update.append(callback)

    def on_error(self, callback: ErrorCallback) -> None:
        self._on_error.append(callback)

    # ---- Connection lifecycle ----------------------------------------------

    async def connect(self) -> None:
        """Connect to the exchange. Handles state transitions and error callbacks."""
        if self._state == ConnectionState.CONNECTED:
            return
        self._state = ConnectionState.CONNECTING
        try:
            await self._do_connect()
            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0
            logger.info("Connected to %s (testnet=%s)", self._exchange_id, self._testnet)
        except Exception as exc:
            self._state = ConnectionState.DISCONNECTED
            logger.error("Failed to connect to %s: %s", self._exchange_id, exc)
            await self._fire_error_callbacks(exc)
            raise ExchangeError(f"Failed to connect to {self._exchange_id}: {exc}") from exc

    async def disconnect(self) -> None:
        """Gracefully disconnect from the exchange."""
        self._state = ConnectionState.CLOSED
        # Cancel all WS tasks
        for name, task in self._ws_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._ws_tasks.clear()
        # Cancel listen-key refresh
        if self._listen_key_task and not self._listen_key_task.done():
            self._listen_key_task.cancel()
            try:
                await self._listen_key_task
            except asyncio.CancelledError:
                pass
        await self._do_disconnect()
        logger.info("Disconnected from %s", self._exchange_id)

    async def reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        if self._state == ConnectionState.RECONNECTING:
            return
        self._state = ConnectionState.RECONNECTING
        logger.warning("Reconnecting to %s (attempt %d/%d)",
                        self._exchange_id, self._reconnect_attempts + 1, self._max_reconnect_attempts)
        while self._reconnect_attempts < self._max_reconnect_attempts:
            delay = min(
                self._reconnect_base_delay * (2 ** self._reconnect_attempts),
                self._max_reconnect_delay,
            )
            await asyncio.sleep(delay)
            self._reconnect_attempts += 1
            try:
                await self._do_disconnect()
                await self._do_connect()
                self._state = ConnectionState.CONNECTED
                self._reconnect_attempts = 0
                logger.info("Reconnected to %s", self._exchange_id)
                return
            except Exception as exc:
                logger.warning("Reconnect attempt %d failed: %s", self._reconnect_attempts, exc)
        self._state = ConnectionState.DISCONNECTED
        raise ExchangeError(f"Failed to reconnect to {self._exchange_id} after {self._max_reconnect_attempts} attempts")

    # ---- Rate-limited request wrapper --------------------------------------

    async def _rate_limited_request(self, coro: Any, weight: int = 1) -> Any:
        """Execute a coroutine after acquiring rate-limit tokens."""
        await self._rate_limiter.acquire(weight)
        return await coro

    # ---- Retry wrapper -----------------------------------------------------

    async def _retry(self, coro_factory: Any, max_retries: int = 3, base_delay: float = 0.5) -> Any:
        """Retry a coroutine factory with exponential backoff on transient errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return await coro_factory()
            except (ExchangeError, OrderError) as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("Request failed (attempt %d/%d), retrying in %.1fs: %s",
                                   attempt + 1, max_retries, delay, exc)
                    await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    # ---- Abstract hooks for subclasses -------------------------------------

    @abstractmethod
    async def _do_connect(self) -> None:
        """Perform the actual exchange connection. Subclasses must implement."""

    @abstractmethod
    async def _do_disconnect(self) -> None:
        """Perform the actual exchange disconnection. Subclasses must implement."""

    # ---- WebSocket management helpers --------------------------------------

    async def _start_ws_stream(self, name: str, ws_coro: Any) -> None:
        """Start a named WebSocket stream task with auto-reconnect."""
        stop_event = asyncio.Event()
        self._ws_stop_events[name] = stop_event

        async def _run() -> None:
            while not stop_event.is_set():
                try:
                    await ws_coro()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    if stop_event.is_set():
                        break
                    logger.warning("WS stream '%s' error, reconnecting in 5s: %s", name, exc)
                    await self._fire_error_callbacks(exc)
                    await asyncio.sleep(5)

        self._ws_tasks[name] = asyncio.create_task(_run())
        logger.info("Started WS stream: %s", name)

    async def _stop_ws_stream(self, name: str) -> None:
        """Stop a named WebSocket stream."""
        if name in self._ws_stop_events:
            self._ws_stop_events[name].set()
        if name in self._ws_tasks:
            self._ws_tasks[name].cancel()
            try:
                await self._ws_tasks[name]
            except asyncio.CancelledError:
                pass
            del self._ws_tasks[name]
            del self._ws_stop_events[name]

    # ---- Callback dispatchers ----------------------------------------------

    async def _fire_candle_callbacks(self, bar: OHLCVBar) -> None:
        for cb in self._on_candle:
            try:
                await cb(bar)
            except Exception as exc:
                logger.error("Candle callback error: %s", exc)

    async def _fire_trade_callbacks(self, tick: Tick) -> None:
        for cb in self._on_trade:
            try:
                await cb(tick)
            except Exception as exc:
                logger.error("Trade callback error: %s", exc)

    async def _fire_orderbook_callbacks(self, book: OrderBookSnapshot) -> None:
        for cb in self._on_orderbook:
            try:
                await cb(book)
            except Exception as exc:
                logger.error("Orderbook callback error: %s", exc)

    async def _fire_order_callbacks(self, order: Order) -> None:
        for cb in self._on_order_update:
            try:
                await cb(order)
            except Exception as exc:
                logger.error("Order callback error: %s", exc)

    async def _fire_balance_callbacks(self, balances: dict[str, BalanceInfo]) -> None:
        for cb in self._on_balance_update:
            try:
                await cb(balances)
            except Exception as exc:
                logger.error("Balance callback error: %s", exc)

    async def _fire_error_callbacks(self, error: Exception) -> None:
        for cb in self._on_error:
            try:
                await cb(error)
            except Exception:
                logger.error("Error callback itself raised", exc_info=True)

    # ---- Utility -----------------------------------------------------------

    @staticmethod
    def _ts_to_dt(ts_ms: int) -> datetime:
        """Convert millisecond timestamp to datetime."""
        return datetime.utcfromtimestamp(ts_ms / 1000)
