"""CCXT Exchange Adapter — Unified crypto exchange connectivity."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import AsyncIterator, Optional

from ..core.enums import OrderState, OrderType, Side, Timeframe
from ..core.errors import ExchangeError, OrderError
from ..core.types import Fill, OHLCVBar, Order, OrderBookLevel, OrderBookSnapshot, Position, Tick
from ..core.interfaces import ExchangeAdapter

logger = logging.getLogger(__name__)

TIMEFRAME_MAP = {
    Timeframe.M1: "1m", Timeframe.M3: "3m", Timeframe.M5: "5m",
    Timeframe.M15: "15m", Timeframe.M30: "30m", Timeframe.H1: "1h",
    Timeframe.H2: "2h", Timeframe.H4: "4h", Timeframe.H6: "6h",
    Timeframe.H8: "8h", Timeframe.H12: "12h", Timeframe.D1: "1d",
    Timeframe.D3: "3d", Timeframe.W1: "1w", Timeframe.MN1: "1M",
}


class CCXTAdapter(ExchangeAdapter):
    """CCXT-based adapter for crypto exchanges (Binance, Bybit, OKX, etc.)."""

    def __init__(self, exchange_id: str, api_key: str = "", api_secret: str = "",
                 passphrase: str = "", testnet: bool = True):
        self._exchange_id = exchange_id
        self._api_key = api_key
        self._api_secret = api_secret
        self._passphrase = passphrase
        self._testnet = testnet
        self._exchange = None
        self._connected = False

    @property
    def name(self) -> str:
        return self._exchange_id

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        try:
            import ccxt.async_support as ccxt_async

            exchange_class = getattr(ccxt_async, self._exchange_id, None)
            if exchange_class is None:
                raise ExchangeError(f"Exchange {self._exchange_id} not supported by CCXT")

            config = {
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
            }
            if self._passphrase:
                config["password"] = self._passphrase
            if self._testnet:
                config["sandbox"] = True

            self._exchange = exchange_class(config)
            await self._exchange.load_markets()
            self._connected = True
            logger.info(f"Connected to {self._exchange_id} (testnet={self._testnet})")

        except Exception as e:
            raise ExchangeError(f"Failed to connect to {self._exchange_id}: {e}")

    async def disconnect(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._connected = False

    async def fetch_candles(
        self, symbol: str, timeframe: Timeframe, since: Optional[datetime] = None, limit: int = 500
    ) -> list[OHLCVBar]:
        tf = TIMEFRAME_MAP.get(timeframe, "1h")
        since_ms = int(since.timestamp() * 1000) if since else None

        try:
            ohlcv = await self._exchange.fetch_ohlcv(symbol, tf, since=since_ms, limit=limit)
            return [
                OHLCVBar(
                    timestamp=datetime.fromtimestamp(candle[0] / 1000),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=candle[1],
                    high=candle[2],
                    low=candle[3],
                    close=candle[4],
                    volume=candle[5],
                    exchange=self._exchange_id,
                )
                for candle in ohlcv
            ]
        except Exception as e:
            raise ExchangeError(f"Failed to fetch candles: {e}")

    async def watch_candles(self, symbol: str, timeframe: Timeframe) -> AsyncIterator[OHLCVBar]:
        tf = TIMEFRAME_MAP.get(timeframe, "1h")
        while True:
            try:
                ohlcv = await self._exchange.fetch_ohlcv(symbol, tf, limit=1)
                if ohlcv:
                    candle = ohlcv[-1]
                    yield OHLCVBar(
                        timestamp=datetime.fromtimestamp(candle[0] / 1000),
                        symbol=symbol, timeframe=timeframe,
                        open=candle[1], high=candle[2], low=candle[3],
                        close=candle[4], volume=candle[5], exchange=self._exchange_id,
                    )
                await asyncio.sleep(timeframe.seconds or 60)
            except Exception as e:
                logger.error(f"Error watching candles: {e}")
                await asyncio.sleep(5)

    async def watch_order_book(self, symbol: str, depth: int = 20) -> AsyncIterator[OrderBookSnapshot]:
        while True:
            try:
                ob = await self._exchange.fetch_order_book(symbol, limit=depth)
                yield OrderBookSnapshot(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    exchange=self._exchange_id,
                    bids=[OrderBookLevel(price=b[0], quantity=b[1]) for b in ob.get("bids", [])],
                    asks=[OrderBookLevel(price=a[0], quantity=a[1]) for a in ob.get("asks", [])],
                )
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error watching order book: {e}")
                await asyncio.sleep(5)

    async def watch_trades(self, symbol: str) -> AsyncIterator[Tick]:
        while True:
            try:
                trades = await self._exchange.fetch_trades(symbol, limit=1)
                for t in trades:
                    yield Tick(
                        timestamp=datetime.fromtimestamp(t["timestamp"] / 1000) if t.get("timestamp") else datetime.utcnow(),
                        symbol=symbol,
                        price=t["price"],
                        quantity=t["amount"],
                        side=Side.BUY if t.get("side") == "buy" else Side.SELL,
                        exchange=self._exchange_id,
                        trade_id=t.get("id", ""),
                    )
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error watching trades: {e}")
                await asyncio.sleep(5)

    async def submit_order(self, order: Order) -> Order:
        try:
            order_type = "market" if order.order_type == OrderType.MARKET else "limit"
            side = "buy" if order.side == Side.BUY else "sell"

            params = {}
            if order.reduce_only:
                params["reduceOnly"] = True

            result = await self._exchange.create_order(
                symbol=order.symbol,
                type=order_type,
                side=side,
                amount=order.quantity,
                price=order.price if order.order_type == OrderType.LIMIT else None,
                params=params,
            )

            order.exchange_order_id = result.get("id")
            order.state = OrderState.SUBMITTED
            return order

        except Exception as e:
            order.state = OrderState.REJECTED
            raise OrderError(f"Order submission failed: {e}")

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return Order(id=order_id, symbol=symbol, state=OrderState.CANCELLED)
        except Exception as e:
            raise OrderError(f"Cancel failed: {e}")

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        try:
            result = await self._exchange.fetch_order(order_id, symbol)
            state_map = {
                "open": OrderState.SUBMITTED, "closed": OrderState.FILLED,
                "canceled": OrderState.CANCELLED, "expired": OrderState.EXPIRED,
            }
            return Order(
                id=order_id, symbol=symbol,
                state=state_map.get(result.get("status", ""), OrderState.SUBMITTED),
                filled_quantity=result.get("filled", 0),
                avg_fill_price=result.get("average", 0) or 0,
            )
        except Exception as e:
            raise OrderError(f"Fetch order failed: {e}")

    async def fetch_positions(self) -> list[Position]:
        try:
            positions = await self._exchange.fetch_positions()
            return [
                Position(
                    symbol=p["symbol"],
                    strategy_id="",
                    side=Side.BUY if p.get("side") == "long" else Side.SELL,
                    quantity=abs(p.get("contracts", 0) or 0),
                    avg_entry_price=p.get("entryPrice", 0) or 0,
                    current_price=p.get("markPrice", 0) or 0,
                    unrealized_pnl=p.get("unrealizedPnl", 0) or 0,
                    exchange=self._exchange_id,
                )
                for p in positions if p.get("contracts") and abs(p["contracts"]) > 0
            ]
        except Exception as e:
            raise ExchangeError(f"Fetch positions failed: {e}")

    async def fetch_balance(self) -> dict[str, float]:
        try:
            balance = await self._exchange.fetch_balance()
            return {
                k: float(v.get("total", 0) or 0)
                for k, v in balance.items()
                if isinstance(v, dict) and float(v.get("total", 0) or 0) > 0
            }
        except Exception as e:
            raise ExchangeError(f"Fetch balance failed: {e}")

    async def fetch_ticker(self, symbol: str) -> dict:
        try:
            return await self._exchange.fetch_ticker(symbol)
        except Exception as e:
            raise ExchangeError(f"Fetch ticker failed: {e}")
