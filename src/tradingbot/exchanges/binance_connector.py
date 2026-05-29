"""Binance Exchange Connector — Production-grade Binance API integration.

Features:
- REST API client with HMAC-SHA256 request signing
- WebSocket streams (klines, trades, order book, user data)
- Order placement (market, limit, stop-loss, OCO)
- Account/balance queries with full asset tracking
- Token-bucket rate limiting integrated with Binance weight system
- Auto-reconnection on WebSocket disconnect
- Exponential backoff retry logic
- Binance testnet (spot + futures) support
- Uses aiohttp for async HTTP and native websockets for WS streams
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from urllib.parse import urlencode

import aiohttp

from ..core.enums import OrderState, OrderType, Side, Timeframe
from ..core.errors import ExchangeError, OrderError
from ..core.types import OHLCVBar, Order, OrderBookLevel, OrderBookSnapshot, Position, Tick

from .base_connector import (
    BalanceInfo,
    BaseExchangeConnector,
    ConnectionState,
)

logger = logging.getLogger(__name__)

# Binance interval map
_INTERVAL_MAP: dict[Timeframe, str] = {
    Timeframe.M1: "1m", Timeframe.M3: "3m", Timeframe.M5: "5m",
    Timeframe.M15: "15m", Timeframe.M30: "30m", Timeframe.H1: "1h",
    Timeframe.H2: "2h", Timeframe.H4: "4h", Timeframe.H6: "6h",
    Timeframe.H8: "8h", Timeframe.H12: "12h", Timeframe.D1: "1d",
    Timeframe.D3: "3d", Timeframe.W1: "1w", Timeframe.MN1: "1M",
}

# Order type mapping to Binance
_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP_MARKET: "STOP_MARKET",
    OrderType.STOP_LIMIT: "STOP_LOSS_LIMIT",
    OrderType.ICEBERG: "LIMIT",
}


class BinanceConnector(BaseExchangeConnector):
    """Full-featured Binance connector supporting spot and futures.

    Parameters
    ----------
    api_key, api_secret : str
        Binance API credentials.
    testnet : bool
        If True, use Binance testnet endpoints.
    market_type : str
        ``"spot"`` or ``"futures"`` (default ``"spot"``).
    """

    # Base URLs
    _SPOT_REST = "https://api.binance.com"
    _SPOT_WS = "wss://stream.binance.com:9443/ws"
    _SPOT_REST_TEST = "https://testnet.binance.vision"
    _SPOT_WS_TEST = "wss://testnet.binance.vision/ws"

    _FUTURES_REST = "https://fapi.binance.com"
    _FUTURES_WS = "wss://fstream.binance.com/ws"
    _FUTURES_REST_TEST = "https://testnet.binancefuture.com"
    _FUTURES_WS_TEST = "wss://stream.binancefuture.com/ws"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        market_type: str = "spot",
        **kwargs: Any,
    ):
        super().__init__(
            exchange_id="binance",
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit=kwargs.pop("rate_limit", 1200),
            **kwargs,
        )
        self._market_type = market_type

        # Resolve base URLs
        if market_type == "futures":
            self._rest_base = self._FUTURES_REST_TEST if testnet else self._FUTURES_REST
            self._ws_base = self._FUTURES_WS_TEST if testnet else self._FUTURES_WS
        else:
            self._rest_base = self._SPOT_REST_TEST if testnet else self._SPOT_REST
            self._ws_base = self._SPOT_WS_TEST if testnet else self._SPOT_WS

    # -----------------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------------

    async def _do_connect(self) -> None:
        """Create aiohttp session and verify connectivity."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._request_timeout),
            headers={"X-MBX-APIKEY": self._api_key},
        )
        # Connectivity ping
        await self._public_request("/api/v3/ping" if self._market_type == "spot" else "/fapi/v1/ping")
        logger.info("Binance %s API reachable (testnet=%s)", self._market_type, self._testnet)

    async def _do_disconnect(self) -> None:
        """Close aiohttp session and stop all WS streams."""
        # Stop all WS streams
        for name in list(self._ws_tasks.keys()):
            await self._stop_ws_stream(name)
        if self._listen_key_task and not self._listen_key_task.done():
            self._listen_key_task.cancel()
            try:
                await self._listen_key_task
            except asyncio.CancelledError:
                pass
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -----------------------------------------------------------------------
    # HMAC-SHA256 signing
    # -----------------------------------------------------------------------

    def _sign(self, params: dict[str, Any]) -> str:
        """Create HMAC-SHA256 signature for a request."""
        query = urlencode(params)
        return hmac.new(
            self._api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # -----------------------------------------------------------------------
    # REST helpers
    # -----------------------------------------------------------------------

    async def _public_request(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Unsigned public API request."""
        url = f"{self._rest_base}{path}"
        async with self._session.get(url, params=params) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise ExchangeError(f"Binance API error {resp.status}: {body}")
            return await resp.json()

    async def _signed_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        weight: int = 1,
    ) -> Any:
        """Signed (authenticated) API request with rate limiting."""
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)

        url = f"{self._rest_base}{path}"

        await self._rate_limiter.acquire(weight)

        if method == "GET":
            async with self._session.get(url, params=params) as resp:
                return await self._handle_response(resp)
        elif method == "POST":
            async with self._session.post(url, params=params) as resp:
                return await self._handle_response(resp)
        elif method == "DELETE":
            async with self._session.delete(url, params=params) as resp:
                return await self._handle_response(resp)
        else:
            raise ExchangeError(f"Unsupported HTTP method: {method}")

    async def _handle_response(self, resp: aiohttp.ClientResponse) -> Any:
        """Process API response, handling Binance error codes."""
        body = await resp.json()
        if resp.status != 200:
            code = body.get("code", resp.status)
            msg = body.get("msg", "Unknown error")
            raise ExchangeError(f"Binance error {code}: {msg}")
        return body

    # -----------------------------------------------------------------------
    # Market data
    # -----------------------------------------------------------------------

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[OHLCVBar]:
        interval = _INTERVAL_MAP.get(timeframe, "1h")
        params: dict[str, Any] = {
            "symbol": self._normalize_symbol(symbol),
            "interval": interval,
            "limit": min(limit, 1500),
        }
        if since:
            params["startTime"] = int(since.timestamp() * 1000)

        path = "/api/v3/klines" if self._market_type == "spot" else "/fapi/v1/klines"
        data = await self._retry(lambda: self._public_request(path, params))

        bars: list[OHLCVBar] = []
        for k in data:
            bars.append(OHLCVBar(
                timestamp=self._ts_to_dt(int(k[0])),
                symbol=symbol,
                timeframe=timeframe,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                exchange="binance",
                trades_count=int(k[8]),
                vwap=float(k[7]) / float(k[5]) if float(k[5]) > 0 else 0.0,
            ))
        return bars

    async def fetch_ticker(self, symbol: str) -> dict:
        sym = self._normalize_symbol(symbol)
        if self._market_type == "spot":
            data = await self._public_request("/api/v3/ticker/24hr", {"symbol": sym})
        else:
            data = await self._public_request("/fapi/v1/ticker/24hr", {"symbol": sym})
        return {
            "symbol": symbol,
            "last": float(data.get("lastPrice", 0)),
            "bid": float(data.get("bidPrice", 0)),
            "ask": float(data.get("askPrice", 0)),
            "high": float(data.get("highPrice", 0)),
            "low": float(data.get("lowPrice", 0)),
            "volume": float(data.get("volume", 0)),
            "quote_volume": float(data.get("quoteVolume", 0)),
            "change_pct": float(data.get("priceChangePercent", 0)),
        }

    async def fetch_order_book(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        sym = self._normalize_symbol(symbol)
        path = "/api/v3/depth" if self._market_type == "spot" else "/fapi/v1/depth"
        params = {"symbol": sym, "limit": depth}
        data = await self._public_request(path, params)
        return OrderBookSnapshot(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            exchange="binance",
            bids=[OrderBookLevel(price=float(b[0]), quantity=float(b[1])) for b in data.get("bids", [])],
            asks=[OrderBookLevel(price=float(a[0]), quantity=float(a[1])) for a in data.get("asks", [])],
        )

    # -----------------------------------------------------------------------
    # WebSocket streams
    # -----------------------------------------------------------------------

    async def watch_candles(self, symbol: str, timeframe: Timeframe) -> AsyncIterator[OHLCVBar]:
        """Stream real-time kline data via WebSocket."""
        interval = _INTERVAL_MAP.get(timeframe, "1h")
        sym = self._normalize_symbol(symbol).lower()
        stream = f"{sym}@kline_{interval}"

        async def _stream() -> None:
            async for raw in self._ws_connect([stream]):
                if "k" not in raw:
                    continue
                k = raw["k"]
                bar = OHLCVBar(
                    timestamp=self._ts_to_dt(int(k["t"])),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(k["o"]),
                    high=float(k["h"]),
                    low=float(k["l"]),
                    close=float(k["c"]),
                    volume=float(k["v"]),
                    exchange="binance",
                    trades_count=int(k["n"]),
                    vwap=float(k["q"]) / float(k["v"]) if float(k["v"]) > 0 else 0.0,
                )
                yield bar

        async for bar in _stream():
            await self._fire_candle_callbacks(bar)
            yield bar

    async def watch_trades(self, symbol: str) -> AsyncIterator[Tick]:
        """Stream real-time trades via WebSocket."""
        sym = self._normalize_symbol(symbol).lower()
        stream = f"{sym}@trade"

        async def _stream() -> None:
            async for raw in self._ws_connect([stream]):
                tick = Tick(
                    timestamp=self._ts_to_dt(int(raw["T"])),
                    symbol=symbol,
                    price=float(raw["p"]),
                    quantity=float(raw["q"]),
                    side=Side.BUY if raw["m"] is False else Side.SELL,
                    exchange="binance",
                    trade_id=str(raw["t"]),
                )
                yield tick

        async for tick in _stream():
            await self._fire_trade_callbacks(tick)
            yield tick

    async def watch_order_book(self, symbol: str, depth: int = 20) -> AsyncIterator[OrderBookSnapshot]:
        """Stream order book diffs via WebSocket."""
        sym = self._normalize_symbol(symbol).lower()
        stream = f"{sym}@depth{depth}@100ms"

        async def _stream() -> None:
            async for raw in self._ws_connect([stream]):
                book = OrderBookSnapshot(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    exchange="binance",
                    bids=[OrderBookLevel(price=float(b[0]), quantity=float(b[1])) for b in raw.get("bids", [])],
                    asks=[OrderBookLevel(price=float(a[0]), quantity=float(a[1])) for a in raw.get("asks", [])],
                )
                yield book

        async for book in _stream():
            await self._fire_orderbook_callbacks(book)
            yield book

    async def watch_user_data(self) -> AsyncIterator[Order]:
        """Stream user data events (order updates, balance changes).

        For spot this uses a listen-key keepalive stream.
        For futures the user data stream includes account updates.
        """
        listen_key = await self._create_listen_key()

        # Keepalive task
        async def _keepalive() -> None:
            while True:
                await asyncio.sleep(1800)  # refresh every 30 min
                try:
                    await self._keepalive_listen_key(listen_key)
                except Exception as exc:
                    logger.warning("Listen key keepalive failed: %s", exc)

        self._listen_key_task = asyncio.create_task(_keepalive())

        async for raw in self._ws_connect([listen_key]):
            event = raw.get("e", "")
            if event == "executionReport":
                order = self._parse_order_update(raw)
                await self._fire_order_callbacks(order)
                yield order
            elif event == "ACCOUNT_UPDATE":
                # Futures account update
                for asset in raw.get("a", {}).get("B", []):
                    bal = BalanceInfo(
                        asset=asset["a"],
                        free=float(asset["wb"]),
                        locked=float(asset["wb"]) - float(asset["cw"]),
                        total=float(asset["wb"]),
                    )
                    await self._fire_balance_callbacks({asset["a"]: bal})

    async def _create_listen_key(self) -> str:
        """Create a listen key for user data stream."""
        if self._market_type == "spot":
            data = await self._signed_request("POST", "/api/v3/userDataStream")
        else:
            data = await self._signed_request("POST", "/fapi/v1/listenKey")
        self._listen_key = data.get("listenKey", "")
        return self._listen_key

    async def _keepalive_listen_key(self, listen_key: str) -> None:
        """Refresh the listen key to keep the user data stream alive."""
        if self._market_type == "spot":
            await self._signed_request("PUT", "/api/v3/userDataStream", {"listenKey": listen_key})
        else:
            await self._signed_request("PUT", "/fapi/v1/listenKey", {"listenKey": listen_key})

    async def _ws_connect(self, streams: list[str]) -> AsyncIterator[dict]:
        """Low-level WebSocket connection with auto-reconnect."""
        url = self._ws_base if len(streams) == 1 else f"{self._ws_base.replace('/ws', '')}/stream"
        if len(streams) > 1:
            url += f"?streams={'/'.join(streams)}"
        else:
            url += f"/{streams[0]}"

        while True:
            try:
                import websockets
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("WS connected: %s", url)
                    async for message in ws:
                        data = json.loads(message)
                        # Combined stream wraps in {"stream": ..., "data": ...}
                        if "data" in data:
                            data = data["data"]
                        yield data
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("WS disconnected (%s), reconnecting in 5s: %s", url, exc)
                await self._fire_error_callbacks(exc)
                await asyncio.sleep(5)

    # -----------------------------------------------------------------------
    # Order management
    # -----------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Place an order on Binance.

        Supports: MARKET, LIMIT, STOP_MARKET, STOP_LIMIT, OCO (via metadata).
        """
        if not self.is_connected:
            raise OrderError("Not connected to Binance")

        try:
            side = "BUY" if order.side == Side.BUY else "SELL"
            params: dict[str, Any] = {
                "symbol": self._normalize_symbol(order.symbol),
                "side": side,
                "quantity": self._format_quantity(order.symbol, order.quantity),
            }

            # Order type and price parameters
            binance_type = _ORDER_TYPE_MAP.get(order.order_type, "MARKET")
            params["type"] = binance_type

            if order.order_type in (OrderType.LIMIT, OrderType.ICEBERG):
                params["price"] = self._format_price(order.symbol, order.price or 0)
                params["timeInForce"] = order.time_in_force or "GTC"
            elif order.order_type == OrderType.STOP_LIMIT:
                params["price"] = self._format_price(order.symbol, order.price or 0)
                params["stopPrice"] = self._format_price(order.symbol, order.stop_price or 0)
                params["timeInForce"] = order.time_in_force or "GTC"
            elif order.order_type == OrderType.STOP_MARKET:
                params["stopPrice"] = self._format_price(order.symbol, order.stop_price or 0)

            if order.reduce_only and self._market_type == "futures":
                params["reduceOnly"] = "true"

            # OCO is a special endpoint
            if order.metadata.get("oco"):
                return await self._submit_oco(order)

            path = "/api/v3/order" if self._market_type == "spot" else "/fapi/v1/order"
            result = await self._retry(lambda: self._signed_request("POST", path, params, weight=1))

            order.exchange_order_id = str(result.get("orderId", ""))
            order.state = OrderState.SUBMITTED
            order.exchange = "binance"

            # Handle immediate fills
            status = result.get("status", "")
            if status == "FILLED":
                order.state = OrderState.FILLED
                order.filled_quantity = float(result.get("executedQty", order.quantity))
                order.avg_fill_price = float(result.get("cummulativeQuoteQty", 0)) / order.filled_quantity if order.filled_quantity > 0 else 0
            elif status == "PARTIALLY_FILLED":
                order.state = OrderState.PARTIAL
                order.filled_quantity = float(result.get("executedQty", 0))
                order.avg_fill_price = float(result.get("cummulativeQuoteQty", 0)) / order.filled_quantity if order.filled_quantity > 0 else 0

            logger.info(
                "Order placed: %s %s %s %s @ %s (id=%s, status=%s)",
                side, order.quantity, order.symbol, binance_type,
                order.price or "market", order.exchange_order_id, status,
            )
            return order

        except ExchangeError:
            order.state = OrderState.REJECTED
            raise
        except Exception as exc:
            order.state = OrderState.REJECTED
            raise OrderError(f"Order submission failed: {exc}") from exc

    async def _submit_oco(self, order: Order) -> Order:
        """Place an OCO (One-Cancels-Other) order."""
        side = "BUY" if order.side == Side.BUY else "SELL"
        params: dict[str, Any] = {
            "symbol": self._normalize_symbol(order.symbol),
            "side": side,
            "quantity": self._format_quantity(order.symbol, order.quantity),
            "price": self._format_price(order.symbol, order.price or 0),
            "stopPrice": self._format_price(order.symbol, order.stop_price or 0),
            "stopLimitPrice": self._format_price(order.symbol, order.metadata.get("stop_limit_price", 0)),
            "stopLimitTimeInForce": "GTC",
        }
        path = "/api/v3/order/oco"
        result = await self._retry(lambda: self._signed_request("POST", path, params, weight=1))
        order_list_id = result.get("orderListId", "")
        orders = result.get("orders", [])
        if orders:
            order.exchange_order_id = str(orders[0].get("orderId", order_list_id))
        order.state = OrderState.SUBMITTED
        order.exchange = "binance"
        order.metadata["oco_order_list_id"] = order_list_id
        return order

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        """Cancel an open order."""
        params: dict[str, Any] = {
            "symbol": self._normalize_symbol(symbol),
            "orderId": int(order_id),
        }
        path = "/api/v3/order" if self._market_type == "spot" else "/fapi/v1/order"
        result = await self._retry(lambda: self._signed_request("DELETE", path, params))
        return Order(
            id=order_id,
            symbol=symbol,
            state=OrderState.CANCELLED,
            exchange="binance",
            exchange_order_id=str(result.get("orderId", order_id)),
        )

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        """Query order status."""
        params: dict[str, Any] = {
            "symbol": self._normalize_symbol(symbol),
            "orderId": int(order_id),
        }
        path = "/api/v3/order" if self._market_type == "spot" else "/fapi/v1/order"
        result = await self._retry(lambda: self._signed_request("GET", path, params, weight=1))
        return self._parse_order_rest(result, symbol)

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """Fetch all open orders, optionally filtered by symbol."""
        params: dict[str, Any] = {}
        if symbol:
            params["symbol"] = self._normalize_symbol(symbol)
        path = "/api/v3/openOrders" if self._market_type == "spot" else "/fapi/v1/openOrders"
        result = await self._retry(lambda: self._signed_request("GET", path, params, weight=3))
        return [self._parse_order_rest(o, symbol or o.get("symbol", "")) for o in result]

    # -----------------------------------------------------------------------
    # Account
    # -----------------------------------------------------------------------

    async def fetch_balance(self) -> dict[str, float]:
        """Fetch account balances."""
        if self._market_type == "spot":
            data = await self._retry(lambda: self._signed_request("GET", "/api/v3/account", weight=10))
            balances = data.get("balances", [])
            result: dict[str, float] = {}
            for b in balances:
                total = float(b["free"]) + float(b["locked"])
                if total > 0:
                    result[b["asset"]] = total
            return result
        else:
            data = await self._retry(lambda: self._signed_request("GET", "/fapi/v2/balance", weight=1))
            return {
                b["asset"]: float(b["balance"])
                for b in data
                if float(b.get("balance", 0)) > 0
            }

    async def fetch_balance_detailed(self) -> dict[str, BalanceInfo]:
        """Fetch detailed balance information."""
        if self._market_type == "spot":
            data = await self._retry(lambda: self._signed_request("GET", "/api/v3/account", weight=10))
            result: dict[str, BalanceInfo] = {}
            for b in data.get("balances", []):
                free = float(b["free"])
                locked = float(b["locked"])
                total = free + locked
                if total > 0:
                    result[b["asset"]] = BalanceInfo(
                        asset=b["asset"], free=free, locked=locked, total=total,
                    )
            return result
        else:
            data = await self._retry(lambda: self._signed_request("GET", "/fapi/v2/balance", weight=1))
            result = {}
            for b in data:
                total = float(b.get("balance", 0))
                if total > 0:
                    result[b["asset"]] = BalanceInfo(
                        asset=b["asset"],
                        free=float(b.get("availableBalance", total)),
                        locked=total - float(b.get("availableBalance", total)),
                        total=total,
                    )
            return result

    async def fetch_positions(self) -> list[Position]:
        """Fetch open positions (futures only)."""
        if self._market_type == "spot":
            return []

        data = await self._retry(lambda: self._signed_request("GET", "/fapi/v2/positionRisk", weight=5))
        positions: list[Position] = []
        for p in data:
            amt = float(p.get("positionAmt", 0))
            if amt == 0:
                continue
            positions.append(Position(
                symbol=p["symbol"],
                strategy_id="",
                side=Side.BUY if amt > 0 else Side.SELL,
                quantity=abs(amt),
                avg_entry_price=float(p.get("entryPrice", 0)),
                current_price=float(p.get("markPrice", 0)),
                unrealized_pnl=float(p.get("unRealizedProfit", 0)),
                exchange="binance",
            ))
        return positions

    # -----------------------------------------------------------------------
    # Funding rate (futures)
    # -----------------------------------------------------------------------

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Fetch current funding rate (futures only)."""
        sym = self._normalize_symbol(symbol)
        data = await self._public_request("/fapi/v1/premiumIndex", {"symbol": sym})
        return {
            "symbol": symbol,
            "funding_rate": float(data.get("lastFundingRate", 0)),
            "next_funding_time": int(data.get("nextFundingTime", 0)),
            "mark_price": float(data.get("markPrice", 0)),
            "index_price": float(data.get("indexPrice", 0)),
        }

    # -----------------------------------------------------------------------
    # Internal parsers
    # -----------------------------------------------------------------------

    def _parse_order_rest(self, data: dict, symbol: str) -> Order:
        """Parse a REST API order response into an Order object."""
        state_map = {
            "NEW": OrderState.SUBMITTED,
            "PARTIALLY_FILLED": OrderState.PARTIAL,
            "FILLED": OrderState.FILLED,
            "CANCELED": OrderState.CANCELLED,
            "EXPIRED": OrderState.EXPIRED,
            "REJECTED": OrderState.REJECTED,
        }
        status = data.get("status", "NEW")
        filled = float(data.get("executedQty", 0))
        quote_qty = float(data.get("cummulativeQuoteQty", 0))
        avg_price = quote_qty / filled if filled > 0 else 0

        return Order(
            id=str(data.get("orderId", "")),
            symbol=symbol,
            side=Side.BUY if data.get("side") == "BUY" else Side.SELL,
            order_type=OrderType.MARKET if data.get("type") == "MARKET" else OrderType.LIMIT,
            quantity=float(data.get("origQty", 0)),
            price=float(data.get("price", 0)) or None,
            state=state_map.get(status, OrderState.SUBMITTED),
            filled_quantity=filled,
            avg_fill_price=avg_price,
            exchange="binance",
            exchange_order_id=str(data.get("orderId", "")),
        )

    def _parse_order_update(self, data: dict) -> Order:
        """Parse a WebSocket executionReport event into an Order."""
        state_map = {
            "NEW": OrderState.SUBMITTED,
            "PARTIALLY_FILLED": OrderState.PARTIAL,
            "FILLED": OrderState.FILLED,
            "CANCELED": OrderState.CANCELLED,
            "EXPIRED": OrderState.EXPIRED,
            "REJECTED": OrderState.REJECTED,
        }
        status = data.get("X", "NEW")
        filled = float(data.get("z", 0))
        quote_qty = float(data.get("Z", 0))
        avg_price = quote_qty / filled if filled > 0 else 0

        return Order(
            id=str(data.get("i", "")),
            symbol=data.get("s", ""),
            side=Side.BUY if data.get("S") == "BUY" else Side.SELL,
            order_type=OrderType.MARKET if data.get("o") == "MARKET" else OrderType.LIMIT,
            quantity=float(data.get("q", 0)),
            price=float(data.get("p", 0)) or None,
            state=state_map.get(status, OrderState.SUBMITTED),
            filled_quantity=filled,
            avg_fill_price=avg_price,
            exchange="binance",
            exchange_order_id=str(data.get("i", "")),
        )

    # -----------------------------------------------------------------------
    # Symbol/price formatting helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Convert 'BTC/USDT' to 'BTCUSDT'."""
        return symbol.replace("/", "").replace("-", "").upper()

    @staticmethod
    def _format_price(symbol: str, price: float) -> str:
        """Format price to Binance precision (simplified)."""
        if price >= 1000:
            return f"{price:.2f}"
        elif price >= 1:
            return f"{price:.4f}"
        else:
            return f"{price:.8f}"

    @staticmethod
    def _format_quantity(symbol: str, quantity: float) -> str:
        """Format quantity to Binance precision (simplified)."""
        if quantity >= 1:
            return f"{quantity:.6f}"
        else:
            return f"{quantity:.8f}"
