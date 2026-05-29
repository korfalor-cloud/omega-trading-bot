"""Bybit Exchange Connector — Production-grade Bybit V5 API integration.

Features:
- V5 API (unified account) for spot and derivatives
- REST API with HMAC-SHA256 authentication
- WebSocket public + private streams
- Order management (market, limit, conditional/stop)
- Position management with leverage/margin controls
- Funding rate queries
- Auto-reconnection on WebSocket disconnect
- Rate limiting and retry logic
- Testnet support
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

import aiohttp

from ..core.enums import OrderState, OrderType, Side, Timeframe
from ..core.errors import ExchangeError, OrderError
from ..core.types import OHLCVBar, Order, OrderBookLevel, OrderBookSnapshot, Position, Tick

from .base_connector import (
    BalanceInfo,
    BaseExchangeConnector,
    ConnectionState,
    PositionInfo,
)

logger = logging.getLogger(__name__)

_INTERVAL_MAP: dict[Timeframe, str] = {
    Timeframe.M1: "1", Timeframe.M3: "3", Timeframe.M5: "5",
    Timeframe.M15: "15", Timeframe.M30: "30", Timeframe.H1: "60",
    Timeframe.H2: "120", Timeframe.H4: "240", Timeframe.H6: "360",
    Timeframe.H8: "480", Timeframe.H12: "720", Timeframe.D1: "D",
    Timeframe.W1: "W", Timeframe.MN1: "M",
}

_ORDER_TYPE_MAP: dict[OrderType, str] = {
    OrderType.MARKET: "Market",
    OrderType.LIMIT: "Limit",
    OrderType.STOP_MARKET: "Market",
    OrderType.STOP_LIMIT: "Limit",
}

_CATEGORY_MAP: dict[str, str] = {
    "spot": "spot",
    "linear": "linear",
    "inverse": "inverse",
    "futures": "linear",
}


class BybitConnector(BaseExchangeConnector):
    """Full-featured Bybit V5 connector.

    Parameters
    ----------
    api_key, api_secret : str
        Bybit API credentials.
    testnet : bool
        Use Bybit testnet.
    category : str
        ``"spot"``, ``"linear"``, ``"inverse"``, or ``"futures"`` (alias for linear).
    """

    _REST_BASE = "https://api.bybit.com"
    _REST_TEST = "https://api-testnet.bybit.com"
    _WS_PUBLIC = "wss://stream.bybit.com/v5/public"
    _WS_PRIVATE = "wss://stream.bybit.com/v5/private"
    _WS_PUBLIC_TEST = "wss://stream-testnet.bybit.com/v5/public"
    _WS_PRIVATE_TEST = "wss://stream-testnet.bybit.com/v5/private"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        category: str = "spot",
        **kwargs: Any,
    ):
        super().__init__(
            exchange_id="bybit",
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            rate_limit=kwargs.pop("rate_limit", 600),
            **kwargs,
        )
        self._category = _CATEGORY_MAP.get(category, category)
        self._rest_base = self._REST_TEST if testnet else self._REST_BASE
        self._ws_public = self._WS_PUBLIC_TEST if testnet else self._WS_PUBLIC
        self._ws_private = self._WS_PRIVATE_TEST if testnet else self._WS_PRIVATE

    # -----------------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------------

    async def _do_connect(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self._request_timeout),
        )
        # Ping to verify connectivity
        await self._public_get("/v5/market/time")
        logger.info("Bybit V5 API reachable (testnet=%s, category=%s)", self._testnet, self._category)

    async def _do_disconnect(self) -> None:
        for name in list(self._ws_tasks.keys()):
            await self._stop_ws_stream(name)
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # -----------------------------------------------------------------------
    # V5 Authentication
    # -----------------------------------------------------------------------

    def _sign_v5(self, timestamp: str, params_str: str) -> str:
        """Create HMAC-SHA256 signature for V5 API.

        Signature = HMAC_SHA256(api_key + timestamp + recv_window + params_str)
        """
        raw = f"{self._api_key}{timestamp}5000{params_str}"
        return hmac.new(
            self._api_secret.encode("utf-8"),
            raw.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, timestamp: str, signature: str) -> dict[str, str]:
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": "5000",
            "Content-Type": "application/json",
        }

    # -----------------------------------------------------------------------
    # REST helpers
    # -----------------------------------------------------------------------

    async def _public_get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self._rest_base}{path}"
        async with self._session.get(url, params=params) as resp:
            return await self._handle_response(resp)

    async def _signed_get(self, path: str, params: dict[str, Any] | None = None, weight: int = 1) -> Any:
        params = params or {}
        # Sort params for consistent signing
        sorted_params = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        ts = str(int(time.time() * 1000))
        sig = self._sign_v5(ts, sorted_params)
        headers = self._auth_headers(ts, sig)

        url = f"{self._rest_base}{path}"
        await self._rate_limiter.acquire(weight)
        async with self._session.get(url, params=params, headers=headers) as resp:
            return await self._handle_response(resp)

    async def _signed_post(self, path: str, body: dict[str, Any], weight: int = 1) -> Any:
        body_str = json.dumps(body, separators=(",", ":"))
        ts = str(int(time.time() * 1000))
        sig = self._sign_v5(ts, body_str)
        headers = self._auth_headers(ts, sig)

        url = f"{self._rest_base}{path}"
        await self._rate_limiter.acquire(weight)
        async with self._session.post(url, data=body_str, headers=headers) as resp:
            return await self._handle_response(resp)

    async def _handle_response(self, resp: aiohttp.ClientResponse) -> Any:
        data = await resp.json()
        ret_code = data.get("retCode", 0)
        if ret_code != 0:
            ret_msg = data.get("retMsg", "Unknown error")
            raise ExchangeError(f"Bybit error {ret_code}: {ret_msg}")
        return data.get("result", data)

    # -----------------------------------------------------------------------
    # Market data
    # -----------------------------------------------------------------------

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: Timeframe,
        since: Optional[datetime] = None,
        limit: int = 200,
    ) -> list[OHLCVBar]:
        interval = _INTERVAL_MAP.get(timeframe, "60")
        params: dict[str, Any] = {
            "category": self._category,
            "symbol": self._normalize_symbol(symbol),
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if since:
            params["start"] = int(since.timestamp() * 1000)

        data = await self._retry(lambda: self._public_get("/v5/market/kline", params))
        bars: list[OHLCVBar] = []
        for k in data.get("list", []):
            # Bybit returns [startTime, open, high, low, close, volume, turnover]
            bars.append(OHLCVBar(
                timestamp=self._ts_to_dt(int(k[0])),
                symbol=symbol,
                timeframe=timeframe,
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                exchange="bybit",
                vwap=float(k[6]) / float(k[5]) if float(k[5]) > 0 else 0.0,
            ))
        # Bybit returns newest first; reverse to chronological
        bars.reverse()
        return bars

    async def fetch_ticker(self, symbol: str) -> dict:
        params = {"category": self._category, "symbol": self._normalize_symbol(symbol)}
        data = await self._retry(lambda: self._public_get("/v5/market/tickers", params))
        tickers = data.get("list", [])
        if not tickers:
            raise ExchangeError(f"No ticker data for {symbol}")
        t = tickers[0]
        return {
            "symbol": symbol,
            "last": float(t.get("lastPrice", 0)),
            "bid": float(t.get("bid1Price", 0)),
            "ask": float(t.get("ask1Price", 0)),
            "high": float(t.get("highPrice24h", 0)),
            "low": float(t.get("lowPrice24h", 0)),
            "volume": float(t.get("volume24h", 0)),
            "quote_volume": float(t.get("turnover24h", 0)),
            "change_pct": float(t.get("price24hPcnt", 0)) * 100,
        }

    async def fetch_order_book(self, symbol: str, depth: int = 25) -> OrderBookSnapshot:
        params = {"category": self._category, "symbol": self._normalize_symbol(symbol), "limit": depth}
        data = await self._retry(lambda: self._public_get("/v5/market/orderbook", params))
        return OrderBookSnapshot(
            timestamp=datetime.utcnow(),
            symbol=symbol,
            exchange="bybit",
            bids=[OrderBookLevel(price=float(b[0]), quantity=float(b[1])) for b in data.get("b", [])],
            asks=[OrderBookLevel(price=float(a[0]), quantity=float(a[1])) for a in data.get("a", [])],
        )

    # -----------------------------------------------------------------------
    # WebSocket streams
    # -----------------------------------------------------------------------

    async def watch_candles(self, symbol: str, timeframe: Timeframe) -> AsyncIterator[OHLCVBar]:
        interval = _INTERVAL_MAP.get(timeframe, "60")
        sym = self._normalize_symbol(symbol)
        topic = f"kline.{interval}.{sym}"

        async for raw in self._ws_public_stream([topic]):
            data_list = raw.get("data", [])
            if isinstance(data_list, dict):
                data_list = [data_list]
            for k in data_list:
                bar = OHLCVBar(
                    timestamp=self._ts_to_dt(int(k["start"])),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(k["open"]),
                    high=float(k["high"]),
                    low=float(k["low"]),
                    close=float(k["close"]),
                    volume=float(k["volume"]),
                    exchange="bybit",
                )
                await self._fire_candle_callbacks(bar)
                yield bar

    async def watch_trades(self, symbol: str) -> AsyncIterator[Tick]:
        sym = self._normalize_symbol(symbol)
        topic = f"publicTrade.{sym}"

        async for raw in self._ws_public_stream([topic]):
            for t in raw.get("data", []):
                tick = Tick(
                    timestamp=self._ts_to_dt(int(t["T"])),
                    symbol=symbol,
                    price=float(t["p"]),
                    quantity=float(t["v"]),
                    side=Side.BUY if t["S"] == "Buy" else Side.SELL,
                    exchange="bybit",
                    trade_id=t.get("i", ""),
                )
                await self._fire_trade_callbacks(tick)
                yield tick

    async def watch_order_book(self, symbol: str, depth: int = 25) -> AsyncIterator[OrderBookSnapshot]:
        sym = self._normalize_symbol(symbol)
        # Bybit supports depth levels: 1, 25, 50, 200, 500
        level = 50 if depth > 25 else 25
        topic = f"orderbook.{level}.{sym}"

        async for raw in self._ws_public_stream([topic]):
            data = raw.get("data", {})
            book = OrderBookSnapshot(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                exchange="bybit",
                bids=[OrderBookLevel(price=float(b[0]), quantity=float(b[1])) for b in data.get("b", [])],
                asks=[OrderBookLevel(price=float(a[0]), quantity=float(a[1])) for a in data.get("a", [])],
            )
            await self._fire_orderbook_callbacks(book)
            yield book

    async def watch_user_data(self) -> AsyncIterator[Order]:
        """Stream private user data (order updates, position changes, wallet)."""
        topics = ["order", "position", "wallet"]

        async for raw in self._ws_private_stream(topics):
            topic = raw.get("topic", "")
            data_list = raw.get("data", [])
            if isinstance(data_list, dict):
                data_list = [data_list]

            if topic == "order":
                for d in data_list:
                    order = self._parse_ws_order(d)
                    await self._fire_order_callbacks(order)
                    yield order
            elif topic == "wallet":
                balances: dict[str, BalanceInfo] = {}
                for d in data_list:
                    for coin in d.get("coin", []):
                        bal = BalanceInfo(
                            asset=coin["coin"],
                            free=float(coin.get("availableToWithdraw", 0)),
                            locked=float(coin.get("locked", 0)),
                            total=float(coin.get("walletBalance", 0)),
                        )
                        balances[coin["coin"]] = bal
                if balances:
                    await self._fire_balance_callbacks(balances)

    async def _ws_public_stream(self, topics: list[str]) -> AsyncIterator[dict]:
        """Subscribe to public WebSocket topics with auto-reconnect."""
        while True:
            try:
                import websockets
                async with websockets.connect(self._ws_public, ping_interval=20) as ws:
                    # Subscribe
                    sub_msg = json.dumps({"op": "subscribe", "args": [f"{t}" for t in topics]})
                    await ws.send(sub_msg)
                    logger.info("Bybit WS public subscribed: %s", topics)

                    async for message in ws:
                        data = json.loads(message)
                        # Skip subscription confirmations and pings
                        if "topic" not in data:
                            if data.get("op") == "ping":
                                await ws.send(json.dumps({"op": "pong"}))
                            continue
                        yield data
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Bybit WS public disconnected, reconnecting in 5s: %s", exc)
                await self._fire_error_callbacks(exc)
                await asyncio.sleep(5)

    async def _ws_private_stream(self, topics: list[str]) -> AsyncIterator[dict]:
        """Subscribe to private WebSocket topics with authentication."""
        while True:
            try:
                import websockets
                async with websockets.connect(self._ws_private, ping_interval=20) as ws:
                    # Authenticate
                    expires = int(time.time() * 1000) + 10000
                    sign_raw = f"GET/realtime{expires}"
                    sig = hmac.new(
                        self._api_secret.encode(),
                        sign_raw.encode(),
                        hashlib.sha256,
                    ).hexdigest()
                    auth_msg = json.dumps({
                        "op": "auth",
                        "args": [self._api_key, expires, sig],
                    })
                    await ws.send(auth_msg)

                    # Wait for auth response
                    auth_resp = json.loads(await ws.recv())
                    if auth_resp.get("success") is not True:
                        raise ExchangeError(f"Bybit WS auth failed: {auth_resp}")

                    # Subscribe
                    sub_msg = json.dumps({"op": "subscribe", "args": topics})
                    await ws.send(sub_msg)
                    logger.info("Bybit WS private subscribed: %s", topics)

                    async for message in ws:
                        data = json.loads(message)
                        if "topic" not in data:
                            if data.get("op") == "ping":
                                await ws.send(json.dumps({"op": "pong"}))
                            continue
                        yield data
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Bybit WS private disconnected, reconnecting in 5s: %s", exc)
                await self._fire_error_callbacks(exc)
                await asyncio.sleep(5)

    # -----------------------------------------------------------------------
    # Order management
    # -----------------------------------------------------------------------

    async def submit_order(self, order: Order) -> Order:
        """Place an order on Bybit V5."""
        if not self.is_connected:
            raise OrderError("Not connected to Bybit")

        try:
            side = "Buy" if order.side == Side.BUY else "Sell"
            order_type = _ORDER_TYPE_MAP.get(order.order_type, "Market")
            is_stop = order.order_type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT)

            body: dict[str, Any] = {
                "category": self._category,
                "symbol": self._normalize_symbol(order.symbol),
                "side": side,
                "orderType": order_type,
                "qty": self._format_quantity(order.quantity),
            }

            if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                body["price"] = self._format_price(order.price or 0)
                body["timeInForce"] = order.time_in_force or "GTC"

            if is_stop:
                body["triggerPrice"] = self._format_price(order.stop_price or 0)
                body["triggerDirection"] = 1 if (order.stop_price or 0) > 0 else 2

            if order.reduce_only:
                body["reduceOnly"] = True

            body["orderLinkId"] = order.id[:36]

            result = await self._retry(lambda: self._signed_post("/v5/order/create", body))

            order.exchange_order_id = result.get("orderId", "")
            order.state = OrderState.SUBMITTED
            order.exchange = "bybit"

            logger.info(
                "Order placed: %s %s %s %s @ %s (id=%s)",
                side, order.quantity, order.symbol, order_type,
                order.price or "market", order.exchange_order_id,
            )
            return order

        except ExchangeError:
            order.state = OrderState.REJECTED
            raise
        except Exception as exc:
            order.state = OrderState.REJECTED
            raise OrderError(f"Order submission failed: {exc}") from exc

    async def cancel_order(self, order_id: str, symbol: str) -> Order:
        body = {
            "category": self._category,
            "symbol": self._normalize_symbol(symbol),
            "orderId": order_id,
        }
        await self._retry(lambda: self._signed_post("/v5/order/cancel", body))
        return Order(
            id=order_id,
            symbol=symbol,
            state=OrderState.CANCELLED,
            exchange="bybit",
        )

    async def fetch_order(self, order_id: str, symbol: str) -> Order:
        params = {
            "category": self._category,
            "symbol": self._normalize_symbol(symbol),
            "orderId": order_id,
        }
        data = await self._retry(lambda: self._signed_get("/v5/order/realtime", params))
        orders = data.get("list", [])
        if not orders:
            raise OrderError(f"Order {order_id} not found")
        return self._parse_rest_order(orders[0])

    async def fetch_open_orders(self, symbol: Optional[str] = None) -> list[Order]:
        params: dict[str, Any] = {"category": self._category}
        if symbol:
            params["symbol"] = self._normalize_symbol(symbol)
        data = await self._retry(lambda: self._signed_get("/v5/order/realtime", params))
        return [self._parse_rest_order(o) for o in data.get("list", [])]

    async def cancel_all_orders(self, symbol: str) -> int:
        """Cancel all open orders for a symbol. Returns count cancelled."""
        body = {
            "category": self._category,
            "symbol": self._normalize_symbol(symbol),
        }
        result = await self._retry(lambda: self._signed_post("/v5/order/cancel-all", body))
        return len(result.get("list", []))

    # -----------------------------------------------------------------------
    # Account / Balance
    # -----------------------------------------------------------------------

    async def fetch_balance(self) -> dict[str, float]:
        params: dict[str, Any] = {"accountType": "UNIFIED" if self._category != "spot" else "SPOT"}
        data = await self._retry(lambda: self._signed_get("/v5/account/wallet-balance", params, weight=5))
        result: dict[str, float] = {}
        for account in data.get("list", []):
            for coin in account.get("coin", []):
                total = float(coin.get("walletBalance", 0))
                if total > 0:
                    result[coin["coin"]] = total
        return result

    async def fetch_balance_detailed(self) -> dict[str, BalanceInfo]:
        params: dict[str, Any] = {"accountType": "UNIFIED" if self._category != "spot" else "SPOT"}
        data = await self._retry(lambda: self._signed_get("/v5/account/wallet-balance", params, weight=5))
        result: dict[str, BalanceInfo] = {}
        for account in data.get("list", []):
            for coin in account.get("coin", []):
                total = float(coin.get("walletBalance", 0))
                if total > 0:
                    result[coin["coin"]] = BalanceInfo(
                        asset=coin["coin"],
                        free=float(coin.get("availableToWithdraw", 0)),
                        locked=float(coin.get("locked", 0)),
                        total=total,
                        usd_value=float(coin.get("usdValue", 0)),
                    )
        return result

    # -----------------------------------------------------------------------
    # Position management
    # -----------------------------------------------------------------------

    async def fetch_positions(self) -> list[Position]:
        """Fetch all open positions."""
        if self._category == "spot":
            return []
        params: dict[str, Any] = {"category": self._category, "settleCoin": "USDT"}
        data = await self._retry(lambda: self._signed_get("/v5/position/list", params, weight=5))
        positions: list[Position] = []
        for p in data.get("list", []):
            size = float(p.get("size", 0))
            if size == 0:
                continue
            positions.append(Position(
                symbol=p["symbol"],
                strategy_id="",
                side=Side.BUY if p.get("side") == "Buy" else Side.SELL,
                quantity=size,
                avg_entry_price=float(p.get("avgPrice", 0)),
                current_price=float(p.get("markPrice", 0)),
                unrealized_pnl=float(p.get("unrealisedPnl", 0)),
                exchange="bybit",
            ))
        return positions

    async def set_leverage(self, symbol: str, leverage: float) -> None:
        """Set leverage for a symbol (linear/inverse only)."""
        if self._category == "spot":
            return
        body = {
            "category": self._category,
            "symbol": self._normalize_symbol(symbol),
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        await self._retry(lambda: self._signed_post("/v5/position/set-leverage", body))

    async def set_margin_mode(self, symbol: str, margin_mode: str = "cross") -> None:
        """Set margin mode (cross/isolated) for a symbol."""
        if self._category == "spot":
            return
        body = {
            "category": self._category,
            "symbol": self._normalize_symbol(symbol),
            "tradeMode": 0 if margin_mode == "cross" else 1,
        }
        await self._retry(lambda: self._signed_post("/v5/position/switch-isolated", body))

    async def set_trading_stop(
        self,
        symbol: str,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        trailing_stop: Optional[float] = None,
    ) -> None:
        """Set TP/SL/trailing stop for a position."""
        body: dict[str, Any] = {
            "category": self._category,
            "symbol": self._normalize_symbol(symbol),
            "positionIdx": 0,
        }
        if take_profit is not None:
            body["takeProfit"] = self._format_price(take_profit)
        if stop_loss is not None:
            body["stopLoss"] = self._format_price(stop_loss)
        if trailing_stop is not None:
            body["trailingStop"] = self._format_price(trailing_stop)
        await self._retry(lambda: self._signed_post("/v5/position/trading-stop", body))

    # -----------------------------------------------------------------------
    # Funding rate
    # -----------------------------------------------------------------------

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Fetch current funding rate."""
        params = {"category": self._category, "symbol": self._normalize_symbol(symbol)}
        data = await self._retry(lambda: self._public_get("/v5/market/tickers", params))
        tickers = data.get("list", [])
        if not tickers:
            raise ExchangeError(f"No data for {symbol}")
        t = tickers[0]
        return {
            "symbol": symbol,
            "funding_rate": float(t.get("fundingRate", 0)),
            "next_funding_time": int(t.get("nextFundingTime", 0)),
            "mark_price": float(t.get("markPrice", 0)),
            "index_price": float(t.get("indexPrice", 0)),
        }

    async def fetch_funding_history(
        self,
        symbol: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Fetch historical funding rate settlements."""
        params = {
            "category": self._category,
            "symbol": self._normalize_symbol(symbol),
            "limit": min(limit, 200),
        }
        data = await self._retry(lambda: self._public_get("/v5/market/funding/history", params))
        return [
            {
                "symbol": f["symbol"],
                "funding_rate": float(f["fundingRate"]),
                "funding_time": int(f["fundingRateTimestamp"]),
            }
            for f in data.get("list", [])
        ]

    # -----------------------------------------------------------------------
    # Parsers
    # -----------------------------------------------------------------------

    def _parse_rest_order(self, data: dict) -> Order:
        state_map = {
            "New": OrderState.SUBMITTED,
            "PartiallyFilled": OrderState.PARTIAL,
            "Filled": OrderState.FILLED,
            "Cancelled": OrderState.CANCELLED,
            "Rejected": OrderState.REJECTED,
            "Untriggered": OrderState.SUBMITTED,
            "Deactivated": OrderState.CANCELLED,
        }
        status = data.get("orderStatus", "New")
        filled = float(data.get("cumExecQty", 0))
        avg_price = float(data.get("avgPrice", 0))

        return Order(
            id=data.get("orderId", ""),
            symbol=data.get("symbol", ""),
            side=Side.BUY if data.get("side") == "Buy" else Side.SELL,
            order_type=OrderType.MARKET if data.get("orderType") == "Market" else OrderType.LIMIT,
            quantity=float(data.get("qty", 0)),
            price=float(data.get("price", 0)) or None,
            stop_price=float(data.get("triggerPrice", 0)) or None,
            state=state_map.get(status, OrderState.SUBMITTED),
            filled_quantity=filled,
            avg_fill_price=avg_price,
            exchange="bybit",
            exchange_order_id=data.get("orderId", ""),
        )

    def _parse_ws_order(self, data: dict) -> Order:
        state_map = {
            "New": OrderState.SUBMITTED,
            "PartiallyFilled": OrderState.PARTIAL,
            "Filled": OrderState.FILLED,
            "Cancelled": OrderState.CANCELLED,
            "Rejected": OrderState.REJECTED,
            "Untriggered": OrderState.SUBMITTED,
            "Deactivated": OrderState.CANCELLED,
        }
        status = data.get("orderStatus", "New")
        filled = float(data.get("cumExecQty", 0))
        avg_price = float(data.get("avgPrice", 0))

        return Order(
            id=data.get("orderId", ""),
            symbol=data.get("symbol", ""),
            side=Side.BUY if data.get("side") == "Buy" else Side.SELL,
            order_type=OrderType.MARKET if data.get("orderType") == "Market" else OrderType.LIMIT,
            quantity=float(data.get("qty", 0)),
            price=float(data.get("price", 0)) or None,
            stop_price=float(data.get("triggerPrice", 0)) or None,
            state=state_map.get(status, OrderState.SUBMITTED),
            filled_quantity=filled,
            avg_fill_price=avg_price,
            exchange="bybit",
            exchange_order_id=data.get("orderId", ""),
        )

    # -----------------------------------------------------------------------
    # Symbol / precision helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Convert 'BTC/USDT' to 'BTCUSDT'."""
        return symbol.replace("/", "").replace("-", "").upper()

    @staticmethod
    def _format_price(price: float) -> str:
        if price >= 1000:
            return f"{price:.2f}"
        elif price >= 1:
            return f"{price:.4f}"
        else:
            return f"{price:.8f}"

    @staticmethod
    def _format_quantity(quantity: float) -> str:
        if quantity >= 1:
            return f"{quantity:.6f}"
        else:
            return f"{quantity:.8f}"
