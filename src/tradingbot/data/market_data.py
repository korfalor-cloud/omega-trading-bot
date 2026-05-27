"""Market Data Pipeline — Fetch, cache, and serve OHLCV data.

Uses ccxt for exchange connectivity with local caching to avoid
redundant API calls during backtesting and evolution.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.enums import Timeframe
from ..core.types import OHLCVBar

logger = logging.getLogger(__name__)

# Map Timeframe to ccxt-compatible string and milliseconds
_TF_MS: dict[str, int] = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000, "3d": 259_200_000,
    "1w": 604_800_000,
}


class MarketDataFetcher:
    """Fetches OHLCV data from exchanges via ccxt.

    Supports pagination for large date ranges and rate limiting.
    """

    def __init__(self, exchange_id: str = "binance", api_key: str = "", api_secret: str = ""):
        self.exchange_id = exchange_id
        self._exchange = None
        self._api_key = api_key
        self._api_secret = api_secret

    async def _get_exchange(self):
        if self._exchange is None:
            try:
                import ccxt.async_support as ccxt_async
            except ImportError:
                import ccxt as ccxt_async
            exchange_class = getattr(ccxt_async, self.exchange_id)
            config = {"enableRateLimit": True}
            if self._api_key:
                config["apiKey"] = self._api_key
                config["secret"] = self._api_secret
            self._exchange = exchange_class(config)
        return self._exchange

    async def fetch_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
        exchange: str = "",
    ) -> list[OHLCVBar]:
        """Fetch OHLCV candles from the exchange.

        Handles pagination automatically for large date ranges.
        """
        ex = await self._get_exchange()
        tf_ms = _TF_MS.get(timeframe, 3_600_000)

        if since is None:
            since = datetime.now(timezone.utc) - timedelta(days=30)

        since_ms = int(since.timestamp() * 1000)
        all_bars: list[OHLCVBar] = []
        current_since = since_ms
        max_candles = ex.rateLimit * 10  # Safety cap

        while len(all_bars) < max_candles:
            try:
                ohlcv = await ex.fetch_ohlcv(symbol, timeframe, since=current_since, limit=limit)
            except Exception as e:
                logger.error(f"Failed to fetch candles for {symbol}: {e}")
                break

            if not ohlcv:
                break

            for candle in ohlcv:
                ts = datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc)
                bar = OHLCVBar(
                    timestamp=ts,
                    symbol=symbol,
                    timeframe=Timeframe(timeframe),
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    exchange=exchange or self.exchange_id,
                )
                all_bars.append(bar)

            # Move to next page
            last_ts = ohlcv[-1][0]
            current_since = last_ts + tf_ms

            if len(ohlcv) < limit:
                break

            # Rate limiting pause
            await asyncio.sleep(ex.rateLimit / 1000)

        logger.info(f"Fetched {len(all_bars)} candles for {symbol} {timeframe}")
        return all_bars

    async def fetch_multiple_symbols(
        self,
        symbols: list[str],
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
    ) -> dict[str, list[OHLCVBar]]:
        """Fetch candles for multiple symbols concurrently."""
        tasks = {
            symbol: asyncio.create_task(
                self.fetch_candles(symbol, timeframe, since, limit)
            )
            for symbol in symbols
        }
        results = {}
        for symbol, task in tasks.items():
            try:
                results[symbol] = await task
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                results[symbol] = []
        return results

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._exchange = None


class CachedDataProvider:
    """Wraps MarketDataFetcher with local file caching.

    Caches OHLCV data as JSON files to avoid redundant API calls.
    Useful for backtesting and evolution where the same data is
    accessed repeatedly.
    """

    def __init__(self, fetcher: MarketDataFetcher, cache_dir: str = "./data/cache"):
        self.fetcher = fetcher
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, list[OHLCVBar]] = {}

    def _cache_key(self, symbol: str, timeframe: str, since: Optional[datetime], limit: int) -> str:
        since_str = since.isoformat() if since else "none"
        raw = f"{symbol}:{timeframe}:{since_str}:{limit}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _bars_to_json(self, bars: list[OHLCVBar]) -> list[dict]:
        return [
            {
                "timestamp": bar.timestamp.isoformat(),
                "symbol": bar.symbol,
                "timeframe": bar.timeframe.value,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "exchange": bar.exchange,
            }
            for bar in bars
        ]

    def _bars_from_json(self, data: list[dict]) -> list[OHLCVBar]:
        return [
            OHLCVBar(
                timestamp=datetime.fromisoformat(d["timestamp"]),
                symbol=d["symbol"],
                timeframe=Timeframe(d["timeframe"]),
                open=d["open"],
                high=d["high"],
                low=d["low"],
                close=d["close"],
                volume=d["volume"],
                exchange=d["exchange"],
            )
            for d in data
        ]

    async def get_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        since: Optional[datetime] = None,
        limit: int = 500,
        max_age_hours: int = 1,
    ) -> list[OHLCVBar]:
        """Get candles, using cache if available and fresh."""
        key = self._cache_key(symbol, timeframe, since, limit)

        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check file cache
        cache_path = self._cache_path(key)
        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < max_age_hours:
                with open(cache_path) as f:
                    data = json.load(f)
                bars = self._bars_from_json(data)
                self._memory_cache[key] = bars
                logger.debug(f"Cache hit for {symbol} {timeframe} ({len(bars)} bars)")
                return bars

        # Fetch from exchange
        bars = await self.fetcher.fetch_candles(symbol, timeframe, since, limit)

        # Cache results
        if bars:
            with open(cache_path, "w") as f:
                json.dump(self._bars_to_json(bars), f)
            self._memory_cache[key] = bars

        return bars

    def clear_cache(self) -> None:
        """Clear all cached data."""
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
        self._memory_cache.clear()
        logger.info("Cache cleared")
