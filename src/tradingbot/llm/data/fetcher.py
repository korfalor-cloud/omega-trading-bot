"""
CandleStick Transformer — Data Fetcher
Fetches OHLCV candlestick data from Binance and Yahoo Finance APIs.
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import requests

from ..config import DataConfig, get_data_config


class BinanceFetcher:
    """Fetch kline (candlestick) data from Binance public API. No API key required."""

    def __init__(self, config: DataConfig = None):
        self.config = config or get_data_config()
        self.base_url = self.config.BINANCE_BASE_URL

    def fetch_klines(
        self,
        symbol: str = None,
        interval: str = None,
        limit: int = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> np.ndarray:
        """
        Fetch klines from Binance.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Candle interval (e.g., "1h", "1d")
            limit: Number of candles (max 1000)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms

        Returns:
            Array of shape (N, 6) — [open, high, low, close, volume, timestamp]
        """
        symbol = symbol or self.config.BINANCE_DEFAULT_SYMBOL
        interval = interval or self.config.BINANCE_DEFAULT_INTERVAL
        limit = limit or self.config.BINANCE_MAX_LIMIT

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        resp = requests.get(f"{self.base_url}/klines", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Binance returns: [open_time, open, high, low, close, volume, close_time, ...]
        candles = []
        for k in data:
            candles.append([
                float(k[1]),   # open
                float(k[2]),   # high
                float(k[3]),   # low
                float(k[4]),   # close
                float(k[5]),   # volume
                float(k[0]),   # timestamp (open_time in ms)
            ])

        return np.array(candles, dtype=np.float64)

    def fetch_all(
        self,
        symbol: str = None,
        interval: str = None,
        days: int = 365,
    ) -> np.ndarray:
        """
        Fetch historical data by paginating through the API.

        Args:
            symbol: Trading pair
            interval: Candle interval
            days: Number of days of history

        Returns:
            Array of shape (N, 6)
        """
        symbol = symbol or self.config.BINANCE_DEFAULT_SYMBOL
        interval = interval or self.config.BINANCE_DEFAULT_INTERVAL

        end_ts = int(datetime.now().timestamp() * 1000)
        start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        all_candles = []
        current_start = start_ts

        while current_start < end_ts:
            candles = self.fetch_klines(
                symbol=symbol,
                interval=interval,
                limit=1000,
                start_time=current_start,
                end_time=end_ts,
            )
            if len(candles) == 0:
                break

            all_candles.append(candles)

            # Move start to after last candle
            last_ts = candles[-1, 5]
            current_start = int(last_ts) + 1

            # Rate limit
            time.sleep(0.2)

        if not all_candles:
            return np.zeros((0, 6), dtype=np.float64)

        return np.vstack(all_candles)


class YahooFetcher:
    """Fetch OHLCV data from Yahoo Finance via yfinance."""

    def __init__(self, config: DataConfig = None):
        self.config = config or get_data_config()

    def fetch(
        self,
        ticker: str = None,
        interval: str = None,
        period: str = "1y",
    ) -> np.ndarray:
        """
        Fetch data from Yahoo Finance.

        Args:
            ticker: Stock/crypto ticker (e.g., "BTC-USD", "AAPL")
            interval: Candle interval (e.g., "1h", "1d")
            period: Time period ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max")

        Returns:
            Array of shape (N, 6) — [open, high, low, close, volume, timestamp]
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
                "yfinance is required for Yahoo data. Install with: pip install yfinance"
            )

        ticker = ticker or self.config.YAHOO_DEFAULT_TICKER
        interval = interval or self.config.YAHOO_DEFAULT_INTERVAL

        # yfinance limits intraday data to 730 days
        data = yf.download(ticker, interval=interval, period=period, progress=False)

        if data.empty:
            return np.zeros((0, 6), dtype=np.float64)

        candles = np.zeros((len(data), 6), dtype=np.float64)
        candles[:, 0] = data["Open"].values.flatten()
        candles[:, 1] = data["High"].values.flatten()
        candles[:, 2] = data["Low"].values.flatten()
        candles[:, 3] = data["Close"].values.flatten()
        candles[:, 4] = data["Volume"].values.flatten()

        # Convert datetime index to unix timestamp in ms
        timestamps = data.index.astype(np.int64) // 10**6  # ns to ms
        candles[:, 5] = timestamps.astype(np.float64)

        return candles
