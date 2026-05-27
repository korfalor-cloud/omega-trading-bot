"""Technical Indicators — Compute features from OHLCV data.

All functions operate on numpy arrays for performance.
No external TA library required — pure numpy/pandas implementation.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..core.types import OHLCVBar

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Compute technical indicators from OHLCV bars."""

    def __init__(self, bars: list[OHLCVBar]):
        self.bars = bars
        self.n = len(bars)
        self.open = np.array([b.open for b in bars])
        self.high = np.array([b.high for b in bars])
        self.low = np.array([b.low for b in bars])
        self.close = np.array([b.close for b in bars])
        self.volume = np.array([b.volume for b in bars])

    # ── Trend Indicators ──────────────────────────────────────────

    def sma(self, period: int = 20) -> np.ndarray:
        """Simple Moving Average."""
        out = np.full(self.n, np.nan)
        if self.n < period:
            return out
        cumsum = np.cumsum(self.close)
        cumsum[period:] = cumsum[period:] - cumsum[:-period]
        out[period - 1:] = cumsum[period - 1:] / period
        return out

    def ema(self, period: int = 20) -> np.ndarray:
        """Exponential Moving Average."""
        out = np.full(self.n, np.nan)
        if self.n < period:
            return out
        alpha = 2.0 / (period + 1)
        out[period - 1] = np.mean(self.close[:period])
        for i in range(period, self.n):
            out[i] = alpha * self.close[i] + (1 - alpha) * out[i - 1]
        return out

    def macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD line, signal line, and histogram."""
        ema_fast = self.ema(fast)
        ema_slow = self.ema(slow)
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD)
        signal_line = np.full(self.n, np.nan)
        start = slow - 1 + signal - 1
        if start < self.n:
            alpha = 2.0 / (signal + 1)
            valid = macd_line[slow - 1:]
            if len(valid) >= signal:
                signal_line[start] = np.nanmean(valid[:signal])
                for i in range(start + 1, self.n):
                    if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i - 1]):
                        signal_line[i] = alpha * macd_line[i] + (1 - alpha) * signal_line[i - 1]

        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def adx(self, period: int = 14) -> np.ndarray:
        """Average Directional Index."""
        out = np.full(self.n, np.nan)
        if self.n < period * 2:
            return out

        # True Range
        tr = np.zeros(self.n)
        tr[0] = self.high[0] - self.low[0]
        for i in range(1, self.n):
            tr[i] = max(
                self.high[i] - self.low[i],
                abs(self.high[i] - self.close[i - 1]),
                abs(self.low[i] - self.close[i - 1]),
            )

        # Directional Movement
        plus_dm = np.zeros(self.n)
        minus_dm = np.zeros(self.n)
        for i in range(1, self.n):
            up = self.high[i] - self.high[i - 1]
            down = self.low[i - 1] - self.low[i]
            plus_dm[i] = up if (up > down and up > 0) else 0
            minus_dm[i] = down if (down > up and down > 0) else 0

        # Smoothed averages (Wilder's smoothing)
        atr = self._wilders_smooth(tr, period)
        plus_di = 100 * self._wilders_smooth(plus_dm, period) / np.where(atr > 0, atr, 1)
        minus_di = 100 * self._wilders_smooth(minus_dm, period) / np.where(atr > 0, atr, 1)

        # DX and ADX
        di_sum = plus_di + minus_di
        dx = 100 * np.abs(plus_di - minus_di) / np.where(di_sum > 0, di_sum, 1)
        adx = self._wilders_smooth(dx, period)

        return adx

    # ── Momentum Indicators ───────────────────────────────────────

    def rsi(self, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        out = np.full(self.n, np.nan)
        if self.n < period + 1:
            return out

        delta = np.diff(self.close)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        if avg_loss == 0:
            out[period] = 100.0
        else:
            out[period] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)

        for i in range(period, len(delta)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                out[i + 1] = 100.0
            else:
                out[i + 1] = 100.0 - 100.0 / (1 + avg_gain / avg_loss)

        return out

    def stochastic(
        self, k_period: int = 14, d_period: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stochastic Oscillator (%K and %D)."""
        k = np.full(self.n, np.nan)
        for i in range(k_period - 1, self.n):
            window_high = np.max(self.high[i - k_period + 1:i + 1])
            window_low = np.min(self.low[i - k_period + 1:i + 1])
            denom = window_high - window_low
            k[i] = 100 * (self.close[i] - window_low) / denom if denom > 0 else 50.0

        # %D = SMA of %K
        d = np.full(self.n, np.nan)
        for i in range(k_period + d_period - 2, self.n):
            window = k[i - d_period + 1:i + 1]
            if not np.any(np.isnan(window)):
                d[i] = np.mean(window)

        return k, d

    def cci(self, period: int = 20) -> np.ndarray:
        """Commodity Channel Index."""
        out = np.full(self.n, np.nan)
        if self.n < period:
            return out

        tp = (self.high + self.low + self.close) / 3
        sma_tp = np.full(self.n, np.nan)
        for i in range(period - 1, self.n):
            sma_tp[i] = np.mean(tp[i - period + 1:i + 1])

        for i in range(period - 1, self.n):
            mean_dev = np.mean(np.abs(tp[i - period + 1:i + 1] - sma_tp[i]))
            if mean_dev > 0:
                out[i] = (tp[i] - sma_tp[i]) / (0.015 * mean_dev)
            else:
                out[i] = 0.0

        return out

    def williams_r(self, period: int = 14) -> np.ndarray:
        """Williams %R."""
        out = np.full(self.n, np.nan)
        for i in range(period - 1, self.n):
            window_high = np.max(self.high[i - period + 1:i + 1])
            window_low = np.min(self.low[i - period + 1:i + 1])
            denom = window_high - window_low
            if denom > 0:
                out[i] = -100 * (window_high - self.close[i]) / denom
            else:
                out[i] = -50.0
        return out

    def mfi(self, period: int = 14) -> np.ndarray:
        """Money Flow Index."""
        out = np.full(self.n, np.nan)
        if self.n < period + 1:
            return out

        tp = (self.high + self.low + self.close) / 3
        mf = tp * self.volume

        for i in range(period, self.n):
            pos_mf = 0.0
            neg_mf = 0.0
            for j in range(i - period + 1, i + 1):
                if tp[j] > tp[j - 1]:
                    pos_mf += mf[j]
                elif tp[j] < tp[j - 1]:
                    neg_mf += mf[j]
            if neg_mf > 0:
                ratio = pos_mf / neg_mf
                out[i] = 100 - 100 / (1 + ratio)
            else:
                out[i] = 100.0

        return out

    def momentum(self, period: int = 10) -> np.ndarray:
        """Price momentum (current - n periods ago)."""
        out = np.full(self.n, np.nan)
        for i in range(period, self.n):
            out[i] = self.close[i] - self.close[i - period]
        return out

    def roc(self, period: int = 10) -> np.ndarray:
        """Rate of Change (percentage)."""
        out = np.full(self.n, np.nan)
        for i in range(period, self.n):
            if self.close[i - period] > 0:
                out[i] = 100 * (self.close[i] - self.close[i - period]) / self.close[i - period]
        return out

    # ── Volatility Indicators ─────────────────────────────────────

    def atr(self, period: int = 14) -> np.ndarray:
        """Average True Range."""
        tr = np.zeros(self.n)
        tr[0] = self.high[0] - self.low[0]
        for i in range(1, self.n):
            tr[i] = max(
                self.high[i] - self.low[i],
                abs(self.high[i] - self.close[i - 1]),
                abs(self.low[i] - self.close[i - 1]),
            )
        return self._wilders_smooth(tr, period)

    def bollinger_bands(
        self, period: int = 20, std_dev: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands (upper, middle, lower)."""
        middle = self.sma(period)
        std = np.full(self.n, np.nan)
        for i in range(period - 1, self.n):
            std[i] = np.std(self.close[i - period + 1:i + 1], ddof=0)

        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    def volatility(self, period: int = 20) -> np.ndarray:
        """Annualized volatility (rolling)."""
        out = np.full(self.n, np.nan)
        if self.n < period + 1:
            return out
        returns = np.diff(np.log(np.maximum(self.close, 1e-10)))
        for i in range(period, len(returns)):
            out[i + 1] = np.std(returns[i - period + 1:i + 1]) * np.sqrt(365)
        return out

    # ── Volume Indicators ─────────────────────────────────────────

    def obv(self) -> np.ndarray:
        """On-Balance Volume."""
        out = np.zeros(self.n)
        for i in range(1, self.n):
            if self.close[i] > self.close[i - 1]:
                out[i] = out[i - 1] + self.volume[i]
            elif self.close[i] < self.close[i - 1]:
                out[i] = out[i - 1] - self.volume[i]
            else:
                out[i] = out[i - 1]
        return out

    def vwap(self) -> np.ndarray:
        """Volume Weighted Average Price (cumulative)."""
        tp = (self.high + self.low + self.close) / 3
        cum_tp_vol = np.cumsum(tp * self.volume)
        cum_vol = np.cumsum(self.volume)
        return np.where(cum_vol > 0, cum_tp_vol / cum_vol, tp)

    # ── Helper ────────────────────────────────────────────────────

    def _wilders_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing (used by ATR, ADX, RSI)."""
        out = np.full(self.n, np.nan)
        if self.n < period:
            return out
        out[period - 1] = np.mean(data[:period])
        for i in range(period, self.n):
            out[i] = (out[i - 1] * (period - 1) + data[i]) / period
        return out


def compute_features(bars: list[OHLCVBar], config: Optional[dict] = None) -> dict[str, list[float]]:
    """Compute a full feature set from OHLCV bars.

    Returns a dict of feature_name -> list[float] aligned with the input bars.
    NaN values are forward-filled and then zero-filled for the start.
    """
    if not bars:
        return {}

    cfg = config or {}
    ti = TechnicalIndicators(bars)
    features: dict[str, list[float]] = {}

    def _add(name: str, arr: np.ndarray) -> None:
        features[name] = _ffill_nan(arr).tolist()

    # Trend
    for p in cfg.get("sma_periods", [20, 50, 200]):
        _add(f"sma_{p}", ti.sma(p))
    for p in cfg.get("ema_periods", [8, 21, 55]):
        _add(f"ema_{p}", ti.ema(p))

    macd_line, signal_line, histogram = ti.macd()
    _add("macd", macd_line)
    _add("macd_signal", signal_line)
    _add("macd_hist", histogram)

    _add("adx_14", ti.adx(14))

    # Momentum
    for p in cfg.get("rsi_periods", [14]):
        _add(f"rsi_{p}", ti.rsi(p))

    stoch_k, stoch_d = ti.stochastic()
    _add("stoch_k", stoch_k)
    _add("stoch_d", stoch_d)

    _add("cci_20", ti.cci(20))
    _add("williams_r_14", ti.williams_r(14))
    _add("mfi_14", ti.mfi(14))
    _add("momentum_10", ti.momentum(10))
    _add("roc_10", ti.roc(10))

    # Volatility
    _add("atr_14", ti.atr(14))
    bb_upper, bb_middle, bb_lower = ti.bollinger_bands()
    _add("bb_upper", bb_upper)
    _add("bb_middle", bb_middle)
    _add("bb_lower", bb_lower)
    _add("volatility", ti.volatility(20))

    # Volume
    _add("obv", ti.obv())
    _add("vwap", ti.vwap())

    # Derived features
    _add("signal_strength", _signal_strength(ti))
    _add("signal_confidence", _signal_confidence(ti))

    return features


def _ffill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values, then fill remaining leading NaNs with 0."""
    out = arr.copy()
    mask = np.isnan(out)
    if mask.all():
        return np.zeros_like(out)
    # Forward fill: each NaN takes the last valid value
    last_valid = 0.0
    for i in range(len(out)):
        if not mask[i]:
            last_valid = out[i]
        else:
            out[i] = last_valid
    return out


def _signal_strength(ti: TechnicalIndicators) -> np.ndarray:
    """Composite signal strength from multiple indicators."""
    rsi = ti.rsi(14)
    macd_line, _, hist = ti.macd()
    bb_upper, _, bb_lower = ti.bollinger_bands()

    strength = np.zeros(ti.n)
    count = np.zeros(ti.n)

    # RSI component: normalized to [-1, 1] where 50 = 0
    valid = ~np.isnan(rsi)
    strength[valid] += (rsi[valid] - 50) / 50
    count[valid] += 1

    # MACD histogram component
    valid = ~np.isnan(hist)
    atr_vals = ti.atr(14)
    atr_safe = np.where(np.isnan(atr_vals) | (atr_vals == 0), 1.0, atr_vals)
    strength[valid] += np.clip(hist[valid] / atr_safe[valid], -1, 1)
    count[valid] += 1

    # Bollinger Band position
    valid = ~(np.isnan(bb_upper) | np.isnan(bb_lower))
    bb_range = bb_upper[valid] - bb_lower[valid]
    bb_range = np.where(bb_range > 0, bb_range, 1.0)
    bb_mid = (bb_upper[valid] + bb_lower[valid]) / 2
    strength[valid] += np.clip((ti.close[valid] - bb_mid) / (bb_range / 2), -1, 1)
    count[valid] += 1

    # Average and clip
    count = np.where(count > 0, count, 1.0)
    strength = np.clip(strength / count, -1, 1)

    return strength


def _signal_confidence(ti: TechnicalIndicators) -> np.ndarray:
    """Signal confidence based on indicator agreement."""
    rsi = ti.rsi(14)
    _, _, hist = ti.macd()
    adx = ti.adx(14)

    confidence = np.full(ti.n, 0.5)

    # ADX strength: higher ADX = more confident trend
    valid = ~np.isnan(adx)
    confidence[valid] = np.clip(adx[valid] / 50, 0.2, 0.9)

    # Agreement bonus: if RSI and MACD agree
    rsi_signal = np.where(np.isnan(rsi), 0, np.sign(rsi - 50))
    macd_signal = np.where(np.isnan(hist), 0, np.sign(hist))
    agree = (rsi_signal == macd_signal) & (rsi_signal != 0)
    confidence[agree] = np.clip(confidence[agree] + 0.1, 0, 0.95)

    return confidence
