"""Advanced Technical Indicators — Ichimoku, Fibonacci, Pivot Points, etc.

Extends the base TechnicalIndicators with more sophisticated indicators.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from ..core.types import OHLCVBar


class AdvancedIndicators:
    """Advanced technical indicators for trading strategies."""

    def __init__(self, bars: list[OHLCVBar]):
        self.bars = bars
        self.n = len(bars)
        self.open = np.array([b.open for b in bars])
        self.high = np.array([b.high for b in bars])
        self.low = np.array([b.low for b in bars])
        self.close = np.array([b.close for b in bars])
        self.volume = np.array([b.volume for b in bars])

    # ── Ichimoku Cloud ────────────────────────────────────────────

    def ichimoku(
        self, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52
    ) -> dict[str, np.ndarray]:
        """Ichimoku Cloud components.

        Returns:
            tenkan_sen: Conversion line (Tenkan-sen)
            kijun_sen: Base line (Kijun-sen)
            senkou_a: Leading span A (displaced 26 bars forward)
            senkou_b: Leading span B (displaced 26 bars forward)
            chikou: Lagging span (displaced 26 bars back)
        """
        tenkan_sen = self._donchian_mid(tenkan)
        kijun_sen = self._donchian_mid(kijun)

        # Senkou Span A = (Tenkan + Kijun) / 2, displaced 26 bars forward
        senkou_a = np.full(self.n, np.nan)
        for i in range(self.n):
            if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
                idx = i + kijun
                if idx < self.n:
                    senkou_a[idx] = (tenkan_sen[i] + kijun_sen[i]) / 2

        # Senkou Span B = Donchian mid of 52 periods, displaced 26 bars forward
        senkou_b_line = self._donchian_mid(senkou_b)
        senkou_b_out = np.full(self.n, np.nan)
        for i in range(self.n):
            if not np.isnan(senkou_b_line[i]):
                idx = i + kijun
                if idx < self.n:
                    senkou_b_out[idx] = senkou_b_line[i]

        # Chikou Span = close displaced 26 bars back
        chikou = np.full(self.n, np.nan)
        for i in range(kijun, self.n):
            chikou[i - kijun] = self.close[i]

        return {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b_out,
            "chikou": chikou,
        }

    def _donchian_mid(self, period: int) -> np.ndarray:
        out = np.full(self.n, np.nan)
        for i in range(period - 1, self.n):
            window_high = np.max(self.high[i - period + 1:i + 1])
            window_low = np.min(self.low[i - period + 1:i + 1])
            out[i] = (window_high + window_low) / 2
        return out

    # ── Fibonacci Retracement ─────────────────────────────────────

    def fibonacci_levels(
        self, lookback: int = 100
    ) -> dict[str, float]:
        """Fibonacci retracement levels from recent high/low."""
        start = max(0, self.n - lookback)
        window_high = np.max(self.high[start:])
        window_low = np.min(self.low[start:])
        diff = window_high - window_low

        return {
            "fib_0": window_low,
            "fib_0.236": window_low + 0.236 * diff,
            "fib_0.382": window_low + 0.382 * diff,
            "fib_0.5": window_low + 0.5 * diff,
            "fib_0.618": window_low + 0.618 * diff,
            "fib_0.786": window_low + 0.786 * diff,
            "fib_1": window_high,
            "fib_1.272": window_high + 0.272 * diff,
            "fib_1.618": window_high + 0.618 * diff,
            "high": window_high,
            "low": window_low,
        }

    def fibonacci_retracement_signal(self, lookback: int = 100) -> float:
        """Signal based on proximity to Fibonacci levels. Returns -1 to 1."""
        levels = self.fibonacci_levels(lookback)
        price = self.close[-1]
        diff = levels["high"] - levels["low"]
        if diff == 0:
            return 0.0

        # Normalize price position to 0-1 range
        position = (price - levels["low"]) / diff

        # Signal: buy near support (0.382-0.618), sell near resistance (0-0.236 or 0.786-1)
        if 0.382 <= position <= 0.618:
            return 0.5  # Buy zone
        elif position < 0.236 or position > 0.786:
            return -0.5  # Sell zone
        else:
            return 0.0  # Neutral

    # ── Pivot Points ──────────────────────────────────────────────

    def pivot_points(self) -> dict[str, float]:
        """Standard pivot points from previous bar."""
        if self.n < 2:
            return {}

        h = self.high[-2]
        l = self.low[-2]
        c = self.close[-2]

        pivot = (h + l + c) / 3

        return {
            "pivot": pivot,
            "r1": 2 * pivot - l,
            "r2": pivot + (h - l),
            "r3": h + 2 * (pivot - l),
            "s1": 2 * pivot - h,
            "s2": pivot - (h - l),
            "s3": l - 2 * (h - pivot),
        }

    def pivot_signal(self) -> float:
        """Signal based on pivot point position. Returns -1 to 1."""
        pp = self.pivot_points()
        if not pp:
            return 0.0

        price = self.close[-1]
        pivot = pp["pivot"]
        r1 = pp["r1"]
        s1 = pp["s1"]

        if price > r1:
            return -0.5  # Overbought
        elif price < s1:
            return 0.5  # Oversold
        elif price > pivot:
            return 0.2  # Mildly bullish
        else:
            return -0.2  # Mildly bearish

    # ── Donchian Channel ──────────────────────────────────────────

    def donchian_channel(
        self, period: int = 20
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Donchian Channel (upper, middle, lower)."""
        upper = np.full(self.n, np.nan)
        lower = np.full(self.n, np.nan)
        middle = np.full(self.n, np.nan)

        for i in range(period - 1, self.n):
            upper[i] = np.max(self.high[i - period + 1:i + 1])
            lower[i] = np.min(self.low[i - period + 1:i + 1])
            middle[i] = (upper[i] + lower[i]) / 2

        return upper, middle, lower

    # ── Keltner Channel ───────────────────────────────────────────

    def keltner_channel(
        self, ema_period: int = 20, atr_period: int = 10, mult: float = 2.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Keltner Channel (upper, middle, lower)."""
        from .technical import TechnicalIndicators
        ti = TechnicalIndicators(self.bars)

        middle = ti.ema(ema_period)
        atr = ti.atr(atr_period)

        upper = middle + mult * atr
        lower = middle - mult * atr

        return upper, middle, lower

    # ── SuperTrend ────────────────────────────────────────────────

    def supertrend(
        self, period: int = 10, multiplier: float = 3.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """SuperTrend indicator.

        Returns:
            supertrend_line: The SuperTrend line values
            direction: 1 for uptrend, -1 for downtrend
        """
        from .technical import TechnicalIndicators
        ti = TechnicalIndicators(self.bars)
        atr = ti.atr(period)

        hl2 = (self.high + self.low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend_line = np.full(self.n, np.nan)
        direction = np.zeros(self.n)

        for i in range(1, self.n):
            if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
                continue

            # Adjust bands
            if lower_band[i] > lower_band[i - 1] or self.close[i - 1] < lower_band[i - 1]:
                pass
            else:
                lower_band[i] = lower_band[i - 1]

            if upper_band[i] < upper_band[i - 1] or self.close[i - 1] > upper_band[i - 1]:
                pass
            else:
                upper_band[i] = upper_band[i - 1]

            # Determine direction
            if i == 1:
                direction[i] = 1 if self.close[i] > upper_band[i] else -1
            else:
                if direction[i - 1] == 1:
                    direction[i] = -1 if self.close[i] < lower_band[i] else 1
                else:
                    direction[i] = 1 if self.close[i] > upper_band[i] else -1

            supertrend_line[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

        return supertrend_line, direction

    # ── Parabolic SAR ─────────────────────────────────────────────

    def parabolic_sar(
        self, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parabolic SAR.

        Returns:
            sar: SAR values
            direction: 1 for long, -1 for short
        """
        sar = np.full(self.n, np.nan)
        direction = np.zeros(self.n)

        if self.n < 2:
            return sar, direction

        # Initialize
        is_long = self.close[1] > self.close[0]
        af = af_start
        ep = self.high[0] if is_long else self.low[0]
        sar[0] = self.low[0] if is_long else self.high[0]
        direction[0] = 1 if is_long else -1

        for i in range(1, self.n):
            prev_sar = sar[i - 1]

            # Calculate SAR
            sar[i] = prev_sar + af * (ep - prev_sar)

            # Ensure SAR doesn't penetrate previous bars
            if is_long:
                sar[i] = min(sar[i], self.low[i - 1])
                if i >= 2:
                    sar[i] = min(sar[i], self.low[i - 2])
            else:
                sar[i] = max(sar[i], self.high[i - 1])
                if i >= 2:
                    sar[i] = max(sar[i], self.high[i - 2])

            # Check for reversal
            if is_long:
                if self.low[i] < sar[i]:
                    is_long = False
                    sar[i] = ep
                    ep = self.low[i]
                    af = af_start
                else:
                    if self.high[i] > ep:
                        ep = self.high[i]
                        af = min(af + af_step, af_max)
            else:
                if self.high[i] > sar[i]:
                    is_long = True
                    sar[i] = ep
                    ep = self.high[i]
                    af = af_start
                else:
                    if self.low[i] < ep:
                        ep = self.low[i]
                        af = min(af + af_step, af_max)

            direction[i] = 1 if is_long else -1

        return sar, direction

    # ── Aroon Indicator ───────────────────────────────────────────

    def aroon(self, period: int = 25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aroon indicator.

        Returns:
            aroon_up: Aroon Up
            aroon_down: Aroon Down
            aroon_osc: Aroon Oscillator (Up - Down)
        """
        up = np.full(self.n, np.nan)
        down = np.full(self.n, np.nan)

        for i in range(period, self.n):
            window_high = self.high[i - period:i + 1]
            window_low = self.low[i - period:i + 1]

            high_idx = np.argmax(window_high)
            low_idx = np.argmin(window_low)

            up[i] = 100 * high_idx / period
            down[i] = 100 * low_idx / period

        osc = up - down
        return up, down, osc

    # ── Chaikin Money Flow ────────────────────────────────────────

    def chaikin_money_flow(self, period: int = 20) -> np.ndarray:
        """Chaikin Money Flow (CMF)."""
        out = np.full(self.n, np.nan)

        for i in range(period - 1, self.n):
            mfv_sum = 0.0
            vol_sum = 0.0
            for j in range(i - period + 1, i + 1):
                hl_range = self.high[j] - self.low[j]
                if hl_range > 0:
                    mfm = ((self.close[j] - self.low[j]) - (self.high[j] - self.close[j])) / hl_range
                    mfv_sum += mfm * self.volume[j]
                    vol_sum += self.volume[j]

            if vol_sum > 0:
                out[i] = mfv_sum / vol_sum

        return out

    # ── Accumulation/Distribution ─────────────────────────────────

    def accumulation_distribution(self) -> np.ndarray:
        """Accumulation/Distribution Line."""
        ad = np.zeros(self.n)
        for i in range(self.n):
            hl_range = self.high[i] - self.low[i]
            if hl_range > 0:
                mfm = ((self.close[i] - self.low[i]) - (self.high[i] - self.close[i])) / hl_range
                ad[i] = (ad[i - 1] if i > 0 else 0) + mfm * self.volume[i]
        return ad

    # ── TRIX ──────────────────────────────────────────────────────

    def trix(self, period: int = 15) -> np.ndarray:
        """TRIX — Triple Exponential Average rate of change."""
        from .technical import TechnicalIndicators
        ti = TechnicalIndicators(self.bars)

        ema1 = ti.ema(period)
        # EMA of EMA
        ema2 = self._ema_of(ema1, period)
        ema3 = self._ema_of(ema2, period)

        out = np.full(self.n, np.nan)
        for i in range(1, self.n):
            if not np.isnan(ema3[i]) and not np.isnan(ema3[i - 1]) and ema3[i - 1] > 0:
                out[i] = (ema3[i] - ema3[i - 1]) / ema3[i - 1] * 10000

        return out

    def _ema_of(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA of an already-computed array."""
        out = np.full(self.n, np.nan)
        alpha = 2.0 / (period + 1)

        # Find first non-NaN
        start = 0
        for i in range(len(data)):
            if not np.isnan(data[i]):
                start = i
                break

        if start + period - 1 >= self.n:
            return out

        out[start + period - 1] = np.nanmean(data[start:start + period])
        for i in range(start + period, self.n):
            if not np.isnan(data[i]) and not np.isnan(out[i - 1]):
                out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]

        return out

    # ── Ultimate Oscillator ───────────────────────────────────────

    def ultimate_oscillator(
        self, p1: int = 7, p2: int = 14, p3: int = 28
    ) -> np.ndarray:
        """Ultimate Oscillator."""
        out = np.full(self.n, np.nan)

        # True range and buying pressure
        bp = np.zeros(self.n)
        tr = np.zeros(self.n)

        for i in range(1, self.n):
            true_low = min(self.low[i], self.close[i - 1])
            true_high = max(self.high[i], self.close[i - 1])
            bp[i] = self.close[i] - true_low
            tr[i] = true_high - true_low

        for i in range(p3, self.n):
            bp1 = np.sum(bp[i - p1 + 1:i + 1])
            tr1 = np.sum(tr[i - p1 + 1:i + 1])
            bp2 = np.sum(bp[i - p2 + 1:i + 1])
            tr2 = np.sum(tr[i - p2 + 1:i + 1])
            bp3 = np.sum(bp[i - p3 + 1:i + 1])
            tr3 = np.sum(tr[i - p3 + 1:i + 1])

            if tr1 > 0 and tr2 > 0 and tr3 > 0:
                avg1 = bp1 / tr1
                avg2 = bp2 / tr2
                avg3 = bp3 / tr3
                out[i] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

        return out

    # ── Vortex Indicator ──────────────────────────────────────────

    def vortex(self, period: int = 14) -> tuple[np.ndarray, np.ndarray]:
        """Vortex Indicator (VI+, VI-)."""
        vi_plus = np.full(self.n, np.nan)
        vi_minus = np.full(self.n, np.nan)

        for i in range(period, self.n):
            vm_plus = 0.0
            vm_minus = 0.0
            tr_sum = 0.0

            for j in range(i - period + 1, i + 1):
                vm_plus += abs(self.high[j] - self.low[j - 1])
                vm_minus += abs(self.low[j] - self.high[j - 1])

                true_high = max(self.high[j], self.close[j - 1])
                true_low = min(self.low[j], self.close[j - 1])
                tr_sum += true_high - true_low

            if tr_sum > 0:
                vi_plus[i] = vm_plus / tr_sum
                vi_minus[i] = vm_minus / tr_sum

        return vi_plus, vi_minus

    # ── Mass Index ────────────────────────────────────────────────

    def mass_index(self, period: int = 9) -> np.ndarray:
        """Mass Index."""
        out = np.full(self.n, np.nan)

        # Single/double EMA of high-low range
        hl_range = self.high - self.low

        from .technical import TechnicalIndicators
        ti = TechnicalIndicators(self.bars)
        # Approximate using the raw array
        ema1 = self._compute_ema(hl_range, period)
        ema2 = self._compute_ema(ema1, period)

        for i in range(2 * period, self.n):
            if not np.isnan(ema1[i]) and not np.isnan(ema2[i]) and ema2[i] > 0:
                ratio = ema1[i] / ema2[i]
                # Sum of ratios over period
                mass_sum = 0.0
                valid = True
                for j in range(i - period + 1, i + 1):
                    if not np.isnan(ema1[j]) and not np.isnan(ema2[j]) and ema2[j] > 0:
                        mass_sum += ema1[j] / ema2[j]
                    else:
                        valid = False
                        break
                if valid:
                    out[i] = mass_sum

        return out

    def _compute_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        out = np.full(len(data), np.nan)
        alpha = 2.0 / (period + 1)
        if len(data) < period:
            return out
        out[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
        return out
