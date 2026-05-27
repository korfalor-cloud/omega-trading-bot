"""Volatility Models.

Implements:
- GARCH(1,1) volatility forecasting
- EWMA volatility
- Realized volatility (Parkinson, Garman-Klass, Yang-Zhang)
- Volatility term structure
- Volatility cone (percentile bands)
- Implied volatility surface estimation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VolForecast:
    """Volatility forecast result."""
    current_vol: float = 0.0
    forecast_vol: float = 0.0
    annualized_vol: float = 0.0
    model: str = ""


class VolatilityModel:
    """Multi-model volatility estimation and forecasting."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.ewma_span = config.get("ewma_span", 20)
        self.garch_omega = config.get("garch_omega", 0.00001)
        self.garch_alpha = config.get("garch_alpha", 0.1)
        self.garch_beta = config.get("garch_beta", 0.85)
        self.annualization = config.get("annualization", 365)  # Crypto = 365

    def ewma(self, returns: np.ndarray) -> np.ndarray:
        """Exponentially weighted moving average volatility."""
        n = len(returns)
        result = np.zeros(n)
        alpha = 2.0 / (self.ewma_span + 1)

        result[0] = returns[0] ** 2
        for i in range(1, n):
            result[i] = alpha * returns[i] ** 2 + (1 - alpha) * result[i - 1]

        return np.sqrt(result)

    def garch(self, returns: np.ndarray) -> np.ndarray:
        """GARCH(1,1) conditional volatility."""
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns) if len(returns) > 1 else returns[0] ** 2

        for i in range(1, n):
            sigma2[i] = (
                self.garch_omega
                + self.garch_alpha * returns[i - 1] ** 2
                + self.garch_beta * sigma2[i - 1]
            )

        return np.sqrt(sigma2)

    def garch_forecast(self, returns: np.ndarray, horizon: int = 1) -> VolForecast:
        """Multi-step GARCH volatility forecast."""
        sigma2 = self.garch(returns) ** 2
        current_var = sigma2[-1]

        alpha = self.garch_alpha
        beta = self.garch_beta
        omega = self.garch_omega

        # Long-run variance
        v_long = omega / (1 - alpha - beta) if (alpha + beta) < 1 else current_var

        # Multi-step forecast
        forecast_var = v_long + (alpha + beta) ** (horizon - 1) * (current_var - v_long)

        current_vol = np.sqrt(current_var)
        forecast_vol = np.sqrt(forecast_var)

        return VolForecast(
            current_vol=current_vol,
            forecast_vol=forecast_vol,
            annualized_vol=forecast_vol * np.sqrt(self.annualization),
            model="garch_1_1",
        )

    def realized_volatility(
        self,
        returns: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Standard realized volatility."""
        n = len(returns)
        result = np.full(n, np.nan)
        for i in range(window, n):
            result[i] = np.std(returns[i - window:i]) * np.sqrt(self.annualization)
        return result

    def parkinson_volatility(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Parkinson volatility estimator (uses high-low range)."""
        n = len(highs)
        result = np.full(n, np.nan)
        log_hl = np.log(highs / lows) ** 2

        for i in range(window, n):
            hl_window = log_hl[i - window:i]
            result[i] = np.sqrt(np.mean(hl_window) / (4 * np.log(2)))

        return result

    def garman_klass_volatility(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Garman-Klass volatility estimator."""
        n = len(opens)
        result = np.full(n, np.nan)

        log_hl = np.log(highs / lows) ** 2
        log_co = np.log(closes / opens) ** 2

        for i in range(window, n):
            hl = log_hl[i - window:i]
            co = log_co[i - window:i]
            result[i] = np.sqrt(np.mean(0.5 * hl - (2 * np.log(2) - 1) * co))

        return result

    def yang_zhang_volatility(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Yang-Zhang volatility estimator (handles overnight jumps)."""
        n = len(opens)
        result = np.full(n, np.nan)

        for i in range(window, n):
            o = opens[i - window:i]
            h = highs[i - window:i]
            l = lows[i - window:i]
            c = closes[i - window:i]

            # Overnight returns
            log_oc = np.log(o[1:] / c[:-1]) ** 2
            # Open-to-close
            log_co = np.log(c / o) ** 2
            # Close-to-open (previous)
            log_cc = np.log(c[1:] / c[:-1]) ** 2

            k = 0.34 / (1.34 + (window + 1) / (window - 1))

            overnight = np.var(np.log(o[1:] / c[:-1])) if len(log_oc) > 1 else 0
            open_close = np.var(np.log(c / o)) if len(log_co) > 1 else 0
            rogers_satchell = np.mean(
                np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
            )

            result[i] = np.sqrt(overnight + k * open_close + (1 - k) * rogers_satchell)

        return result

    def volatility_cone(
        self,
        returns: np.ndarray,
        windows: list[int] = None,
        percentiles: list[float] = None,
    ) -> dict[str, dict[float, float]]:
        """Volatility cone — percentile bands at different horizons."""
        if windows is None:
            windows = [5, 10, 20, 40, 60]
        if percentiles is None:
            percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]

        result = {}
        for w in windows:
            vols = []
            for i in range(w, len(returns)):
                v = np.std(returns[i - w:i]) * np.sqrt(self.annualization)
                vols.append(v)

            if vols:
                cone = {}
                for p in percentiles:
                    cone[p] = float(np.percentile(vols, p * 100))
                result[f"{w}d"] = cone

        return result

    def vol_of_vol(self, returns: np.ndarray, window: int = 20) -> float:
        """Volatility of volatility — how much vol changes."""
        vols = self.ewma(returns)
        if len(vols) < window * 2:
            return 0.0
        vol_changes = np.diff(vols[-window * 2:])
        return float(np.std(vol_changes))

    def vol_regime_thresholds(
        self,
        returns: np.ndarray,
        lookback: int = 252,
    ) -> tuple[float, float]:
        """Return low/high volatility thresholds based on historical percentiles."""
        vols = self.ewma(returns)
        recent = vols[-lookback:] if len(vols) >= lookback else vols
        low = float(np.percentile(recent, 33))
        high = float(np.percentile(recent, 67))
        return low, high
