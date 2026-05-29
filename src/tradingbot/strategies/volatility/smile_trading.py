"""Volatility Smile Trading Strategy.

Implements:
- Volatility smile / skew detection across the strike surface
- Strike selection for relative-value trades on the smile
- Vega-neutral positioning to isolate skew/convexity exposure
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome, Tick
from ...core.interfaces import Strategy

logger = logging.getLogger(__name__)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class SmileTradingStrategy(Strategy):
    """Trade relative value on the volatility smile.

    Strategy:
        1. Build a simple smile model (quadratic in log-moneyness)
        2. Detect strikes that are cheap vs rich relative to the fitted curve
        3. Enter vega-neutral butterfly or risk-reversal structures to
           capture the mispricing without directional vol exposure
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}

        self._expiry_days = feats.get("expiry_days", 30)
        self._risk_free_rate = feats.get("risk_free_rate", 0.05)

        # Smile parameters
        self._n_strikes = feats.get("n_strikes", 7)  # number of strikes to model
        self._moneyness_range = feats.get("moneyness_range", 0.20)  # +/- 20 %

        # Entry thresholds
        self._min_residual_sigma = feats.get("min_residual_sigma", 1.5)  # std devs from fit
        self._max_vega_imbalance = feats.get("max_vega_imbalance", 10.0)

        # Internal state
        self._bar_buffer: list[OHLCVBar] = []
        self._lookback = feats.get("lookback", 30)
        self._in_position = False
        self._cheap_strike = 0.0
        self._rich_strike = 0.0
        self._position_vega = 0.0

    # ------------------------------------------------------------------
    # Black-Scholes helpers
    # ------------------------------------------------------------------
    def _bs_d1d2(self, S: float, K: float, T: float, sigma: float) -> tuple[float, float]:
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (math.log(S / K) + (self._risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def _bs_vega(self, S: float, K: float, T: float, sigma: float) -> float:
        d1, _ = self._bs_d1d2(S, K, T, sigma)
        return S * _norm_pdf(d1) * math.sqrt(T) / 100.0

    def _bs_call(self, S: float, K: float, T: float, sigma: float) -> float:
        d1, d2 = self._bs_d1d2(S, K, T, sigma)
        return S * _norm_cdf(d1) - K * math.exp(-self._risk_free_rate * T) * _norm_cdf(d2)

    # ------------------------------------------------------------------
    # Smile modelling
    # ------------------------------------------------------------------
    def _build_smile(self, S: float, T: float, iv_atm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a synthetic smile surface around the current price.

        In production this would read from a live vol surface.
        Here we simulate a realistic skew using a simple parameterisation:
            iv(K) = iv_atm * (1 + skew * log(K/S) + convexity * log(K/S)^2)
        """
        strikes = np.array([
            round(S * (1.0 + offset), 2)
            for offset in np.linspace(-self._moneyness_range, self._moneyness_range, self._n_strikes)
        ])
        log_moneyness = np.log(strikes / S)

        # Simulate smile: negative skew (OTM puts have higher IV), convex wings
        skew = -0.15
        convexity = 0.50
        ivs = iv_atm * (1.0 + skew * log_moneyness + convexity * log_moneyness ** 2)
        ivs = np.maximum(ivs, 0.05)  # floor at 5 %

        return strikes, log_moneyness, ivs

    def _fit_smile(self, log_moneyness: np.ndarray, ivs: np.ndarray) -> tuple[float, float, float]:
        """Fit quadratic: iv = a + b*k + c*k^2  (k = log-moneyness)."""
        A = np.column_stack([np.ones_like(log_moneyness), log_moneyness, log_moneyness ** 2])
        coeffs, _, _, _ = np.linalg.lstsq(A, ivs, rcond=None)
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    def _smile_residuals(
        self, log_moneyness: np.ndarray, ivs: np.ndarray, a: float, b: float, c: float
    ) -> np.ndarray:
        """Residuals from the fitted smile."""
        fitted = a + b * log_moneyness + c * log_moneyness ** 2
        return ivs - fitted

    # ------------------------------------------------------------------
    # Core strategy interface
    # ------------------------------------------------------------------
    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._lookback:
            return None

        if len(self._bar_buffer) > 200:
            self._bar_buffer = self._bar_buffer[-150:]

        prices = np.array([b.close for b in self._bar_buffer[-self._lookback:]])
        returns = np.diff(np.log(prices))
        realized_vol = float(np.std(returns) * np.sqrt(365))
        current_price = bar.close
        iv_atm = bar.vwap if bar.vwap > 0 else realized_vol
        T = self._expiry_days / 365.0

        # --- EXIT ------------------------------------------------------
        if self._in_position:
            # Rebuild smile and check if the mispricing has closed
            strikes, log_m, ivs = self._build_smile(current_price, T, iv_atm)
            a, b, c = self._fit_smile(log_m, ivs)
            residuals = self._smile_residuals(log_m, ivs, a, b, c)
            residual_std = float(np.std(residuals))

            cheap_idx = int(np.argmin(residuals))
            rich_idx = int(np.argmax(residuals))
            cheap_res = residuals[cheap_idx]
            rich_res = residuals[rich_idx]

            # Mispricing has mean-reverted (both near zero or flip sides)
            if abs(cheap_res) < 0.5 * residual_std and abs(rich_res) < 0.5 * residual_std:
                self._in_position = False
                logger.info("Smile trading exit: residuals converged")
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL,
                    strength=0.6,
                    confidence=0.65,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={"strategy": "smile_trading", "reason": "residual_convergence"},
                )
            return None

        # --- ENTRY -----------------------------------------------------
        # Build smile and look for outlier strikes
        strikes, log_m, ivs = self._build_smile(current_price, T, iv_atm)
        a, b, c = self._fit_smile(log_m, ivs)
        residuals = self._smile_residuals(log_m, ivs, a, b, c)
        residual_std = float(np.std(residuals))

        if residual_std == 0:
            return None

        cheap_idx = int(np.argmin(residuals))  # most negative residual = cheap
        rich_idx = int(np.argmax(residuals))  # most positive residual = rich
        cheap_res = residuals[cheap_idx]
        rich_res = residuals[rich_idx]

        # Require at least one strike to be significantly mispriced
        if abs(cheap_res) < self._min_residual_sigma * residual_std:
            return None
        if abs(rich_res) < self._min_residual_sigma * residual_std:
            return None

        cheap_strike = float(strikes[cheap_idx])
        rich_strike = float(strikes[rich_idx])

        # Vega-neutral: match vega of long cheap leg to short rich leg
        vega_cheap = self._bs_vega(current_price, cheap_strike, T, float(ivs[cheap_idx]))
        vega_rich = self._bs_vega(current_price, rich_strike, T, float(ivs[rich_idx]))

        if vega_rich == 0:
            return None

        ratio = vega_cheap / vega_rich
        if abs(1.0 - ratio) > self._max_vega_imbalance / 100.0:
            logger.debug("Vega imbalance too large (%.2f), skipping", ratio)
            return None

        self._cheap_strike = cheap_strike
        self._rich_strike = rich_strike
        self._position_vega = 0.0  # approximately vega-neutral
        self._in_position = True

        logger.info(
            "Smile entry: cheap=%.2f (res=%.4f) rich=%.2f (res=%.4f) vega_ratio=%.2f",
            cheap_strike, cheap_res, rich_strike, rich_res, ratio,
        )
        return Signal(
            strategy_id=self.strategy_id,
            symbol=bar.symbol,
            side=Side.BUY,  # buy cheap vol, sell rich vol
            strength=min(1.0, (abs(cheap_res) + abs(rich_res)) / (4 * residual_std)),
            confidence=0.60,
            signal_type=SignalType.ENTRY,
            timeframe=Timeframe.H1,
            metadata={
                "strategy": "smile_trading",
                "cheap_strike": cheap_strike,
                "rich_strike": rich_strike,
                "cheap_residual": float(cheap_res),
                "rich_residual": float(rich_res),
                "vega_ratio": ratio,
                "smile_coeffs": {"a": a, "b": b, "c": c},
            },
        )

    async def on_tick(self, tick: Tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        base = self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC"
        return [f"{base}/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
