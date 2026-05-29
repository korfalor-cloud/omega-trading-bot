"""Options Ratio Spread Strategy.

Implements:
- Asymmetric payoff via unequal long / short legs
- Vega exposure management and monitoring
- Entry / exit conditions based on IV and price proximity to short strike
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


class RatioSpreadStrategy(Strategy):
    """Ratio spread — asymmetric options structure.

    Classic 1x2 call ratio spread:
        Long  1 ATM call
        Short 2 OTM calls

    Characteristics:
        - Can be entered for a small credit or zero-cost
        - Unlimited risk on the upside beyond the short strike
        - Maximum profit at the short strike at expiry
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}

        self._expiry_days = feats.get("expiry_days", 30)
        self._risk_free_rate = feats.get("risk_free_rate", 0.05)

        # Ratio: long _n_long, short _n_short (typical 1x2)
        self._n_long = feats.get("n_long", 1)
        self._n_short = feats.get("n_short", 2)
        self._otm_offset_pct = feats.get("otm_offset_pct", 0.05)  # 5 % OTM

        # Vega management
        self._max_abs_vega = feats.get("max_abs_vega", 50.0)
        self._vega_exit_threshold = feats.get("vega_exit_threshold", 80.0)

        # IV thresholds
        self._iv_entry_min = feats.get("iv_entry_min", 0.30)
        self._iv_exit = feats.get("iv_exit", 0.15)

        # Internal state
        self._bar_buffer: list[OHLCVBar] = []
        self._lookback = feats.get("lookback", 30)
        self._in_position = False
        self._long_strike = 0.0
        self._short_strike = 0.0
        self._position_vega = 0.0

    # ------------------------------------------------------------------
    # Black-Scholes
    # ------------------------------------------------------------------
    def _bs_d1d2(self, S: float, K: float, T: float, sigma: float) -> tuple[float, float]:
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (math.log(S / K) + (self._risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def _bs_call(self, S: float, K: float, T: float, sigma: float) -> float:
        d1, d2 = self._bs_d1d2(S, K, T, sigma)
        return S * _norm_cdf(d1) - K * math.exp(-self._risk_free_rate * T) * _norm_cdf(d2)

    def _bs_call_greeks(self, S: float, K: float, T: float, sigma: float) -> dict[str, float]:
        d1, d2 = self._bs_d1d2(S, K, T, sigma)
        delta = _norm_cdf(d1)
        gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T)) if (S * sigma * math.sqrt(T)) > 0 else 0.0
        vega = S * _norm_pdf(d1) * math.sqrt(T) / 100.0
        theta = (
            -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
            - self._risk_free_rate * K * math.exp(-self._risk_free_rate * T) * _norm_cdf(d2)
        ) / 365.0
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    # ------------------------------------------------------------------
    # Ratio spread aggregation
    # ------------------------------------------------------------------
    def _ratio_greeks(self, S: float, iv: float, T: float) -> dict[str, float]:
        g_long = self._bs_call_greeks(S, self._long_strike, T, iv)
        g_short = self._bs_call_greeks(S, self._short_strike, T, iv)
        return {
            "delta": self._n_long * g_long["delta"] - self._n_short * g_short["delta"],
            "gamma": self._n_long * g_long["gamma"] - self._n_short * g_short["gamma"],
            "vega": self._n_long * g_long["vega"] - self._n_short * g_short["vega"],
            "theta": self._n_long * g_long["theta"] - self._n_short * g_short["theta"],
        }

    def _max_profit(self) -> float:
        """Max profit at the short strike."""
        wing = self._short_strike - self._long_strike
        return self._n_long * wing  # approximate (ignores net debit/credit)

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
        iv = bar.vwap if bar.vwap > 0 else realized_vol
        T = self._expiry_days / 365.0

        # --- EXIT / MANAGE --------------------------------------------
        if self._in_position:
            greeks = self._ratio_greeks(current_price, iv, T)
            self._position_vega = greeks["vega"]

            # Exit on IV crush (profit from net short vega)
            if iv < self._iv_exit:
                self._in_position = False
                logger.info("Ratio spread exit: iv crush %.2f", iv)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL,
                    strength=0.6,
                    confidence=0.65,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={"strategy": "ratio_spread", "reason": "iv_crush", "iv": iv},
                )

            # Exit if price blows through short strike (unlimited risk zone)
            if current_price > self._short_strike * 1.05:
                self._in_position = False
                logger.warning("Ratio spread: price above short strike %.2f", current_price)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.BUY,
                    strength=0.8,
                    confidence=0.75,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={"strategy": "ratio_spread", "reason": "upside_breach"},
                )

            # Vega management — hedge if vega exposure gets too large
            if abs(greeks["vega"]) > self._vega_exit_threshold:
                hedge_side = Side.SELL if greeks["vega"] > 0 else Side.BUY
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=hedge_side,
                    strength=0.4,
                    confidence=0.55,
                    signal_type=SignalType.HEDGE,
                    timeframe=Timeframe.H1,
                    metadata={"vega_hedge": True, "portfolio_vega": greeks["vega"]},
                )
            return None

        # --- ENTRY -----------------------------------------------------
        if iv > self._iv_entry_min:
            self._long_strike = round(current_price, 2)  # ATM
            self._short_strike = round(current_price * (1.0 + self._otm_offset_pct), 2)

            greeks = self._ratio_greeks(current_price, iv, T)
            self._position_vega = greeks["vega"]
            self._in_position = True

            logger.info(
                "Ratio spread entry: long=%.2f short=%.2f ratio=%dx%d iv=%.2f",
                self._long_strike, self._short_strike,
                self._n_long, self._n_short, iv,
            )
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=min(1.0, (iv - self._iv_entry_min) / 0.3),
                confidence=0.60,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                target_price=self._short_strike,
                metadata={
                    "strategy": "ratio_spread",
                    "long_strike": self._long_strike,
                    "short_strike": self._short_strike,
                    "ratio": f"{self._n_long}x{self._n_short}",
                    "greeks": greeks,
                    "max_profit": self._max_profit(),
                    "iv": iv,
                },
            )

        return None

    async def on_tick(self, tick: Tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        base = self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC"
        return [f"{base}/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
