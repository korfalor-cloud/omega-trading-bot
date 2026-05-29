"""Butterfly Spread Strategy.

Implements:
- Options butterfly spread with ATM body + OTM wings
- Strike selection based on current price and wing width
- P&L calculation across the payoff structure
- Greeks monitoring (delta, gamma, theta, vega)
- Entry/exit logic driven by implied volatility regime
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
    """Standard normal cumulative distribution function (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class ButterflySpreadStrategy(Strategy):
    """Options butterfly spread — defined-risk volatility bet.

    Structure:  long 1 lower wing call, short 2 ATM body calls,
                long 1 upper wing call.  Maximum profit when the
                underlying expires exactly at the body strike.
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}

        # Strike selection parameters
        self._wing_width_pct = feats.get("wing_width_pct", 0.03)  # 3 % from ATM
        self._expiry_days = feats.get("expiry_days", 30)
        self._risk_free_rate = feats.get("risk_free_rate", 0.05)

        # IV thresholds for entry / exit
        self._iv_high_threshold = feats.get("iv_high_threshold", 0.60)
        self._iv_low_threshold = feats.get("iv_low_threshold", 0.25)
        self._iv_exit_threshold = feats.get("iv_exit_threshold", 0.35)

        # Greeks limits
        self._max_abs_portfolio_delta = feats.get("max_abs_portfolio_delta", 0.15)

        # Position tracking
        self._bar_buffer: list[OHLCVBar] = []
        self._lookback = feats.get("lookback", 30)
        self._in_position = False
        self._body_strike = 0.0
        self._lower_strike = 0.0
        self._upper_strike = 0.0
        self._entry_price = 0.0
        self._entry_iv = 0.0

    # ------------------------------------------------------------------
    # Black-Scholes helpers
    # ------------------------------------------------------------------
    def _bs_d1d2(self, S: float, K: float, T: float, sigma: float) -> tuple[float, float]:
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (math.log(S / K) + (self._risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def _bs_call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        d1, d2 = self._bs_d1d2(S, K, T, sigma)
        return S * _norm_cdf(d1) - K * math.exp(-self._risk_free_rate * T) * _norm_cdf(d2)

    def _bs_greeks(self, S: float, K: float, T: float, sigma: float) -> dict[str, float]:
        d1, d2 = self._bs_d1d2(S, K, T, sigma)
        delta = _norm_cdf(d1)
        gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T)) if (S * sigma * math.sqrt(T)) > 0 else 0.0
        vega = S * _norm_pdf(d1) * math.sqrt(T) / 100.0  # per 1 % vol move
        theta = (
            -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
            - self._risk_free_rate * K * math.exp(-self._risk_free_rate * T) * _norm_cdf(d2)
        ) / 365.0
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    # ------------------------------------------------------------------
    # Butterfly P&L
    # ------------------------------------------------------------------
    def _butterfly_pnl(self, S: float, iv: float, T: float) -> float:
        """Net P&L of the butterfly at spot S."""
        lower = self._bs_call_price(S, self._lower_strike, T, iv)
        body = self._bs_call_price(S, self._body_strike, T, iv)
        upper = self._bs_call_price(S, self._upper_strike, T, iv)
        # Long 1 lower, short 2 body, long 1 upper
        return lower - 2.0 * body + upper

    def _butterfly_greeks(self, S: float, iv: float, T: float) -> dict[str, float]:
        g_lower = self._bs_greeks(S, self._lower_strike, T, iv)
        g_body = self._bs_greeks(S, self._body_strike, T, iv)
        g_upper = self._bs_greeks(S, self._upper_strike, T, iv)
        return {
            "delta": g_lower["delta"] - 2.0 * g_body["delta"] + g_upper["delta"],
            "gamma": g_lower["gamma"] - 2.0 * g_body["gamma"] + g_upper["gamma"],
            "vega": g_lower["vega"] - 2.0 * g_body["vega"] + g_upper["vega"],
            "theta": g_lower["theta"] - 2.0 * g_body["theta"] + g_upper["theta"],
        }

    def _max_profit(self) -> float:
        """Max profit = wing width - net debit (approx wing width for ATM body)."""
        return self._body_strike - self._lower_strike

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

        # Use vwap as proxy for implied vol if available; fall back to realized
        iv = bar.vwap if bar.vwap > 0 else realized_vol

        T = self._expiry_days / 365.0
        wing_width = current_price * self._wing_width_pct

        # --- EXIT logic ------------------------------------------------
        if self._in_position:
            greeks = self._butterfly_greeks(current_price, iv, T)
            pnl = self._butterfly_pnl(current_price, iv, T)
            max_pnl = self._max_profit()

            # Exit on IV collapse (favourable vol crush) or breach of delta
            if iv < self._iv_exit_threshold or pnl > 0.75 * max_pnl:
                self._in_position = False
                logger.info(
                    "Butterfly exit: iv=%.2f pnl=%.4f max_pnl=%.4f", iv, pnl, max_pnl
                )
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL,
                    strength=min(1.0, pnl / max_pnl) if max_pnl > 0 else 0.5,
                    confidence=0.70,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={
                        "strategy": "butterfly_spread",
                        "pnl": pnl,
                        "max_profit": max_pnl,
                        "greeks": greeks,
                        "iv": iv,
                    },
                )

            # Adjust if portfolio delta drifts too far
            if abs(greeks["delta"]) > self._max_abs_portfolio_delta:
                hedge_side = Side.SELL if greeks["delta"] > 0 else Side.BUY
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=hedge_side,
                    strength=0.4,
                    confidence=0.60,
                    signal_type=SignalType.HEDGE,
                    timeframe=Timeframe.H1,
                    metadata={"delta_hedge": True, "portfolio_delta": greeks["delta"]},
                )
            return None

        # --- ENTRY logic -----------------------------------------------
        # Enter when IV is elevated — the butterfly is short vega, so we
        # profit from an IV drop; high IV also makes the structure cheaper
        if iv > self._iv_high_threshold:
            self._body_strike = round(current_price, 2)
            self._lower_strike = round(current_price - wing_width, 2)
            self._upper_strike = round(current_price + wing_width, 2)
            self._entry_price = current_price
            self._entry_iv = iv
            self._in_position = True

            greeks = self._butterfly_greeks(current_price, iv, T)
            logger.info(
                "Butterfly entry: body=%.2f wings=[%.2f, %.2f] iv=%.2f",
                self._body_strike, self._lower_strike, self._upper_strike, iv,
            )
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,  # net debit structure
                strength=min(1.0, (iv - self._iv_high_threshold) / 0.3),
                confidence=0.65,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                target_price=self._body_strike,
                metadata={
                    "strategy": "butterfly_spread",
                    "body_strike": self._body_strike,
                    "lower_strike": self._lower_strike,
                    "upper_strike": self._upper_strike,
                    "wing_width": wing_width,
                    "iv": iv,
                    "greeks": greeks,
                    "max_profit": self._max_profit(),
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
