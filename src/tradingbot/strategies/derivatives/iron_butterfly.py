"""Iron Butterfly Strategy.

Implements:
- Short ATM straddle + long OTM wings (put and call)
- Max profit = net credit received (when spot pins at ATM at expiry)
- Max loss = wing width - net credit
- Adjustment triggers on delta drift and price beyond breakeven
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


class IronButterflyStrategy(Strategy):
    """Iron butterfly — short ATM straddle with protective wings.

    Payoff:
        Max profit  = net credit at ATM strike
        Max loss    = wing width - net credit
        Breakevens  = ATM +/- net credit

    Profits from low realised vol and time decay (positive theta).
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}

        self._wing_width_pct = feats.get("wing_width_pct", 0.05)  # 5 %
        self._expiry_days = feats.get("expiry_days", 30)
        self._risk_free_rate = feats.get("risk_free_rate", 0.05)

        # IV thresholds
        self._iv_entry_min = feats.get("iv_entry_min", 0.40)
        self._iv_exit = feats.get("iv_exit", 0.20)

        # Adjustment triggers
        self._delta_trigger = feats.get("delta_trigger", 0.30)
        self._price_breach_pct = feats.get("price_breach_pct", 0.03)

        # Internal state
        self._bar_buffer: list[OHLCVBar] = []
        self._lookback = feats.get("lookback", 30)
        self._in_position = False
        self._atm_strike = 0.0
        self._put_wing = 0.0
        self._call_wing = 0.0
        self._net_credit = 0.0

    # ------------------------------------------------------------------
    # Black-Scholes building blocks
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

    def _bs_put(self, S: float, K: float, T: float, sigma: float) -> float:
        d1, d2 = self._bs_d1d2(S, K, T, sigma)
        return K * math.exp(-self._risk_free_rate * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    def _bs_greeks(self, S: float, K: float, T: float, sigma: float, is_call: bool) -> dict[str, float]:
        d1, d2 = self._bs_d1d2(S, K, T, sigma)
        gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T)) if (S * sigma * math.sqrt(T)) > 0 else 0.0
        vega = S * _norm_pdf(d1) * math.sqrt(T) / 100.0

        if is_call:
            delta = _norm_cdf(d1)
            theta = (
                -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
                - self._risk_free_rate * K * math.exp(-self._risk_free_rate * T) * _norm_cdf(d2)
            ) / 365.0
        else:
            delta = _norm_cdf(d1) - 1.0
            theta = (
                -(S * _norm_pdf(d1) * sigma) / (2.0 * math.sqrt(T))
                + self._risk_free_rate * K * math.exp(-self._risk_free_rate * T) * _norm_cdf(-d2)
            ) / 365.0
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    # ------------------------------------------------------------------
    # Iron-butterfly aggregation
    # ------------------------------------------------------------------
    def _iron_bfly_greeks(self, S: float, iv: float, T: float) -> dict[str, float]:
        # Short ATM straddle:  short call + short put at ATM
        g_short_call = self._bs_greeks(S, self._atm_strike, T, iv, is_call=True)
        g_short_put = self._bs_greeks(S, self._atm_strike, T, iv, is_call=False)
        # Long wings
        g_long_put = self._bs_greeks(S, self._put_wing, T, iv, is_call=False)
        g_long_call = self._bs_greeks(S, self._call_wing, T, iv, is_call=True)

        def _combine(key: str, sign_short: int = -1) -> float:
            return (
                sign_short * g_short_call[key]
                + sign_short * g_short_put[key]
                + g_long_put[key]
                + g_long_call[key]
            )

        return {
            "delta": _combine("delta"),
            "gamma": _combine("gamma"),
            "vega": _combine("vega"),
            "theta": _combine("theta"),
        }

    def _max_profit(self) -> float:
        return self._net_credit

    def _max_loss(self) -> float:
        wing_width = self._atm_strike - self._put_wing
        return max(0.0, wing_width - self._net_credit)

    def _breakevens(self) -> tuple[float, float]:
        lower = self._atm_strike - self._net_credit
        upper = self._atm_strike + self._net_credit
        return lower, upper

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

        # --- EXIT / ADJUST --------------------------------------------
        if self._in_position:
            greeks = self._iron_bfly_greeks(current_price, iv, T)
            be_low, be_high = self._breakevens()

            # Exit on IV crush (profit from short vega)
            if iv < self._iv_exit:
                self._in_position = False
                logger.info("Iron butterfly exit: iv crush %.2f", iv)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.BUY,
                    strength=0.7,
                    confidence=0.70,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={"strategy": "iron_butterfly", "reason": "iv_crush", "iv": iv},
                )

            # Exit if price breaches wing
            if current_price <= self._put_wing or current_price >= self._call_wing:
                self._in_position = False
                logger.warning("Iron butterfly: price breached wing at %.2f", current_price)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.BUY,
                    strength=0.8,
                    confidence=0.75,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={"strategy": "iron_butterfly", "reason": "wing_breach"},
                )

            # Adjustment trigger: delta drift
            if abs(greeks["delta"]) > self._delta_trigger:
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

        # --- ENTRY -----------------------------------------------------
        # Iron butterfly profits from elevated IV collapsing; enter when IV
        # is above threshold so that theta income is rich and we have room
        # for an IV crush.
        if iv > self._iv_entry_min:
            wing_width = current_price * self._wing_width_pct
            self._atm_strike = round(current_price, 2)
            self._put_wing = round(current_price - wing_width, 2)
            self._call_wing = round(current_price + wing_width, 2)

            # Estimate net credit: ATM straddle premium minus wing premiums
            atm_call = self._bs_call(current_price, self._atm_strike, T, iv)
            atm_put = self._bs_put(current_price, self._atm_strike, T, iv)
            wing_put = self._bs_put(current_price, self._put_wing, T, iv)
            wing_call = self._bs_call(current_price, self._call_wing, T, iv)
            self._net_credit = (atm_call + atm_put) - (wing_put + wing_call)

            if self._net_credit <= 0:
                return None  # negative credit — skip

            self._in_position = True
            greeks = self._iron_bfly_greeks(current_price, iv, T)
            be_low, be_high = self._breakevens()

            logger.info(
                "Iron butterfly entry: atm=%.2f wings=[%.2f, %.2f] credit=%.2f",
                self._atm_strike, self._put_wing, self._call_wing, self._net_credit,
            )
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,  # net credit structure
                strength=min(1.0, self._net_credit / wing_width),
                confidence=0.65,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                target_price=self._atm_strike,
                metadata={
                    "strategy": "iron_butterfly",
                    "atm_strike": self._atm_strike,
                    "put_wing": self._put_wing,
                    "call_wing": self._call_wing,
                    "net_credit": self._net_credit,
                    "max_profit": self._max_profit(),
                    "max_loss": self._max_loss(),
                    "breakevens": [be_low, be_high],
                    "greeks": greeks,
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
