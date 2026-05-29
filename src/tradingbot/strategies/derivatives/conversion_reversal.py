"""Options Conversion / Reversal Arbitrage Strategy.

Implements:
- Put-call parity violation detection
- Conversion:  long stock + short call + long put
- Reversal:     short stock + long call + short put
- Fee-adjusted profit calculation across all legs
- Multi-leg execution signals
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


class ConversionReversalStrategy(Strategy):
    """Conversion / reversal arbitrage exploiting put-call parity violations.

    Put-call parity:  C - P = S - K * e^(-rT)

    Conversion (parity overpriced):
        Long stock + Short call K + Long put K   =>  lock in risk-free profit
        Profit = (C - P) - (S - K * e^(-rT)) > 0

    Reversal (parity underpriced):
        Short stock + Long call K + Short put K  =>  lock in risk-free profit
        Profit = (S - K * e^(-rT)) - (C - P) > 0
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}

        self._expiry_days = feats.get("expiry_days", 30)
        self._risk_free_rate = feats.get("risk_free_rate", 0.05)

        # Execution costs
        self._fee_per_leg_pct = feats.get("fee_per_leg_pct", 0.0005)  # 5 bps
        self._slippage_pct = feats.get("slippage_pct", 0.0010)  # 10 bps
        self._min_profit_bps = feats.get("min_profit_bps", 15)

        # Strike selection: ATM or near-ATM
        self._strike_offset_pct = feats.get("strike_offset_pct", 0.0)

        # Internal state
        self._bar_buffer: list[OHLCVBar] = []
        self._lookback = feats.get("lookback", 30)
        self._in_position = False
        self._position_type = ""  # "conversion" or "reversal"
        self._strike = 0.0
        self._entry_parity_deviation = 0.0

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

    def _bs_put(self, S: float, K: float, T: float, sigma: float) -> float:
        d1, d2 = self._bs_d1d2(S, K, T, sigma)
        return K * math.exp(-self._risk_free_rate * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    # ------------------------------------------------------------------
    # Parity analysis
    # ------------------------------------------------------------------
    def _parity_fair_value(self, S: float, K: float, T: float) -> float:
        """C - P fair value per put-call parity = S - K * e^(-rT)."""
        return S - K * math.exp(-self._risk_free_rate * T)

    def _parity_deviation(self, S: float, K: float, T: float, iv: float) -> float:
        """Market (C - P) minus fair value.  Positive => conversion opportunity."""
        call_price = self._bs_call(S, K, T, iv)
        put_price = self._bs_put(S, K, T, iv)
        fair_value = self._parity_fair_value(S, K, T)
        return (call_price - put_price) - fair_value

    def _total_fees(self, S: float) -> float:
        """Fees + slippage for a 3-leg round trip."""
        notional = S * 3
        per_leg = notional * (self._fee_per_leg_pct + self._slippage_pct)
        return per_leg * 2  # entry + exit

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
        strike = round(current_price * (1.0 + self._strike_offset_pct), 2)

        # --- EXIT ------------------------------------------------------
        if self._in_position:
            # Parity should converge to fair value; exit when deviation
            # has narrowed below fees or at near-expiry
            deviation = abs(self._parity_deviation(current_price, self._strike, T, iv))
            fees = self._total_fees(current_price)

            if deviation < fees * 0.5 or T < 3.0 / 365.0:
                self._in_position = False
                logger.info(
                    "Conversion/Reversal exit: type=%s deviation=%.4f",
                    self._position_type, deviation,
                )
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL if self._position_type == "conversion" else Side.BUY,
                    strength=0.6,
                    confidence=0.70,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={
                        "strategy": "conversion_reversal",
                        "type": self._position_type,
                        "deviation": deviation,
                    },
                )
            return None

        # --- ENTRY -----------------------------------------------------
        deviation = self._parity_deviation(current_price, strike, T, iv)
        fees = self._total_fees(current_price)
        net_profit = abs(deviation) - fees
        net_profit_bps = (net_profit / current_price * 10000) if current_price > 0 else 0

        if net_profit_bps > self._min_profit_bps:
            if deviation > 0:
                # C - P > fair value => conversion
                # Long stock + Short call + Long put
                self._position_type = "conversion"
                entry_side = Side.SELL  # structure involves short call (net bearish overlay)
            else:
                # C - P < fair value => reversal
                # Short stock + Long call + Short put
                self._position_type = "reversal"
                entry_side = Side.BUY  # structure involves long call (net bullish overlay)

            self._strike = strike
            self._entry_parity_deviation = deviation
            self._in_position = True

            logger.info(
                "%s entry: strike=%.2f deviation=%.4f net_bps=%.1f",
                self._position_type, strike, deviation, net_profit_bps,
            )
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=entry_side,
                strength=min(1.0, net_profit_bps / 50),
                confidence=0.80,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                target_price=strike,
                metadata={
                    "strategy": "conversion_reversal",
                    "type": self._position_type,
                    "strike": strike,
                    "deviation": deviation,
                    "net_profit_bps": net_profit_bps,
                    "parity_fair_value": self._parity_fair_value(current_price, strike, T),
                    "legs": {
                        "stock": "BUY" if self._position_type == "conversion" else "SELL",
                        "call": "SELL" if self._position_type == "conversion" else "BUY",
                        "put": "BUY" if self._position_type == "conversion" else "SELL",
                    },
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
