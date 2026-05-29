"""Box Spread Arbitrage Strategy.

Implements:
- Synthetic long (long call + short put at same strike) +
  synthetic short (short call + long put at another strike)
- Captures the risk-free rate implied by the box price vs actual rates
- Fee-aware execution that accounts for commissions and slippage
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


class BoxSpreadStrategy(Strategy):
    """Box spread arbitrage — lock in synthetic risk-free rate.

    Structure:
        Long box  = long call K1 + short put K1 + short call K2 + long put K2
        Theoretical value = (K2 - K1) * e^(-rT)

    If market price of the box deviates from the theoretical value we can
    capture the arbitrage profit, adjusted for fees and slippage.
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}

        self._expiry_days = feats.get("expiry_days", 30)
        self._risk_free_rate = feats.get("risk_free_rate", 0.05)
        self._strike_width_pct = feats.get("strike_width_pct", 0.10)  # 10 %

        # Fee and execution parameters
        self._fee_per_leg_pct = feats.get("fee_per_leg_pct", 0.0005)  # 5 bps per leg
        self._slippage_pct = feats.get("slippage_pct", 0.0010)  # 10 bps
        self._min_profit_bps = feats.get("min_profit_bps", 20)  # minimum 20 bps after fees

        # Internal state
        self._bar_buffer: list[OHLCVBar] = []
        self._lookback = feats.get("lookback", 30)
        self._in_position = False
        self._K1 = 0.0
        self._K2 = 0.0
        self._entry_box_price = 0.0

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
    # Box valuation
    # ------------------------------------------------------------------
    def _theoretical_box_value(self, T: float) -> float:
        """Present value of the box = (K2 - K1) * e^(-rT)."""
        return (self._K2 - self._K1) * math.exp(-self._risk_free_rate * T)

    def _market_box_price(self, S: float, iv: float, T: float) -> float:
        """Market price of the box (cost to enter all four legs)."""
        call_K1 = self._bs_call(S, self._K1, T, iv)
        put_K1 = self._bs_put(S, self._K1, T, iv)
        call_K2 = self._bs_call(S, self._K2, T, iv)
        put_K2 = self._bs_put(S, self._K2, T, iv)
        # Long call K1 + Short put K1 + Short call K2 + Long put K2
        return call_K1 - put_K1 - call_K2 + put_K2

    def _total_fees(self, S: float) -> float:
        """Total round-trip cost of fees + slippage (4 legs, each direction)."""
        notional = S * 4  # 4 legs
        per_leg_fee = notional * (self._fee_per_leg_pct + self._slippage_pct)
        return per_leg_fee * 2  # entry + exit

    def _implied_rate(self, S: float, T: float) -> float:
        """Rate implied by the box price vs strike width."""
        box_market = self._market_box_price(S, 0.5, T)  # use 50 % iv as proxy
        strike_width = self._K2 - self._K1
        if box_market <= 0 or T <= 0:
            return 0.0
        return -math.log(box_market / strike_width) / T

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

        # --- EXIT ------------------------------------------------------
        if self._in_position:
            # Box spreads converge to intrinsic at expiry; exit when
            # convergence profit exceeds entry cost + fees
            T_remaining = max(T * 0.5, T - 1.0 / 365.0)  # simulate time decay
            theo_val = (self._K2 - self._K1) * math.exp(-self._risk_free_rate * T_remaining)
            pnl = theo_val - self._entry_box_price
            fees = self._total_fees(current_price)

            if pnl > fees or T_remaining < 5.0 / 365.0:
                self._in_position = False
                logger.info("Box spread exit: pnl=%.4f fees=%.4f", pnl, fees)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL,
                    strength=min(1.0, pnl / fees) if fees > 0 else 0.5,
                    confidence=0.70,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={"strategy": "box_spread", "pnl": pnl, "fees": fees},
                )
            return None

        # --- ENTRY -----------------------------------------------------
        # Construct strikes
        self._K1 = round(current_price * (1.0 - self._strike_width_pct / 2), 2)
        self._K2 = round(current_price * (1.0 + self._strike_width_pct / 2), 2)

        # Calculate market box price and theoretical value
        market_price = self._market_box_price(current_price, iv, T)
        theo_value = self._theoretical_box_value(T)
        fees = self._total_fees(current_price)

        # Profit = |market_price - theo_value| - fees
        deviation = abs(market_price - theo_value)
        net_profit = deviation - fees
        net_profit_bps = (net_profit / theo_value * 10000) if theo_value > 0 else 0

        if net_profit_bps > self._min_profit_bps:
            self._entry_box_price = market_price
            self._in_position = True

            # Determine direction: if market price < theo, buy box; else sell
            if market_price < theo_value:
                side = Side.BUY
            else:
                side = Side.SELL

            logger.info(
                "Box spread entry: K1=%.2f K2=%.2f market=%.4f theo=%.4f net_bps=%.1f",
                self._K1, self._K2, market_price, theo_value, net_profit_bps,
            )
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=side,
                strength=min(1.0, net_profit_bps / 100),
                confidence=0.75,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={
                    "strategy": "box_spread",
                    "K1": self._K1,
                    "K2": self._K2,
                    "market_price": market_price,
                    "theoretical_value": theo_value,
                    "net_profit_bps": net_profit_bps,
                    "implied_rate": self._implied_rate(current_price, T),
                    "risk_free_rate": self._risk_free_rate,
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
