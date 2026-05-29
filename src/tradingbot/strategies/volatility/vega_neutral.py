"""Vega-Neutral Volatility Trading Strategy.

Implements:
- Vega hedging to maintain delta-neutral, vega-neutral exposure
- Gamma capture via delta-hedged long gamma positions
- P&L attribution decomposed into gamma, theta, and vega components
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


class VegaNeutralStrategy(Strategy):
    """Vega-neutral volatility trading with gamma capture.

    Structure:
        - Long gamma (via ATM straddle or similar) to profit from
          realized moves
        - Hedge vega with a calendar spread or additional options
        - Delta-hedge the combined position to isolate gamma P&L

    P&L attribution:
        Gamma P&L  ~= 0.5 * gamma * (dS)^2
        Theta P&L  ~= theta * dt
        Vega P&L   ~= vega * dIV
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}

        self._expiry_days = feats.get("expiry_days", 30)
        self._risk_free_rate = feats.get("risk_free_rate", 0.05)

        # Hedging parameters
        self._hedge_threshold = feats.get("hedge_threshold", 0.10)
        self._vega_hedge_tolerance = feats.get("vega_hedge_tolerance", 5.0)
        self._gamma_target = feats.get("gamma_target", 0.02)

        # P&L tracking
        self._lookback = feats.get("lookback", 30)

        # Internal state
        self._bar_buffer: list[OHLCVBar] = []
        self._in_position = False
        self._delta = 0.0
        self._gamma = 0.0
        self._vega = 0.0
        self._theta = 0.0
        self._gamma_pnl = 0.0
        self._theta_pnl = 0.0
        self._vega_pnl = 0.0
        self._prev_iv = 0.0
        self._position_size = 0.0

    # ------------------------------------------------------------------
    # Black-Scholes
    # ------------------------------------------------------------------
    def _bs_d1d2(self, S: float, K: float, T: float, sigma: float) -> tuple[float, float]:
        if T <= 0 or sigma <= 0:
            return 0.0, 0.0
        d1 = (math.log(S / K) + (self._risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def _bs_greeks(self, S: float, K: float, T: float, sigma: float, is_call: bool = True) -> dict[str, float]:
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

    def _straddle_greeks(self, S: float, K: float, T: float, sigma: float) -> dict[str, float]:
        """Greeks of a long ATM straddle (long call + long put at K)."""
        call_g = self._bs_greeks(S, K, T, sigma, is_call=True)
        put_g = self._bs_greeks(S, K, T, sigma, is_call=False)
        return {
            "delta": call_g["delta"] + put_g["delta"],
            "gamma": call_g["gamma"] + put_g["gamma"],
            "vega": call_g["vega"] + put_g["vega"],
            "theta": call_g["theta"] + put_g["theta"],
        }

    def _compute_pnl_attribution(self, dS: float, dIV: float, dt: float) -> dict[str, float]:
        """Decompose P&L into gamma, theta, and vega components."""
        gamma_pnl = 0.5 * self._gamma * dS ** 2 * self._position_size
        theta_pnl = self._theta * dt * self._position_size
        vega_pnl = self._vega * dIV * self._position_size
        return {
            "gamma_pnl": gamma_pnl,
            "theta_pnl": theta_pnl,
            "vega_pnl": vega_pnl,
            "total_pnl": gamma_pnl + theta_pnl + vega_pnl,
        }

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

        atm_strike = round(current_price, 2)
        straddle_greeks = self._straddle_greeks(current_price, atm_strike, T, iv)

        # --- P&L attribution (while in position) -----------------------
        if self._in_position and len(self._bar_buffer) > 1:
            prev_price = self._bar_buffer[-2].close
            dS = current_price - prev_price
            dIV = iv - self._prev_iv
            dt = 1.0  # 1 bar
            attr = self._compute_pnl_attribution(dS, dIV, dt)
            self._gamma_pnl += attr["gamma_pnl"]
            self._theta_pnl += attr["theta_pnl"]
            self._vega_pnl += attr["vega_pnl"]

        self._prev_iv = iv

        # --- EXIT ------------------------------------------------------
        if self._in_position:
            total_attr_pnl = self._gamma_pnl + self._theta_pnl + self._vega_pnl

            # Exit when cumulative theta overwhelms gamma (time decay wins)
            if self._theta_pnl < -abs(self._gamma_pnl) * 0.8:
                self._in_position = False
                logger.info(
                    "Vega-neutral exit: theta dominates. gamma_pnl=%.4f theta_pnl=%.4f",
                    self._gamma_pnl, self._theta_pnl,
                )
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL,
                    strength=0.65,
                    confidence=0.70,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={
                        "strategy": "vega_neutral",
                        "reason": "theta_decay",
                        "gamma_pnl": self._gamma_pnl,
                        "theta_pnl": self._theta_pnl,
                        "vega_pnl": self._vega_pnl,
                    },
                )

            # Delta hedge
            if abs(self._delta) > self._hedge_threshold:
                hedge_side = Side.SELL if self._delta > 0 else Side.BUY
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=hedge_side,
                    strength=0.4,
                    confidence=0.60,
                    signal_type=SignalType.HEDGE,
                    timeframe=Timeframe.H1,
                    metadata={"delta_hedge": True, "portfolio_delta": self._delta},
                )

            # Vega hedge: adjust if vega drifts beyond tolerance
            if abs(self._vega) > self._vega_hedge_tolerance:
                hedge_side = Side.SELL if self._vega > 0 else Side.BUY
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=hedge_side,
                    strength=0.35,
                    confidence=0.55,
                    signal_type=SignalType.HEDGE,
                    timeframe=Timeframe.H1,
                    metadata={"vega_hedge": True, "portfolio_vega": self._vega},
                )
            return None

        # --- ENTRY -----------------------------------------------------
        # Enter a long gamma position when realized vol is below implied
        # (we expect to capture gamma P&L that exceeds theta decay)
        if realized_vol < iv * 0.85:
            # Scale position to target gamma
            per_unit_gamma = straddle_greeks["gamma"]
            if per_unit_gamma > 0:
                self._position_size = self._gamma_target / per_unit_gamma
            else:
                self._position_size = 1.0

            self._gamma = straddle_greeks["gamma"]
            self._vega = straddle_greeks["vega"]
            self._theta = straddle_greeks["theta"]
            self._delta = straddle_greeks["delta"]
            self._gamma_pnl = 0.0
            self._theta_pnl = 0.0
            self._vega_pnl = 0.0
            self._in_position = True

            logger.info(
                "Vega-neutral entry: rv=%.2f iv=%.2f gamma=%.4f vega=%.4f theta=%.4f size=%.2f",
                realized_vol, iv, self._gamma, self._vega, self._theta, self._position_size,
            )
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,  # long straddle
                strength=min(1.0, (iv - realized_vol) / iv),
                confidence=0.65,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                target_price=atm_strike,
                metadata={
                    "strategy": "vega_neutral",
                    "realized_vol": realized_vol,
                    "implied_vol": iv,
                    "gamma": self._gamma,
                    "vega": self._vega,
                    "theta": self._theta,
                    "delta": self._delta,
                    "position_size": self._position_size,
                    "pnl_attribution": {
                        "gamma_pnl": self._gamma_pnl,
                        "theta_pnl": self._theta_pnl,
                        "vega_pnl": self._vega_pnl,
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
