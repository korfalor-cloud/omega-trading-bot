"""Volatility Term Structure Trading Strategy.

Implements:
- Term structure slope measurement (front vs back month IV)
- Calendar spread on vol (long back month, short front month, or vice versa)
- Mean-reversion on the term structure slope using z-scores
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


class TermStructureStrategy(Strategy):
    """Trade mean-reversion in the volatility term structure.

    Concept:
        - Measure the slope between near-term and longer-term implied vol
        - When the slope is steep (front >> back), sell front / buy back
          (contango mean-reversion)
        - When the slope is inverted (front << back), buy front / sell back
          (backwardation mean-reversion)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}

        self._near_expiry_days = feats.get("near_expiry_days", 7)
        self._far_expiry_days = feats.get("far_expiry_days", 60)
        self._risk_free_rate = feats.get("risk_free_rate", 0.05)

        # Mean-reversion parameters
        self._slope_lookback = feats.get("slope_lookback", 30)
        self._entry_zscore = feats.get("entry_zscore", 2.0)
        self._exit_zscore = feats.get("exit_zscore", 0.5)

        # Internal state
        self._bar_buffer: list[OHLCVBar] = []
        self._lookback = feats.get("lookback", 30)
        self._slope_history: list[float] = []
        self._in_position = False
        self._trade_direction = ""  # "contango" or "backwardation"

    # ------------------------------------------------------------------
    # Black-Scholes
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

    # ------------------------------------------------------------------
    # Term structure analysis
    # ------------------------------------------------------------------
    def _simulate_term_structure(self, S: float, iv_atm: float) -> tuple[float, float]:
        """Simulate near and far expiry IV from the ATM level.

        In production this would read from the live vol surface.
        We add a realistic term-structure shape: near-term vol is more
        sensitive to current conditions, longer-term is more anchored.
        """
        # Typical contango: near < far when markets are calm,
        # inverted when fear is high
        fear_factor = max(0.0, iv_atm - 0.30)  # higher IV => more inversion
        near_iv = iv_atm + fear_factor * 0.5
        far_iv = iv_atm - fear_factor * 0.3
        return max(near_iv, 0.05), max(far_iv, 0.05)

    def _term_structure_slope(self, near_iv: float, far_iv: float) -> float:
        """Slope = (far_iv - near_iv) / near_iv.  Positive = contango."""
        if near_iv == 0:
            return 0.0
        return (far_iv - near_iv) / near_iv

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
        T_near = self._near_expiry_days / 365.0
        T_far = self._far_expiry_days / 365.0

        near_iv, far_iv = self._simulate_term_structure(current_price, iv_atm)
        slope = self._term_structure_slope(near_iv, far_iv)
        self._slope_history.append(slope)

        if len(self._slope_history) < self._slope_lookback:
            return None

        slopes = np.array(self._slope_history[-self._slope_lookback:])
        mean_slope = float(np.mean(slopes))
        std_slope = float(np.std(slopes))
        if std_slope == 0:
            return None
        zscore = (slope - mean_slope) / std_slope

        # --- EXIT ------------------------------------------------------
        if self._in_position:
            exit_z = self._exit_zscore
            if self._trade_direction == "contango" and zscore > -exit_z:
                self._in_position = False
                logger.info("Term structure exit: contango zscore=%.2f", zscore)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.BUY,  # close short front month
                    strength=0.6,
                    confidence=0.65,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={
                        "strategy": "term_structure",
                        "reason": "slope_normalised",
                        "zscore": zscore,
                    },
                )
            if self._trade_direction == "backwardation" and zscore < exit_z:
                self._in_position = False
                logger.info("Term structure exit: backwardation zscore=%.2f", zscore)
                return Signal(
                    strategy_id=self.strategy_id,
                    symbol=bar.symbol,
                    side=Side.SELL,  # close long front month
                    strength=0.6,
                    confidence=0.65,
                    signal_type=SignalType.EXIT,
                    timeframe=Timeframe.H1,
                    metadata={
                        "strategy": "term_structure",
                        "reason": "slope_normalised",
                        "zscore": zscore,
                    },
                )
            return None

        # --- ENTRY -----------------------------------------------------
        # Steep contango (positive zscore) => sell near vol, buy far vol
        if zscore > self._entry_zscore:
            self._in_position = True
            self._trade_direction = "contango"
            logger.info(
                "Term structure entry: contango zscore=%.2f near_iv=%.2f far_iv=%.2f",
                zscore, near_iv, far_iv,
            )
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,  # sell front month (short near vol)
                strength=min(1.0, abs(zscore) / 4),
                confidence=0.60,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={
                    "strategy": "term_structure",
                    "direction": "contango",
                    "near_iv": near_iv,
                    "far_iv": far_iv,
                    "slope": slope,
                    "zscore": zscore,
                    "calendar_spread": {
                        "front_leg": "SELL",
                        "back_leg": "BUY",
                    },
                },
            )

        # Inverted term structure (negative zscore) => buy near vol, sell far vol
        if zscore < -self._entry_zscore:
            self._in_position = True
            self._trade_direction = "backwardation"
            logger.info(
                "Term structure entry: backwardation zscore=%.2f near_iv=%.2f far_iv=%.2f",
                zscore, near_iv, far_iv,
            )
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,  # buy front month (long near vol)
                strength=min(1.0, abs(zscore) / 4),
                confidence=0.60,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                metadata={
                    "strategy": "term_structure",
                    "direction": "backwardation",
                    "near_iv": near_iv,
                    "far_iv": far_iv,
                    "slope": slope,
                    "zscore": zscore,
                    "calendar_spread": {
                        "front_leg": "BUY",
                        "back_leg": "SELL",
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
