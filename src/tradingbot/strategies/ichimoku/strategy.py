"""Ichimoku Cloud Strategy.

Trades based on Ichimoku Cloud signals:
- Tenkan/Kijun cross
- Price position relative to cloud
- Cloud color (Senkou A vs B)
- Chikou span confirmation
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy
from ...features.advanced_indicators import AdvancedIndicators

logger = logging.getLogger(__name__)


class IchimokuStrategy(Strategy):
    """Ichimoku Cloud trading strategy.

    BUY when:
    - Price above cloud
    - Tenkan-sen crosses above Kijun-sen
    - Cloud is bullish (Senkou A > Senkou B)
    - Chikou span above price

    SELL when opposite conditions.

    Parameters (from genome.features):
        tenkan: Tenkan period (default 9)
        kijun: Kijun period (default 26)
        senkou_b: Senkou B period (default 52)
        use_chikou: Require chikou confirmation (default True)
    """

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        feats = genome.features[0] if genome.features else {}
        self._tenkan_period = feats.get("tenkan", 9)
        self._kijun_period = feats.get("kijun", 26)
        self._senkou_b_period = feats.get("senkou_b", 52)
        self._use_chikou = feats.get("use_chikou", True)
        self._atr_mult = genome.stop_loss_param
        self._bar_buffer: list[OHLCVBar] = []
        self._min_bars = self._senkou_b_period + self._kijun_period + 5

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < self._min_bars:
            return None

        if len(self._bar_buffer) > 500:
            self._bar_buffer = self._bar_buffer[-300:]

        ai = AdvancedIndicators(self._bar_buffer)
        ich = ai.ichimoku(self._tenkan_period, self._kijun_period, self._senkou_b_period)

        tenkan = ich["tenkan_sen"]
        kijun = ich["kijun_sen"]
        senkou_a = ich["senkou_a"]
        senkou_b = ich["senkou_b"]
        chikou = ich["chikou"]

        # Current values
        curr_tenkan = tenkan[-1]
        curr_kijun = kijun[-1]
        prev_tenkan = tenkan[-2] if len(tenkan) > 1 else np.nan
        prev_kijun = kijun[-2] if len(kijun) > 1 else np.nan
        curr_senkou_a = senkou_a[-1]
        curr_senkou_b = senkou_b[-1]
        curr_chikou = chikou[-1] if not np.isnan(chikou[-1]) else 0

        if any(x != x for x in [curr_tenkan, curr_kijun, prev_tenkan, prev_kijun]):
            return None

        price = bar.close
        cloud_top = max(curr_senkou_a, curr_senkou_b) if not (np.isnan(curr_senkou_a) or np.isnan(curr_senkou_b)) else price
        cloud_bottom = min(curr_senkou_a, curr_senkou_b) if not (np.isnan(curr_senkou_a) or np.isnan(curr_senkou_b)) else price

        # Tenkan/Kijun crossover
        tk_cross_up = prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun
        tk_cross_down = prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun

        # Price relative to cloud
        above_cloud = price > cloud_top
        below_cloud = price < cloud_bottom

        # Cloud bullish/bearish
        cloud_bullish = not np.isnan(curr_senkou_a) and not np.isnan(curr_senkou_b) and curr_senkou_a > curr_senkou_b
        cloud_bearish = not np.isnan(curr_senkou_a) and not np.isnan(curr_senkou_b) and curr_senkou_a < curr_senkou_b

        # Chikou confirmation
        chikou_bullish = not self._use_chikou or (curr_chikou > 0 and curr_chikou > price)
        chikou_bearish = not self._use_chikou or (curr_chikou > 0 and curr_chikou < price)

        # Compute ATR for stops
        from ...features.technical import TechnicalIndicators
        ti = TechnicalIndicators(self._bar_buffer)
        atr = ti.atr(14)[-1]
        if atr is None or atr != atr:
            atr = price * 0.02

        # BUY signal
        if tk_cross_up and above_cloud and cloud_bullish and chikou_bullish:
            strength = min(1.0, (price - cloud_top) / atr) if atr > 0 else 0.5
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.BUY,
                strength=max(0.3, strength),
                confidence=0.7,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=cloud_bottom - self._atr_mult * atr,
                take_profit=price + self._atr_mult * atr * self.genome.take_profit_ratio,
            )

        # SELL signal
        if tk_cross_down and below_cloud and cloud_bearish and chikou_bearish:
            strength = min(1.0, (cloud_bottom - price) / atr) if atr > 0 else 0.5
            return Signal(
                strategy_id=self.strategy_id,
                symbol=bar.symbol,
                side=Side.SELL,
                strength=max(0.3, strength),
                confidence=0.7,
                signal_type=SignalType.ENTRY,
                timeframe=Timeframe.H1,
                stop_loss=cloud_top + self._atr_mult * atr,
                take_profit=price - self._atr_mult * atr * self.genome.take_profit_ratio,
            )

        return None

    async def on_tick(self, tick) -> Optional[Signal]:
        return None

    def required_symbols(self) -> list[str]:
        return [self.genome.name.split("_")[0] if "_" in self.genome.name else "BTC/USDT"]

    def required_timeframes(self) -> list[Timeframe]:
        return [Timeframe.H1]
