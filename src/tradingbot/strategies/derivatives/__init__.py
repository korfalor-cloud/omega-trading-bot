"""Derivatives trading strategies."""
from .funding_arb import FundingRateArbitrage
from .basis_trade import BasisTradeStrategy
from .calendar_spread import CalendarSpreadStrategy
from .gamma_scalp import GammaScalpStrategy
from .vol_arb import VolArbStrategy
from .carry_trade import CarryTradeStrategy
from .butterfly_spread import ButterflySpreadStrategy
from .iron_butterfly import IronButterflyStrategy
from .ratio_spread import RatioSpreadStrategy
from .box_spread import BoxSpreadStrategy
from .conversion_reversal import ConversionReversalStrategy

__all__ = [
    "FundingRateArbitrage", "BasisTradeStrategy",
    "CalendarSpreadStrategy", "GammaScalpStrategy", "VolArbStrategy", "CarryTradeStrategy",
    "ButterflySpreadStrategy", "IronButterflyStrategy", "RatioSpreadStrategy",
    "BoxSpreadStrategy", "ConversionReversalStrategy",
]
