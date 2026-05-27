"""Derivatives trading strategies."""
from .funding_arb import FundingRateArbitrage
from .basis_trade import BasisTradeStrategy

__all__ = ["FundingRateArbitrage", "BasisTradeStrategy"]
