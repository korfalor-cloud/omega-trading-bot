"""Arbitrage strategies."""
from .triangular import TriangularArbitrageStrategy
from .cross_exchange import CrossExchangeArbitrage

__all__ = ["TriangularArbitrageStrategy", "CrossExchangeArbitrage"]
