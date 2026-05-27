"""Trading strategies — concrete implementations of the Strategy ABC."""
from .trend.following import TrendFollowingStrategy
from .mean_reversion.bollinger import BollingerMeanReversion
from .ml.gradient_boost import GradientBoostStrategy

__all__ = [
    "TrendFollowingStrategy",
    "BollingerMeanReversion",
    "GradientBoostStrategy",
]
