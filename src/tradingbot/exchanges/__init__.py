"""Exchange adapters for live and paper trading."""
from .ccxt_adapter import CCXTAdapter
from .paper_exchange import PaperExecutionBackend
from .multi_exchange import MultiExchangeRouter, VenueQuote, RoutingDecision
from .base_connector import (
    BaseExchangeConnector,
    BalanceInfo,
    ConnectionState,
    PositionInfo,
    RateLimitRule,
)
from .binance_connector import BinanceConnector
from .bybit_connector import BybitConnector
from .paper_trading import EnhancedPaperTrading, PaperTradingConfig, PnLTracker

__all__ = [
    # Legacy
    "CCXTAdapter",
    "PaperExecutionBackend",
    "MultiExchangeRouter",
    "VenueQuote",
    "RoutingDecision",
    # Base
    "BaseExchangeConnector",
    "BalanceInfo",
    "ConnectionState",
    "PositionInfo",
    "RateLimitRule",
    # Connectors
    "BinanceConnector",
    "BybitConnector",
    # Paper trading
    "EnhancedPaperTrading",
    "PaperTradingConfig",
    "PnLTracker",
]
