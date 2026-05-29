"""Asyncio production trading engine.

Provides the main event-loop-driven engine for live/paper trading:
- Concurrent strategy execution
- Multi-stream data feed management
- Async order routing pipeline with pre-trade risk gates
- Graceful lifecycle management with signal handling
"""
from __future__ import annotations

from .async_engine import AsyncTradingEngine
from .data_feed import DataFeedManager, DataQualityReport
from .order_pipeline import OrderPipeline

__all__ = [
    "AsyncTradingEngine",
    "DataFeedManager",
    "DataQualityReport",
    "OrderPipeline",
]
