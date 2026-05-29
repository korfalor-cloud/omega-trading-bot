"""Order Book Viewer — live order book display.

Implements:
- Order book snapshot
- Depth visualization
- Imbalance detection
- Spread monitoring
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Order book snapshot."""
    symbol: str = ""
    bids: list = None  # [(price, qty), ...]
    asks: list = None
    mid_price: float = 0.0
    spread: float = 0.0
    spread_bps: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    imbalance: float = 0.0

    def __post_init__(self):
        if self.bids is None:
            self.bids = []
        if self.asks is None:
            self.asks = []


class OrderBookViewer:
    """Order book visualization and analysis."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._books: dict[str, OrderBookSnapshot] = {}

    def update(self, symbol: str, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> OrderBookSnapshot:
        """Update order book."""
        bids = sorted(bids, key=lambda x: x[0], reverse=True)
        asks = sorted(asks, key=lambda x: x[0])

        best_bid = bids[0][0] if bids else 0
        best_ask = asks[0][0] if asks else 0
        mid = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
        spread = best_ask - best_bid
        spread_bps = spread / mid * 10000 if mid > 0 else 0

        bid_depth = sum(qty for _, qty in bids)
        ask_depth = sum(qty for _, qty in asks)
        imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0

        snapshot = OrderBookSnapshot(
            symbol=symbol, bids=bids, asks=asks,
            mid_price=mid, spread=spread, spread_bps=spread_bps,
            bid_depth=bid_depth, ask_depth=ask_depth, imbalance=imbalance,
        )
        self._books[symbol] = snapshot
        return snapshot

    def get_snapshot(self, symbol: str) -> OrderBookSnapshot:
        return self._books.get(symbol, OrderBookSnapshot())

    def format_book(self, symbol: str, levels: int = 5) -> str:
        """Format order book as text."""
        book = self._books.get(symbol)
        if not book:
            return "No data"

        lines = [f"\n{'='*40}", f"  {symbol} Order Book", f"{'='*40}"]

        # Asks (reversed)
        for price, qty in reversed(book.asks[:levels]):
            lines.append(f"  {'':>12}{price:>12.2f}  {'█' * int(qty * 10):<10}  {qty:.4f}")

        lines.append(f"  {'─'*40}")
        lines.append(f"  Mid: {book.mid_price:.2f}  Spread: {book.spread_bps:.1f}bps")
        lines.append(f"  {'─'*40}")

        # Bids
        for price, qty in book.bids[:levels]:
            lines.append(f"  {qty:>12.4f}  {'█' * int(qty * 10):<10}  {price:>12.2f}")

        lines.append(f"\n  Bid Depth: {book.bid_depth:.4f}  Ask Depth: {book.ask_depth:.4f}")
        lines.append(f"  Imbalance: {book.imbalance:+.2f}")
        return "\n".join(lines)
