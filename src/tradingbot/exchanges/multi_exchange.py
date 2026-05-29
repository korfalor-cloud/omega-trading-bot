"""Multi-Exchange Router.

Implements:
- Best price routing across exchanges
- Order splitting across venues
- Latency-aware routing
- Fee-adjusted price comparison
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VenueQuote:
    """Quote from an exchange venue."""
    exchange: str = ""
    bid: float = 0.0
    ask: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    fee_rate: float = 0.001
    latency_ms: float = 0.0


@dataclass
class RoutingDecision:
    """Order routing decision."""
    exchange: str = ""
    price: float = 0.0
    quantity: float = 0.0
    fee: float = 0.0
    net_price: float = 0.0
    reason: str = ""


class MultiExchangeRouter:
    """Route orders to the best venue."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_latency_ms = config.get("max_latency_ms", 500)
        self.min_liquidity = config.get("min_liquidity", 0.01)
        self._quotes: dict[str, VenueQuote] = {}

    def update_quote(self, quote: VenueQuote) -> None:
        """Update quote from an exchange."""
        self._quotes[quote.exchange] = quote

    def best_bid(self) -> Optional[VenueQuote]:
        """Find best bid across all venues."""
        if not self._quotes:
            return None
        return max(self._quotes.values(), key=lambda q: q.bid - q.fee_rate * q.bid)

    def best_ask(self) -> Optional[VenueQuote]:
        """Find best ask across all venues."""
        if not self._quotes:
            return None
        return min(self._quotes.values(), key=lambda q: q.ask + q.fee_rate * q.ask)

    def route_buy(
        self,
        quantity: float,
        max_exchanges: int = 3,
    ) -> list[RoutingDecision]:
        """Route a buy order across venues for best execution."""
        # Sort by ask (lowest first), filter by latency and liquidity
        valid = [
            q for q in self._quotes.values()
            if q.latency_ms <= self.max_latency_ms
            and q.ask_size >= self.min_liquidity
            and q.ask > 0
        ]
        valid.sort(key=lambda q: q.ask + q.fee_rate * q.ask)

        decisions = []
        remaining = quantity

        for quote in valid[:max_exchanges]:
            if remaining <= 0:
                break

            fill_qty = min(remaining, quote.ask_size)
            fee = fill_qty * quote.ask * quote.fee_rate
            net = quote.ask * fill_qty + fee

            decisions.append(RoutingDecision(
                exchange=quote.exchange,
                price=quote.ask,
                quantity=fill_qty,
                fee=fee,
                net_price=net / fill_qty if fill_qty > 0 else 0,
                reason=f"Best ask: {quote.ask}",
            ))
            remaining -= fill_qty

        return decisions

    def route_sell(
        self,
        quantity: float,
        max_exchanges: int = 3,
    ) -> list[RoutingDecision]:
        """Route a sell order across venues for best execution."""
        valid = [
            q for q in self._quotes.values()
            if q.latency_ms <= self.max_latency_ms
            and q.bid_size >= self.min_liquidity
            and q.bid > 0
        ]
        valid.sort(key=lambda q: q.bid - q.fee_rate * q.bid, reverse=True)

        decisions = []
        remaining = quantity

        for quote in valid[:max_exchanges]:
            if remaining <= 0:
                break

            fill_qty = min(remaining, quote.bid_size)
            fee = fill_qty * quote.bid * quote.fee_rate
            net = quote.bid * fill_qty - fee

            decisions.append(RoutingDecision(
                exchange=quote.exchange,
                price=quote.bid,
                quantity=fill_qty,
                fee=fee,
                net_price=net / fill_qty if fill_qty > 0 else 0,
                reason=f"Best bid: {quote.bid}",
            ))
            remaining -= fill_qty

        return decisions

    def get_venues(self) -> list[str]:
        return list(self._quotes.keys())

    def get_spread_comparison(self) -> dict[str, float]:
        """Compare spreads across venues."""
        return {
            q.exchange: q.ask - q.bid
            for q in self._quotes.values()
            if q.ask > 0 and q.bid > 0
        }
