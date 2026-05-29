"""Smart Order Router — multi-venue order routing.

Implements:
- Best price routing
- Fee-adjusted routing
- Latency-aware routing
- Iceberg order support
- Bracket orders
- OCO orders
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    """Routing decision."""
    venue: str = ""
    price: float = 0.0
    quantity: float = 0.0
    fee: float = 0.0
    reason: str = ""


@dataclass
class BracketOrder:
    """Bracket order (entry + stop + target)."""
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    quantity: float = 0.0
    side: str = ""


class SmartOrderRouter:
    """Intelligent multi-venue order routing."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.default_fee = config.get("default_fee", 0.001)
        self._venues: dict[str, dict] = {}

    def register_venue(self, name: str, fee_rate: float = 0.001, latency_ms: float = 50) -> None:
        self._venues[name] = {"fee_rate": fee_rate, "latency_ms": latency_ms, "connected": True}

    def route_buy(self, quantity: float, prices: dict[str, float], max_venues: int = 3) -> list[RouterDecision]:
        """Route buy order for best execution."""
        valid = [(v, p) for v, p in prices.items() if v in self._venues and self._venues[v]["connected"]]
        valid.sort(key=lambda x: x[1] * (1 + self._venues[x[0]]["fee_rate"]))

        decisions = []
        remaining = quantity
        for venue, price in valid[:max_venues]:
            if remaining <= 0:
                break
            fee = remaining * price * self._venues[venue]["fee_rate"]
            decisions.append(RouterDecision(venue=venue, price=price, quantity=remaining, fee=fee))
            remaining = 0

        return decisions

    def route_sell(self, quantity: float, prices: dict[str, float], max_venues: int = 3) -> list[RouterDecision]:
        """Route sell order for best execution."""
        valid = [(v, p) for v, p in prices.items() if v in self._venues and self._venues[v]["connected"]]
        valid.sort(key=lambda x: x[1] * (1 - self._venues[x[0]]["fee_rate"]), reverse=True)

        decisions = []
        remaining = quantity
        for venue, price in valid[:max_venues]:
            if remaining <= 0:
                break
            fee = remaining * price * self._venues[venue]["fee_rate"]
            decisions.append(RouterDecision(venue=venue, price=price, quantity=remaining, fee=fee))
            remaining = 0

        return decisions

    def create_bracket(self, side: str, entry: float, stop_pct: float = 0.02, target_pct: float = 0.04, quantity: float = 0) -> BracketOrder:
        """Create bracket order."""
        if side == "buy":
            return BracketOrder(entry_price=entry, stop_price=entry * (1 - stop_pct), target_price=entry * (1 + target_pct), quantity=quantity, side=side)
        return BracketOrder(entry_price=entry, stop_price=entry * (1 + stop_pct), target_price=entry * (1 - target_pct), quantity=quantity, side=side)

    def create_oco(self, stop_price: float, target_price: float, quantity: float, side: str) -> dict:
        """Create OCO (one-cancels-other) order."""
        return {"stop_price": stop_price, "target_price": target_price, "quantity": quantity, "side": side, "type": "oco"}

    def iceberg_slice(self, total_qty: float, slice_pct: float = 0.1) -> float:
        """Calculate iceberg order slice size."""
        return total_qty * slice_pct

    def pegged_order(self, side: str, offset_bps: float = 0, ref_price: float = 0) -> dict:
        """Create pegged order."""
        if side == "buy":
            price = ref_price * (1 - offset_bps / 10000)
        else:
            price = ref_price * (1 + offset_bps / 10000)
        return {"side": side, "price": price, "offset_bps": offset_bps, "ref_price": ref_price}
