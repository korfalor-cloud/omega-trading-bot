"""Multi-Strategy Orchestrator.

Implements:
- Strategy portfolio management
- Capital allocation across strategies
- Strategy performance monitoring
- Dynamic strategy weighting
- Strategy conflict resolution
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyAllocation:
    """Allocation to a single strategy."""
    strategy_id: str = ""
    weight: float = 0.0
    allocated_capital: float = 0.0
    current_pnl: float = 0.0
    sharpe: float = 0.0
    win_rate: float = 0.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    is_active: bool = True


@dataclass
class OrchestratorSignal:
    """Signal from the orchestrator."""
    strategy_id: str = ""
    symbol: str = ""
    side: str = ""
    strength: float = 0.0
    confidence: float = 0.0
    allocated_size: float = 0.0
    metadata: dict = field(default_factory=dict)


class StrategyOrchestrator:
    """Multi-strategy portfolio orchestrator.

    Manages multiple strategies, allocates capital,
    and resolves conflicting signals.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.total_capital = config.get("total_capital", 100_000.0)
        self.rebalance_interval = config.get("rebalance_interval", 7)  # days
        self.min_weight = config.get("min_weight", 0.05)
        self.max_weight = config.get("max_weight", 0.40)
        self.max_strategies = config.get("max_strategies", 10)
        self._allocations: dict[str, StrategyAllocation] = {}
        self._signal_history: list[OrchestratorSignal] = []
        self._performance_history: dict[str, list[float]] = {}

    def register_strategy(
        self,
        strategy_id: str,
        initial_weight: float = 0.1,
    ) -> StrategyAllocation:
        """Register a new strategy."""
        if len(self._allocations) >= self.max_strategies:
            raise ValueError(f"Max strategies ({self.max_strategies}) reached")

        weight = max(self.min_weight, min(self.max_weight, initial_weight))
        allocation = StrategyAllocation(
            strategy_id=strategy_id,
            weight=weight,
            allocated_capital=self.total_capital * weight,
        )
        self._allocations[strategy_id] = allocation
        self._performance_history[strategy_id] = []
        return allocation

    def remove_strategy(self, strategy_id: str) -> bool:
        """Remove a strategy and redistribute its weight."""
        if strategy_id not in self._allocations:
            return False

        removed_weight = self._allocations[strategy_id].weight
        del self._allocations[strategy_id]
        del self._performance_history[strategy_id]

        # Redistribute weight
        if self._allocations:
            n = len(self._allocations)
            add_per = removed_weight / n
            for alloc in self._allocations.values():
                alloc.weight = min(self.max_weight, alloc.weight + add_per)
            self._normalize_weights()

        return True

    def update_performance(
        self,
        strategy_id: str,
        pnl: float,
        sharpe: float = 0.0,
        win_rate: float = 0.0,
        max_drawdown: float = 0.0,
        trade_count: int = 0,
    ) -> None:
        """Update strategy performance metrics."""
        alloc = self._allocations.get(strategy_id)
        if not alloc:
            return

        alloc.current_pnl = pnl
        alloc.sharpe = sharpe
        alloc.win_rate = win_rate
        alloc.max_drawdown = max_drawdown
        alloc.trade_count = trade_count

        if strategy_id not in self._performance_history:
            self._performance_history[strategy_id] = []
        self._performance_history[strategy_id].append(pnl)

    def compute_signal(
        self,
        signals: list[OrchestratorSignal],
    ) -> dict[str, OrchestratorSignal]:
        """Resolve conflicting signals and compute final allocations.

        If multiple strategies signal on the same symbol:
        - Same direction: combine (weighted average)
        - Opposing: cancel out, take the stronger one
        """
        # Group by symbol
        symbol_signals: dict[str, list[OrchestratorSignal]] = {}
        for sig in signals:
            if sig.symbol not in symbol_signals:
                symbol_signals[sig.symbol] = []
            symbol_signals[sig.symbol].append(sig)

        final_signals = {}
        for symbol, sigs in symbol_signals.items():
            buy_sigs = [s for s in sigs if s.side == "buy"]
            sell_sigs = [s for s in sigs if s.side == "sell"]

            buy_strength = sum(
                s.strength * self._allocations.get(s.strategy_id, StrategyAllocation()).weight
                for s in buy_sigs
            )
            sell_strength = sum(
                s.strength * self._allocations.get(s.strategy_id, StrategyAllocation()).weight
                for s in sell_sigs
            )

            net_strength = buy_strength - sell_strength

            if abs(net_strength) < 0.01:
                continue  # Signals cancel out

            side = "buy" if net_strength > 0 else "sell"
            strength = min(1.0, abs(net_strength))

            # Find the contributing strategy with highest confidence
            contributing = buy_sigs if side == "buy" else sell_sigs
            best = max(contributing, key=lambda s: s.confidence) if contributing else None

            if best:
                alloc = self._allocations.get(best.strategy_id)
                allocated_size = (alloc.allocated_capital * strength / 100000) if alloc else 0

                final_signals[symbol] = OrchestratorSignal(
                    strategy_id=best.strategy_id,
                    symbol=symbol,
                    side=side,
                    strength=strength,
                    confidence=best.confidence,
                    allocated_size=allocated_size,
                    metadata={
                        "n_contributing": len(contributing),
                        "net_strength": net_strength,
                    },
                )

        return final_signals

    def rebalance_weights(self) -> dict[str, float]:
        """Rebalance strategy weights based on performance.

        Uses risk-adjusted returns (Sharpe) to weight strategies.
        """
        if not self._allocations:
            return {}

        sharpes = {}
        for sid, alloc in self._allocations.items():
            if alloc.is_active:
                sharpes[sid] = max(0.01, alloc.sharpe + 1)  # Shift to positive

        if not sharpes:
            return {}

        total = sum(sharpes.values())
        for sid, sharpe in sharpes.items():
            new_weight = sharpe / total
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            self._allocations[sid].weight = new_weight

        self._normalize_weights()

        # Update allocated capital
        for alloc in self._allocations.values():
            alloc.allocated_capital = self.total_capital * alloc.weight

        return {sid: alloc.weight for sid, alloc in self._allocations.items()}

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(a.weight for a in self._allocations.values())
        if total > 0:
            for alloc in self._allocations.values():
                alloc.weight /= total

    def get_allocation(self, strategy_id: str) -> Optional[StrategyAllocation]:
        return self._allocations.get(strategy_id)

    def get_all_allocations(self) -> list[StrategyAllocation]:
        return list(self._allocations.values())

    def get_portfolio_summary(self) -> dict:
        """Get orchestrator portfolio summary."""
        total_pnl = sum(a.current_pnl for a in self._allocations.values())
        total_trades = sum(a.trade_count for a in self._allocations.values())

        return {
            "n_strategies": len(self._allocations),
            "total_capital": self.total_capital,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "strategies": {
                sid: {
                    "weight": a.weight,
                    "allocated": a.allocated_capital,
                    "pnl": a.current_pnl,
                    "sharpe": a.sharpe,
                    "win_rate": a.win_rate,
                    "active": a.is_active,
                }
                for sid, a in self._allocations.items()
            },
        }
