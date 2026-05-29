"""Strategy Factory — dynamic strategy instantiation.

Creates strategy instances from genomes, manages their lifecycle,
and handles hot-swapping of live strategies.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from ..core.types import StrategyGenome

logger = logging.getLogger(__name__)


@dataclass
class StrategyInstance:
    """A live strategy instance."""
    id: str = ""
    name: str = ""
    genome: dict = field(default_factory=dict)
    status: str = "idle"  # idle, running, paused, stopped, errored
    allocated_capital: float = 0.0
    current_pnl: float = 0.0
    total_trades: int = 0
    sharpe: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_signal_at: Optional[datetime] = None
    error_count: int = 0


class StrategyFactory:
    """Dynamic strategy creation and lifecycle management."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.max_strategies = config.get("max_strategies", 20)
        self.default_capital = config.get("default_capital", 10000)
        self._instances: dict[str, StrategyInstance] = {}
        self._strategy_classes: dict[str, type] = {}

    def register_class(self, name: str, cls: type) -> None:
        """Register a strategy class for instantiation."""
        self._strategy_classes[name] = cls

    def create_from_genome(
        self,
        genome: dict,
        strategy_id: str = "",
        capital: float = None,
    ) -> StrategyInstance:
        """Create a strategy instance from a genome."""
        if len(self._instances) >= self.max_strategies:
            raise ValueError(f"Max strategies ({self.max_strategies}) reached")

        sid = strategy_id or f"evolved_{len(self._instances)}"
        instance = StrategyInstance(
            id=sid,
            name=genome.get("entry_indicator", "unknown"),
            genome=genome,
            allocated_capital=capital or self.default_capital,
        )
        self._instances[sid] = instance
        return instance

    def get_instance(self, strategy_id: str) -> Optional[StrategyInstance]:
        return self._instances.get(strategy_id)

    def get_all_instances(self) -> list[StrategyInstance]:
        return list(self._instances.values())

    def get_running(self) -> list[StrategyInstance]:
        return [i for i in self._instances.values() if i.status == "running"]

    def update_performance(
        self,
        strategy_id: str,
        pnl: float = 0,
        trades: int = 0,
        sharpe: float = 0,
    ) -> None:
        inst = self._instances.get(strategy_id)
        if inst:
            inst.current_pnl += pnl
            inst.total_trades += trades
            inst.sharpe = sharpe

    def stop_strategy(self, strategy_id: str) -> bool:
        inst = self._instances.get(strategy_id)
        if inst:
            inst.status = "stopped"
            return True
        return False

    def pause_strategy(self, strategy_id: str) -> bool:
        inst = self._instances.get(strategy_id)
        if inst:
            inst.status = "paused"
            return True
        return False

    def resume_strategy(self, strategy_id: str) -> bool:
        inst = self._instances.get(strategy_id)
        if inst and inst.status == "paused":
            inst.status = "running"
            return True
        return False

    def remove_strategy(self, strategy_id: str) -> bool:
        if strategy_id in self._instances:
            del self._instances[strategy_id]
            return True
        return False

    def get_summary(self) -> dict:
        instances = list(self._instances.values())
        return {
            "total": len(instances),
            "running": sum(1 for i in instances if i.status == "running"),
            "paused": sum(1 for i in instances if i.status == "paused"),
            "stopped": sum(1 for i in instances if i.status == "stopped"),
            "total_pnl": sum(i.current_pnl for i in instances),
            "total_trades": sum(i.total_trades for i in instances),
            "avg_sharpe": np.mean([i.sharpe for i in instances]) if instances else 0,
        }
