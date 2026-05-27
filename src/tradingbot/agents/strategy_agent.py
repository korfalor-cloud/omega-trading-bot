"""Strategy Agent — Autonomous agent for strategy management.

Each agent manages a portfolio of strategies, monitors their performance,
and makes decisions about promotion, demotion, and parameter adjustment.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..core.enums import StrategyStatus
from ..core.types import OHLCVBar, PortfolioState, Signal, StrategyGenome

logger = logging.getLogger(__name__)


@dataclass
class AgentDecision:
    """A decision made by a strategy agent."""
    agent_id: str
    decision_type: str  # promote, demote, retrain, adjust, retire
    strategy_id: str
    reason: str
    confidence: float
    parameters: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class StrategyPerformance:
    """Performance tracking for a managed strategy."""
    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    consecutive_losses: int = 0
    last_signal_time: Optional[datetime] = None
    bars_since_last_trade: int = 0


class StrategyAgent:
    """Autonomous agent that manages a set of strategies.

    Responsibilities:
    - Monitor strategy performance in real-time
    - Decide when to promote strategies (paper → live)
    - Decide when to demote/retire underperforming strategies
    - Trigger retraining when performance degrades
    - Coordinate with other agents to avoid correlation
    """

    def __init__(self, agent_id: str = "", config: Optional[dict] = None):
        self.agent_id = agent_id or str(uuid.uuid4())[:8]
        cfg = config or {}
        self.min_trades_for_promotion = cfg.get("min_trades_for_promotion", 50)
        self.min_sharpe_for_promotion = cfg.get("min_sharpe_for_promotion", 1.0)
        self.max_drawdown_for_retirement = cfg.get("max_drawdown_for_retirement", 0.20)
        self.max_consecutive_losses = cfg.get("max_consecutive_losses", 10)
        self.correlation_threshold = cfg.get("correlation_threshold", 0.8)

        self._managed_strategies: dict[str, StrategyGenome] = {}
        self._performance: dict[str, StrategyPerformance] = {}
        self._decisions: list[AgentDecision] = []
        self._active_signals: list[Signal] = []

    def register_strategy(self, genome: StrategyGenome) -> None:
        """Register a strategy for management."""
        self._managed_strategies[genome.id] = genome
        self._performance[genome.id] = StrategyPerformance(strategy_id=genome.id)
        logger.info(f"Agent {self.agent_id}: registered strategy {genome.name} ({genome.id[:8]})")

    def unregister_strategy(self, strategy_id: str) -> None:
        """Remove a strategy from management."""
        self._managed_strategies.pop(strategy_id, None)
        self._performance.pop(strategy_id, None)

    def update_performance(self, strategy_id: str, trade_pnl: float, equity: float) -> None:
        """Update performance metrics after a trade."""
        if strategy_id not in self._performance:
            return

        perf = self._performance[strategy_id]
        perf.total_trades += 1
        perf.total_pnl += trade_pnl
        perf.bars_since_last_trade = 0

        if trade_pnl > 0:
            perf.winning_trades += 1
            perf.consecutive_losses = 0
        else:
            perf.consecutive_losses += 1

        perf.last_signal_time = datetime.now(timezone.utc)

    def tick(self, bar: OHLCVBar, portfolio: PortfolioState) -> Optional[AgentDecision]:
        """Called on each bar — evaluate strategies and make decisions."""
        for strategy_id, perf in self._performance.items():
            perf.bars_since_last_trade += 1

        decisions = []

        for strategy_id, genome in self._managed_strategies.items():
            perf = self._performance.get(strategy_id)
            if not perf:
                continue

            decision = self._evaluate_strategy(genome, perf, portfolio)
            if decision:
                decisions.append(decision)

        # Return the highest-urgency decision
        if decisions:
            decisions.sort(key=lambda d: d.confidence, reverse=True)
            decision = decisions[0]
            self._decisions.append(decision)
            return decision

        return None

    def _evaluate_strategy(
        self,
        genome: StrategyGenome,
        perf: StrategyPerformance,
        portfolio: PortfolioState,
    ) -> Optional[AgentDecision]:
        """Evaluate a single strategy and decide if action is needed."""

        # Check for retirement (too many consecutive losses)
        if perf.consecutive_losses >= self.max_consecutive_losses:
            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="retire",
                strategy_id=genome.id,
                reason=f"Consecutive losses ({perf.consecutive_losses}) exceeded limit",
                confidence=0.9,
            )

        # Check for demotion (excessive drawdown)
        if perf.max_drawdown > self.max_drawdown_for_retirement:
            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="demote",
                strategy_id=genome.id,
                reason=f"Drawdown ({perf.max_drawdown:.1%}) exceeded limit",
                confidence=0.85,
            )

        # Check for promotion (enough trades, good performance)
        if (genome.status == StrategyStatus.PAPER.value and
            perf.total_trades >= self.min_trades_for_promotion):
            win_rate = perf.winning_trades / perf.total_trades if perf.total_trades > 0 else 0
            if win_rate > 0.55 and perf.sharpe_ratio > self.min_sharpe_for_promotion:
                return AgentDecision(
                    agent_id=self.agent_id,
                    decision_type="promote",
                    strategy_id=genome.id,
                    reason=f"Strong performance: win_rate={win_rate:.1%}, sharpe={perf.sharpe_ratio:.2f}",
                    confidence=min(1.0, perf.sharpe_ratio / 3.0),
                )

        # Check for retraining (performance degradation)
        if (perf.total_trades > 30 and
            perf.winning_trades / max(1, perf.total_trades) < 0.4):
            return AgentDecision(
                agent_id=self.agent_id,
                decision_type="retrain",
                strategy_id=genome.id,
                reason=f"Low win rate ({perf.winning_trades}/{perf.total_trades})",
                confidence=0.6,
            )

        return None

    def get_managed_strategies(self) -> list[StrategyGenome]:
        """Get all managed strategies."""
        return list(self._managed_strategies.values())

    def get_performance(self, strategy_id: str) -> Optional[StrategyPerformance]:
        """Get performance for a specific strategy."""
        return self._performance.get(strategy_id)

    def get_decision_history(self) -> list[AgentDecision]:
        """Get all decisions made by this agent."""
        return list(self._decisions)

    def get_status(self) -> dict:
        """Get agent status summary."""
        return {
            "agent_id": self.agent_id,
            "strategies_managed": len(self._managed_strategies),
            "total_decisions": len(self._decisions),
            "strategies": {
                sid: {
                    "status": g.status,
                    "trades": self._performance[sid].total_trades,
                    "pnl": self._performance[sid].total_pnl,
                    "win_rate": (
                        self._performance[sid].winning_trades / max(1, self._performance[sid].total_trades)
                    ),
                }
                for sid, g in self._managed_strategies.items()
                if sid in self._performance
            },
        }


class AgentCoordinator:
    """Coordinates multiple strategy agents.

    Prevents duplicate strategies, manages resource allocation,
    and ensures portfolio-level constraints.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.max_agents = cfg.get("max_agents", 10)
        self.max_strategies_per_agent = cfg.get("max_strategies_per_agent", 20)
        self._agents: dict[str, StrategyAgent] = {}

    def create_agent(self, config: Optional[dict] = None) -> StrategyAgent:
        """Create and register a new agent."""
        agent = StrategyAgent(config=config)
        self._agents[agent.agent_id] = agent
        logger.info(f"Coordinator: created agent {agent.agent_id}")
        return agent

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent."""
        self._agents.pop(agent_id, None)

    def get_agent(self, agent_id: str) -> Optional[StrategyAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def get_all_strategies(self) -> list[StrategyGenome]:
        """Get all strategies across all agents."""
        strategies = []
        for agent in self._agents.values():
            strategies.extend(agent.get_managed_strategies())
        return strategies

    def check_correlation(self, new_genome: StrategyGenome) -> bool:
        """Check if a new strategy is too similar to existing ones."""
        existing = self.get_all_strategies()
        for existing_genome in existing:
            if self._genomes_similar(new_genome, existing_genome):
                return True
        return False

    def _genomes_similar(self, a: StrategyGenome, b: StrategyGenome) -> bool:
        """Check if two genomes are too similar."""
        # Compare risk parameters
        if (a.stop_loss_method == b.stop_loss_method and
            abs(a.stop_loss_param - b.stop_loss_param) < 0.5 and
            abs(a.take_profit_ratio - b.take_profit_ratio) < 0.5 and
            a.primary_timeframe == b.primary_timeframe):
            return True
        return False

    def tick_all(self, bar: OHLCVBar, portfolio: PortfolioState) -> list[AgentDecision]:
        """Tick all agents and collect decisions."""
        decisions = []
        for agent in self._agents.values():
            decision = agent.tick(bar, portfolio)
            if decision:
                decisions.append(decision)
        return decisions

    def get_status(self) -> dict:
        """Get coordinator status."""
        return {
            "agents": len(self._agents),
            "total_strategies": len(self.get_all_strategies()),
            "agents_status": {
                aid: agent.get_status()
                for aid, agent in self._agents.items()
            },
        }
