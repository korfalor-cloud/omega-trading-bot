"""Tests for multi-agent strategy management."""
from __future__ import annotations

import pytest

from tradingbot.agents.strategy_agent import (
    AgentCoordinator,
    AgentDecision,
    StrategyAgent,
    StrategyPerformance,
)
from tradingbot.core.enums import StrategyStatus
from tradingbot.core.types import PortfolioState
from tradingbot.genome.strategy_genome import create_random_genome
from datetime import datetime, timezone


class TestStrategyAgent:
    @pytest.fixture
    def agent(self):
        return StrategyAgent("test-agent")

    def test_register_strategy(self, agent):
        genome = create_random_genome("test")
        agent.register_strategy(genome)
        assert len(agent.get_managed_strategies()) == 1

    def test_unregister_strategy(self, agent):
        genome = create_random_genome("test")
        agent.register_strategy(genome)
        agent.unregister_strategy(genome.id)
        assert len(agent.get_managed_strategies()) == 0

    def test_performance_update(self, agent):
        genome = create_random_genome("test")
        agent.register_strategy(genome)

        agent.update_performance(genome.id, 100.0, 100100)
        agent.update_performance(genome.id, -50.0, 100050)

        perf = agent.get_performance(genome.id)
        assert perf.total_trades == 2
        assert perf.winning_trades == 1
        assert perf.total_pnl == 50.0

    def test_consecutive_loss_retirement(self, agent):
        agent.max_consecutive_losses = 3
        genome = create_random_genome("loser")
        agent.register_strategy(genome)

        # 3 consecutive losses
        for _ in range(3):
            agent.update_performance(genome.id, -100.0, 90000)

        portfolio = PortfolioState(
            timestamp=datetime.now(timezone.utc),
            total_equity=90000, cash=90000, positions_value=0,
            unrealized_pnl=0, realized_pnl=0,
        )
        decision = agent.tick(
            type("Bar", (), {"symbol": "BTC/USDT"})(),
            portfolio,
        )
        assert decision is not None
        assert decision.decision_type == "retire"

    def test_status(self, agent):
        genome = create_random_genome("test")
        agent.register_strategy(genome)
        status = agent.get_status()
        assert status["agent_id"] == "test-agent"
        assert status["strategies_managed"] == 1


class TestAgentCoordinator:
    @pytest.fixture
    def coordinator(self):
        return AgentCoordinator()

    def test_create_agent(self, coordinator):
        agent = coordinator.create_agent()
        assert agent.agent_id in coordinator._agents

    def test_remove_agent(self, coordinator):
        agent = coordinator.create_agent()
        coordinator.remove_agent(agent.agent_id)
        assert agent.agent_id not in coordinator._agents

    def test_get_all_strategies(self, coordinator):
        agent1 = coordinator.create_agent()
        agent2 = coordinator.create_agent()
        agent1.register_strategy(create_random_genome("a1"))
        agent2.register_strategy(create_random_genome("a2"))
        assert len(coordinator.get_all_strategies()) == 2

    def test_correlation_check(self, coordinator):
        agent = coordinator.create_agent()
        genome1 = create_random_genome("g1")
        genome1.stop_loss_method = "atr"
        genome1.stop_loss_param = 2.0
        genome1.take_profit_ratio = 2.0
        genome1.primary_timeframe = "1h"
        agent.register_strategy(genome1)

        # Similar genome
        genome2 = create_random_genome("g2")
        genome2.stop_loss_method = "atr"
        genome2.stop_loss_param = 2.2
        genome2.take_profit_ratio = 2.1
        genome2.primary_timeframe = "1h"

        assert coordinator.check_correlation(genome2)

    def test_status(self, coordinator):
        coordinator.create_agent()
        status = coordinator.get_status()
        assert status["agents"] == 1
