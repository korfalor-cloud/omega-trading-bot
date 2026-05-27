"""Tests for configuration loading."""
from __future__ import annotations

import pytest

from tradingbot.config import (
    BacktestConfig,
    EvolutionConfig,
    MonitoringConfig,
    OmegaConfig,
    RiskConfig,
    load_config,
)


class TestOmegaConfig:
    def test_default_values(self):
        config = OmegaConfig()
        assert config.mode == "paper"
        assert config.log_level == "INFO"
        assert len(config.symbols) > 0
        assert config.risk.max_position_pct > 0

    def test_risk_defaults(self):
        config = OmegaConfig()
        assert config.risk.max_drawdown_pct == 0.15
        assert config.risk.daily_loss_limit_pct == 0.05
        assert config.risk.max_leverage == 3.0

    def test_evolution_defaults(self):
        config = OmegaConfig()
        assert config.evolution.population_size == 1000
        assert config.evolution.mutation_rate == 0.25
        assert config.evolution.crossover_rate == 0.70

    def test_backtest_defaults(self):
        config = OmegaConfig()
        assert config.backtest.initial_capital == 100000.0
        assert config.backtest.slippage_bps == 5.0

    def test_monitoring_defaults(self):
        config = OmegaConfig()
        assert config.monitoring.heartbeat_interval_seconds == 30
        assert config.monitoring.prometheus_port == 9090

    def test_model_dump(self):
        config = OmegaConfig()
        data = config.model_dump()
        assert "mode" in data
        assert "risk" in data
        assert "evolution" in data


class TestLoadConfig:
    def test_load_default(self):
        config = load_config("configs/default.yaml")
        assert isinstance(config, OmegaConfig)
        assert config.mode in ("paper", "live", "backtest")

    def test_load_nonexistent(self):
        config = load_config("nonexistent.yaml")
        assert isinstance(config, OmegaConfig)
        assert config.mode == "paper"  # Default
