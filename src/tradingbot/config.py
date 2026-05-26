from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ExchangeConfig(BaseModel):
    name: str
    api_key: str = ""
    api_secret: str = ""
    passphrase: str = ""
    testnet: bool = True
    rate_limit: int = 6000


class RiskConfig(BaseModel):
    max_position_pct: float = 0.05
    max_gross_exposure: float = 2.0
    max_drawdown_pct: float = 0.15
    daily_loss_limit_pct: float = 0.05
    max_leverage: float = 3.0
    var_confidence: float = 0.95
    circuit_breaker_dd_pct: float = 0.10


class EvolutionConfig(BaseModel):
    population_size: int = 1000
    num_islands: int = 10
    generations: int = 1000
    mutation_rate: float = 0.25
    crossover_rate: float = 0.70
    elitism_pct: float = 0.10
    immigration_pct: float = 0.05
    tournament_size: int = 5
    speciation_threshold: float = 0.3
    # Fitness weights
    sharpe_weight: float = 0.35
    sortino_weight: float = 0.25
    max_dd_weight: float = 0.20
    win_rate_weight: float = 0.10
    stability_weight: float = 0.10


class RegimeConfig(BaseModel):
    hmm_states: int = 4
    bocpd_hazard_rate: float = 0.01
    lookback_days: int = 90
    vol_lookback: int = 20
    correlation_lookback: int = 30


class WorldModelConfig(BaseModel):
    causal_update_interval_hours: int = 24
    scenario_count: int = 10000
    simulation_horizon_days: int = 30
    participant_model_update_hours: int = 6


class SwarmConfig(BaseModel):
    ant_colony_size: int = 100
    particle_swarm_size: int = 500
    bee_colony_size: int = 50
    firefly_count: int = 100
    wolf_pack_size: int = 30
    max_iterations: int = 1000


class ConsciousnessConfig(BaseModel):
    reflection_interval_hours: int = 1
    goal_review_interval_hours: int = 24
    uncertainty_threshold: float = 0.7
    min_confidence_for_live: float = 0.6


class BacktestConfig(BaseModel):
    initial_capital: float = 100_000.0
    slippage_bps: float = 5.0
    commission_bps: float = 10.0
    lookback_days: int = 365
    walk_forward_train_days: int = 180
    walk_forward_test_days: int = 30


class MonitoringConfig(BaseModel):
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    prometheus_port: int = 9090
    grafana_port: int = 3000
    heartbeat_interval_seconds: int = 30


class OmegaConfig(BaseSettings):
    """Master configuration for the Omega Trading Intelligence."""
    # Mode
    mode: str = "paper"  # paper, live, backtest
    log_level: str = "INFO"

    # Paths
    data_dir: str = "./data"
    model_dir: str = "./models"
    log_dir: str = "./logs"

    # Exchanges
    exchanges: dict[str, ExchangeConfig] = Field(default_factory=dict)

    # Symbols to trade
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])

    # Subsystem configs
    risk: RiskConfig = Field(default_factory=RiskConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    consciousness: ConsciousnessConfig = Field(default_factory=ConsciousnessConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    # Evolution schedule
    evolution_interval_hours: int = 168  # Weekly
    paper_trial_days: int = 21  # 3 weeks
    meta_learning_interval_hours: int = 720  # Monthly

    model_config = {"env_prefix": "OMEGA_", "env_nested_delimiter": "__"}


def load_config(config_path: Optional[str] = None) -> OmegaConfig:
    """Load configuration from YAML file and environment variables."""
    config_data: dict[str, Any] = {}

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}

    # Load exchange configs from separate files
    exchanges_dir = Path(config_path).parent / "exchanges" if config_path else None
    if exchanges_dir and exchanges_dir.exists():
        for f in exchanges_dir.glob("*.yaml"):
            with open(f) as fh:
                ex_data = yaml.safe_load(fh) or {}
                name = f.stem
                config_data.setdefault("exchanges", {})[name] = ex_data

    return OmegaConfig(**config_data)
