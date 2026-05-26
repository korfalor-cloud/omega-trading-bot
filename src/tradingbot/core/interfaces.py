from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from .enums import Timeframe
from .types import (
    ConsciousnessState,
    EvolutionState,
    Fill,
    OHLCVBar,
    Order,
    OrderBookSnapshot,
    PortfolioState,
    Position,
    RegimeState,
    RiskAlert,
    RiskCheck,
    Signal,
    StrategyGenome,
    Tick,
    WorldModelState,
)


class ExchangeAdapter(ABC):
    """Unified interface for all exchange/broker connectivity."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def fetch_candles(
        self, symbol: str, timeframe: Timeframe, since: Optional[datetime] = None, limit: int = 500
    ) -> list[OHLCVBar]: ...

    @abstractmethod
    async def watch_candles(self, symbol: str, timeframe: Timeframe) -> AsyncIterator[OHLCVBar]: ...

    @abstractmethod
    async def watch_order_book(self, symbol: str, depth: int = 20) -> AsyncIterator[OrderBookSnapshot]: ...

    @abstractmethod
    async def watch_trades(self, symbol: str) -> AsyncIterator[Tick]: ...

    @abstractmethod
    async def submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Order: ...

    @abstractmethod
    async def fetch_order(self, order_id: str, symbol: str) -> Order: ...

    @abstractmethod
    async def fetch_positions(self) -> list[Position]: ...

    @abstractmethod
    async def fetch_balance(self) -> dict[str, float]: ...

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> dict: ...


class Strategy(ABC):
    """Base class for all trading strategies (created from genomes)."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        self.strategy_id = strategy_id
        self.genome = genome
        self._is_active = True
        self._positions: dict[str, Position] = {}
        self._signals_generated = 0

    @abstractmethod
    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]: ...

    @abstractmethod
    async def on_tick(self, tick: Tick) -> Optional[Signal]: ...

    async def on_order_book(self, book: OrderBookSnapshot) -> Optional[Signal]:
        return None

    async def on_fill(self, fill: Fill) -> None:
        pass

    async def on_position_update(self, position: Position) -> None:
        self._positions[f"{position.symbol}:{position.strategy_id}"] = position

    async def on_regime_change(self, regime: RegimeState) -> None:
        pass

    @abstractmethod
    def required_symbols(self) -> list[str]: ...

    @abstractmethod
    def required_timeframes(self) -> list[Timeframe]: ...

    def is_active(self) -> bool:
        return self._is_active

    async def activate(self) -> None:
        self._is_active = True

    async def deactivate(self) -> None:
        self._is_active = False


class ExecutionBackend(ABC):
    """Paper or live execution."""

    @abstractmethod
    async def execute(self, order: Order) -> Order: ...

    @abstractmethod
    async def cancel(self, order: Order) -> Order: ...

    @abstractmethod
    async def get_positions(self, strategy_id: str) -> list[Position]: ...

    @abstractmethod
    async def get_balance(self) -> dict[str, float]: ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order: ...


class RiskEngine(ABC):
    """Risk management interface."""

    @abstractmethod
    async def pre_trade_check(self, signal: Signal, portfolio: PortfolioState) -> RiskCheck: ...

    @abstractmethod
    async def post_trade_update(self, fill: Fill) -> None: ...

    @abstractmethod
    async def get_portfolio_state(self) -> PortfolioState: ...

    @abstractmethod
    async def check_drawdown(self) -> Optional[RiskAlert]: ...


class EvolutionEngine(ABC):
    """Interface for strategy evolution engines."""

    @abstractmethod
    async def initialize_population(self, size: int) -> list[StrategyGenome]: ...

    @abstractmethod
    async def evaluate_fitness(self, genome: StrategyGenome) -> float: ...

    @abstractmethod
    async def evolve_generation(self, population: list[StrategyGenome]) -> list[StrategyGenome]: ...

    @abstractmethod
    async def get_best_strategies(self, n: int) -> list[StrategyGenome]: ...


class WorldModelEngine(ABC):
    """Interface for the world model."""

    @abstractmethod
    async def update(self, market_data: Any) -> None: ...

    @abstractmethod
    async def predict_scenario(self, horizon: int) -> list[dict]: ...

    @abstractmethod
    async def get_causal_graph(self) -> dict: ...

    @abstractmethod
    async def get_state(self) -> WorldModelState: ...


class RegimeDetector(ABC):
    """Interface for regime detection."""

    @abstractmethod
    async def detect(self, market_data: Any) -> RegimeState: ...

    @abstractmethod
    async def get_transition_probabilities(self) -> dict: ...


class ConsciousnessLayer(ABC):
    """Interface for self-awareness."""

    @abstractmethod
    async def reflect(self, experience: Any) -> None: ...

    @abstractmethod
    async def set_goals(self, context: Any) -> list[str]: ...

    @abstractmethod
    async def explain_decision(self, decision: Any) -> str: ...

    @abstractmethod
    async def get_state(self) -> ConsciousnessState: ...
