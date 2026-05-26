from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .enums import AssetClass, OrderState, OrderType, Side, SignalType, Timeframe


@dataclass(frozen=True)
class OHLCVBar:
    timestamp: datetime
    symbol: str
    timeframe: Timeframe
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str
    trades_count: int = 0
    vwap: float = 0.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass(frozen=True)
class Tick:
    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    side: Side
    exchange: str
    trade_id: str = ""


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    quantity: float
    order_count: int = 0


@dataclass(frozen=True)
class OrderBookSnapshot:
    timestamp: datetime
    symbol: str
    exchange: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_bps(self) -> Optional[float]:
        mid = self.mid_price
        spread = self.spread
        if mid and spread and mid > 0:
            return (spread / mid) * 10000
        return None

    @property
    def bid_depth(self) -> float:
        return sum(l.quantity for l in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(l.quantity for l in self.asks)

    @property
    def imbalance(self) -> Optional[float]:
        total = self.bid_depth + self.ask_depth
        if total > 0:
            return (self.bid_depth - self.ask_depth) / total
        return None


@dataclass
class Signal:
    strategy_id: str
    symbol: str
    side: Side
    strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    signal_type: SignalType = SignalType.ENTRY
    timeframe: Timeframe = Timeframe.H1
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_atr_mult: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Order:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    symbol: str = ""
    side: Side = Side.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    state: OrderState = OrderState.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    exchange_order_id: Optional[str] = None
    exchange: str = ""
    asset_class: AssetClass = AssetClass.CRYPTO
    time_in_force: str = "GTC"
    reduce_only: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    @property
    def is_active(self) -> bool:
        return self.state in (OrderState.PENDING, OrderState.SUBMITTED, OrderState.PARTIAL)

    @property
    def notional_value(self) -> float:
        return self.quantity * (self.price or self.avg_fill_price or 0.0)


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: Side
    price: float
    quantity: float
    commission: float
    exchange: str
    timestamp: datetime
    fill_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Position:
    symbol: str
    strategy_id: str
    side: Side
    quantity: float
    avg_entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    exchange: str = ""
    asset_class: AssetClass = AssetClass.CRYPTO
    opened_at: datetime = field(default_factory=datetime.utcnow)
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0

    @property
    def notional_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        multiplier = 1 if self.side == Side.BUY else -1
        return multiplier * (self.current_price - self.avg_entry_price) / self.avg_entry_price

    def update_price(self, price: float) -> None:
        self.current_price = price
        multiplier = 1 if self.side == Side.BUY else -1
        self.unrealized_pnl = multiplier * (price - self.avg_entry_price) * self.quantity


@dataclass
class RiskCheck:
    approved: bool
    reason: str = ""
    max_allowed_quantity: float = 0.0
    risk_score: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class PortfolioState:
    timestamp: datetime
    total_equity: float
    cash: float
    positions_value: float
    unrealized_pnl: float
    realized_pnl: float
    positions: list[Position] = field(default_factory=list)
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    portfolio_delta: float = 0.0
    portfolio_gamma: float = 0.0
    portfolio_vega: float = 0.0
    portfolio_theta: float = 0.0


@dataclass
class RiskAlert:
    level: str
    message: str
    metric: str
    current_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    action_taken: str = ""


@dataclass
class StrategyGenome:
    """The DNA of a strategy — everything needed to create and evaluate it."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    # Rule tree (JSON-serializable AST)
    signal_tree: dict = field(default_factory=dict)
    # Risk parameters
    stop_loss_method: str = "atr"  # atr, fixed, trailing, none
    stop_loss_param: float = 2.0
    take_profit_ratio: float = 2.0
    position_sizing: str = "kelly"  # kelly, fixed, atr, volatility
    max_position_pct: float = 0.05
    # Timing parameters
    primary_timeframe: str = "1h"
    confirmation_timeframe: str = "4h"
    cooldown_bars: int = 3
    # Regime filters
    active_regimes: list[str] = field(default_factory=lambda: ["bull_low_vol", "trending"])
    min_volatility: float = 0.0
    max_volatility: float = 1.0
    # Fitness scores
    fitness: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    # Metadata
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "dormant"
    # Feature genome (evolved features)
    features: list[dict] = field(default_factory=list)


@dataclass
class RegimeState:
    """Current market regime information."""
    regime: str
    confidence: float
    volatility_percentile: float
    trend_strength: float
    correlation_regime: str
    transition_probability: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


@dataclass
class WorldModelState:
    """State of the world model."""
    causal_graph: dict = field(default_factory=dict)
    regime_probabilities: dict = field(default_factory=dict)
    participant_states: dict = field(default_factory=dict)
    predicted_scenarios: list[dict] = field(default_factory=list)
    uncertainty: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConsciousnessState:
    """Self-awareness state."""
    confidence_in_predictions: float = 0.0
    uncertainty_about_market: float = 0.0
    strategy_health: dict = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EvolutionState:
    """State of the evolution cycle."""
    generation: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity_score: float = 0.0
    strategies_alive: int = 0
    strategies_retired: int = 0
    strategies_promoted: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
