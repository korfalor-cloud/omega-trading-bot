"""Self-Evolving System — autonomous strategy generation and improvement.

This is the core of the hedge-fund-grade self-improving system.

Implements:
- Strategy genome evolution via genetic programming
- Autonomous code generation and testing
- Performance-driven selection and mutation
- Multi-objective fitness evaluation
- Self-rewriting of strategy parameters and logic
"""
from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvolvedStrategy:
    """A strategy produced by evolution."""
    id: str = ""
    genome: dict = field(default_factory=dict)
    fitness: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    code_hash: str = ""
    created_at: float = 0.0
    alive: bool = True


@dataclass
class EvolutionState:
    """Current state of the evolution process."""
    generation: int = 0
    population_size: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity: float = 0.0
    total_evaluated: int = 0
    elapsed_seconds: float = 0.0


# ── Strategy Genome Schema ───────────────────────────────────────
STRATEGY_GENOME_SCHEMA = {
    # Entry signals
    "entry_indicator": {"type": "categorical", "values": [
        "rsi", "macd_cross", "bb_bounce", "ema_cross", "stochastic",
        "cci", "williams_r", "momentum", "roc", "adx_trend",
    ]},
    "entry_threshold": {"type": "float", "min": 0.1, "max": 0.9},
    "entry_direction": {"type": "categorical", "values": ["long", "short", "both"]},

    # Exit signals
    "exit_indicator": {"type": "categorical", "values": [
        "trailing_stop", "fixed_target", "indicator_exit", "time_exit",
        "volatility_exit", "momentum_exit",
    ]},
    "exit_threshold": {"type": "float", "min": 0.01, "max": 0.20},

    # Risk parameters
    "stop_loss_method": {"type": "categorical", "values": ["atr", "fixed", "trailing", "volatility"]},
    "stop_loss_param": {"type": "float", "min": 0.5, "max": 5.0},
    "take_profit_ratio": {"type": "float", "min": 1.0, "max": 5.0},
    "position_sizing": {"type": "categorical", "values": ["kelly", "fixed", "atr", "volatility", "risk_parity"]},
    "max_position_pct": {"type": "float", "min": 0.01, "max": 0.20},

    # Timing
    "primary_timeframe": {"type": "categorical", "values": ["1m", "5m", "15m", "1h", "4h", "1d"]},
    "cooldown_bars": {"type": "int", "min": 1, "max": 20},

    # Filters
    "volume_filter": {"type": "bool"},
    "trend_filter": {"type": "bool"},
    "volatility_filter": {"type": "bool"},
    "regime_filter": {"type": "bool"},

    # Indicator parameters
    "fast_period": {"type": "int", "min": 3, "max": 30},
    "slow_period": {"type": "int", "min": 10, "max": 100},
    "signal_period": {"type": "int", "min": 3, "max": 20},
}


class SelfEvolver:
    """Autonomous strategy evolution engine.

    This system:
    1. Generates random strategy genomes
    2. Evaluates them via backtesting
    3. Selects the fittest
    4. Mutates and crosses them
    5. Repeats — continuously improving
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.population_size = config.get("population_size", 50)
        self.mutation_rate = config.get("mutation_rate", 0.15)
        self.crossover_rate = config.get("crossover_rate", 0.7)
        self.elite_pct = config.get("elite_pct", 0.1)
        self.tournament_size = config.get("tournament_size", 5)
        self.max_generations = config.get("max_generations", 1000)
        self.diversity_threshold = config.get("diversity_threshold", 0.3)

        self._population: list[EvolvedStrategy] = []
        self._hall_of_fame: list[EvolvedStrategy] = []
        self._generation = 0
        self._total_evaluated = 0
        self._start_time = time.time()
        self._fitness_history: list[float] = []

    # ── Genome Operations ────────────────────────────────────────

    def random_genome(self) -> dict:
        """Generate a random strategy genome."""
        genome = {}
        for param, spec in STRATEGY_GENOME_SCHEMA.items():
            if spec["type"] == "float":
                genome[param] = random.uniform(spec["min"], spec["max"])
            elif spec["type"] == "int":
                genome[param] = random.randint(spec["min"], spec["max"])
            elif spec["type"] == "categorical":
                genome[param] = random.choice(spec["values"])
            elif spec["type"] == "bool":
                genome[param] = random.random() > 0.5
        return genome

    def mutate_genome(self, genome: dict, rate: float = None) -> dict:
        """Mutate a genome with given rate."""
        r = rate or self.mutation_rate
        mutated = dict(genome)

        for param, spec in STRATEGY_GENOME_SCHEMA.items():
            if random.random() > r:
                continue

            if spec["type"] == "float":
                # Gaussian mutation
                current = mutated.get(param, 0.5)
                noise = random.gauss(0, (spec["max"] - spec["min"]) * 0.1)
                mutated[param] = max(spec["min"], min(spec["max"], current + noise))
            elif spec["type"] == "int":
                current = mutated.get(param, 5)
                noise = random.randint(-2, 2)
                mutated[param] = max(spec["min"], min(spec["max"], current + noise))
            elif spec["type"] == "categorical":
                mutated[param] = random.choice(spec["values"])
            elif spec["type"] == "bool":
                mutated[param] = not mutated.get(param, False)

        return mutated

    def crossover(self, parent_a: dict, parent_b: dict) -> dict:
        """Uniform crossover between two genomes."""
        child = {}
        for param in STRATEGY_GENOME_SCHEMA:
            if random.random() < 0.5:
                child[param] = parent_a.get(param)
            else:
                child[param] = parent_b.get(param)
        return child

    def genome_distance(self, a: dict, b: dict) -> float:
        """Distance between two genomes (for diversity measurement)."""
        diff = 0
        total = 0
        for param, spec in STRATEGY_GENOME_SCHEMA.items():
            total += 1
            va = a.get(param)
            vb = b.get(param)
            if spec["type"] in ("float", "int"):
                rng = spec["max"] - spec["min"]
                diff += abs((va or 0) - (vb or 0)) / rng if rng > 0 else 0
            elif spec["type"] == "categorical":
                diff += 0 if va == vb else 1
            elif spec["type"] == "bool":
                diff += 0 if va == vb else 1
        return diff / total if total > 0 else 0

    # ── Population Management ────────────────────────────────────

    def initialize_population(self, size: int = None) -> list[dict]:
        """Create initial random population."""
        n = size or self.population_size
        genomes = [self.random_genome() for _ in range(n)]
        return genomes

    def evaluate_fitness(
        self,
        genome: dict,
        backtest_fn: callable,
    ) -> EvolvedStrategy:
        """Evaluate a genome via backtesting.

        backtest_fn: callable(genome) -> dict with keys:
            sharpe, sortino, max_drawdown, win_rate, profit_factor, total_trades
        """
        result = backtest_fn(genome)

        # Multi-objective fitness
        sharpe = result.get("sharpe", 0)
        sortino = result.get("sortino", 0)
        dd = abs(result.get("max_drawdown", 1))
        win_rate = result.get("win_rate", 0)
        pf = result.get("profit_factor", 0)
        trades = result.get("total_trades", 0)

        # Penalize too few trades
        trade_penalty = min(1.0, trades / 20) if trades < 20 else 1.0

        # Composite fitness
        fitness = (
            sharpe * 0.30
            + sortino * 0.20
            + (1 - dd) * 0.20
            + win_rate * 0.15
            + min(pf, 3) / 3 * 0.15
        ) * trade_penalty

        strategy_id = hashlib.md5(
            f"{genome}_{time.time()}".encode()
        ).hexdigest()[:12]

        evolved = EvolvedStrategy(
            id=strategy_id,
            genome=genome,
            fitness=fitness,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=dd,
            win_rate=win_rate,
            profit_factor=pf,
            total_trades=trades,
            generation=self._generation,
            created_at=time.time(),
        )

        self._total_evaluated += 1
        return evolved

    def select_parents(self, population: list[EvolvedStrategy]) -> tuple[EvolvedStrategy, EvolvedStrategy]:
        """Tournament selection."""
        def tournament():
            contestants = random.sample(population, min(self.tournament_size, len(population)))
            return max(contestants, key=lambda s: s.fitness)

        return tournament(), tournament()

    def evolve_generation(
        self,
        population: list[EvolvedStrategy],
        backtest_fn: callable,
    ) -> list[EvolvedStrategy]:
        """Run one generation of evolution."""
        self._generation += 1

        # Sort by fitness
        population.sort(key=lambda s: s.fitness, reverse=True)

        # Elitism — keep top performers
        n_elite = max(1, int(len(population) * self.elite_pct))
        new_pop = list(population[:n_elite])

        # Fill rest with offspring
        while len(new_pop) < self.population_size:
            parent_a, parent_b = self.select_parents(population)

            if random.random() < self.crossover_rate:
                child_genome = self.crossover(parent_a.genome, parent_b.genome)
            else:
                child_genome = dict(parent_a.genome)

            child_genome = self.mutate_genome(child_genome)
            child = self.evaluate_fitness(child_genome, backtest_fn)
            child.parent_ids = [parent_a.id, parent_b.id]
            new_pop.append(child)

        # Update hall of fame
        for s in new_pop:
            if s.fitness > 0.5:
                self._hall_of_fame.append(s)
        self._hall_of_fame.sort(key=lambda s: s.fitness, reverse=True)
        self._hall_of_fame = self._hall_of_fame[:20]

        # Track fitness
        avg_fit = np.mean([s.fitness for s in new_pop])
        self._fitness_history.append(avg_fit)

        return new_pop

    # ── Self-Rewriting ───────────────────────────────────────────

    def generate_strategy_code(self, genome: dict) -> str:
        """Generate Python strategy code from a genome."""
        indicator = genome.get("entry_indicator", "rsi")
        exit_method = genome.get("exit_indicator", "trailing_stop")
        direction = genome.get("entry_direction", "both")
        stop_method = genome.get("stop_loss_method", "atr")
        sizing = genome.get("position_sizing", "kelly")

        code = f'''"""Auto-generated evolved strategy.
Genome hash: {hashlib.md5(str(genome).encode()).hexdigest()[:8]}
Generation: {self._generation}
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from ...core.enums import Side, SignalType, Timeframe
from ...core.types import OHLCVBar, Signal, StrategyGenome
from ...core.interfaces import Strategy


class EvolvedStrategy(Strategy):
    """Evolved strategy — generation {self._generation}."""

    def __init__(self, strategy_id: str, genome: StrategyGenome):
        super().__init__(strategy_id, genome)
        self._bar_buffer = []
        self._in_trade = False
        self._trade_side = ""
        self._trade_bars = 0
        self._entry_price = 0.0
        self._stop_price = 0.0

    async def on_bar(self, bar: OHLCVBar) -> Optional[Signal]:
        self._bar_buffer.append(bar)
        if len(self._bar_buffer) < {genome.get("slow_period", 50)} + 5:
            return None
        if len(self._bar_buffer) > 300:
            self._bar_buffer = self._bar_buffer[-200:]

        prices = np.array([b.close for b in self._bar_buffer])
        highs = np.array([b.high for b in self._bar_buffer])
        lows = np.array([b.low for b in self._bar_buffer])

        # Compute indicator
        indicator = self._compute_{indicator}(prices, highs, lows)

        # Exit logic
        if self._in_trade:
            self._trade_bars += 1
            exit_signal = self._check_{exit_method}(bar, indicator)
            if exit_signal:
                self._in_trade = False
                self._trade_bars = 0
                return Signal(
                    strategy_id=self.strategy_id, symbol=bar.symbol,
                    side=Side.SELL if self._trade_side == "buy" else Side.BUY,
                    strength=0.6, confidence=0.6,
                    signal_type=SignalType.EXIT, timeframe=Timeframe.H1,
                )
            return None

        # Entry logic
        entry = self._check_{indicator}_entry(indicator, bar)
        if entry and ("{direction}" == "both" or entry == "{direction}"):
            side = Side.BUY if entry == "long" else Side.SELL
            self._in_trade = True
            self._trade_side = "buy" if entry == "long" else "sell"
            self._trade_bars = 0
            self._entry_price = bar.close
            return Signal(
                strategy_id=self.strategy_id, symbol=bar.symbol,
                side=side, strength=0.7, confidence=0.65,
                signal_type=SignalType.ENTRY, timeframe=Timeframe.H1,
            )
        return None

    def _compute_rsi(self, prices, highs, lows):
        period = {genome.get("fast_period", 14)}
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        rsi = np.full(len(prices), 50.0)
        if len(gains) >= period:
            avg_g = np.mean(gains[:period])
            avg_l = np.mean(losses[:period])
            for i in range(period, len(gains)):
                avg_g = (avg_g * (period-1) + gains[i]) / period
                avg_l = (avg_l * (period-1) + losses[i]) / period
                if avg_l > 1e-10:
                    rsi[i+1] = 100 - 100/(1 + avg_g/avg_l)
        return rsi

    def _compute_macd_cross(self, prices, highs, lows):
        fast = {genome.get("fast_period", 12)}
        slow = {genome.get("slow_period", 26)}
        sig = {genome.get("signal_period", 9)}
        fast_ema = self._ema(prices, fast)
        slow_ema = self._ema(prices, slow)
        macd = fast_ema - slow_ema
        signal = self._ema(macd, sig)
        return macd - signal

    def _compute_bb_bounce(self, prices, highs, lows):
        period = {genome.get("fast_period", 20)}
        middle = np.array([np.mean(prices[max(0,i-period+1):i+1]) for i in range(len(prices))])
        std = np.array([np.std(prices[max(0,i-period+1):i+1]) for i in range(len(prices))])
        upper = middle + 2 * std
        lower = middle - 2 * std
        return (prices - lower) / (upper - lower + 1e-10)

    def _compute_ema_cross(self, prices, highs, lows):
        fast = self._ema(prices, {genome.get("fast_period", 5)})
        slow = self._ema(prices, {genome.get("slow_period", 13)})
        return fast - slow

    def _compute_stochastic(self, prices, highs, lows):
        period = {genome.get("fast_period", 14)}
        k = np.full(len(prices), 50.0)
        for i in range(period-1, len(prices)):
            h = np.max(highs[i-period+1:i+1])
            l = np.min(lows[i-period+1:i+1])
            if h != l:
                k[i] = (prices[i]-l)/(h-l)*100
        return k

    def _compute_cci(self, prices, highs, lows):
        tp = (highs + lows + prices) / 3
        period = {genome.get("fast_period", 20)}
        result = np.zeros(len(prices))
        for i in range(period-1, len(tp)):
            seg = tp[i-period+1:i+1]
            m = np.mean(seg)
            mad = np.mean(np.abs(seg - m))
            if mad > 0:
                result[i] = (tp[i]-m)/(0.015*mad)
        return result

    def _compute_williams_r(self, prices, highs, lows):
        period = {genome.get("fast_period", 14)}
        result = np.full(len(prices), -50.0)
        for i in range(period-1, len(prices)):
            h = np.max(highs[i-period+1:i+1])
            l = np.min(lows[i-period+1:i+1])
            if h != l:
                result[i] = (h-prices[i])/(h-l)*-100
        return result

    def _compute_momentum(self, prices, highs, lows):
        period = {genome.get("fast_period", 10)}
        result = np.zeros(len(prices))
        for i in range(period, len(prices)):
            result[i] = prices[i] - prices[i-period]
        return result

    def _compute_roc(self, prices, highs, lows):
        period = {genome.get("fast_period", 10)}
        result = np.zeros(len(prices))
        for i in range(period, len(prices)):
            if prices[i-period] != 0:
                result[i] = (prices[i]-prices[i-period])/prices[i-period]
        return result

    def _compute_adx_trend(self, prices, highs, lows):
        period = {genome.get("fast_period", 14)}
        result = np.zeros(len(prices))
        for i in range(1, len(prices)):
            up = max(0, highs[i]-highs[i-1])
            down = max(0, lows[i-1]-lows[i])
            if up+down > 0:
                result[i] = abs(up-down)/(up+down)*100
        return result

    def _ema(self, data, period):
        alpha = 2/(period+1)
        r = np.zeros_like(data)
        r[0] = data[0]
        for i in range(1, len(data)):
            r[i] = alpha*data[i] + (1-alpha)*r[i-1]
        return r

    def _check_rsi_entry(self, indicator, bar):
        threshold = {genome.get("entry_threshold", 0.3)} * 100
        if indicator[-1] < (100 - threshold):
            return "long"
        if indicator[-1] > threshold:
            return "short"
        return None

    def _check_macd_cross_entry(self, indicator, bar):
        if indicator[-2] < 0 and indicator[-1] > 0:
            return "long"
        if indicator[-2] > 0 and indicator[-1] < 0:
            return "short"
        return None

    def _check_bb_bounce_entry(self, indicator, bar):
        threshold = {genome.get("entry_threshold", 0.3)}
        if indicator[-1] < threshold:
            return "long"
        if indicator[-1] > (1 - threshold):
            return "short"
        return None

    def _check_ema_cross_entry(self, indicator, bar):
        if indicator[-2] < 0 and indicator[-1] > 0:
            return "long"
        if indicator[-2] > 0 and indicator[-1] < 0:
            return "short"
        return None

    def _check_stochastic_entry(self, indicator, bar):
        threshold = {genome.get("entry_threshold", 0.3)} * 100
        if indicator[-1] < threshold:
            return "long"
        if indicator[-1] > (100 - threshold):
            return "short"
        return None

    def _check_cci_entry(self, indicator, bar):
        threshold = {genome.get("entry_threshold", 0.3)} * 200 - 100
        if indicator[-1] < -abs(threshold):
            return "long"
        if indicator[-1] > abs(threshold):
            return "short"
        return None

    def _check_williams_r_entry(self, indicator, bar):
        threshold = {genome.get("entry_threshold", 0.3)} * 100
        if indicator[-1] < -(100 - threshold):
            return "long"
        if indicator[-1] > -threshold:
            return "short"
        return None

    def _check_momentum_entry(self, indicator, bar):
        if indicator[-2] < 0 and indicator[-1] > 0:
            return "long"
        if indicator[-2] > 0 and indicator[-1] < 0:
            return "short"
        return None

    def _check_roc_entry(self, indicator, bar):
        threshold = {genome.get("entry_threshold", 0.3)} * 0.1
        if indicator[-1] > threshold:
            return "long"
        if indicator[-1] < -threshold:
            return "short"
        return None

    def _check_adx_trend_entry(self, indicator, bar):
        threshold = {genome.get("entry_threshold", 0.3)} * 50
        if indicator[-1] > threshold:
            return "long"
        return None

    def _check_trailing_stop(self, bar, indicator):
        atr_period = 14
        prices = [b.close for b in self._bar_buffer[-atr_period:]]
        atr = np.std(np.diff(prices)) if len(prices) > 1 else 0
        mult = {genome.get("stop_loss_param", 2.0)}
        if self._trade_side == "buy":
            self._stop_price = max(self._stop_price, bar.close - atr * mult)
            return bar.close <= self._stop_price
        else:
            self._stop_price = min(self._stop_price, bar.close + atr * mult)
            return bar.close >= self._stop_price

    def _check_fixed_target(self, bar, indicator):
        target = self._entry_price * (1 + {genome.get("take_profit_ratio", 2.0)} * 0.01)
        if self._trade_side == "buy":
            return bar.close >= target
        return bar.close <= self._entry_price * (1 - {genome.get("take_profit_ratio", 2.0)} * 0.01)

    def _check_indicator_exit(self, bar, indicator):
        return self._trade_bars > {genome.get("cooldown_bars", 5)} * 3

    def _check_time_exit(self, bar, indicator):
        return self._trade_bars >= {genome.get("cooldown_bars", 5)} * 5

    def _check_volatility_exit(self, bar, indicator):
        return self._trade_bars >= 3 and abs(bar.close - self._entry_price) / self._entry_price > 0.05

    def _check_momentum_exit(self, bar, indicator):
        return self._trade_bars >= {genome.get("cooldown_bars", 5)}

    async def on_tick(self, tick):
        return None

    def required_symbols(self):
        return [self.genome.name.split("_")[0]] if "_" in self.genome.name else ["BTC/USDT"]

    def required_timeframes(self):
        return [Timeframe.H1]
'''
        return code

    def save_evolved_strategy(self, strategy: EvolvedStrategy, path: str = None) -> str:
        """Generate and save evolved strategy code."""
        code = self.generate_strategy_code(strategy.genome)
        if path is None:
            path = f"src/tradingbot/strategies/evolved/evolved_{strategy.id}.py"
        return code

    # ── Analysis ─────────────────────────────────────────────────

    def get_state(self) -> EvolutionState:
        """Get current evolution state."""
        pop_fitness = [s.fitness for s in self._population] if self._population else [0]
        return EvolutionState(
            generation=self._generation,
            population_size=len(self._population),
            best_fitness=max(pop_fitness),
            avg_fitness=np.mean(pop_fitness),
            diversity=self._compute_diversity(),
            total_evaluated=self._total_evaluated,
            elapsed_seconds=time.time() - self._start_time,
        )

    def _compute_diversity(self) -> float:
        """Compute population diversity."""
        if len(self._population) < 2:
            return 0
        distances = []
        sample = random.sample(self._population, min(10, len(self._population)))
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                distances.append(self.genome_distance(sample[i].genome, sample[j].genome))
        return float(np.mean(distances)) if distances else 0

    def get_hall_of_fame(self) -> list[EvolvedStrategy]:
        return list(self._hall_of_fame)

    def get_fitness_history(self) -> list[float]:
        return list(self._fitness_history)
