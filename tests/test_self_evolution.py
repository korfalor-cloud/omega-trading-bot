"""Tests for the self-evolving system."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.evolution.self_evolver import SelfEvolver, EvolvedStrategy, EvolutionState, STRATEGY_GENOME_SCHEMA
from tradingbot.evolution.strategy_factory import StrategyFactory, StrategyInstance
from tradingbot.evolution.parameter_optimizer import ParameterOptimizer, OptimizationResult
from tradingbot.evolution.meta_learner import MetaLearner, StrategyScore
from tradingbot.evolution.regime_allocator import RegimeAdaptiveAllocator, AllocationResult


class TestSelfEvolver:
    @pytest.fixture
    def evolver(self):
        return SelfEvolver(config={"population_size": 10, "mutation_rate": 0.2})

    def test_random_genome(self, evolver):
        genome = evolver.random_genome()
        assert isinstance(genome, dict)
        assert "entry_indicator" in genome
        assert "stop_loss_method" in genome

    def test_genome_has_all_params(self, evolver):
        genome = evolver.random_genome()
        for param in STRATEGY_GENOME_SCHEMA:
            assert param in genome

    def test_mutate_genome(self, evolver):
        original = evolver.random_genome()
        mutated = evolver.mutate_genome(original, rate=1.0)  # Mutate all
        # At least some params should differ
        diffs = sum(1 for k in original if original[k] != mutated[k])
        assert diffs > 0

    def test_crossover(self, evolver):
        a = evolver.random_genome()
        b = evolver.random_genome()
        child = evolver.crossover(a, b)
        assert isinstance(child, dict)
        assert len(child) == len(a)

    def test_genome_distance(self, evolver):
        a = evolver.random_genome()
        b = evolver.random_genome()
        dist = evolver.genome_distance(a, b)
        assert 0 <= dist <= 1

    def test_same_genome_distance_zero(self, evolver):
        a = evolver.random_genome()
        assert evolver.genome_distance(a, a) == 0

    def test_initialize_population(self, evolver):
        pop = evolver.initialize_population(5)
        assert len(pop) == 5
        for genome in pop:
            assert isinstance(genome, dict)

    def test_evaluate_fitness(self, evolver):
        genome = evolver.random_genome()
        def mock_backtest(g):
            return {"sharpe": 1.5, "sortino": 2.0, "max_drawdown": 0.1, "win_rate": 0.6, "profit_factor": 1.5, "total_trades": 30}
        result = evolver.evaluate_fitness(genome, mock_backtest)
        assert isinstance(result, EvolvedStrategy)
        assert result.fitness > 0
        assert result.sharpe == 1.5

    def test_evolve_generation(self, evolver):
        pop = evolver.initialize_population(5)
        def mock_backtest(g):
            return {"sharpe": np.random.uniform(0, 2), "sortino": np.random.uniform(0, 2), "max_drawdown": np.random.uniform(0, 0.3), "win_rate": np.random.uniform(0.3, 0.7), "profit_factor": np.random.uniform(0.5, 2), "total_trades": 30}
        evaluated = [evolver.evaluate_fitness(g, mock_backtest) for g in pop]
        new_pop = evolver.evolve_generation(evaluated, mock_backtest)
        assert len(new_pop) >= 5

    def test_generate_strategy_code(self, evolver):
        genome = evolver.random_genome()
        code = evolver.generate_strategy_code(genome)
        assert "class EvolvedStrategy" in code
        assert "async def on_bar" in code
        assert "def required_symbols" in code

    def test_get_state(self, evolver):
        state = evolver.get_state()
        assert isinstance(state, EvolutionState)
        assert state.generation == 0

    def test_hall_of_fame(self, evolver):
        assert evolver.get_hall_of_fame() == []


class TestStrategyFactory:
    @pytest.fixture
    def factory(self):
        return StrategyFactory(config={"max_strategies": 5})

    def test_create_from_genome(self, factory):
        genome = {"entry_indicator": "rsi", "stop_loss_method": "atr"}
        inst = factory.create_from_genome(genome, "test_1")
        assert isinstance(inst, StrategyInstance)
        assert inst.id == "test_1"
        assert inst.name == "rsi"

    def test_max_strategies(self, factory):
        for i in range(5):
            factory.create_from_genome({}, f"s_{i}")
        with pytest.raises(ValueError):
            factory.create_from_genome({}, "s_5")

    def test_get_instance(self, factory):
        factory.create_from_genome({}, "test")
        assert factory.get_instance("test") is not None
        assert factory.get_instance("nonexistent") is None

    def test_update_performance(self, factory):
        factory.create_from_genome({}, "test")
        factory.update_performance("test", pnl=100, trades=5, sharpe=1.5)
        inst = factory.get_instance("test")
        assert inst.current_pnl == 100
        assert inst.total_trades == 5

    def test_stop_strategy(self, factory):
        factory.create_from_genome({}, "test")
        assert factory.stop_strategy("test") is True
        assert factory.get_instance("test").status == "stopped"

    def test_pause_resume(self, factory):
        factory.create_from_genome({}, "test")
        factory.pause_strategy("test")
        assert factory.get_instance("test").status == "paused"
        factory.resume_strategy("test")
        assert factory.get_instance("test").status == "running"

    def test_remove_strategy(self, factory):
        factory.create_from_genome({}, "test")
        assert factory.remove_strategy("test") is True
        assert factory.get_instance("test") is None

    def test_get_summary(self, factory):
        factory.create_from_genome({}, "s1")
        factory.create_from_genome({}, "s2")
        summary = factory.get_summary()
        assert summary["total"] == 2


class TestParameterOptimizer:
    @pytest.fixture
    def optimizer(self):
        return ParameterOptimizer(config={"n_trials": 20})

    def test_grid_search(self, optimizer):
        def objective(params):
            return -(params["x"] - 5) ** 2 + 10
        result = optimizer.grid_search({"x": list(range(10))}, objective)
        assert isinstance(result, OptimizationResult)
        assert result.best_params["x"] == 5

    def test_random_search(self, optimizer):
        def objective(params):
            return -(params["x"] - 0.5) ** 2
        result = optimizer.random_search({"x": (0, 1)}, objective, n_trials=50)
        assert abs(result.best_params["x"] - 0.5) < 0.3

    def test_hill_climb(self, optimizer):
        def objective(params):
            return -(params["x"] - 3) ** 2
        result = optimizer.hill_climb({"x": 0}, {"x": (0, 10)}, objective, n_iterations=50)
        assert abs(result.best_params["x"] - 3) < 2

    def test_record(self, optimizer):
        optimizer.record("test", {"x": 1}, 0.5)
        history = optimizer.get_history("test")
        assert len(history) == 1


class TestMetaLearner:
    @pytest.fixture
    def learner(self):
        return MetaLearner(config={"min_history": 5})

    def test_record_outcome(self, learner):
        learner.record_outcome("strat_1", 100, "bull_low_vol")
        learner.record_outcome("strat_1", -50, "bear_high_vol")
        stats = learner.get_strategy_stats("strat_1")
        assert stats["n_trades"] == 2

    def test_score_strategy_insufficient(self, learner):
        score = learner.score_strategy("unknown")
        assert score.confidence == 0.1

    def test_score_strategy(self, learner):
        rng = np.random.default_rng(42)
        for _ in range(20):
            learner.record_outcome("strat_1", rng.uniform(-5, 15), "bull_low_vol")
        score = learner.score_strategy("strat_1", "bull_low_vol")
        assert score.confidence > 0
        assert isinstance(score.expected_sharpe, float)

    def test_rank_strategies(self, learner):
        for _ in range(20):
            learner.record_outcome("good", 10)
            learner.record_outcome("bad", -5)
        ranking = learner.rank_strategies()
        assert len(ranking) == 2

    def test_should_retire(self, learner):
        for _ in range(30):
            learner.record_outcome("loser", -10)
        assert learner.should_retire("loser") == True

    def test_should_promote(self, learner):
        for _ in range(20):
            learner.record_outcome("winner", 5)
        assert learner.should_promote("winner") == True

    def test_regime_mapping(self, learner):
        for _ in range(10):
            learner.record_outcome("strat", 5, "bull_low_vol")
        stats = learner.get_strategy_stats("strat")
        assert "bull_low_vol" in stats["regime_map"]


class TestRegimeAdaptiveAllocator:
    @pytest.fixture
    def allocator(self):
        return RegimeAdaptiveAllocator()

    def test_detect_regime(self, allocator):
        returns = np.random.normal(0.001, 0.02, 100)  # Bull
        regime, conf = allocator.detect_regime(returns)
        assert regime in RegimeAdaptiveAllocator.REGIMES
        assert 0 <= conf <= 1

    def test_detect_bear(self, allocator):
        rng = np.random.default_rng(42)
        returns = rng.normal(-0.005, 0.01, 100)  # Clearly negative
        regime, _ = allocator.detect_regime(returns)
        assert regime in ("bear_low_vol", "bear_high_vol", "crisis")

    def test_allocate(self, allocator):
        allocator.register_strategy_regime_score("trend", "bull_low_vol", 0.8)
        allocator.register_strategy_regime_score("mr", "sideways_low_vol", 0.7)
        result = allocator.allocate(["trend", "mr"], 100000, "bull_low_vol")
        assert isinstance(result, AllocationResult)
        assert result.strategy_weights["trend"] > result.strategy_weights["mr"]

    def test_allocate_crisis(self, allocator):
        result = allocator.allocate(["s1", "s2"], 100000, "crisis")
        assert result.total_allocated < 1.0  # Cash reserve

    def test_get_regime_description(self, allocator):
        desc = allocator.get_regime_description("bull_low_vol")
        assert len(desc) > 0

    def test_equal_weight_fallback(self, allocator):
        result = allocator.allocate(["s1", "s2", "s3"], 100000)
        assert len(result.strategy_weights) == 3
