#!/usr/bin/env python3
"""Test suite for Omega Trading Intelligence."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from datetime import datetime, timedelta
import asyncio

passed = 0
failed = 0

def test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  PASS: {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL: {name} -> {e}")
        failed += 1

# === Test 1: Core imports ===
def test_core_imports():
    from tradingbot.core.enums import Side, OrderType, Timeframe, AssetClass, Regime, NodeType
    from tradingbot.core.types import OHLCVBar, Signal, StrategyGenome, Fill, Position, PortfolioState
    from tradingbot.core.events import EventBus, Event
    from tradingbot.core.interfaces import ExchangeAdapter, Strategy, ExecutionBackend
    from tradingbot.core.errors import OmegaError, ExchangeError, GenomeError

print("\n=== Core Imports ===")
test("core_imports", test_core_imports)

# === Test 2: Genome creation ===
def test_genome_creation():
    from tradingbot.genome.strategy_genome import create_random_genome
    g = create_random_genome()
    assert g.id, "Genome must have an id"
    assert isinstance(g.signal_tree, dict), "signal_tree must be dict"
    assert g.max_position_pct > 0, "position pct must be positive"
    assert g.stop_loss_param > 0, "stop loss must be positive"
    print(f"    Genome: id={g.id}, position_sizing={g.position_sizing}, stop={g.stop_loss_method}")

print("\n=== Genome Creation ===")
test("create_random_genome", test_genome_creation)

# === Test 3: Genome crossover ===
def test_crossover():
    from tradingbot.genome.strategy_genome import create_random_genome, crossover_genomes
    g1 = create_random_genome()
    g2 = create_random_genome()
    child = crossover_genomes(g1, g2)
    assert child.id != g1.id and child.id != g2.id, "Child must have unique id"
    assert child.generation == 1, f"Child generation should be 1, got {child.generation}"
    print(f"    Child: id={child.id}, gen={child.generation}")

print("\n=== Genome Crossover ===")
test("crossover", test_crossover)

# === Test 4: Genome mutation ===
def test_mutation():
    from tradingbot.genome.strategy_genome import create_random_genome, mutate_genome
    g = create_random_genome()
    mutated = mutate_genome(g, mutation_rate=0.5)
    assert mutated.id != g.id, "Mutated genome must have different id"
    print(f"    Original: {g.id}, Mutated: {mutated.id}")

print("\n=== Genome Mutation ===")
test("mutation", test_mutation)

# === Test 5: Rule tree ===
def test_rule_tree():
    from tradingbot.genome.rule_tree import random_tree, TreeNode
    tree_dict = random_tree(max_depth=3)
    node = TreeNode.from_dict(tree_dict)
    d = node.depth()
    s = node.size()
    assert d >= 0, "Depth must be non-negative"
    assert s >= 1, "Size must be at least 1"
    print(f"    Tree: depth={d}, size={s}")

print("\n=== Rule Tree ===")
test("rule_tree", test_rule_tree)

# === Test 6: Fitness evaluation ===
def test_fitness():
    from tradingbot.population.fitness import FitnessEvaluator
    evaluator = FitnessEvaluator({"fitness": {
        "sharpe_weight": 0.35, "sortino_weight": 0.25,
        "max_dd_weight": 0.20, "win_rate_weight": 0.10, "stability_weight": 0.10
    }})
    np.random.seed(42)
    equity = [100000.0]
    for _ in range(500):
        equity.append(equity[-1] * (1 + np.random.normal(0.0003, 0.015)))
    trades = np.random.normal(0.001, 0.02, 100).tolist()
    result = evaluator.evaluate(equity, trades)
    assert -1 <= result.composite_fitness <= 1, "Fitness must be bounded"
    print(f"    Fitness={result.composite_fitness:.4f}, Sharpe={result.sharpe_ratio:.4f}, DD={result.max_drawdown:.4f}")

print("\n=== Fitness Evaluation ===")
test("fitness", test_fitness)

# === Test 7: Backtesting engine ===
def test_backtest():
    from tradingbot.core.enums import Timeframe
    from tradingbot.core.types import OHLCVBar
    from tradingbot.genome.strategy_genome import create_random_genome
    from tradingbot.backtesting.engine import BacktestEngine

    np.random.seed(42)
    bars = []
    price = 50000.0
    for i in range(200):
        change = np.random.normal(0.0002, 0.02)
        price *= (1 + change)
        bars.append(OHLCVBar(
            timestamp=datetime(2024, 1, 1) + timedelta(hours=i),
            symbol="BTC/USDT", timeframe=Timeframe.H1,
            open=price / (1 + change), high=price * 1.005,
            low=price * 0.995, close=price,
            volume=np.random.uniform(100, 1000), exchange="binance",
        ))

    n = len(bars)
    features = {
        "rsi_14": np.clip(np.random.normal(50, 15, n), 0, 100).tolist(),
        "ema_21": np.random.normal(0, 1, n).tolist(),
        "atr_14": np.abs(np.random.normal(0.02, 0.005, n)).tolist(),
        "adx_14": np.clip(np.random.normal(25, 10, n), 0, 100).tolist(),
        "volatility": np.abs(np.random.normal(0.02, 0.01, n)).tolist(),
        "signal_strength": np.random.uniform(0, 1, n).tolist(),
        "signal_confidence": np.random.uniform(0.3, 0.9, n).tolist(),
    }

    genome = create_random_genome()
    engine = BacktestEngine({"backtest": {"initial_capital": 100000, "commission_bps": 10}})
    result = asyncio.run(engine.run(genome, bars, features))
    assert len(result.equity_curve) > 0, "Equity curve must not be empty"
    print(f"    Fitness={result.fitness.composite_fitness:.4f}, Trades={result.n_trades}, EquityCurve={len(result.equity_curve)}")

print("\n=== Backtesting Engine ===")
test("backtest", test_backtest)

# === Test 8: Config loading ===
def test_config():
    from tradingbot.config import load_config
    config = load_config("configs/default.yaml")
    assert config.mode in ("paper", "live", "backtest"), f"Invalid mode: {config.mode}"
    assert len(config.symbols) > 0, "Must have symbols"
    assert config.evolution.population_size > 0, "Population must be positive"
    print(f"    mode={config.mode}, symbols={config.symbols}, pop={config.evolution.population_size}")

print("\n=== Config Loading ===")
test("config", test_config)

# === Test 9: Risk manager ===
def test_risk():
    from tradingbot.core.events import EventBus
    from tradingbot.risk.risk_manager import RiskManager
    bus = EventBus()
    rm = RiskManager({"max_position_pct": 0.05, "max_drawdown_pct": 0.15}, bus)
    assert not rm.is_circuit_breaker_active, "CB should not be active initially"
    assert not rm.is_emergency_stop, "ES should not be active initially"
    print(f"    max_position_pct=0.05, max_dd=0.15, cb_active=False, es_active=False")

print("\n=== Risk Manager ===")
test("risk_manager", test_risk)

# === Test 10: Regime detection ===
def test_regime():
    from tradingbot.regime.hmm_detector import HMMRegimeDetector
    detector = HMMRegimeDetector({"regime": {"n_regimes": 4}})
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, 100)
    regime = detector.simple_detect(returns)
    assert regime is not None, "Regime must not be None"
    print(f"    Detected regime: {regime}")

print("\n=== Regime Detection ===")
test("regime_detection", test_regime)

# === Test 11: World model ===
def test_world_model():
    from tradingbot.world_model.market_simulator import MarketSimulator
    sim = MarketSimulator({"world_model": {"n_scenarios": 50, "horizon": 30}})
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.02, 300)
    scenarios = sim.simulate_scenarios(50000.0, returns, n_scenarios=50, horizon=30)
    assert len(scenarios) == 50, f"Expected 50 scenarios, got {len(scenarios)}"
    assert len(scenarios[0]) == 30, f"Expected 30 steps, got {len(scenarios[0])}"
    var = sim.monte_carlo_var(returns, confidence=0.95)
    assert "var" in var and "cvar" in var, "VaR result must have var and cvar"
    print(f"    Scenarios={len(scenarios)}, VaR={var['var']:.4f}, CVaR={var['cvar']:.4f}")

print("\n=== World Model ===")
test("world_model", test_world_model)

# === Test 12: Consciousness ===
def test_consciousness():
    from tradingbot.consciousness.metacognition import MetacognitionEngine
    engine = MetacognitionEngine({"consciousness": {}})
    state = engine.assess_state(
        prediction_confidence=0.7, regime_uncertainty=0.3,
        strategy_health=0.8, knowledge_gaps=0.2,
    )
    assert state is not None, "State must not be None"
    reflection = engine.reflect_on_trade(
        symbol="BTC/USDT", side="buy", entry_price=50000,
        exit_price=51000, pnl=1000, rationale="momentum",
    )
    assert reflection is not None, "Reflection must not be None"
    print(f"    State assessed, reflection generated")

print("\n=== Consciousness ===")
test("consciousness", test_consciousness)

# === Test 13: Swarm ===
def test_swarm():
    from tradingbot.swarm.particle_swarm import ParticleSwarmOptimizer
    def sphere(x):
        return -sum(xi**2 for xi in x)
    pso = ParticleSwarmOptimizer({"swarm": {"n_particles": 10, "n_dimensions": 2, "bounds": [(-5, 5)] * 2}})
    result = pso.optimize(sphere, n_iterations=30)
    assert "best_fitness" in result, "Must have best_fitness"
    print(f"    PSO best_fitness={result['best_fitness']:.4f}")

print("\n=== Swarm Intelligence ===")
test("swarm", test_swarm)

# === Test 14: Full engine init ===
def test_engine_init():
    from tradingbot.config import load_config
    from tradingbot.main import OmegaEngine
    config = load_config("configs/default.yaml")
    engine = OmegaEngine(config)
    assert engine.gp_engine is not None, "GP engine must exist"
    assert engine.backtest_engine is not None, "Backtest engine must exist"
    assert engine.risk_manager is not None, "Risk manager must exist"
    assert engine.hmm_detector is not None, "HMM detector must exist"
    assert engine.causal_engine is not None, "Causal engine must exist"
    assert engine.market_simulator is not None, "Market simulator must exist"
    assert engine.consciousness is not None, "Consciousness must exist"
    assert engine.ant_colony is not None, "Ant colony must exist"
    assert engine.particle_swarm is not None, "PSO must exist"
    print(f"    All 9 subsystems initialized")

print("\n=== Full Engine Init ===")
test("engine_init", test_engine_init)

# === Test 15: All modules importable ===
def test_all_imports():
    import importlib
    import pkgutil
    import tradingbot
    errors = []
    for importer, modname, ispkg in pkgutil.walk_packages(tradingbot.__path__, prefix="tradingbot."):
        try:
            importlib.import_module(modname)
        except Exception as e:
            errors.append(f"{modname}: {e}")
    if errors:
        for e in errors:
            print(f"    IMPORT ERROR: {e}")
        raise ImportError(f"{len(errors)} modules failed to import")
    print(f"    All modules imported successfully")

print("\n=== All Module Imports ===")
test("all_imports", test_all_imports)

# === Summary ===
print(f"\n{'='*50}")
print(f"  RESULTS: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*50}")
sys.exit(0 if failed == 0 else 1)
