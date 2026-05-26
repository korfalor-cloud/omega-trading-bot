"""Omega Trading Intelligence — Main Entrypoint.

The master orchestrator that wires all subsystems together:
- Evolution engines (GP, NEAT, LLM, Swarm)
- Regime detection (HMM, BOCPD)
- World model (causal graph, market simulator)
- Consciousness (metacognition, goal setting, reflection)
- Execution (order management, risk, monitoring)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .backtesting.engine import BacktestEngine
from .config import OmegaConfig, load_config
from .consciousness.metacognition import MetacognitionEngine
from .core.events import Event, EventBus
from .core.types import EvolutionState, PortfolioState, RegimeState, StrategyGenome
from .evolution.gp_engine import GPEngine
from .evolution.llm_architect import LLMStrategist
from .execution.order_manager import OrderManager
from .genome.strategy_genome import create_random_genome
from .monitoring.telegram import TelegramNotifier
from .population.fitness import FitnessEvaluator
from .regime.hmm_detector import HMMRegimeDetector
from .risk.risk_manager import RiskManager
from .swarm.ant_colony import AntColonyOptimizer
from .swarm.particle_swarm import ParticleSwarmOptimizer
from .world_model.causal_graph import CausalGraphEngine
from .world_model.market_simulator import MarketSimulator

logger = logging.getLogger("omega")


class OmegaEngine:
    """The master orchestrator — wires all subsystems and runs the evolution cycle.

    This is the "consciousness" of the trading system. It:
    1. Initializes all subsystems
    2. Collects market data
    3. Detects regime
    4. Evolves strategies (GP + NEAT + LLM + Swarm)
    5. Backtests candidates
    6. Promotes winners to paper/live
    7. Monitors performance
    8. Reflects and adapts
    """

    def __init__(self, config: OmegaConfig):
        self.config = config
        self.event_bus = EventBus()

        # Core subsystems
        self.gp_engine = GPEngine(config.evolution.model_dump())
        self.backtest_engine = BacktestEngine(config.backtest.model_dump())
        self.fitness_evaluator = FitnessEvaluator(config.evolution.model_dump())
        self.risk_manager = RiskManager(config.risk.model_dump(), self.event_bus)
        self.order_manager = OrderManager(self.event_bus)

        # Intelligence subsystems
        self.hmm_detector = HMMRegimeDetector(config.regime.model_dump())
        self.causal_engine = CausalGraphEngine(config.world_model.model_dump())
        self.market_simulator = MarketSimulator(config.world_model.model_dump())
        self.consciousness = MetacognitionEngine(config.consciousness.model_dump())
        self.llm_strategist = LLMStrategist({})

        # Swarm
        self.ant_colony = AntColonyOptimizer(config.swarm.model_dump())
        self.particle_swarm = ParticleSwarmOptimizer(config.swarm.model_dump())

        # Monitoring
        self.telegram = TelegramNotifier(config.monitoring.model_dump())

        # State
        self._running = False
        self._current_regime: Optional[RegimeState] = None
        self._active_strategies: list[StrategyGenome] = []
        self._population: list[StrategyGenome] = []
        self._evolution_state = EvolutionState()

    async def start(self) -> None:
        """Start all subsystems and run the main loop."""
        logger.info("=" * 60)
        logger.info("  OMEGA TRADING INTELLIGENCE")
        logger.info("  Self-Creating, Self-Improving Autonomous Trading Consciousness")
        logger.info("=" * 60)
        logger.info(f"Mode: {self.config.mode}")
        logger.info(f"Symbols: {self.config.symbols}")
        logger.info(f"Population: {self.config.evolution.population_size}")

        self._running = True

        # Start event bus
        bus_task = asyncio.create_task(self.event_bus.run())

        # Start Telegram
        await self.telegram.start()

        # Subscribe to events
        self.event_bus.subscribe(Event.ORDER_FILLED, self._on_order_filled)
        self.event_bus.subscribe(Event.RISK_ALERT, self._on_risk_alert)

        # Initialize population
        logger.info("Initializing strategy population...")
        self._population = await self.gp_engine.initialize_population()
        logger.info(f"Population initialized: {len(self._population)} genomes")

        # Start main loops
        tasks = [
            asyncio.create_task(self._evolution_loop()),
            asyncio.create_task(self._monitoring_loop()),
        ]

        logger.info("=== OMEGA READY ===")
        await self.telegram.send_message("🚀 *Omega Trading Intelligence Started*\nMode: " + self.config.mode)

        # Main loop
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            for task in tasks:
                task.cancel()
            await self.event_bus.stop()
            bus_task.cancel()
            await self.telegram.stop()

    async def stop(self) -> None:
        """Gracefully stop."""
        logger.info("=== OMEGA SHUTTING DOWN ===")
        self._running = False
        await self.event_bus.publish(Event.SHUTDOWN, None)

    async def _evolution_loop(self) -> None:
        """Main evolution cycle — runs continuously."""
        while self._running:
            try:
                logger.info("--- Starting evolution cycle ---")

                # Step 1: Generate random historical data for backtesting
                # In production, this would come from the data pipeline
                bars = self._generate_sample_data()
                features = self._generate_sample_features(len(bars))

                # Step 2: Evaluate current population
                fitness_scores = {}
                for genome in self._population[:100]:  # Evaluate top 100
                    result = await self.backtest_engine.run(genome, bars, features)
                    fitness_scores[genome.id] = result.fitness.composite_fitness

                # Step 3: Evolve next generation
                self._population = await self.gp_engine.evolve_generation(fitness_scores)

                # Step 4: Get best strategies
                best = await self.gp_engine.get_best_strategies(5)
                if best:
                    logger.info(f"Best strategy: fitness={best[0].fitness:.4f}, gen={best[0].generation}")
                    self._active_strategies = best

                    # Notify
                    await self.telegram.notify_evolution(
                        generation=self.gp_engine.state.generation,
                        best_fitness=best[0].fitness,
                        avg_fitness=self.gp_engine.state.avg_fitness,
                        population_size=len(self._population),
                    )

                # Step 5: Consciousness reflection
                self._evolution_state = self.gp_engine.state

                # Wait before next cycle (1 hour in production)
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Evolution loop error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _monitoring_loop(self) -> None:
        """Monitoring and heartbeat loop."""
        while self._running:
            try:
                # Log status
                logger.info(
                    f"Status: population={len(self._population)}, "
                    f"active_strategies={len(self._active_strategies)}, "
                    f"generation={self._evolution_state.generation}, "
                    f"best_fitness={self._evolution_state.best_fitness:.4f}"
                )

                await asyncio.sleep(self.config.monitoring.heartbeat_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(30)

    async def _on_order_filled(self, fill) -> None:
        """Handle order fill event."""
        await self.risk_manager.post_trade_update(fill)
        await self.telegram.notify_trade(
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            price=fill.price,
            strategy_id="",
        )

    async def _on_risk_alert(self, alert) -> None:
        """Handle risk alert event."""
        await self.telegram.notify_risk_alert(alert)

    def _generate_sample_data(self):
        """Generate sample OHLCV data for backtesting (placeholder)."""
        import numpy as np
        from .core.enums import Timeframe
        from .core.types import OHLCVBar

        bars = []
        price = 50000.0
        for i in range(1000):
            change = np.random.normal(0.0002, 0.02)
            price *= (1 + change)
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            bars.append(OHLCVBar(
                timestamp=datetime(2024, 1, 1) + __import__("datetime").timedelta(hours=i),
                symbol="BTC/USDT",
                timeframe=Timeframe.H1,
                open=price / (1 + change),
                high=high,
                low=low,
                close=price,
                volume=np.random.uniform(100, 1000),
                exchange="binance",
            ))
        return bars

    def _generate_sample_features(self, n: int) -> dict:
        """Generate sample features for backtesting (placeholder)."""
        import numpy as np
        return {
            "rsi_14": np.clip(np.random.normal(50, 15, n), 0, 100).tolist(),
            "ema_21": np.random.normal(0, 1, n).tolist(),
            "atr_14": np.abs(np.random.normal(0.02, 0.005, n)).tolist(),
            "adx_14": np.clip(np.random.normal(25, 10, n), 0, 100).tolist(),
            "volatility": np.abs(np.random.normal(0.02, 0.01, n)).tolist(),
            "signal_strength": np.random.uniform(0, 1, n).tolist(),
            "signal_confidence": np.random.uniform(0.3, 0.9, n).tolist(),
        }


def cli() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="Omega Trading Intelligence")
    parser.add_argument("--config", "-c", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--mode", "-m", choices=["paper", "live", "backtest"], default=None, help="Trading mode")
    parser.add_argument("--log-level", "-l", default="INFO", help="Log level")
    parser.add_argument("--evolve", action="store_true", help="Run evolution only (no live trading)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("omega.log"),
        ],
    )

    # Load config
    config = load_config(args.config)
    if args.mode:
        config.mode = args.mode
    config.log_level = args.log_level

    # Run
    engine = OmegaEngine(config)
    try:
        asyncio.run(engine.start())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")


if __name__ == "__main__":
    cli()
