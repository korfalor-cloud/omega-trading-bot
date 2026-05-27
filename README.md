# Omega Trading Intelligence

An autonomous, self-evolving crypto trading system that uses genetic programming, machine learning, and swarm intelligence to discover and optimize trading strategies.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Omega Engine (main.py)                    │
│  Master orchestrator — wires all subsystems, runs evolution  │
├─────────┬──────────┬──────────┬──────────┬─────────────────┤
│  Data   │ Features │ Strategy │  Risk    │  Portfolio      │
│ Pipeline│ Engine   │ Genome   │ Manager  │  Manager        │
│ (ccxt)  │ (numpy)  │ (GP/ML)  │ (limits) │  (positions)    │
├─────────┴──────────┴──────────┴──────────┴─────────────────┤
│              Backtesting Engine (event-driven)               │
├─────────────────────────────────────────────────────────────┤
│  Evolution: GP Engine │ LLM Architect │ Swarm Optimizers    │
├─────────────────────────────────────────────────────────────┤
│  Intelligence: Regime Detection │ World Model │ Consciousness│
├─────────────────────────────────────────────────────────────┤
│  Execution: Order Manager │ Exchange Adapters │ Monitoring   │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Genetic Programming**: Evolves trading strategies as rule trees (AST), with crossover, mutation, and selection
- **Technical Indicators**: Pure numpy implementation — RSI, EMA, MACD, Bollinger Bands, ATR, ADX, Stochastic, CCI, Williams %R, MFI, OBV, VWAP, and more
- **Feature Engineering**: Automated feature computation from OHLCV data with configurable indicator sets
- **Backtesting Engine**: Event-driven with realistic slippage, commission, and stop-loss/take-profit simulation
- **Multi-Objective Fitness**: Evaluates strategies on Sharpe, Sortino, max drawdown, win rate, and stability
- **Risk Management**: Position limits, drawdown circuit breakers, daily loss limits, leverage caps, emergency stop
- **Portfolio Management**: Real-time position tracking, P&L calculation, exposure monitoring
- **Market Data Pipeline**: ccxt-based fetching with local file caching
- **Multiple Strategies**: Trend following, mean reversion, and ML-based (gradient boosting)
- **Regime Detection**: HMM-based market regime classification
- **Swarm Intelligence**: Ant colony and particle swarm optimization for parameter tuning
- **Monitoring**: Telegram notifications and Prometheus metrics

## Installation

```bash
# Clone
git clone https://github.com/your-org/omega-trading-bot.git
cd omega-trading-bot

# Install
pip install -e .

# Install with optional dependencies
pip install -e ".[ml]"       # XGBoost, LightGBM, scikit-learn
pip install -e ".[torch]"    # PyTorch for deep learning strategies
```

## Quick Start

```bash
# Paper trading (default)
omega

# Backtest mode
omega --mode backtest

# With custom config
omega --config configs/my_config.yaml --mode paper

# Evolution only (no trading)
omega --evolve

# Debug logging
omega --log-level DEBUG
```

## Configuration

Configuration is loaded from YAML files with environment variable overrides:

```yaml
# configs/default.yaml
mode: paper
symbols:
  - BTC/USDT
  - ETH/USDT

risk:
  max_position_pct: 0.05
  max_drawdown_pct: 0.15
  daily_loss_limit_pct: 0.05

evolution:
  population_size: 1000
  generations: 1000
  mutation_rate: 0.25
  crossover_rate: 0.70

backtest:
  initial_capital: 100000
  slippage_bps: 5
  commission_bps: 10
```

Environment variables use `OMEGA_` prefix with `__` for nesting:
```bash
OMEGA_MODE=live OMEGA_RISK__MAX_LEVERAGE=2.0 omega
```

## Project Structure

```
src/tradingbot/
├── main.py                 # CLI entrypoint and OmegaEngine orchestrator
├── config.py               # Pydantic config models
├── core/
│   ├── types.py            # OHLCVBar, Signal, Order, Position, etc.
│   ├── enums.py            # Side, Timeframe, NodeType, etc.
│   ├── events.py           # Async pub/sub EventBus
│   └── interfaces.py       # Abstract base classes
├── genome/
│   ├── rule_tree.py        # AST node types and random tree generation
│   ├── strategy_genome.py  # Genome creation, crossover, mutation
│   └── genome_encoder.py   # Evaluates rule trees against market data
├── features/
│   └── technical.py        # Technical indicators (RSI, EMA, MACD, etc.)
├── strategies/
│   ├── trend/              # EMA crossover + ADX trend following
│   ├── mean_reversion/     # Bollinger Band mean reversion
│   └── ml/                 # Gradient boosting ML strategy
├── backtesting/
│   └── engine.py           # Event-driven backtesting
├── population/
│   └── fitness.py          # Multi-objective fitness evaluation
├── portfolio/
│   └── portfolio_manager.py # Position tracking and P&L
├── risk/
│   └── risk_manager.py     # Pre-trade checks and circuit breakers
├── data/
│   └── market_data.py      # ccxt fetcher with caching
├── evolution/              # GP engine, LLM architect
├── regime/                 # HMM regime detection
├── world_model/            # Causal graph, market simulator
├── consciousness/          # Metacognition, goal setting
├── swarm/                  # Ant colony, particle swarm
├── execution/              # Order management
├── exchanges/              # Exchange adapters
└── monitoring/             # Telegram, Prometheus
```

## Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=tradingbot --cov-report=term-missing

# Type checking
mypy src/tradingbot/

# Linting
ruff check src/ tests/
```

## How Evolution Works

1. **Initialize**: Generate a random population of strategy genomes (rule trees)
2. **Evaluate**: Backtest each genome against historical data, compute fitness
3. **Select**: Tournament selection picks parents based on fitness
4. **Crossover**: Combine two parent genomes to create children (subtree swap)
5. **Mutate**: Randomly modify genomes (point, node, subtree, insertion, deletion)
6. **Repeat**: Continue for N generations, tracking best strategies
7. **Promote**: Best strategies move to paper trading, then live if validated

## License

MIT
