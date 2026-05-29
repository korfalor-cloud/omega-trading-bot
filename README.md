# 🧬 Omega Trading Bot

**Hedge-fund grade autonomous crypto trading system with self-improving AI**

[![Tests](https://img.shields.io/badge/tests-1110%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 📊 System Overview

Omega is a fully autonomous trading system that **evolves its own strategies** using genetic programming, manages risk across 26 modules, connects to major exchanges, and monitors itself with production-grade observability.

| Component | Count | Description |
|---|---|---|
| **Source Modules** | 211 | Python modules across 15 packages |
| **Test Files** | 90 | Comprehensive test coverage |
| **Tests** | 1,110 | All passing |
| **Strategies** | 47 | From trend following to options Greeks |
| **Risk Modules** | 26 | VaR to stress testing |
| **Data Sources** | 23 | On-chain, sentiment, news, whale alerts |
| **Infrastructure** | 18 | Auth, metrics, tracing, health checks |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Async Trading Engine                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │  Data Feed   │  │  Strategies │  │ Order Pipeline│  │ Monitoring │ │
│  │  Manager     │  │  (47 types) │  │ (risk→route) │  │ Dashboard  │ │
│  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘ │
│         │                 │                │                 │        │
│  ┌──────▼─────────────────▼────────────────▼─────────────────▼──────┐ │
│  │                      Event Bus (pub/sub)                         │ │
│  └──────┬─────────────────┬────────────────┬─────────────────┬──────┘ │
│         │                 │                │                 │        │
│  ┌──────▼───────┐  ┌──────▼──────┐  ┌──────▼───────┐  ┌─────▼─────┐  │
│  │  Exchanges   │  │    Risk     │  │  Evolution   │  │    ML     │  │
│  │ Binance/Bybit│  │  26 modules │  │  Self-Evolve │  │ Transformer│ │
│  └──────────────┘  └─────────────┘  └──────────────┘  └───────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │  Infrastructure: Auth • Security • Metrics • Tracing • Health   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/korfalor-cloud/omega-trading-bot.git
cd omega-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
# Edit config
nano config.yaml
```

**Key configuration:**

```yaml
exchange:
  name: binance
  api_key: "your-api-key"
  api_secret: "your-api-secret"
  testnet: true  # Use testnet first!

trading:
  mode: paper     # paper or live
  initial_capital: 100000

risk:
  max_drawdown: 0.15
  max_daily_loss: 0.03
  max_leverage: 3.0
```

### 3. Run

```bash
# Paper trading (default)
python -m tradingbot.main

# With environment overrides
OMEGA_TRADING_MODE=paper python -m tradingbot.main

# Live trading (be careful!)
OMEGA_TRADING_MODE=live python -m tradingbot.main
```

### 4. Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f omega-bot

# Stop
docker-compose down
```

---

## 📈 Trading Strategies (47)

### Trend Following
| Strategy | Description |
|---|---|
| `TrendFollowingStrategy` | Dual MA crossover with ADX filter |
| `SuperTrendStrategy` | SuperTrend indicator with MACD confirmation |
| `IchimokuStrategy` | Ichimoku cloud with Tenkan/Kijun cross |
| `AdaptiveMAStrategy` | Kaufman AMA with efficiency ratio |

### Mean Reversion
| Strategy | Description |
|---|---|
| `RSIMeanReversionStrategy` | RSI overbought/oversold with volume |
| `BollingerMeanReversion` | Bollinger Band bounce with RSI confirmation |
| `CandlestickStrategy` | Engulfing, hammer, shooting star patterns |
| `KalmanMeanReversion` | Kalman filter hedge ratio mean reversion |
| `SeasonalityStrategy` | Hour/day-of-week seasonal patterns |

### Momentum
| Strategy | Description |
|---|---|
| `ROCMomentumStrategy` | Rate of change with volume confirmation |
| `CrossAssetMomentumStrategy` | Multi-asset relative strength |
| `SectorRotationStrategy` | Rotate across crypto sectors |

### Scalping
| Strategy | Description |
|---|---|
| `ScalpingStrategy` | EMA cross + RSI + Bollinger edge |
| `EMACrossScalpStrategy` | Fast/slow EMA crossover |
| `OrderFlowScalpStrategy` | Order flow imbalance detection |
| `MicrostructureScalpStrategy` | Trade intensity signals |

### Arbitrage
| Strategy | Description |
|---|---|
| `TriangularArbitrageStrategy` | 3-pair triangular arbitrage |
| `CrossExchangeArbitrage` | Price discrepancy across exchanges |
| `FundingRateArbitrage` | Funding rate differential |
| `BasisTradeStrategy` | Spot-futures basis |

### Options & Derivatives
| Strategy | Description |
|---|---|
| `IronCondorStrategy` | Iron condor with IV filter |
| `StraddleStrategy` | Long/short straddle |
| `CoveredCallStrategy` | Covered call writing |
| `ButterflySpreadStrategy` | ATM butterfly with Greeks |
| `IronButterflyStrategy` | Short straddle + wings |
| `RatioSpreadStrategy` | 1x2 call ratio |
| `BoxSpreadStrategy` | Risk-free rate capture |
| `ConversionReversalStrategy` | Put-call parity arbitrage |

### Volatility
| Strategy | Description |
|---|---|
| `VolArbStrategy` | Implied vs realized vol spread |
| `SmileTradingStrategy` | Volatility smile mispricing |
| `TermStructureStrategy` | Vol term structure |
| `VegaNeutralStrategy` | Delta/vega hedged gamma capture |
| `GammaScalpStrategy` | Delta-hedged gamma trading |

### Other
| Strategy | Description |
|---|---|
| `GridTradingStrategy` | Grid for range-bound markets |
| `DCAStrategy` | Smart DCA with RSI filter |
| `PairsTradingStrategy` | Spread z-score mean reversion |
| `StatArbStrategy` | Z-score mean reversion with Kalman |
| `MarketMakingStrategy` | Spread-based quoting |
| `LiquidityProvisionStrategy` | Inventory-skewed quoting |
| `NewsDrivenStrategy` | Sentiment-based trading |
| `CarryTradeStrategy` | Interest rate differential |
| `CalendarSpreadStrategy` | Futures calendar spread |
| `VolumeBreakoutStrategy` | Price breakout with volume |

---

## 🛡️ Risk Management (26 modules)

### Core Risk
| Module | Description |
|---|---|
| `VaRModel` | Historical, parametric, Cornish-Fisher, Monte Carlo VaR/CVaR |
| `BlackScholesCalculator` | Options pricing, Greeks (delta/gamma/theta/vega/rho) |
| `TailRiskAnalyzer` | EVT, GPD fitting, max drawdown distribution |
| `VolatilityModel` | GARCH, EWMA, Parkinson, Garman-Klass, Yang-Zhang |
| `FactorModel` | CAPM, Fama-French 3-factor, PCA decomposition |
| `RiskBudgeter` | Risk parity, risk budgeting, max diversification |
| `PortfolioOptimizer` | Mean-variance, min variance, HRP, Black-Litterman |

### Position & Loss Limits
| Module | Description |
|---|---|
| `RiskLimitsEngine` | Position size, daily loss, drawdown, leverage limits |
| `DrawdownMonitor` | Real-time drawdown tracking, circuit breaker |
| `PositionSizer` | Kelly criterion, vol-targeting, ATR-based sizing |
| `MarginCalculator` | Margin requirements, liquidation price |
| `StressTester` | 5 default scenarios (crash, flash crash, etc.) |

### Advanced Risk
| Module | Description |
|---|---|
| `CorrelationAnalyzer` | Rolling correlation, breakdown detection |
| `LiquidityRiskAnalyzer` | Market impact, liquidity scoring |
| `BetaManager` | Portfolio beta, hedging |
| `CounterpartyRiskManager` | Exchange risk scoring |
| `FundingRiskManager` | Margin call prediction |
| `SectorExposureManager` | Sector concentration limits |
| `OmegaRatioCalculator` | Omega ratio optimization |

---

## 📊 Data Sources (23 modules)

### On-Chain Analytics
| Module | Description |
|---|---|
| `OnChainAnalyzer` | NVT ratio, MVRV ratio, SOPR, Puell Multiple, Stock-to-Flow |
| `WhaleAlertAnalyzer` | Large transaction detection, accumulation signals |
| `ExchangeFlowAnalyzer` | Exchange inflow/outflow, net flow signals |

### Market Sentiment
| Module | Description |
|---|---|
| `SentimentIndex` | Fear & Greed Index, volatility index |
| `SocialSentimentAnalyzer` | Twitter/Reddit sentiment, trending detection |
| `NewsAPIAnalyzer` | News sentiment scoring, event detection |

### Market Data
| Module | Description |
|---|---|
| `OrderBookAnalyzer` | Mid price, spread, depth, imbalance |
| `VolumeProfileAnalyzer` | POC, value area high/low |
| `FuturesCurveAnalyzer` | Term structure, contango/backwardation |
| `OpenInterestAnalyzer` | OI tracking, divergence detection |
| `LiquidationFeed` | Liquidation monitoring, cascade risk |
| `DataQualityChecker` | OHLCV validation, gap/spike detection |

---

## 🧬 Self-Evolution System

```
┌─────────────────────────────────────────────────────┐
│                  SELF-IMPROVEMENT LOOP               │
│                                                      │
│  1. Generate random strategy genomes                 │
│  2. Backtest each genome                             │
│  3. Score with multi-objective fitness               │
│  4. Select fittest (tournament selection)            │
│  5. Crossover + mutate → new generation              │
│  6. Auto-generate Python code for best strategies    │
│  7. Deploy to paper trading                          │
│  8. Meta-learner tracks performance per regime       │
│  9. Regime allocator shifts capital                  │
│ 10. Retire losers, promote winners                   │
│ 11. Optimize parameters continuously                 │
│ 12. Repeat forever → strategies improve over time    │
└─────────────────────────────────────────────────────┘
```

| Component | Description |
|---|---|
| `SelfEvolver` | Genetic programming — genome mutation, crossover, code generation |
| `StrategyFactory` | Dynamic strategy instantiation and lifecycle |
| `ParameterOptimizer` | Grid search, random search, hill climbing, walk-forward |
| `MetaLearner` | Strategy-regime mapping, auto-retirement/promotion |
| `RegimeAdaptiveAllocator` | Regime detection, dynamic capital allocation |

---

## 🔧 Production Infrastructure (18 modules)

### Authentication & Security
| Module | Description |
|---|---|
| `AuthManager` | API key management, JWT tokens, RBAC (admin/trader/viewer) |
| `SecurityManager` | Input validation, XSS/SQL injection prevention, CORS |
| `CredentialStore` | Encrypted credential storage, rotation, multi-environment |
| `RecoveryManager` | Crash detection, idempotent execution, state reconstruction |

### Observability
| Module | Description |
|---|---|
| `MetricsServer` | Prometheus-compatible counters, gauges, histograms |
| `Tracer` | Distributed tracing with span propagation |
| `HealthChecker` | Liveness/readiness probes, dependency checks |
| `CircuitBreaker` | Failure detection, automatic recovery |
| `BackpressureQueue` | Priority-based load shedding |
| `GracefulShutdown` | Signal handling, state persistence |

### Data & Config
| Module | Description |
|---|---|
| `Database` | SQLite persistence (trades, signals, equity, config) |
| `ConfigManager` | YAML hot-reload, environment variable overrides |
| `StructuredLogger` | JSON-formatted structured logging |
| `EventBus` | Decoupled pub/sub for system events |
| `RateLimiter` | Token bucket rate limiting |

---

## 🔌 Exchange Connectors (7)

| Connector | Description |
|---|---|
| `BinanceConnector` | Full Binance REST + WebSocket, HMAC signing |
| `BybitConnector` | Bybit V5 API, unified trading |
| `BaseConnector` | Abstract interface for all exchanges |
| `PaperTrading` | Realistic fill simulation with slippage/fees |
| `MultiExchangeRouter` | Best-price routing across venues |

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | System status |
| `/api/portfolio` | GET | Portfolio overview |
| `/api/strategies` | GET | List all strategies |
| `/api/strategies/{id}` | GET | Strategy details |
| `/api/strategies/{id}/start` | POST | Start strategy |
| `/api/strategies/{id}/stop` | POST | Stop strategy |
| `/api/trades` | GET | Trade history |
| `/api/signals` | GET | Signal history |
| `/api/risk` | GET | Risk metrics |
| `/api/evolution` | GET | Evolution status |
| `/api/equity` | GET | Equity curve |
| `/ws` | WebSocket | Real-time updates |
| `/metrics` | GET | Prometheus metrics |
| `/health` | GET | Health check |

---

## ⚙️ Configuration

```yaml
system:
  name: Omega Trading Bot
  version: "1.0.0"
  log_level: INFO
  db_path: data/omega.db

exchange:
  name: binance
  api_key: ""
  api_secret: ""
  testnet: true
  rate_limit: 1200

trading:
  mode: paper          # paper or live
  initial_capital: 100000
  max_positions: 10
  max_position_pct: 0.10
  default_stop_loss: 0.02
  default_take_profit: 0.04

risk:
  max_drawdown: 0.15
  max_daily_loss: 0.03
  max_leverage: 3.0
  max_concentration: 0.40
  circuit_breaker_enabled: true

evolution:
  enabled: true
  population_size: 50
  mutation_rate: 0.15
  crossover_rate: 0.7
  elite_pct: 0.1
  max_generations: 1000

strategies:
  active:
    - trend_following
    - mean_reversion
    - momentum
  max_active: 5
  min_sharpe: 0.5

monitoring:
  alerts_enabled: true
  telegram_enabled: false
  dashboard_port: 8080
```

### Environment Variables

```bash
OMEGA_SYSTEM_LOG_LEVEL=INFO
OMEGA_TRADING_MODE=paper
OMEGA_EXCHANGE_API_KEY=your-key
OMEGA_EXCHANGE_API_SECRET=your-secret
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/tradingbot --cov-report=html

# Run specific test file
pytest tests/test_self_evolution.py -v

# Run tests matching pattern
pytest tests/ -k "test_rsi" -v
```

**Coverage: 1,110 tests across 90 files**

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t omega-bot .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f omega-bot

# Stop
docker-compose down
```

---

## 📊 ML Models (12)

| Model | Description |
|---|---|
| `TransformerPredictor` | Multi-head attention for price prediction |
| `LSTMPredictor` | LSTM sequence prediction |
| `RLTradingAgent` | DQN reinforcement learning |
| `EnsembleModel` | Model averaging, stacking |
| `AutoencoderDetector` | Anomaly detection via reconstruction error |
| `KMeansClusterer` | Market regime clustering |
| `MetaLearner` | Strategy-context matching |
| `XGBModel` | Gradient boosting classification |
| `FeatureImportanceAnalyzer` | Permutation, correlation, mutual information |
| `AutoFeatureEngine` | Automated feature engineering |

---

## 🔄 CI/CD

### GitHub Actions

**Test Pipeline** (`.github/workflows/test.yml`):
- Python 3.11 setup with pip caching
- flake8 linting
- mypy type checking
- pytest with coverage

**Deploy Pipeline** (`.github/workflows/deploy.yml`):
- Docker build with layer caching
- Push to container registry
- Deploy to staging
- Smoke tests
- Deploy to production (manual approval)

---

## 🔐 Security Features

- **API Key Encryption**: Fernet encryption with master key
- **JWT Authentication**: HMAC-SHA256 signed tokens
- **RBAC**: Admin, Trader, Viewer roles
- **Input Validation**: SQL injection, XSS prevention
- **Request Signing**: HMAC-SHA256 request verification
- **CORS**: Configurable origin allowlisting
- **Secure Headers**: HSTS, CSP, X-Frame-Options
- **Audit Logging**: Full action trail

---

## 📁 Project Structure

```
omega-trading-bot/
├── src/tradingbot/
│   ├── engine/               # Async trading engine (3 modules)
│   ├── strategies/           # Trading strategies (47 modules)
│   │   ├── trend/            # Trend following
│   │   ├── mean_reversion/   # RSI, Bollinger, candlestick, Kalman
│   │   ├── momentum/         # ROC, cross-asset, sector rotation
│   │   ├── scalping/         # EMA cross, order flow, microstructure
│   │   ├── breakout/         # Volume breakout
│   │   ├── grid/             # Grid trading
│   │   ├── dca/              # Dollar cost averaging
│   │   ├── pairs/            # Pairs trading
│   │   ├── stat_arb/         # Statistical arbitrage
│   │   ├── market_making/    # Market making, liquidity provision
│   │   ├── arbitrage/        # Triangular, cross-exchange
│   │   ├── derivatives/      # Options, futures, carry trade
│   │   ├── volatility/       # Vol arb, smile, term structure
│   │   ├── options/          # Iron condor, straddle, covered call
│   │   ├── news/             # News-driven
│   │   ├── multi_strategy/   # Orchestrator
│   │   └── multi_timeframe/  # Multi-TF analysis
│   ├── risk/                 # Risk management (26 modules)
│   ├── data/                 # Data & analytics (23 modules)
│   ├── ml/                   # ML & AI (12 modules)
│   ├── evolution/            # Self-improvement (7 modules)
│   ├── exchanges/            # Exchange connectors (7 modules)
│   ├── execution/            # Execution (7 modules)
│   ├── monitoring/           # Monitoring (16 modules)
│   ├── infrastructure/       # Production infra (18 modules)
│   ├── backtesting/          # Backtesting (4 modules)
│   └── features/             # Feature engineering (4 modules)
├── tests/                    # 90 test files (1110 tests)
├── config.yaml               # Configuration
├── Dockerfile                # Container
├── docker-compose.yml        # Deployment
├── requirements.txt          # Dependencies
└── .github/workflows/        # CI/CD
```

---

## 📜 License

MIT License

---

## ⚠️ Disclaimer

This software is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk. Always test thoroughly with paper trading before using real funds.
