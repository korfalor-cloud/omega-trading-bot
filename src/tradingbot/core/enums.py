from enum import Enum, auto


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"


class OrderState(Enum):
    PENDING = auto()
    SUBMITTED = auto()
    PARTIAL = auto()
    FILLED = auto()
    CANCELLED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class Timeframe(Enum):
    TICK = "tick"
    S1 = "1s"
    S5 = "5s"
    S15 = "15s"
    S30 = "30s"
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H6 = "6h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    D3 = "3d"
    W1 = "1w"
    MN1 = "1M"

    @property
    def seconds(self) -> int:
        _map = {
            "tick": 0, "1s": 1, "5s": 5, "15s": 15, "30s": 30,
            "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
            "8h": 28800, "12h": 43200, "1d": 86400, "3d": 259200,
            "1w": 604800, "1M": 2592000,
        }
        return _map[self.value]


class AssetClass(Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    FUTURES = "futures"
    OPTIONS = "options"
    BOND = "bond"


class Regime(Enum):
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    MEAN_REVERTING = "mean_reverting"
    TRENDING = "trending"
    CRISIS = "crisis"


class ExecutionMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class SignalType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    SCALING = "scaling"
    HEDGE = "hedge"


class StrategyStatus(Enum):
    DORMANT = "dormant"          # Not yet evaluated
    EVALUATING = "evaluating"    # Being backtested
    PAPER = "paper"              # Paper trading
    LIVE = "live"                # Live trading
    RETIRED = "retired"          # Performance degraded, removed
    FAILED = "failed"            # Failed validation


class NodeType(Enum):
    """Node types for strategy rule trees."""
    # Logical operators
    AND = "and"
    OR = "or"
    NOT = "not"
    # Comparison operators
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"
    EQ = "eq"
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"
    # Indicators
    RSI = "rsi"
    EMA = "ema"
    SMA = "sma"
    MACD = "macd"
    MACD_SIGNAL = "macd_signal"
    MACD_HIST = "macd_hist"
    BB_UPPER = "bb_upper"
    BB_LOWER = "bb_lower"
    BB_MIDDLE = "bb_middle"
    ATR = "atr"
    ADX = "adx"
    STOCH_K = "stoch_k"
    STOCH_D = "stoch_d"
    CCI = "cci"
    WILLIAMS_R = "williams_r"
    MFI = "mfi"
    OBV = "obv"
    VWAP = "vwap"
    MOMENTUM = "momentum"
    ROC = "roc"
    # Order book features
    BOOK_IMBALANCE = "book_imbalance"
    SPREAD = "spread"
    BID_DEPTH = "bid_depth"
    ASK_DEPTH = "ask_depth"
    # Microstructure
    VPIN = "vpin"
    KYLE_LAMBDA = "kyle_lambda"
    TRADE_FLOW = "trade_flow"
    # Cross-asset
    CORRELATION = "correlation"
    BETA = "beta"
    # Alternative data
    SENTIMENT = "sentiment"
    ON_CHAIN = "on_chain"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    # Regime
    REGIME_STATE = "regime_state"
    VOLATILITY = "volatility"
    # Price
    CLOSE = "close"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    VOLUME = "volume"
    # Constants
    CONSTANT = "constant"
    # Time
    HOUR = "hour"
    DAY_OF_WEEK = "day_of_week"
