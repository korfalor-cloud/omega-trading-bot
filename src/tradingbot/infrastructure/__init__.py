"""Infrastructure — database, config, logging, API, events, auth, security, credentials, recovery, observability."""
from .database import Database
from .config_manager import ConfigManager, DEFAULT_CONFIG
from .logger import StructuredLogger, setup_logging
from .api_server import APIServer
from .event_bus import EventBus, Event, EventType
from .rate_limiter import RateLimiter, TokenBucket
from .report_generator import ReportGenerator, ReportData
from .websocket_manager import WebSocketManager, WSConnection
from .auth import (
    AuthManager, JWTService, Encryptor, PasswordHasher,
    Role, User, APIKey, Session, AuthRateLimiter,
)
from .security import (
    SecurityManager, AuditLogger, AuditAction, AuditEntry,
    InputValidator, XSSSanitiser, SQLInjectionDetector, QueryBuilder,
    RequestSigner, CORSConfig, SecureHeaders,
)
from .credentials import (
    CredentialStore, CredentialMeta, Environment,
    load_exchange_credentials,
)
from .recovery import (
    RecoveryManager, CrashStore, CrashDetector, StateManager,
    IdempotentExecutor, OrderReconstructor, ShutdownManager,
    OperationRecord, OperationStatus, StateSnapshot, ShutdownPhase,
)
from .metrics import (
    MetricsRegistry, MetricsServer,
    Counter, Gauge, Histogram, Summary,
    REGISTRY as METRICS_REGISTRY,
    TRADES_TOTAL, ORDERS_TOTAL, ERRORS_TOTAL,
    OPEN_POSITIONS, EQUITY, DRAWDOWN_PCT,
    ORDER_LATENCY, SLIPPAGE_BPS, PNL_DISTRIBUTION,
)
from .tracing import (
    Tracer, Span, SpanContext, SpanEvent, SpanLink,
    OperationTimer, timed,
    DEFAULT_TRACER, get_tracer,
)
from .health import (
    HealthChecker, HealthServer, HealthStatus, ComponentHealth,
    exchange_check_fn, database_check_fn, strategy_check_fn,
)
from .circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry,
    CircuitState, CircuitBreakerError, CallResult,
    REGISTRY as CB_REGISTRY, get_circuit_breaker,
)
from .backpressure import (
    BackpressureQueue, AdaptiveRateLimiter, LoadShedder,
    Priority, QueueMessage, create_backpressure_system,
)
from .graceful_shutdown import (
    GracefulShutdown, OrderTracker, PositionReconciler,
    CleanupCallback, ShutdownState, ShutdownPhase,
)

__all__ = [
    "Database", "ConfigManager", "DEFAULT_CONFIG",
    "StructuredLogger", "setup_logging",
    "APIServer", "EventBus", "Event", "EventType",
    "RateLimiter", "TokenBucket",
    "ReportGenerator", "ReportData",
    "WebSocketManager", "WSConnection",
    "AuthManager", "JWTService", "Encryptor", "PasswordHasher",
    "Role", "User", "APIKey", "Session", "AuthRateLimiter",
    "SecurityManager", "AuditLogger", "AuditAction", "AuditEntry",
    "InputValidator", "XSSSanitiser", "SQLInjectionDetector", "QueryBuilder",
    "RequestSigner", "CORSConfig", "SecureHeaders",
    "CredentialStore", "CredentialMeta", "Environment",
    "load_exchange_credentials",
    "RecoveryManager", "CrashStore", "CrashDetector", "StateManager",
    "IdempotentExecutor", "OrderReconstructor", "ShutdownManager",
    "OperationRecord", "OperationStatus", "StateSnapshot", "ShutdownPhase",
    # Metrics
    "MetricsRegistry", "MetricsServer",
    "Counter", "Gauge", "Histogram", "Summary",
    "METRICS_REGISTRY",
    "TRADES_TOTAL", "ORDERS_TOTAL", "ERRORS_TOTAL",
    "OPEN_POSITIONS", "EQUITY", "DRAWDOWN_PCT",
    "ORDER_LATENCY", "SLIPPAGE_BPS", "PNL_DISTRIBUTION",
    # Tracing
    "Tracer", "Span", "SpanContext", "SpanEvent", "SpanLink",
    "OperationTimer", "timed",
    "DEFAULT_TRACER", "get_tracer",
    # Health
    "HealthChecker", "HealthServer", "HealthStatus", "ComponentHealth",
    "exchange_check_fn", "database_check_fn", "strategy_check_fn",
    # Circuit breaker
    "CircuitBreaker", "CircuitBreakerConfig", "CircuitBreakerRegistry",
    "CircuitState", "CircuitBreakerError", "CallResult",
    "CB_REGISTRY", "get_circuit_breaker",
    # Backpressure
    "BackpressureQueue", "AdaptiveRateLimiter", "LoadShedder",
    "Priority", "QueueMessage", "create_backpressure_system",
    # Graceful shutdown
    "GracefulShutdown", "OrderTracker", "PositionReconciler",
    "CleanupCallback", "ShutdownState",
]
