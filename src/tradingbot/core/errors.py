class OmegaError(Exception):
    """Base exception for all Omega Trading errors."""
    pass


class ExchangeError(OmegaError):
    """Exchange connectivity or execution error."""
    pass


class OrderError(OmegaError):
    """Order lifecycle error."""
    pass


class InsufficientFundsError(OrderError):
    """Not enough balance to execute order."""
    pass


class OrderRejectedError(OrderError):
    """Order rejected by exchange or risk check."""
    pass


class GenomeError(OmegaError):
    """Strategy genome error."""
    pass


class EvolutionError(OmegaError):
    """Evolution engine error."""
    pass


class DataError(OmegaError):
    """Data pipeline error."""
    pass


class RiskLimitExceededError(OmegaError):
    """Risk limit breached."""
    pass


class ConfigError(OmegaError):
    """Configuration error."""
    pass


class BacktestError(OmegaError):
    """Backtesting engine error."""
    pass


class MLError(OmegaError):
    """Machine learning pipeline error."""
    pass


class RegimeDetectionError(OmegaError):
    """Regime detection error."""
    pass


class WorldModelError(OmegaError):
    """World model error."""
    pass


class ConsciousnessError(OmegaError):
    """Consciousness layer error."""
    pass


class SwarmError(OmegaError):
    """Swarm intelligence error."""
    pass
