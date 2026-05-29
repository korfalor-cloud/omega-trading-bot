"""Evolution engine — self-improving strategy system."""
from .self_evolver import SelfEvolver, EvolvedStrategy, EvolutionState, STRATEGY_GENOME_SCHEMA
from .strategy_factory import StrategyFactory, StrategyInstance
from .parameter_optimizer import ParameterOptimizer, OptimizationResult
from .meta_learner import MetaLearner, StrategyScore
from .regime_allocator import RegimeAdaptiveAllocator, AllocationResult

__all__ = [
    "SelfEvolver", "EvolvedStrategy", "EvolutionState", "STRATEGY_GENOME_SCHEMA",
    "StrategyFactory", "StrategyInstance",
    "ParameterOptimizer", "OptimizationResult",
    "MetaLearner", "StrategyScore",
    "RegimeAdaptiveAllocator", "AllocationResult",
]
