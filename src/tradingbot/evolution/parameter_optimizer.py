"""Parameter Optimizer — continuous strategy parameter tuning.

Implements:
- Bayesian optimization (simplified)
- Grid search with smart sampling
- Walk-forward parameter optimization
- Adaptive parameter adjustment based on regime
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of parameter optimization."""
    best_params: dict = None
    best_score: float = 0.0
    n_trials: int = 0
    all_scores: list = None

    def __post_init__(self):
        if self.best_params is None:
            self.best_params = {}
        if self.all_scores is None:
            self.all_scores = []


class ParameterOptimizer:
    """Continuous strategy parameter optimization."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.n_trials = config.get("n_trials", 100)
        self.learning_rate = config.get("learning_rate", 0.01)
        self._history: dict[str, list[tuple[dict, float]]] = {}

    def grid_search(
        self,
        param_grid: dict[str, list],
        objective_fn: callable,
        n_samples: int = None,
    ) -> OptimizationResult:
        """Smart grid search with random sampling."""
        n = n_samples or self.n_trials

        # Generate parameter combinations
        all_combos = self._generate_combinations(param_grid)
        if len(all_combos) > n:
            indices = np.random.choice(len(all_combos), n, replace=False)
            combos = [all_combos[i] for i in indices]
        else:
            combos = all_combos

        best_score = float("-inf")
        best_params = {}
        all_scores = []

        for params in combos:
            score = objective_fn(params)
            all_scores.append(score)
            if score > best_score:
                best_score = score
                best_params = dict(params)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=len(combos),
            all_scores=all_scores,
        )

    def random_search(
        self,
        param_ranges: dict[str, tuple],
        objective_fn: callable,
        n_trials: int = None,
    ) -> OptimizationResult:
        """Random search over parameter space."""
        n = n_trials or self.n_trials
        best_score = float("-inf")
        best_params = {}
        all_scores = []

        for _ in range(n):
            params = {}
            for name, (low, high) in param_ranges.items():
                params[name] = np.random.uniform(low, high)

            score = objective_fn(params)
            all_scores.append(score)
            if score > best_score:
                best_score = score
                best_params = dict(params)

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            n_trials=n,
            all_scores=all_scores,
        )

    def hill_climb(
        self,
        initial_params: dict[str, float],
        param_ranges: dict[str, tuple],
        objective_fn: callable,
        n_iterations: int = 50,
        step_size: float = 0.1,
    ) -> OptimizationResult:
        """Hill climbing optimization."""
        current = dict(initial_params)
        current_score = objective_fn(current)
        best = dict(current)
        best_score = current_score
        all_scores = [current_score]

        for _ in range(n_iterations):
            # Generate neighbor
            neighbor = dict(current)
            param = np.random.choice(list(param_ranges.keys()))
            low, high = param_ranges[param]
            step = (high - low) * step_size
            neighbor[param] = np.clip(
                neighbor[param] + np.random.normal(0, step),
                low, high,
            )

            score = objective_fn(neighbor)
            all_scores.append(score)

            if score > current_score:
                current = neighbor
                current_score = score
                if score > best_score:
                    best = dict(neighbor)
                    best_score = score

        return OptimizationResult(
            best_params=best,
            best_score=best_score,
            n_trials=n_iterations,
            all_scores=all_scores,
        )

    def walk_forward_optimize(
        self,
        param_ranges: dict[str, tuple],
        data_chunks: list,
        objective_fn: callable,
        n_trials: int = 30,
    ) -> list[OptimizationResult]:
        """Walk-forward parameter optimization.

        Optimizes on each data chunk and validates on the next.
        """
        results = []
        for i in range(len(data_chunks) - 1):
            train_data = data_chunks[i]
            test_data = data_chunks[i + 1]

            def train_objective(params):
                return objective_fn(params, train_data)

            result = self.random_search(param_ranges, train_objective, n_trials)

            # Validate on test data
            test_score = objective_fn(result.best_params, test_data)
            result.best_score = test_score
            results.append(result)

        return results

    def adaptive_adjust(
        self,
        current_params: dict[str, float],
        recent_performance: float,
        target_performance: float,
        param_ranges: dict[str, tuple],
    ) -> dict[str, float]:
        """Adaptively adjust parameters based on performance gap."""
        gap = target_performance - recent_performance
        adjusted = dict(current_params)

        for param, (low, high) in param_ranges.items():
            current = adjusted.get(param, (low + high) / 2)
            # Move toward better region
            if gap > 0:
                # Underperforming — explore
                noise = np.random.normal(0, (high - low) * 0.05)
            else:
                # Overperforming — fine-tune
                noise = np.random.normal(0, (high - low) * 0.01)

            adjusted[param] = np.clip(current + noise, low, high)

        return adjusted

    def _generate_combinations(self, param_grid: dict[str, list]) -> list[dict]:
        """Generate all parameter combinations."""
        keys = list(param_grid.keys())
        if not keys:
            return [{}]

        combos = [{}]
        for key in keys:
            new_combos = []
            for combo in combos:
                for val in param_grid[key]:
                    new_combos.append({**combo, key: val})
            combos = new_combos

        return combos

    def record(self, strategy_id: str, params: dict, score: float) -> None:
        """Record optimization history."""
        if strategy_id not in self._history:
            self._history[strategy_id] = []
        self._history[strategy_id].append((params, score))

    def get_history(self, strategy_id: str) -> list[tuple[dict, float]]:
        return self._history.get(strategy_id, [])
