"""Ant Colony Optimization for Strategy Discovery.

Ants explore the strategy space, leaving pheromone trails
on successful strategy regions. The colony converges on
optimal strategy territories through stigmergy.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.types import StrategyGenome
from ..genome.strategy_genome import create_random_genome, mutate_genome

logger = logging.getLogger(__name__)


@dataclass
class Ant:
    """An ant that explores the strategy space."""
    id: int
    position: list[float]  # Position in strategy parameter space
    fitness: float = 0.0
    path: list[int] = field(default_factory=list)  # Trail of visited nodes


class AntColonyOptimizer:
    """Ant Colony Optimization for discovering promising strategy regions.

    The strategy space is discretized into a graph where:
    - Nodes represent strategy parameter configurations
    - Edges have pheromone levels indicating success
    - Ants traverse the graph building strategies
    - Pheromones evaporate over time (forget bad regions)
    - Successful ants deposit more pheromones
    """

    def __init__(self, config: dict):
        self.colony_size = config.get("ant_colony_size", 100)
        self.max_iterations = config.get("max_iterations", 1000)
        self.evaporation_rate = config.get("evaporation_rate", 0.1)
        self.alpha = config.get("alpha", 1.0)  # Pheromone importance
        self.beta = config.get("beta", 2.0)  # Heuristic importance
        self.q0 = config.get("q0", 0.3)  # Exploitation vs exploration

        self._pheromone_trails: dict[str, float] = {}
        self._best_solutions: list[tuple[list[float], float]] = []
        self._dimension_sizes = [10, 10, 10, 10, 10]  # Discretization per dimension

    async def optimize(
        self,
        fitness_fn,
        n_dimensions: int = 5,
        bounds: list[tuple[float, float]] = None,
    ) -> list[tuple[list[float], float]]:
        """Run ACO optimization.

        Args:
            fitness_fn: Async function that evaluates a parameter vector
            n_dimensions: Number of parameter dimensions
            bounds: Min/max for each dimension

        Returns:
            List of (parameters, fitness) tuples, sorted by fitness
        """
        if bounds is None:
            bounds = [(0, 1)] * n_dimensions

        # Initialize pheromone trails
        self._init_pheromones(n_dimensions)

        best_solutions = []

        for iteration in range(self.max_iterations):
            # Each ant constructs a solution
            solutions = []
            for ant_id in range(self.colony_size):
                params = self._construct_solution(n_dimensions, bounds)
                fitness = await fitness_fn(params)
                solutions.append((params, fitness))

            # Sort by fitness
            solutions.sort(key=lambda x: x[1], reverse=True)

            # Update best solutions
            if solutions[0][1] > (best_solutions[0][1] if best_solutions else 0):
                best_solutions = solutions[:10]

            # Update pheromones
            self._evaporate_pheromones()
            self._deposit_pheromones(solutions, bounds)

            if iteration % 100 == 0:
                logger.info(f"ACO iteration {iteration}: best fitness = {solutions[0][1]:.4f}")

        return best_solutions

    def _init_pheromones(self, n_dimensions: int) -> None:
        """Initialize pheromone trails."""
        self._pheromone_trails = {}
        for dim in range(n_dimensions):
            for level in range(self._dimension_sizes[dim]):
                key = f"{dim}_{level}"
                self._pheromone_trails[key] = 0.5  # Initial pheromone

    def _construct_solution(self, n_dimensions: int, bounds: list[tuple[float, float]]) -> list[float]:
        """An ant constructs a solution by traversing the graph."""
        params = []
        for dim in range(n_dimensions):
            # Calculate probabilities for each level
            probs = []
            for level in range(self._dimension_sizes[dim]):
                key = f"{dim}_{level}"
                pheromone = self._pheromone_trails.get(key, 0.5)
                # Heuristic: prefer middle values
                heuristic = 1.0 - abs(level / self._dimension_sizes[dim] - 0.5)
                prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probs.append(prob)

            # Normalize probabilities
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / len(probs)] * len(probs)

            # Select level
            if random.random() < self.q0:
                # Exploitation: choose best
                level = probs.index(max(probs))
            else:
                # Exploration: sample from distribution
                level = random.choices(range(self._dimension_sizes[dim]), weights=probs)[0]

            # Convert level to parameter value
            min_val, max_val = bounds[dim]
            param = min_val + (level / (self._dimension_sizes[dim] - 1)) * (max_val - min_val)
            params.append(param)

        return params

    def _evaporate_pheromones(self) -> None:
        """Evaporate pheromones (forget bad regions)."""
        for key in self._pheromone_trails:
            self._pheromone_trails[key] *= (1 - self.evaporation_rate)

    def _deposit_pheromones(self, solutions: list[tuple[list[float], float]], bounds: list[tuple[float, float]]) -> None:
        """Deposit pheromones on successful paths."""
        for params, fitness in solutions[:10]:  # Top 10 deposit
            deposit = fitness  # Better fitness = more pheromone
            for dim, param in enumerate(params):
                min_val, max_val = bounds[dim]
                level = int((param - min_val) / (max_val - min_val) * (self._dimension_sizes[dim] - 1))
                level = max(0, min(self._dimension_sizes[dim] - 1, level))
                key = f"{dim}_{level}"
                self._pheromone_trails[key] += deposit
