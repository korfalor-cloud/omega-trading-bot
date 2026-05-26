"""Particle Swarm Optimization for Parameter Tuning.

Particles explore the parameter space, attracted to their
own best position and the swarm's global best.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """A particle in the swarm."""
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float = -float("inf")
    current_fitness: float = -float("inf")


class ParticleSwarmOptimizer:
    """PSO for optimizing strategy parameters.

    Each particle represents a parameter configuration.
    Particles are attracted to:
    1. Their personal best position (cognitive)
    2. The swarm's global best position (social)
    3. Random exploration (inertia)
    """

    def __init__(self, config: dict):
        self.swarm_size = config.get("particle_swarm_size", 500)
        self.max_iterations = config.get("max_iterations", 1000)
        self.inertia_weight = config.get("inertia_weight", 0.7)
        self.cognitive_coeff = config.get("cognitive_coeff", 1.5)
        self.social_coeff = config.get("social_coeff", 1.5)
        self.inertia_decay = config.get("inertia_decay", 0.99)

        self._particles: list[Particle] = []
        self._global_best_position: Optional[np.ndarray] = None
        self._global_best_fitness: float = -float("inf")

    async def optimize(
        self,
        fitness_fn: Callable,
        n_dimensions: int,
        bounds: list[tuple[float, float]],
    ) -> tuple[np.ndarray, float]:
        """Run PSO optimization.

        Returns:
            (best_parameters, best_fitness)
        """
        # Initialize particles
        self._initialize(n_dimensions, bounds)

        for iteration in range(self.max_iterations):
            # Evaluate all particles
            for particle in self._particles:
                fitness = await fitness_fn(particle.position.tolist())
                particle.current_fitness = fitness

                # Update personal best
                if fitness > particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position.copy()

                # Update global best
                if fitness > self._global_best_fitness:
                    self._global_best_fitness = fitness
                    self._global_best_position = particle.position.copy()

            # Update velocities and positions
            for particle in self._particles:
                self._update_particle(particle, bounds)

            # Decay inertia
            self.inertia_weight *= self.inertia_decay

            if iteration % 100 == 0:
                logger.info(f"PSO iteration {iteration}: global best = {self._global_best_fitness:.4f}")

        return self._global_best_position.tolist(), self._global_best_fitness

    def _initialize(self, n_dimensions: int, bounds: list[tuple[float, float]]) -> None:
        """Initialize the swarm."""
        self._particles = []
        for _ in range(self.swarm_size):
            position = np.array([
                random.uniform(bounds[d][0], bounds[d][1])
                for d in range(n_dimensions)
            ])
            velocity = np.array([
                random.uniform(-(bounds[d][1] - bounds[d][0]) * 0.1, (bounds[d][1] - bounds[d][0]) * 0.1)
                for d in range(n_dimensions)
            ])
            self._particles.append(Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
            ))

    def _update_particle(self, particle: Particle, bounds: list[tuple[float, float]]) -> None:
        """Update particle velocity and position."""
        r1 = np.random.random(len(particle.position))
        r2 = np.random.random(len(particle.position))

        # Velocity update
        cognitive = self.cognitive_coeff * r1 * (particle.best_position - particle.position)
        social = self.social_coeff * r2 * (self._global_best_position - particle.position)

        particle.velocity = self.inertia_weight * particle.velocity + cognitive + social

        # Position update
        particle.position += particle.velocity

        # Clamp to bounds
        for d in range(len(bounds)):
            particle.position[d] = max(bounds[d][0], min(bounds[d][1], particle.position[d]))
