"""Genetic Programming Engine — Evolves strategy trees through natural selection."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Optional

from ..core.enums import StrategyStatus
from ..core.types import EvolutionState, StrategyGenome
from ..genome.strategy_genome import (
    crossover_genomes,
    create_random_genome,
    mutate_genome,
)

logger = logging.getLogger(__name__)


class GPEngine:
    """Genetic Programming engine for evolving trading strategies.

    Maintains a population of strategy genomes and evolves them through:
    - Tournament selection
    - Subtree crossover
    - Various mutation operators
    - Elitism (preserve top performers)
    - Immigration (inject random strategies for diversity)
    - Speciation (cluster similar strategies)
    """

    def __init__(self, config: dict):
        self.config = config
        self.population_size = config.get("population_size", 1000)
        self.num_islands = config.get("num_islands", 10)
        self.mutation_rate = config.get("mutation_rate", 0.25)
        self.crossover_rate = config.get("crossover_rate", 0.70)
        self.elitism_pct = config.get("elitism_pct", 0.10)
        self.immigration_pct = config.get("immigration_pct", 0.05)
        self.tournament_size = config.get("tournament_size", 5)
        self.max_tree_depth = config.get("max_tree_depth", 6)

        self._islands: list[list[StrategyGenome]] = [[] for _ in range(self.num_islands)]
        self._generation = 0
        self._best_ever: Optional[StrategyGenome] = None
        self._state = EvolutionState()

    async def initialize_population(self) -> list[StrategyGenome]:
        """Create initial random population distributed across islands."""
        per_island = self.population_size // self.num_islands
        all_genomes = []

        for i in range(self.num_islands):
            island = [create_random_genome(f"island{i}_gen0_{j}") for j in range(per_island)]
            self._islands[i] = island
            all_genomes.extend(island)

        logger.info(f"Initialized population: {len(all_genomes)} genomes across {self.num_islands} islands")
        self._update_state()
        return all_genomes

    async def evolve_generation(self, fitness_scores: dict[str, float]) -> list[StrategyGenome]:
        """Evolve one generation across all islands."""
        self._generation += 1
        all_new = []

        for island_idx in range(self.num_islands):
            island = self._islands[island_idx]
            if not island:
                continue

            # Update fitness scores
            for genome in island:
                if genome.id in fitness_scores:
                    genome.fitness = fitness_scores[genome.id]

            # Sort by fitness
            island.sort(key=lambda g: g.fitness, reverse=True)

            # Track best ever
            if island[0].fitness > (self._best_ever.fitness if self._best_ever else 0):
                self._best_ever = island[0]

            new_island = await self._evolve_island(island, island_idx)
            self._islands[island_idx] = new_island
            all_new.extend(new_island)

        # Migration: exchange top strategies between islands
        if self._generation % 10 == 0:
            await self._migrate()

        self._update_state()
        logger.info(
            f"Generation {self._generation}: best_fitness={self._state.best_fitness:.4f}, "
            f"avg_fitness={self._state.avg_fitness:.4f}, diversity={self._state.diversity_score:.3f}"
        )
        return all_new

    async def _evolve_island(self, island: list[StrategyGenome], island_idx: int) -> list[StrategyGenome]:
        """Evolve a single island."""
        n = len(island)
        if n == 0:
            return []

        new_island: list[StrategyGenome] = []

        # Elitism: keep top performers
        elite_count = max(1, int(n * self.elitism_pct))
        elites = island[:elite_count]
        new_island.extend([g for g in elites])

        # Immigration: inject random strategies
        immigrant_count = max(1, int(n * self.immigration_pct))
        for _ in range(immigrant_count):
            new_island.append(create_random_genome(f"island{island_idx}_immigrant_{self._generation}"))

        # Fill the rest with crossover and mutation
        while len(new_island) < n:
            r = random.random()
            if r < self.crossover_rate:
                parent_a = self._tournament_select(island)
                parent_b = self._tournament_select(island)
                child = crossover_genomes(parent_a, parent_b)
                # Possibly mutate the child
                if random.random() < self.mutation_rate:
                    child = mutate_genome(child)
                new_island.append(child)
            elif r < self.crossover_rate + self.mutation_rate:
                parent = self._tournament_select(island)
                child = mutate_genome(parent)
                new_island.append(child)
            else:
                # Reproduction (copy as-is)
                parent = self._tournament_select(island)
                new_island.append(parent)

        return new_island[:n]

    def _tournament_select(self, population: list[StrategyGenome]) -> StrategyGenome:
        """Select a genome via tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda g: g.fitness)

    async def _migrate(self) -> None:
        """Migrate top strategies between islands."""
        migrants_per_island = max(1, self.population_size // self.num_islands // 20)

        for i in range(self.num_islands):
            source = self._islands[i]
            if not source:
                continue
            source.sort(key=lambda g: g.fitness, reverse=True)
            migrants = source[:migrants_per_island]

            # Send to next island
            target_idx = (i + 1) % self.num_islands
            target = self._islands[target_idx]

            # Replace worst in target
            target.sort(key=lambda g: g.fitness)
            for j, migrant in enumerate(migrants):
                if j < len(target):
                    target[j] = migrant.clone()
                else:
                    target.append(migrant.clone())

            self._islands[target_idx] = target

    def _update_state(self) -> None:
        """Update the evolution state."""
        all_genomes = [g for island in self._islands for g in island]
        if not all_genomes:
            return

        fitnesses = [g.fitness for g in all_genomes]
        self._state = EvolutionState(
            generation=self._generation,
            population_size=len(all_genomes),
            best_fitness=max(fitnesses),
            avg_fitness=sum(fitnesses) / len(fitnesses),
            diversity_score=self._calculate_diversity(all_genomes),
            strategies_alive=len(all_genomes),
            strategies_retired=0,
            strategies_promoted=0,
        )

    def _calculate_diversity(self, population: list[StrategyGenome]) -> float:
        """Calculate population diversity (simplified)."""
        if len(population) < 2:
            return 0.0

        # Diversity based on fitness variance
        fitnesses = [g.fitness for g in population]
        mean = sum(fitnesses) / len(fitnesses)
        variance = sum((f - mean) ** 2 for f in fitnesses) / len(fitnesses)
        return min(1.0, variance ** 0.5)

    async def get_best_strategies(self, n: int = 10) -> list[StrategyGenome]:
        """Get the N best strategies across all islands."""
        all_genomes = [g for island in self._islands for g in island]
        all_genomes.sort(key=lambda g: g.fitness, reverse=True)
        return all_genomes[:n]

    @property
    def state(self) -> EvolutionState:
        return self._state

    @property
    def best_ever(self) -> Optional[StrategyGenome]:
        return self._best_ever
