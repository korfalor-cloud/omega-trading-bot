"""Strategy Genome — The complete DNA of a trading strategy."""
from __future__ import annotations

import copy
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.enums import StrategyStatus, Timeframe
from ..core.types import StrategyGenome
from .rule_tree import TreeNode, random_tree


def create_random_genome(name: str = "") -> StrategyGenome:
    """Create a completely random strategy genome."""
    return StrategyGenome(
        id=str(uuid.uuid4()),
        name=name or f"strategy_{str(uuid.uuid4())[:8]}",
        signal_tree=random_tree(max_depth=4).to_dict(),
        stop_loss_method=random.choice(["atr", "fixed", "trailing", "percent"]),
        stop_loss_param=random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0]),
        take_profit_ratio=random.choice([1.0, 1.5, 2.0, 2.5, 3.0]),
        position_sizing=random.choice(["kelly", "fixed", "atr", "volatility"]),
        max_position_pct=random.uniform(0.01, 0.10),
        primary_timeframe=random.choice(["5m", "15m", "1h", "4h"]),
        confirmation_timeframe=random.choice(["1h", "4h", "1d"]),
        cooldown_bars=random.randint(1, 10),
        active_regimes=random.sample(
            ["bull_low_vol", "bull_high_vol", "bear_low_vol", "bear_high_vol",
             "mean_reverting", "trending", "crisis"],
            k=random.randint(2, 5),
        ),
        min_volatility=random.uniform(0.0, 0.3),
        max_volatility=random.uniform(0.5, 1.0),
        generation=0,
        status=StrategyStatus.DORMANT.value,
    )


def crossover_genomes(parent_a: StrategyGenome, parent_b: StrategyGenome) -> StrategyGenome:
    """Create a child genome by combining two parents."""
    tree_a = TreeNode.from_dict(parent_a.signal_tree)
    tree_b = TreeNode.from_dict(parent_b.signal_tree)

    # Subtree crossover
    nodes_a = tree_a.collect_nodes()
    nodes_b = tree_b.collect_nodes()

    if len(nodes_a) > 1 and len(nodes_b) > 1:
        # Pick random crossover points
        point_a = random.choice(nodes_a[1:])  # Skip root
        point_b = random.choice(nodes_b[1:])

        # Swap subtrees
        child_tree = tree_a.clone()
        child_nodes = child_tree.collect_nodes()
        for node in child_nodes:
            for i, child in enumerate(node.children):
                if child.id == point_a.id:
                    node.children[i] = point_b.clone()
                    break

        child_signal_tree = child_tree.to_dict()
    else:
        child_signal_tree = random.choice([parent_a.signal_tree, parent_b.signal_tree])

    # Mix risk parameters
    child = StrategyGenome(
        id=str(uuid.uuid4()),
        name=f"child_{str(uuid.uuid4())[:8]}",
        signal_tree=child_signal_tree,
        stop_loss_method=random.choice([parent_a.stop_loss_method, parent_b.stop_loss_method]),
        stop_loss_param=random.choice([parent_a.stop_loss_param, parent_b.stop_loss_param]),
        take_profit_ratio=random.choice([parent_a.take_profit_ratio, parent_b.take_profit_ratio]),
        position_sizing=random.choice([parent_a.position_sizing, parent_b.position_sizing]),
        max_position_pct=random.choice([parent_a.max_position_pct, parent_b.max_position_pct]),
        primary_timeframe=random.choice([parent_a.primary_timeframe, parent_b.primary_timeframe]),
        confirmation_timeframe=random.choice([parent_a.confirmation_timeframe, parent_b.confirmation_timeframe]),
        cooldown_bars=random.choice([parent_a.cooldown_bars, parent_b.cooldown_bars]),
        active_regimes=random.choice([parent_a.active_regimes, parent_b.active_regimes]),
        min_volatility=random.choice([parent_a.min_volatility, parent_b.min_volatility]),
        max_volatility=random.choice([parent_a.max_volatility, parent_b.max_volatility]),
        generation=max(parent_a.generation, parent_b.generation) + 1,
        parent_ids=[parent_a.id, parent_b.id],
        status=StrategyStatus.DORMANT.value,
    )
    return child


def mutate_genome(genome: StrategyGenome, mutation_strength: float = 1.0) -> StrategyGenome:
    """Mutate a genome with various mutation types."""
    mutated = copy.deepcopy(genome)
    mutated.id = str(uuid.uuid4())
    mutated.name = f"mutant_{str(uuid.uuid4())[:8]}"
    mutated.generation = genome.generation + 1
    mutated.parent_ids = [genome.id]

    tree = TreeNode.from_dict(mutated.signal_tree)

    mutation_type = random.random()

    if mutation_type < 0.3:
        # Point mutation: change a parameter
        _point_mutate(tree, mutation_strength)
    elif mutation_type < 0.5:
        # Node mutation: change operator type
        _node_mutate(tree)
    elif mutation_type < 0.7:
        # Subtree mutation: replace a subtree
        _subtree_mutate(tree, mutation_strength)
    elif mutation_type < 0.85:
        # Insertion: add a new condition
        _insertion_mutate(tree, mutation_strength)
    else:
        # Deletion: remove a condition
        _deletion_mutate(tree)

    mutated.signal_tree = tree.to_dict()

    # Mutate risk parameters
    if random.random() < 0.3:
        mutated.stop_loss_param += random.gauss(0, 0.3 * mutation_strength)
        mutated.stop_loss_param = max(0.5, min(5.0, mutated.stop_loss_param))

    if random.random() < 0.2:
        mutated.take_profit_ratio += random.gauss(0, 0.3 * mutation_strength)
        mutated.take_profit_ratio = max(0.5, min(5.0, mutated.take_profit_ratio))

    if random.random() < 0.2:
        mutated.max_position_pct += random.gauss(0, 0.01 * mutation_strength)
        mutated.max_position_pct = max(0.005, min(0.20, mutated.max_position_pct))

    if random.random() < 0.1:
        mutated.primary_timeframe = random.choice(["5m", "15m", "1h", "4h"])

    return mutated


def _point_mutate(tree: TreeNode, strength: float) -> None:
    """Change a parameter value."""
    nodes = tree.collect_nodes()
    node = random.choice(nodes)

    if node.node_type.value in ("rsi", "cci", "williams_r", "mfi"):
        node.params["period"] = max(2, int(node.params.get("period", 14) + random.gauss(0, 3 * strength)))
    elif node.node_type.value in ("ema", "sma"):
        node.params["period"] = max(2, int(node.params.get("period", 20) + random.gauss(0, 10 * strength)))
    elif node.node_type.value == "atr":
        node.params["period"] = max(2, int(node.params.get("period", 14) + random.gauss(0, 3 * strength)))
    elif node.node_type.value == "constant":
        node.value += random.gauss(0, 5 * strength)
        node.value = round(node.value, 2)


def _node_mutate(tree: TreeNode) -> None:
    """Change a node's type."""
    from ..core.enums import NodeType
    nodes = tree.collect_nodes()
    node = random.choice(nodes)

    if node.node_type in {NodeType.GT, NodeType.LT, NodeType.GTE, NodeType.LTE}:
        node.node_type = random.choice([NodeType.GT, NodeType.LT, NodeType.GTE, NodeType.LTE])
    elif node.node_type in {NodeType.AND, NodeType.OR}:
        node.node_type = random.choice([NodeType.AND, NodeType.OR])


def _subtree_mutate(tree: TreeNode, strength: float) -> None:
    """Replace a subtree with a new random one."""
    if not tree.children:
        return

    idx = random.randint(0, len(tree.children) - 1)
    tree.children[idx] = random_tree(max_depth=max(1, int(3 * strength)))


def _insertion_mutate(tree: TreeNode, strength: float) -> None:
    """Insert a new condition."""
    from ..core.enums import NodeType
    from .rule_tree import random_comparison_node

    new_node = TreeNode(
        node_type=random.choice([NodeType.AND, NodeType.OR]),
        children=[tree.clone(), random_comparison_node()],
    )
    tree.node_type = new_node.node_type
    tree.children = new_node.children


def _deletion_mutate(tree: TreeNode) -> None:
    """Remove a condition (promote a child)."""
    if not tree.children:
        return

    child = random.choice(tree.children)
    tree.node_type = child.node_type
    tree.value = child.value
    tree.params = child.params
    tree.children = child.children
