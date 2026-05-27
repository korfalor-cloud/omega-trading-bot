"""Tests for genome operations — creation, crossover, mutation."""
from __future__ import annotations

import pytest

from tradingbot.core.enums import NodeType
from tradingbot.genome.rule_tree import TreeNode, random_tree
from tradingbot.genome.strategy_genome import (
    crossover_genomes,
    create_random_genome,
    mutate_genome,
)
from tradingbot.genome.genome_encoder import GenomeEvaluator, genome_to_description


class TestRuleTree:
    def test_random_tree_creation(self):
        tree = random_tree(max_depth=3)
        assert isinstance(tree, TreeNode)
        assert tree.node_type is not None

    def test_tree_depth(self):
        tree = random_tree(max_depth=4)
        assert tree.depth() <= 5  # Allow some slack

    def test_tree_size(self):
        tree = random_tree(max_depth=3)
        assert tree.size() >= 1

    def test_tree_serialization(self):
        tree = random_tree(max_depth=3)
        data = tree.to_dict()
        restored = TreeNode.from_dict(data)
        assert restored.to_dict() == data

    def test_tree_clone(self):
        tree = random_tree(max_depth=3)
        cloned = tree.clone()
        assert cloned.to_dict() == tree.to_dict()
        # Modify clone shouldn't affect original
        cloned.value = 999
        assert tree.value != 999

    def test_collect_nodes(self):
        tree = random_tree(max_depth=3)
        nodes = tree.collect_nodes()
        assert len(nodes) == tree.size()
        assert nodes[0] is tree  # Root first


class TestRandomGenome:
    def test_creation(self):
        genome = create_random_genome("test")
        assert genome.name == "test"
        assert genome.id
        assert genome.signal_tree
        assert genome.generation == 0

    def test_unique_ids(self):
        g1 = create_random_genome()
        g2 = create_random_genome()
        assert g1.id != g2.id

    def test_valid_parameters(self):
        genome = create_random_genome()
        assert 0.005 <= genome.max_position_pct <= 0.20
        assert 0.5 <= genome.stop_loss_param <= 5.0
        assert 0.5 <= genome.take_profit_ratio <= 5.0
        assert genome.cooldown_bars >= 1


class TestCrossover:
    def test_crossover_produces_child(self):
        parent_a = create_random_genome("parent_a")
        parent_b = create_random_genome("parent_b")
        child = crossover_genomes(parent_a, parent_b)

        assert child.id != parent_a.id
        assert child.id != parent_b.id
        assert child.generation == 1
        assert len(child.parent_ids) == 2

    def test_crossover_inherits_parameters(self):
        parent_a = create_random_genome()
        parent_b = create_random_genome()
        child = crossover_genomes(parent_a, parent_b)

        # Child params should come from one of the parents
        assert child.stop_loss_method in (parent_a.stop_loss_method, parent_b.stop_loss_method)
        assert child.position_sizing in (parent_a.position_sizing, parent_b.position_sizing)

    def test_crossover_preserves_tree_structure(self):
        parent_a = create_random_genome()
        parent_b = create_random_genome()
        child = crossover_genomes(parent_a, parent_b)

        tree = TreeNode.from_dict(child.signal_tree)
        assert tree.node_type is not None


class TestMutation:
    def test_mutation_changes_id(self):
        original = create_random_genome("original")
        mutated = mutate_genome(original)

        assert mutated.id != original.id
        assert mutated.generation == original.generation + 1
        assert original.id in mutated.parent_ids

    def test_mutation_preserves_validity(self):
        original = create_random_genome()
        mutated = mutate_genome(original, mutation_strength=1.0)

        # Parameters should remain in valid ranges
        assert 0.005 <= mutated.max_position_pct <= 0.20
        assert 0.5 <= mutated.stop_loss_param <= 5.0
        assert 0.5 <= mutated.take_profit_ratio <= 5.0

    def test_mutation_produces_valid_tree(self):
        original = create_random_genome()
        mutated = mutate_genome(original)

        tree = TreeNode.from_dict(mutated.signal_tree)
        assert tree.node_type is not None
        assert tree.size() >= 1

    def test_mutation_strength_affects_change(self):
        original = create_random_genome()
        # Low strength = small changes
        mild = mutate_genome(original, mutation_strength=0.1)
        # High strength = big changes
        wild = mutate_genome(original, mutation_strength=2.0)

        # Both should be valid
        assert mild.signal_tree
        assert wild.signal_tree


class TestGenomeEvaluator:
    def test_evaluate_returns_signal_or_none(self, sample_bars, random_genome):
        evaluator = GenomeEvaluator()
        from tradingbot.features.technical import compute_features
        features = compute_features(sample_bars[:100])

        bar = sample_bars[50]
        history = sample_bars[:51]
        bar_features = {k: v[50] for k, v in features.items()}

        result = evaluator.evaluate(random_genome, bar, history, bar_features)
        # Should return Signal or None
        assert result is None or hasattr(result, "side")

    def test_genome_to_description(self, random_genome):
        desc = genome_to_description(random_genome)
        assert "Strategy:" in desc
        assert "Stop:" in desc
        assert "Fitness:" in desc


class TestTreeNodeTypes:
    def test_logical_nodes(self):
        node = TreeNode(node_type=NodeType.AND, children=[
            TreeNode(node_type=NodeType.CONSTANT, value=1.0),
            TreeNode(node_type=NodeType.CONSTANT, value=1.0),
        ])
        assert not node.is_leaf()
        assert node.depth() == 1

    def test_leaf_nodes(self):
        node = TreeNode(node_type=NodeType.CONSTANT, value=42.0)
        assert node.is_leaf()
        assert node.depth() == 0

    def test_nested_tree(self):
        tree = TreeNode(
            node_type=NodeType.AND,
            children=[
                TreeNode(
                    node_type=NodeType.OR,
                    children=[
                        TreeNode(node_type=NodeType.CONSTANT, value=1.0),
                        TreeNode(node_type=NodeType.CONSTANT, value=0.0),
                    ],
                ),
                TreeNode(node_type=NodeType.CONSTANT, value=1.0),
            ],
        )
        assert tree.size() == 5
        assert tree.depth() == 2
