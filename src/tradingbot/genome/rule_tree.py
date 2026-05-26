"""Rule Tree — AST for trading strategy rules.

A strategy is encoded as a tree of nodes. Each node is either:
- A logical operator (AND, OR, NOT) with children
- A comparison operator (GT, LT, CROSS_ABOVE, etc.) with two children
- An indicator leaf (RSI, EMA, MACD, etc.) with parameters
- A constant leaf (numeric value)

The tree is evaluated against market data to produce a boolean signal.
"""
from __future__ import annotations

import copy
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.enums import NodeType


@dataclass
class TreeNode:
    """A node in the strategy rule tree."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    node_type: NodeType = NodeType.CONSTANT
    value: float = 0.0  # For CONSTANT nodes
    params: dict[str, Any] = field(default_factory=dict)  # For indicator nodes
    children: list[TreeNode] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "value": self.value,
            "params": self.params,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TreeNode:
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            node_type=NodeType(data["node_type"]),
            value=data.get("value", 0.0),
            params=data.get("params", {}),
            children=[cls.from_dict(c) for c in data.get("children", [])],
        )

    def clone(self) -> TreeNode:
        return TreeNode.from_dict(self.to_dict())

    def collect_nodes(self) -> list[TreeNode]:
        """Collect all nodes in the tree (pre-order traversal)."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.collect_nodes())
        return nodes

    def random_subtree(self) -> TreeNode:
        """Pick a random node from the tree."""
        nodes = self.collect_nodes()
        return random.choice(nodes)


# Logical operators (boolean → boolean)
LOGICAL_TYPES = {NodeType.AND, NodeType.OR, NodeType.NOT}

# Comparison operators (value × value → boolean)
COMPARISON_TYPES = {
    NodeType.GT, NodeType.LT, NodeType.GTE, NodeType.LTE,
    NodeType.EQ, NodeType.CROSS_ABOVE, NodeType.CROSS_BELOW,
}

# Indicator nodes (params → value)
INDICATOR_TYPES = {
    NodeType.RSI, NodeType.EMA, NodeType.SMA, NodeType.MACD,
    NodeType.MACD_SIGNAL, NodeType.MACD_HIST, NodeType.BB_UPPER,
    NodeType.BB_LOWER, NodeType.BB_MIDDLE, NodeType.ATR, NodeType.ADX,
    NodeType.STOCH_K, NodeType.STOCH_D, NodeType.CCI, NodeType.WILLIAMS_R,
    NodeType.MFI, NodeType.OBV, NodeType.VWAP, NodeType.MOMENTUM, NodeType.ROC,
}

# Order book / microstructure nodes
MICRO_TYPES = {
    NodeType.BOOK_IMBALANCE, NodeType.SPREAD, NodeType.BID_DEPTH,
    NodeType.ASK_DEPTH, NodeType.VPIN, NodeType.KYLE_LAMBDA, NodeType.TRADE_FLOW,
}

# Cross-asset / alternative data nodes
CROSS_TYPES = {
    NodeType.CORRELATION, NodeType.BETA, NodeType.SENTIMENT,
    NodeType.ON_CHAIN, NodeType.FUNDING_RATE, NodeType.OPEN_INTEREST,
}

# Price/volume leaf nodes (no params)
PRICE_TYPES = {
    NodeType.CLOSE, NodeType.OPEN, NodeType.HIGH, NodeType.LOW, NodeType.VOLUME,
}

# Meta nodes
META_TYPES = {NodeType.REGIME_STATE, NodeType.VOLATILITY, NodeType.HOUR, NodeType.DAY_OF_WEEK}

# All value-producing nodes
VALUE_TYPES = INDICATOR_TYPES | PRICE_TYPES | MICRO_TYPES | CROSS_TYPES | META_TYPES | {NodeType.CONSTANT}


def random_indicator_node() -> TreeNode:
    """Create a random indicator node with sensible default params."""
    node_type = random.choice(list(INDICATOR_TYPES))
    params: dict[str, Any] = {}

    if node_type in (NodeType.RSI, NodeType.CCI, NodeType.WILLIAMS_R, NodeType.MFI):
        params["period"] = random.choice([7, 14, 21, 28])
    elif node_type in (NodeType.EMA, NodeType.SMA):
        params["period"] = random.choice([8, 12, 21, 34, 55, 100, 200])
    elif node_type == NodeType.ATR:
        params["period"] = random.choice([7, 14, 21])
    elif node_type == NodeType.ADX:
        params["period"] = random.choice([14, 21, 28])
    elif node_type in (NodeType.STOCH_K, NodeType.STOCH_D):
        params["k_period"] = random.choice([9, 14, 21])
        params["d_period"] = 3
    elif node_type in (NodeType.MACD, NodeType.MACD_SIGNAL, NodeType.MACD_HIST):
        params["fast"] = 12
        params["slow"] = 26
        params["signal"] = 9
    elif node_type in (NodeType.BB_UPPER, NodeType.BB_LOWER, NodeType.BB_MIDDLE):
        params["period"] = random.choice([14, 20, 30])
        params["std"] = random.choice([1.5, 2.0, 2.5])
    elif node_type == NodeType.MOMENTUM:
        params["period"] = random.choice([5, 10, 14, 20])
    elif node_type == NodeType.ROC:
        params["period"] = random.choice([5, 10, 14, 20])
    elif node_type == NodeType.CORRELATION:
        params["symbol"] = random.choice(["BTC/USDT", "ETH/USDT", "SPY"])
        params["period"] = random.choice([14, 30, 60])
    elif node_type == NodeType.BETA:
        params["symbol"] = random.choice(["BTC/USDT", "ETH/USDT", "SPY"])
        params["period"] = random.choice([30, 60, 90])
    elif node_type == NodeType.FUNDING_RATE:
        params["exchange"] = "binance"
    elif node_type == NodeType.OPEN_INTEREST:
        params["exchange"] = "binance"
    elif node_type == NodeType.SENTIMENT:
        params["source"] = random.choice(["twitter", "reddit", "news"])
        params["period"] = random.choice([1, 4, 24])  # hours

    return TreeNode(node_type=node_type, params=params)


def random_price_node() -> TreeNode:
    """Create a random price/volume leaf node."""
    node_type = random.choice(list(PRICE_TYPES))
    return TreeNode(node_type=node_type)


def random_constant_node() -> TreeNode:
    """Create a random constant node."""
    value = random.uniform(-100, 100)
    return TreeNode(node_type=NodeType.CONSTANT, value=round(value, 2))


def random_value_node() -> TreeNode:
    """Create a random value-producing node."""
    choice = random.random()
    if choice < 0.5:
        return random_indicator_node()
    elif choice < 0.7:
        return random_price_node()
    elif choice < 0.85:
        return TreeNode(node_type=random.choice(list(MICRO_TYPES)))
    else:
        return random_constant_node()


def random_comparison_node() -> TreeNode:
    """Create a random comparison node with two value children."""
    node_type = random.choice(list(COMPARISON_TYPES))
    return TreeNode(
        node_type=node_type,
        children=[random_value_node(), random_value_node()],
    )


def random_logical_node(max_depth: int = 3) -> TreeNode:
    """Create a random logical node with children."""
    if max_depth <= 0:
        return random_comparison_node()

    node_type = random.choice(list(LOGICAL_TYPES))
    if node_type == NodeType.NOT:
        return TreeNode(node_type=node_type, children=[random_comparison_node()])

    children = [random_comparison_node()]
    if random.random() < 0.5:
        children.append(random_comparison_node())
    else:
        children.append(random_logical_node(max_depth - 1))

    return TreeNode(node_type=node_type, children=children)


def random_tree(max_depth: int = 4) -> TreeNode:
    """Generate a random rule tree."""
    if max_depth <= 0:
        return random_comparison_node()

    choice = random.random()
    if choice < 0.3:
        return random_comparison_node()
    elif choice < 0.7:
        return random_logical_node(max_depth - 1)
    else:
        return TreeNode(
            node_type=random.choice([NodeType.AND, NodeType.OR]),
            children=[
                random_logical_node(max_depth - 1),
                random_logical_node(max_depth - 1),
            ],
        )
