"""Genome Encoder — Converts strategy genomes into executable strategy instances."""
from __future__ import annotations

from typing import Optional

from ..core.enums import NodeType, Side, Timeframe
from ..core.types import OHLCVBar, Signal, StrategyGenome
from .rule_tree import TreeNode


class GenomeEvaluator:
    """Evaluates a rule tree against market data to produce signals."""

    def __init__(self):
        self._indicator_cache: dict[str, dict] = {}  # Cache indicator values

    def evaluate(
        self,
        genome: StrategyGenome,
        bar: OHLCVBar,
        history: list[OHLCVBar],
        features: dict[str, float],
    ) -> Optional[Signal]:
        """Evaluate a genome against current market data."""
        if not genome.signal_tree:
            return None

        tree = TreeNode.from_dict(genome.signal_tree)
        context = self._build_context(bar, history, features)

        result = self._evaluate_node(tree, context)

        if result is None:
            return None

        # result is a boolean: True = buy signal, False = sell signal
        side = Side.BUY if result else Side.SELL

        return Signal(
            strategy_id=genome.id,
            symbol=bar.symbol,
            side=side,
            strength=abs(features.get("signal_strength", 0.5)),
            confidence=features.get("signal_confidence", 0.5),
            timeframe=Timeframe(genome.primary_timeframe),
            metadata={"genome_id": genome.id, "generation": genome.generation},
        )

    def _build_context(
        self, bar: OHLCVBar, history: list[OHLCVBar], features: dict[str, float]
    ) -> dict[str, float]:
        """Build evaluation context from market data."""
        ctx: dict[str, float] = {
            "close": bar.close,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "volume": bar.volume,
            "hour": bar.timestamp.hour,
            "day_of_week": bar.timestamp.weekday(),
        }

        # Add computed features
        ctx.update(features)

        # Add basic computed values from history
        if len(history) >= 2:
            ctx["prev_close"] = history[-1].close
            ctx["price_change"] = (bar.close - history[-1].close) / history[-1].close

        return ctx

    def _evaluate_node(self, node: TreeNode, ctx: dict[str, float]) -> Optional[bool | float]:
        """Recursively evaluate a tree node."""
        node_type = node.node_type

        # Logical operators
        if node_type == NodeType.AND:
            left = self._evaluate_node(node.children[0], ctx)
            right = self._evaluate_node(node.children[1], ctx) if len(node.children) > 1 else None
            if left is None or right is None:
                return None
            return bool(left) and bool(right)

        elif node_type == NodeType.OR:
            left = self._evaluate_node(node.children[0], ctx)
            right = self._evaluate_node(node.children[1], ctx) if len(node.children) > 1 else None
            if left is None or right is None:
                return None
            return bool(left) or bool(right)

        elif node_type == NodeType.NOT:
            child = self._evaluate_node(node.children[0], ctx) if node.children else None
            if child is None:
                return None
            return not bool(child)

        # Comparison operators
        elif node_type in {NodeType.GT, NodeType.LT, NodeType.GTE, NodeType.LTE, NodeType.EQ}:
            left = self._evaluate_node(node.children[0], ctx) if node.children else None
            right = self._evaluate_node(node.children[1], ctx) if len(node.children) > 1 else None
            if left is None or right is None:
                return None
            if node_type == NodeType.GT:
                return float(left) > float(right)
            elif node_type == NodeType.LT:
                return float(left) < float(right)
            elif node_type == NodeType.GTE:
                return float(left) >= float(right)
            elif node_type == NodeType.LTE:
                return float(left) <= float(right)
            else:
                return abs(float(left) - float(right)) < 1e-10

        elif node_type in {NodeType.CROSS_ABOVE, NodeType.CROSS_BELOW}:
            # Cross detection needs history
            return None  # TODO: Implement with historical values

        # Value-producing nodes
        elif node_type == NodeType.CONSTANT:
            return node.value

        elif node_type == NodeType.CLOSE:
            return ctx.get("close")
        elif node_type == NodeType.OPEN:
            return ctx.get("open")
        elif node_type == NodeType.HIGH:
            return ctx.get("high")
        elif node_type == NodeType.LOW:
            return ctx.get("low")
        elif node_type == NodeType.VOLUME:
            return ctx.get("volume")
        elif node_type == NodeType.HOUR:
            return ctx.get("hour")
        elif node_type == NodeType.DAY_OF_WEEK:
            return ctx.get("day_of_week")

        # Indicators (look up from features)
        elif node_type in {NodeType.RSI, NodeType.EMA, NodeType.SMA, NodeType.MACD,
                          NodeType.MACD_SIGNAL, NodeType.MACD_HIST, NodeType.BB_UPPER,
                          NodeType.BB_LOWER, NodeType.BB_MIDDLE, NodeType.ATR, NodeType.ADX,
                          NodeType.STOCH_K, NodeType.STOCH_D, NodeType.CCI, NodeType.WILLIAMS_R,
                          NodeType.MFI, NodeType.OBV, NodeType.VWAP, NodeType.MOMENTUM, NodeType.ROC}:
            period = node.params.get("period", 14)
            key = f"{node_type.value}_{period}"
            return ctx.get(key)

        # Microstructure
        elif node_type in {NodeType.BOOK_IMBALANCE, NodeType.SPREAD, NodeType.BID_DEPTH,
                          NodeType.ASK_DEPTH, NodeType.VPIN, NodeType.KYLE_LAMBDA,
                          NodeType.TRADE_FLOW}:
            return ctx.get(node_type.value)

        # Cross-asset
        elif node_type in {NodeType.CORRELATION, NodeType.BETA}:
            symbol = node.params.get("symbol", "BTC/USDT").replace("/", "_")
            period = node.params.get("period", 30)
            key = f"{node_type.value}_{symbol}_{period}"
            return ctx.get(key)

        # Alternative data
        elif node_type == NodeType.SENTIMENT:
            source = node.params.get("source", "news")
            return ctx.get(f"sentiment_{source}")
        elif node_type == NodeType.ON_CHAIN:
            return ctx.get("on_chain_score")
        elif node_type == NodeType.FUNDING_RATE:
            return ctx.get("funding_rate")
        elif node_type == NodeType.OPEN_INTEREST:
            return ctx.get("open_interest_change")

        # Regime
        elif node_type == NodeType.REGIME_STATE:
            return ctx.get("regime_score")
        elif node_type == NodeType.VOLATILITY:
            return ctx.get("volatility")

        return None


def genome_to_description(genome: StrategyGenome) -> str:
    """Convert a genome to human-readable description."""
    tree = TreeNode.from_dict(genome.signal_tree)
    parts = []

    def _describe_node(node: TreeNode, depth: int = 0) -> str:
        indent = "  " * depth
        if node.node_type == NodeType.AND:
            children = [_describe_node(c, depth + 1) for c in node.children]
            return f"{indent}AND(\n" + ",\n".join(children) + f"\n{indent})"
        elif node.node_type == NodeType.OR:
            children = [_describe_node(c, depth + 1) for c in node.children]
            return f"{indent}OR(\n" + ",\n".join(children) + f"\n{indent})"
        elif node.node_type == NodeType.NOT:
            child = _describe_node(node.children[0], depth + 1) if node.children else "?"
            return f"{indent}NOT({child})"
        elif node.node_type in {NodeType.GT, NodeType.LT, NodeType.GTE, NodeType.LTE}:
            op = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<="}[node.node_type.value]
            left = _describe_node(node.children[0], 0) if node.children else "?"
            right = _describe_node(node.children[1], 0) if len(node.children) > 1 else "?"
            return f"{left} {op} {right}"
        elif node.node_type == NodeType.CONSTANT:
            return str(node.value)
        else:
            params = ", ".join(f"{k}={v}" for k, v in node.params.items())
            return f"{node.node_type.value}({params})" if params else node.node_type.value

    signal_desc = _describe_node(tree)
    return (
        f"Strategy: {genome.name}\n"
        f"Signal: {signal_desc}\n"
        f"Stop: {genome.stop_loss_method} ({genome.stop_loss_param})\n"
        f"TP Ratio: {genome.take_profit_ratio}\n"
        f"Sizing: {genome.position_sizing} (max {genome.max_position_pct:.1%})\n"
        f"Timeframe: {genome.primary_timeframe} / {genome.confirmation_timeframe}\n"
        f"Regimes: {', '.join(genome.active_regimes)}\n"
        f"Fitness: {genome.fitness:.3f} (gen {genome.generation})"
    )
