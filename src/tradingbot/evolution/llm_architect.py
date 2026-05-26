"""LLM Strategy Architect — Uses LLMs to reason about markets and design strategies.

The LLM doesn't trade. It RESEARCHES and DESIGNS. It's the Chief Scientist
of the trading lab — analyzing markets, hypothesizing, and proposing experiments.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from ..core.types import StrategyGenome
from ..genome.strategy_genome import create_random_genome

logger = logging.getLogger(__name__)


class LLMStrategist:
    """Uses LLMs to reason about markets and generate strategy hypotheses.

    Capabilities:
    1. Market Analysis — "What's happening in the market right now?"
    2. Hypothesis Generation — "I think X causes Y because Z"
    3. Strategy Design — "Based on this hypothesis, here's a strategy"
    4. Failure Analysis — "This strategy failed because..."
    5. Feature Discovery — "We should try combining X and Y features"
    """

    def __init__(self, config: dict):
        self.model = config.get("model", "gpt-4")
        self.temperature = config.get("temperature", 0.7)
        self._conversation_history: list[dict] = []
        self._proposed_strategies: list[dict] = []

    async def analyze_market(self, market_data: dict) -> dict:
        """Analyze current market conditions and generate insights."""
        prompt = self._build_market_analysis_prompt(market_data)

        # In production, this would call an LLM API
        # For now, return a structured analysis
        analysis = {
            "regime": self._infer_regime(market_data),
            "key_drivers": self._identify_drivers(market_data),
            "risks": self._identify_risks(market_data),
            "opportunities": self._identify_opportunities(market_data),
            "confidence": 0.6,
        }

        self._conversation_history.append({"role": "assistant", "content": json.dumps(analysis)})
        return analysis

    async def propose_strategy(self, context: dict) -> Optional[StrategyGenome]:
        """Propose a new strategy based on market analysis."""
        analysis = context.get("market_analysis", {})
        regime = analysis.get("regime", "unknown")

        # Generate strategy hypothesis
        hypothesis = self._generate_hypothesis(analysis)

        # Convert hypothesis to genome
        genome = create_random_genome(f"llm_proposed_{len(self._proposed_strategies)}")

        # Customize genome based on hypothesis
        if hypothesis.get("type") == "trend_following":
            genome.active_regimes = ["trending", "bull_low_vol"]
            genome.stop_loss_method = "atr"
            genome.stop_loss_param = 2.5
        elif hypothesis.get("type") == "mean_reversion":
            genome.active_regimes = ["mean_reverting", "bear_low_vol"]
            genome.stop_loss_method = "atr"
            genome.stop_loss_param = 1.5
        elif hypothesis.get("type") == "momentum":
            genome.active_regimes = ["bull_high_vol", "trending"]
            genome.stop_loss_method = "trailing"
            genome.stop_loss_param = 3.0

        self._proposed_strategies.append({
            "hypothesis": hypothesis,
            "genome_id": genome.id,
            "regime_context": regime,
        })

        logger.info(f"LLM proposed strategy: {hypothesis.get('description', 'unknown')}")
        return genome

    async def analyze_failure(self, genome: StrategyGenome, trade_history: list[dict]) -> dict:
        """Analyze why a strategy failed."""
        analysis = {
            "strategy_id": genome.id,
            "possible_causes": [],
            "suggestions": [],
        }

        # Analyze trade history
        if not trade_history:
            analysis["possible_causes"].append("No trades executed")
            return analysis

        losses = [t for t in trade_history if t.get("pnl", 0) < 0]
        wins = [t for t in trade_history if t.get("pnl", 0) > 0]

        if len(losses) > len(wins) * 2:
            analysis["possible_causes"].append("Win rate too low")
            analysis["suggestions"].add("Tighten entry conditions")

        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0

        if abs(avg_loss) > avg_win * 2:
            analysis["possible_causes"].append("Average loss much larger than average win")
            analysis["suggestions"].append("Tighten stop loss or widen take profit")

        # Check if regime mismatch
        analysis["possible_causes"].append("Possible regime mismatch — strategy may not suit current market")

        return analysis

    async def discover_features(self, current_features: list[str]) -> list[dict]:
        """Propose new feature combinations to try."""
        proposals = []

        # Propose feature interactions
        for i, f1 in enumerate(current_features[:10]):
            for f2 in current_features[i + 1:10]:
                proposals.append({
                    "name": f"interaction_{f1}_{f2}",
                    "formula": f"{f1} * {f2}",
                    "hypothesis": f"Interaction between {f1} and {f2} may capture non-linear effects",
                })

        # Propose temporal features
        for f in current_features[:5]:
            proposals.append({
                "name": f"momentum_{f}",
                "formula": f"{f} - {f}[-5]",
                "hypothesis": f"Momentum of {f} may predict future direction",
            })

        return proposals[:20]  # Return top 20 proposals

    def _build_market_analysis_prompt(self, market_data: dict) -> str:
        """Build prompt for market analysis."""
        return f"""Analyze the current market conditions:

Prices: {json.dumps(market_data.get('prices', {}), indent=2)}
Recent returns: {json.dumps(market_data.get('returns', {}), indent=2)}
Volatility: {json.dumps(market_data.get('volatility', {}), indent=2)}
Volume: {json.dumps(market_data.get('volume', {}), indent=2)}

What regime are we in? What are the key drivers? What risks exist?
What opportunities exist for trading strategies?"""

    def _infer_regime(self, market_data: dict) -> str:
        """Infer market regime from data."""
        returns = market_data.get("returns", {})
        vol = market_data.get("volatility", {})

        avg_return = sum(returns.values()) / len(returns) if returns else 0
        avg_vol = sum(vol.values()) / len(vol) if vol else 0

        if avg_return > 0.001 and avg_vol < 0.02:
            return "bull_low_vol"
        elif avg_return > 0.001:
            return "bull_high_vol"
        elif avg_vol < 0.02:
            return "bear_low_vol"
        else:
            return "bear_high_vol"

    def _identify_drivers(self, market_data: dict) -> list[str]:
        """Identify key market drivers."""
        return ["momentum", "volatility_regime", "cross_asset_correlation"]

    def _identify_risks(self, market_data: dict) -> list[str]:
        """Identify market risks."""
        return ["regime_transition", "liquidity_dry_up", "correlation_spike"]

    def _identify_opportunities(self, market_data: dict) -> list[str]:
        """Identify trading opportunities."""
        return ["trend_continuation", "mean_reversion_setup", "volatility_compression"]

    def _generate_hypothesis(self, analysis: dict) -> dict:
        """Generate a trading hypothesis from market analysis."""
        regime = analysis.get("regime", "unknown")

        if "bull" in regime:
            return {
                "type": "trend_following",
                "description": "Follow the upward trend with momentum confirmation",
                "expected_edge": "Trend persistence in low-vol bull market",
            }
        elif "bear" in regime:
            return {
                "type": "mean_reversion",
                "description": "Trade bounces in oversold conditions",
                "expected_edge": "Oversold bounces in declining market",
            }
        else:
            return {
                "type": "momentum",
                "description": "Capture momentum in trending conditions",
                "expected_edge": "Momentum factor in trending market",
            }
