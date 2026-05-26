"""Metacognition — Self-monitoring and uncertainty tracking."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from ..core.types import ConsciousnessState, PortfolioState, StrategyGenome

logger = logging.getLogger(__name__)


class MetacognitionEngine:
    """The self-awareness engine. Monitors what the system knows and doesn't know.

    Tracks:
    - Prediction confidence across all models
    - Regime uncertainty
    - Strategy health (which are working, which aren't)
    - Knowledge gaps (what the system hasn't seen before)
    - Risk appetite based on uncertainty
    """

    def __init__(self, config: dict):
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.7)
        self.min_confidence = config.get("min_confidence_for_live", 0.6)
        self._prediction_history: list[dict] = []
        self._regime_history: list[dict] = []
        self._strategy_health: dict[str, dict] = {}
        self._knowledge_gaps: list[str] = []
        self._reflections: list[str] = []

    async def assess_state(
        self,
        portfolio: PortfolioState,
        active_strategies: list[StrategyGenome],
        regime_confidence: float,
        model_confidences: dict[str, float],
    ) -> ConsciousnessState:
        """Assess the current state of self-awareness."""
        # Overall prediction confidence
        if model_confidences:
            avg_confidence = sum(model_confidences.values()) / len(model_confidences)
            min_confidence = min(model_confidences.values())
        else:
            avg_confidence = 0.0
            min_confidence = 0.0

        # Strategy health assessment
        strategy_health = {}
        for genome in active_strategies:
            health = self._assess_strategy_health(genome)
            strategy_health[genome.id] = health

        # Identify knowledge gaps
        gaps = self._identify_knowledge_gaps(
            portfolio, active_strategies, regime_confidence, model_confidences
        )

        # Calculate overall uncertainty
        uncertainty = 1.0 - (
            0.3 * avg_confidence
            + 0.3 * regime_confidence
            + 0.2 * (1.0 - portfolio.current_drawdown)
            + 0.2 * self._strategy_health_score(strategy_health)
        )

        state = ConsciousnessState(
            confidence_in_predictions=avg_confidence,
            uncertainty_about_market=uncertainty,
            strategy_health=strategy_health,
            goals=self._current_goals(uncertainty),
            reflections=self._reflections[-10:],
            knowledge_gaps=gaps,
        )

        # Log if uncertainty is high
        if uncertainty > self.uncertainty_threshold:
            logger.warning(f"HIGH UNCERTAINTY: {uncertainty:.2f} — reducing risk appetite")
            self._reflections.append(
                f"[{datetime.utcnow().isoformat()}] High uncertainty ({uncertainty:.2f}). "
                f"Regime confidence: {regime_confidence:.2f}, "
                f"Model confidence: {avg_confidence:.2f}, "
                f"Drawdown: {portfolio.current_drawdown:.2%}"
            )

        return state

    def _assess_strategy_health(self, genome: StrategyGenome) -> dict:
        """Assess the health of a single strategy."""
        return {
            "fitness": genome.fitness,
            "generation": genome.generation,
            "status": genome.status,
            "trades": genome.total_trades,
            "win_rate": genome.win_rate,
            "sharpe": genome.sharpe,
            "is_healthy": genome.fitness > 0.3 and genome.total_trades > 10,
        }

    def _strategy_health_score(self, health: dict[str, dict]) -> float:
        """Calculate overall strategy health score."""
        if not health:
            return 0.0
        healthy = sum(1 for h in health.values() if h.get("is_healthy", False))
        return healthy / len(health)

    def _identify_knowledge_gaps(
        self,
        portfolio: PortfolioState,
        strategies: list[StrategyGenome],
        regime_confidence: float,
        model_confidences: dict[str, float],
    ) -> list[str]:
        """Identify what the system doesn't know."""
        gaps = []

        if regime_confidence < 0.5:
            gaps.append("Uncertain about current market regime")

        low_conf_models = [k for k, v in model_confidences.items() if v < 0.4]
        if low_conf_models:
            gaps.append(f"Low confidence models: {', '.join(low_conf_models)}")

        if portfolio.current_drawdown > 0.05:
            gaps.append(f"In drawdown ({portfolio.current_drawdown:.1%}) — may not understand current conditions")

        if not strategies:
            gaps.append("No active strategies — need to evolve new ones")

        return gaps

    def _current_goals(self, uncertainty: float) -> list[str]:
        """Generate current goals based on state."""
        goals = ["Maintain positive risk-adjusted returns"]

        if uncertainty > self.uncertainty_threshold:
            goals.append("Reduce uncertainty through exploration")
            goals.append("Tighten risk limits")

        if uncertainty < 0.3:
            goals.append("Increase alpha generation")
            goals.append("Explore new strategy space")

        return goals

    async def reflect_on_trade(self, trade_result: dict) -> None:
        """Reflect on a completed trade."""
        pnl = trade_result.get("pnl", 0)
        strategy_id = trade_result.get("strategy_id", "unknown")
        reason = trade_result.get("reason", "")

        if pnl < 0:
            reflection = (
                f"[{datetime.utcnow().isoformat()}] Lost {abs(pnl):.2f} on {strategy_id}. "
                f"Reason: {reason}. Need to understand why."
            )
            self._reflections.append(reflection)
        elif pnl > 0:
            reflection = (
                f"[{datetime.utcnow().isoformat()}] Gained {pnl:.2f} on {strategy_id}. "
                f"What worked: {reason}"
            )
            self._reflections.append(reflection)

    async def reflect_on_regime_change(self, old_regime: str, new_regime: str) -> None:
        """Reflect on a regime transition."""
        self._reflections.append(
            f"[{datetime.utcnow().isoformat()}] Regime change: {old_regime} → {new_regime}. "
            f"Need to adapt strategy allocation."
        )
        self._regime_history.append({
            "old": old_regime,
            "new": new_regime,
            "timestamp": datetime.utcnow().isoformat(),
        })
