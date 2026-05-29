"""Position Concentration — HHI-based concentration limits.

Implements:
- Herfindahl-Hirschman Index (HHI) calculation for portfolio concentration
- Concentration alerts with configurable thresholds
- Diversification scoring (inverse HHI / effective N)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConcentrationAlert:
    """Alert for concentration limit breach."""
    alert_type: str = ""  # single_position, sector, top_n
    symbol: str = ""
    current_weight: float = 0.0
    limit: float = 0.0
    severity: str = "warning"  # warning, critical


@dataclass
class ConcentrationReport:
    """Full concentration analysis report."""
    hhi: float = 0.0
    effective_n: float = 0.0
    diversification_score: float = 0.0
    max_single_weight: float = 0.0
    top_3_weight: float = 0.0
    top_5_weight: float = 0.0
    n_positions: int = 0
    alerts: list[ConcentrationAlert] = field(default_factory=list)
    gini_coefficient: float = 0.0
    risk_level: str = "low"


class PositionConcentrationManager:
    """HHI-based position concentration management."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.max_single_weight = config.get("max_single_weight", 0.25)
        self.max_top3_weight = config.get("max_top3_weight", 0.60)
        self.max_top5_weight = config.get("max_top5_weight", 0.75)
        self.hhi_warning = config.get("hhi_warning", 0.15)
        self.hhi_critical = config.get("hhi_critical", 0.25)
        self.min_effective_n = config.get("min_effective_n", 4)
        self._positions: dict[str, float] = {}  # symbol -> notional

    def update_position(self, symbol: str, notional: float) -> None:
        """Update position notional."""
        if abs(notional) > 0:
            self._positions[symbol] = abs(notional)
        elif symbol in self._positions:
            del self._positions[symbol]

    def clear_positions(self) -> None:
        """Clear all positions."""
        self._positions.clear()

    def compute_weights(self) -> dict[str, float]:
        """Compute portfolio weights from notional values."""
        total = sum(self._positions.values())
        if total == 0:
            return {}
        return {s: v / total for s, v in self._positions.items()}

    def compute_hhi(self) -> float:
        """Compute Herfindahl-Hirschman Index.

        HHI = sum(w_i^2) for all positions
        Range: 1/N (perfectly diversified) to 1.0 (single position)
        """
        weights = self.compute_weights()
        if not weights:
            return 0.0

        hhi = sum(w ** 2 for w in weights.values())
        logger.debug("HHI=%.4f across %d positions", hhi, len(weights))
        return float(hhi)

    def compute_effective_n(self) -> float:
        """Compute effective number of positions.

        Effective N = 1 / HHI
        Equivalent to the number of equal-weight positions
        that would produce the same concentration.
        """
        hhi = self.compute_hhi()
        if hhi == 0:
            return 0.0
        return float(1.0 / hhi)

    def compute_gini_coefficient(self) -> float:
        """Compute Gini coefficient of weight distribution.

        0 = perfectly equal weights, 1 = maximum inequality.
        """
        weights = self.compute_weights()
        if len(weights) < 2:
            return 0.0

        sorted_weights = sorted(weights.values())
        n = len(sorted_weights)
        cumulative = np.cumsum(sorted_weights)
        total = cumulative[-1]

        if total == 0:
            return 0.0

        # Gini = (2 * sum(i * w_i) / (n * sum(w_i))) - (n + 1) / n
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_weights) / (n * total)) - (n + 1) / n
        return float(np.clip(gini, 0.0, 1.0))

    def compute_diversification_score(self) -> float:
        """Compute diversification score (0-100).

        Score = (1 - normalized_HHI) * 100
        Normalized HHI accounts for number of positions.
        """
        hhi = self.compute_hhi()
        n = len(self._positions)

        if n <= 1:
            return 0.0

        # Minimum possible HHI for n positions = 1/n
        min_hhi = 1.0 / n
        # Maximum HHI = 1.0
        # Normalized: how close to perfectly diversified?
        if hhi <= min_hhi:
            return 100.0

        score = (1.0 - hhi) / (1.0 - min_hhi) * 100.0
        return float(np.clip(score, 0.0, 100.0))

    def check_alerts(self) -> list[ConcentrationAlert]:
        """Check for concentration limit breaches."""
        weights = self.compute_weights()
        alerts = []

        # Sort by weight descending
        sorted_items = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        # Single position limit
        for symbol, weight in sorted_items:
            if weight > self.max_single_weight:
                alerts.append(ConcentrationAlert(
                    alert_type="single_position",
                    symbol=symbol,
                    current_weight=round(weight, 4),
                    limit=self.max_single_weight,
                    severity="critical" if weight > self.max_single_weight * 1.5 else "warning",
                ))

        # Top-N concentration
        top_weights = [w for _, w in sorted_items]
        if len(top_weights) >= 3:
            top3 = sum(top_weights[:3])
            if top3 > self.max_top3_weight:
                alerts.append(ConcentrationAlert(
                    alert_type="top_3",
                    symbol=", ".join(s for s, _ in sorted_items[:3]),
                    current_weight=round(top3, 4),
                    limit=self.max_top3_weight,
                    severity="critical" if top3 > self.max_top3_weight * 1.2 else "warning",
                ))

        if len(top_weights) >= 5:
            top5 = sum(top_weights[:5])
            if top5 > self.max_top5_weight:
                alerts.append(ConcentrationAlert(
                    alert_type="top_5",
                    symbol=", ".join(s for s, _ in sorted_items[:5]),
                    current_weight=round(top5, 4),
                    limit=self.max_top5_weight,
                    severity="warning",
                ))

        # HHI alert
        hhi = self.compute_hhi()
        if hhi >= self.hhi_critical:
            alerts.append(ConcentrationAlert(
                alert_type="hhi",
                symbol="portfolio",
                current_weight=round(hhi, 4),
                limit=self.hhi_critical,
                severity="critical",
            ))
        elif hhi >= self.hhi_warning:
            alerts.append(ConcentrationAlert(
                alert_type="hhi",
                symbol="portfolio",
                current_weight=round(hhi, 4),
                limit=self.hhi_warning,
                severity="warning",
            ))

        return alerts

    def analyze(self) -> ConcentrationReport:
        """Run full concentration analysis."""
        weights = self.compute_weights()
        sorted_weights = sorted(weights.values(), reverse=True)

        hhi = self.compute_hhi()
        effective_n = self.compute_effective_n()
        gini = self.compute_gini_coefficient()
        score = self.compute_diversification_score()
        alerts = self.check_alerts()

        max_weight = sorted_weights[0] if sorted_weights else 0.0
        top3 = sum(sorted_weights[:3])
        top5 = sum(sorted_weights[:5])

        # Risk level
        has_critical = any(a.severity == "critical" for a in alerts)
        if has_critical or hhi >= self.hhi_critical:
            risk_level = "high"
        elif alerts or hhi >= self.hhi_warning:
            risk_level = "medium"
        else:
            risk_level = "low"

        return ConcentrationReport(
            hhi=round(hhi, 4),
            effective_n=round(effective_n, 2),
            diversification_score=round(score, 2),
            max_single_weight=round(max_weight, 4),
            top_3_weight=round(top3, 4),
            top_5_weight=round(top5, 4),
            n_positions=len(self._positions),
            alerts=alerts,
            gini_coefficient=round(gini, 4),
            risk_level=risk_level,
        )
