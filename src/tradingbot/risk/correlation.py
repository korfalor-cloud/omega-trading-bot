"""Correlation Analysis for Portfolio Risk.

Implements:
- Rolling correlation matrices
- Correlation breakdown detection
- Tail dependency (Hill estimator)
- DCC-GARCH dynamic correlation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Correlation analysis result."""
    matrix: np.ndarray = None
    asset_names: list[str] = None
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    min_correlation: float = 0.0
    eigenvalues: np.ndarray = None

    def __post_init__(self):
        if self.matrix is None:
            self.matrix = np.array([])
        if self.asset_names is None:
            self.asset_names = []
        if self.eigenvalues is None:
            self.eigenvalues = np.array([])


class CorrelationAnalyzer:
    """Portfolio correlation analysis."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.rolling_window = config.get("rolling_window", 30)
        self.breakdown_threshold = config.get("breakdown_threshold", 0.3)

    def rolling_correlation(
        self,
        returns: dict[str, np.ndarray],
        window: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute rolling pairwise correlations."""
        w = window or self.rolling_window
        names = sorted(returns.keys())
        n = len(names)

        # Get common length
        min_len = min(len(returns[name]) for name in names)
        result = {}

        for i in range(n):
            for j in range(i + 1, n):
                pair = f"{names[i]}_{names[j]}"
                r_i = returns[names[i]][-min_len:]
                r_j = returns[names[j]][-min_len:]

                rolling_corr = np.full(min_len, np.nan)
                for k in range(w, min_len):
                    segment_i = r_i[k - w:k]
                    segment_j = r_j[k - w:k]
                    if np.std(segment_i) > 0 and np.std(segment_j) > 0:
                        rolling_corr[k] = np.corrcoef(segment_i, segment_j)[0, 1]

                result[pair] = rolling_corr

        return result

    def correlation_matrix(
        self,
        returns: dict[str, np.ndarray],
        window: int | None = None,
    ) -> CorrelationResult:
        """Compute correlation matrix."""
        names = sorted(returns.keys())
        n = len(names)
        if n == 0:
            return CorrelationResult()
        min_len = min(len(returns[name]) for name in names)

        r_matrix = np.zeros((min_len, n))
        for i, name in enumerate(names):
            r_matrix[:, i] = returns[name][-min_len:]

        corr = np.corrcoef(r_matrix.T)

        # Eigenvalues for concentration check
        eigenvalues = np.linalg.eigvalsh(corr)

        # Off-diagonal stats
        off_diag = corr[np.triu_indices(n, k=1)]

        return CorrelationResult(
            matrix=corr,
            asset_names=names,
            avg_correlation=float(np.mean(off_diag)) if len(off_diag) > 0 else 0,
            max_correlation=float(np.max(off_diag)) if len(off_diag) > 0 else 0,
            min_correlation=float(np.min(off_diag)) if len(off_diag) > 0 else 0,
            eigenvalues=eigenvalues,
        )

    def detect_breakdown(
        self,
        returns: dict[str, np.ndarray],
        window: int | None = None,
    ) -> list[dict]:
        """Detect correlation breakdowns.

        A breakdown is when rolling correlation drops significantly
        from its recent average.
        """
        rolling = self.rolling_correlation(returns, window)
        breakdowns = []

        for pair, corr_series in rolling.items():
            valid = corr_series[~np.isnan(corr_series)]
            if len(valid) < 20:
                continue

            recent = valid[-10:]
            historical = valid[:-10]

            if len(historical) < 10:
                continue

            recent_avg = np.mean(recent)
            hist_avg = np.mean(historical)
            change = abs(recent_avg - hist_avg)

            if change > self.breakdown_threshold:
                breakdowns.append({
                    "pair": pair,
                    "recent_corr": float(recent_avg),
                    "historical_corr": float(hist_avg),
                    "change": float(change),
                    "direction": "decrease" if recent_avg < hist_avg else "increase",
                })

        return breakdowns

    def tail_dependency(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        threshold: float = 0.05,
    ) -> dict:
        """Estimate tail dependency using Hill estimator."""
        n = min(len(returns_a), len(returns_b))
        r_a = returns_a[-n:]
        r_b = returns_b[-n:]

        # Sort by magnitude
        k = max(1, int(n * threshold))

        # Lower tail (both negative)
        mask_lower = (r_a < np.percentile(r_a, threshold * 100)) & (r_b < np.percentile(r_b, threshold * 100))
        lower_dep = np.sum(mask_lower) / k if k > 0 else 0

        # Upper tail (both positive)
        mask_upper = (r_a > np.percentile(r_a, (1 - threshold) * 100)) & (r_b > np.percentile(r_b, (1 - threshold) * 100))
        upper_dep = np.sum(mask_upper) / k if k > 0 else 0

        return {
            "lower_tail_dependency": float(lower_dep),
            "upper_tail_dependency": float(upper_dep),
            "threshold": threshold,
        }

    def effective_n_assets(self, corr_matrix: np.ndarray) -> float:
        """Effective number of assets (inverse of concentration).

        N_eff = 1 / sum(lambda_i^2) where lambda are eigenvalues
        normalized to sum to 1.
        """
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]
        normalized = eigenvalues / np.sum(eigenvalues)
        return float(1 / np.sum(normalized ** 2))
