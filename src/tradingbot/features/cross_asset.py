"""Cross-Asset Features — Correlation, cointegration, lead-lag analysis.

Implements:
- Rolling correlation matrices
- Cointegration tests (Engle-Granger)
- Lead-lag analysis
- Beta computation
- Regime-conditional correlations
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    correlation_matrix: np.ndarray
    asset_names: list[str]
    avg_correlation: float = 0.0
    max_correlation: float = 0.0
    min_correlation: float = 0.0


@dataclass
class CointegrationResult:
    """Result of cointegration test."""
    asset_a: str
    asset_b: str
    is_cointegrated: bool = False
    hedge_ratio: float = 0.0
    spread_mean: float = 0.0
    spread_std: float = 0.0
    half_life: float = 0.0  # Mean reversion half-life
    adf_statistic: float = 0.0


class CrossAssetAnalyzer:
    """Cross-asset analysis toolkit.

    Computes correlations, cointegration, and lead-lag relationships
    between multiple assets.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.default_lookback = config.get("lookback", 60)

    def rolling_correlation(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        window: int = 20,
    ) -> np.ndarray:
        """Compute rolling correlation between two return series."""
        n = min(len(returns_a), len(returns_b))
        a = returns_a[:n]
        b = returns_b[:n]

        result = np.full(n, np.nan)
        for i in range(window, n):
            wa = a[i - window:i]
            wb = b[i - window:i]
            if np.std(wa) > 0 and np.std(wb) > 0:
                result[i] = float(np.corrcoef(wa, wb)[0, 1])

        return result

    def correlation_matrix(
        self,
        returns_dict: dict[str, np.ndarray],
        window: int = 60,
    ) -> CorrelationResult:
        """Compute correlation matrix for multiple assets."""
        names = list(returns_dict.keys())
        n_assets = len(names)

        # Align all series to same length
        min_len = min(len(v) for v in returns_dict.values())
        returns_matrix = np.column_stack([
            returns_dict[name][-min_len:] for name in names
        ])

        # Use last `window` observations
        if min_len > window:
            returns_matrix = returns_matrix[-window:]

        corr = np.corrcoef(returns_matrix, rowvar=False)

        # Extract off-diagonal elements
        off_diag = []
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                off_diag.append(corr[i, j])

        off_diag = np.array(off_diag) if off_diag else np.array([0.0])

        return CorrelationResult(
            correlation_matrix=corr,
            asset_names=names,
            avg_correlation=float(np.mean(off_diag)),
            max_correlation=float(np.max(off_diag)),
            min_correlation=float(np.min(off_diag)),
        )

    def beta(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        window: int = 60,
    ) -> float:
        """Compute asset beta relative to market."""
        n = min(len(asset_returns), len(market_returns))
        if n < window:
            return 1.0

        a = asset_returns[-window:]
        m = market_returns[-window:]

        cov_am = np.cov(a, m)[0, 1]
        var_m = np.var(m)

        if var_m < 1e-10:
            return 1.0

        return float(cov_am / var_m)

    def rolling_beta(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        window: int = 60,
    ) -> np.ndarray:
        """Compute rolling beta."""
        n = min(len(asset_returns), len(market_returns))
        a = asset_returns[:n]
        m = market_returns[:n]

        result = np.full(n, np.nan)
        for i in range(window, n):
            wa = a[i - window:i]
            wm = m[i - window:i]
            var_m = np.var(wm)
            if var_m > 1e-10:
                result[i] = float(np.cov(wa, wm)[0, 1] / var_m)

        return result

    def cointegration_test(
        self,
        prices_a: np.ndarray,
        prices_b: np.ndarray,
        significance: float = 0.05,
    ) -> CointegrationResult:
        """Simple Engle-Granger cointegration test.

        Tests whether two price series share a common stochastic trend.
        """
        n = min(len(prices_a), len(prices_b))
        a = prices_a[:n]
        b = prices_b[:n]

        if n < 30:
            return CointegrationResult(asset_a="", asset_b="")

        # Step 1: OLS regression to find hedge ratio
        # a = hedge_ratio * b + residual
        b_matrix = np.column_stack([np.ones(n), b])
        try:
            coeffs = np.linalg.lstsq(b_matrix, a, rcond=None)[0]
            hedge_ratio = coeffs[1]
        except np.linalg.LinAlgError:
            return CointegrationResult(asset_a="", asset_b="")

        # Step 2: Compute spread (residual)
        spread = a - hedge_ratio * b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)

        # Step 3: ADF-like test on spread
        # Simple version: check if spread is mean-reverting
        spread_demeaned = spread - spread_mean
        lagged = spread_demeaned[:-1]
        diff = np.diff(spread_demeaned)

        if len(lagged) < 10:
            return CointegrationResult(asset_a="", asset_b="")

        # OLS: diff = phi * lagged + error
        try:
            phi = np.dot(lagged, diff) / np.dot(lagged, lagged)
        except ZeroDivisionError:
            phi = 0

        # Step 4: Half-life of mean reversion
        if phi < 0:
            half_life = -np.log(2) / phi
        else:
            half_life = float("inf")

        # Step 5: Determine if cointegrated
        # Simplified: phi < 0 and half-life is reasonable
        is_cointegrated = phi < 0 and half_life < 100 and abs(phi) > 0.01

        return CointegrationResult(
            asset_a="",
            asset_b="",
            is_cointegrated=is_cointegrated,
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
            half_life=half_life,
            adf_statistic=phi,
        )

    def lead_lag(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        max_lag: int = 10,
    ) -> tuple[int, float]:
        """Find the lead-lag relationship between two assets.

        Returns:
            best_lag: Positive = A leads B, Negative = B leads A
            correlation: Cross-correlation at best lag
        """
        n = min(len(returns_a), len(returns_b))
        a = returns_a[:n]
        b = returns_b[:n]

        best_lag = 0
        best_corr = 0.0

        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                a_shifted = a[:n - lag] if lag > 0 else a
                b_shifted = b[lag:]
            else:
                a_shifted = a[-lag:]
                b_shifted = b[:n + lag]

            if len(a_shifted) < 10:
                continue

            if np.std(a_shifted) > 0 and np.std(b_shifted) > 0:
                corr = abs(float(np.corrcoef(a_shifted, b_shifted)[0, 1]))
                if corr > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

        return best_lag, best_corr

    def regime_conditional_correlation(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        regime_indicator: np.ndarray,
    ) -> dict[str, float]:
        """Compute correlation conditioned on market regime.

        Args:
            regime_indicator: Array of regime labels (e.g., 0=bear, 1=bull)
        """
        n = min(len(returns_a), len(returns_b), len(regime_indicator))
        a = returns_a[:n]
        b = returns_b[:n]
        regime = regime_indicator[:n]

        result = {}
        unique_regimes = np.unique(regime)

        for r in unique_regimes:
            mask = regime == r
            if np.sum(mask) < 10:
                continue
            a_r = a[mask]
            b_r = b[mask]
            if np.std(a_r) > 0 and np.std(b_r) > 0:
                result[f"regime_{r}"] = float(np.corrcoef(a_r, b_r)[0, 1])

        return result
