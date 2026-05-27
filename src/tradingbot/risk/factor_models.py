"""Factor Models for Risk Decomposition.

Implements:
- CAPM (single factor)
- Fama-French 3-factor model
- Statistical factor model (PCA)
- Factor exposure analysis
- Factor-based risk decomposition
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FactorResult:
    """Factor model regression result."""
    alpha: float = 0.0
    betas: dict[str, float] = None
    r_squared: float = 0.0
    residual_vol: float = 0.0
    factor_contributions: dict[str, float] = None

    def __post_init__(self):
        if self.betas is None:
            self.betas = {}
        if self.factor_contributions is None:
            self.factor_contributions = {}


@dataclass
class PCAResult:
    """PCA decomposition result."""
    eigenvalues: np.ndarray = None
    eigenvectors: np.ndarray = None
    explained_variance_ratio: np.ndarray = None
    n_components_90: int = 0  # Components for 90% variance

    def __post_init__(self):
        if self.eigenvalues is None:
            self.eigenvalues = np.array([])
        if self.eigenvectors is None:
            self.eigenvectors = np.array([])
        if self.explained_variance_ratio is None:
            self.explained_variance_ratio = np.array([])


class FactorModel:
    """Factor model analysis toolkit."""

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.risk_free_rate = config.get("risk_free_rate", 0.0)
        self.annualization = config.get("annualization", 365)

    def capm(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
    ) -> FactorResult:
        """CAPM single-factor model.

        R_i = alpha + beta * R_market + epsilon
        """
        n = min(len(asset_returns), len(market_returns))
        r = asset_returns[:n] - self.risk_free_rate / self.annualization
        m = market_returns[:n] - self.risk_free_rate / self.annualization

        # OLS regression
        X = np.column_stack([np.ones(n), m])
        try:
            coeffs = np.linalg.lstsq(X, r, rcond=None)[0]
        except np.linalg.LinAlgError:
            return FactorResult()

        alpha = coeffs[0]
        beta = coeffs[1]

        # R-squared
        predicted = X @ coeffs
        ss_res = np.sum((r - predicted) ** 2)
        ss_tot = np.sum((r - np.mean(r)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        residual_vol = np.std(r - predicted) * np.sqrt(self.annualization)

        return FactorResult(
            alpha=alpha * self.annualization,
            betas={"market": beta},
            r_squared=r_squared,
            residual_vol=residual_vol,
            factor_contributions={
                "market": beta * np.std(m) * np.sqrt(self.annualization),
            },
        )

    def fama_french_3(
        self,
        asset_returns: np.ndarray,
        market_returns: np.ndarray,
        smb_returns: np.ndarray,
        hml_returns: np.ndarray,
    ) -> FactorResult:
        """Fama-French 3-factor model.

        R_i = alpha + beta_mkt * R_mkt + beta_smb * SMB + beta_hml * HML + epsilon
        """
        n = min(len(asset_returns), len(market_returns), len(smb_returns), len(hml_returns))
        r = asset_returns[:n]
        mkt = market_returns[:n]
        smb = smb_returns[:n]
        hml = hml_returns[:n]

        X = np.column_stack([np.ones(n), mkt, smb, hml])
        try:
            coeffs = np.linalg.lstsq(X, r, rcond=None)[0]
        except np.linalg.LinAlgError:
            return FactorResult()

        alpha = coeffs[0]
        betas = {
            "market": coeffs[1],
            "smb": coeffs[2],
            "hml": coeffs[3],
        }

        predicted = X @ coeffs
        ss_res = np.sum((r - predicted) ** 2)
        ss_tot = np.sum((r - np.mean(r)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        residual_vol = np.std(r - predicted) * np.sqrt(self.annualization)

        # Factor contributions (beta * factor vol)
        contributions = {}
        factor_names = ["market", "smb", "hml"]
        factor_returns = [mkt, smb, hml]
        for name, fret in zip(factor_names, factor_returns):
            contributions[name] = betas[name] * np.std(fret) * np.sqrt(self.annualization)

        return FactorResult(
            alpha=alpha * self.annualization,
            betas=betas,
            r_squared=r_squared,
            residual_vol=residual_vol,
            factor_contributions=contributions,
        )

    def pca(
        self,
        returns_matrix: np.ndarray,
        n_components: int = 0,
    ) -> PCAResult:
        """PCA-based statistical factor model.

        Args:
            returns_matrix: Shape (n_obs, n_assets)
            n_components: Number of components (0 = all)
        """
        # Center returns
        centered = returns_matrix - np.mean(returns_matrix, axis=0)

        # Covariance matrix
        cov = np.cov(centered, rowvar=False)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Explained variance
        total_var = np.sum(eigenvalues)
        explained = eigenvalues / total_var if total_var > 0 else eigenvalues

        # Components for 90% variance
        cumvar = np.cumsum(explained)
        n_90 = int(np.searchsorted(cumvar, 0.9)) + 1

        if n_components > 0:
            eigenvalues = eigenvalues[:n_components]
            eigenvectors = eigenvectors[:, :n_components]
            explained = explained[:n_components]

        return PCAResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            explained_variance_ratio=explained,
            n_components_90=n_90,
        )

    def factor_exposure(
        self,
        portfolio_returns: np.ndarray,
        factor_returns: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Compute portfolio factor exposures (betas)."""
        exposures = {}
        for name, fret in factor_returns.items():
            n = min(len(portfolio_returns), len(fret))
            p = portfolio_returns[:n]
            f = fret[:n]

            cov_pf = np.cov(p, f)[0, 1]
            var_f = np.var(f)
            if var_f > 1e-10:
                exposures[name] = float(cov_pf / var_f)
            else:
                exposures[name] = 0.0

        return exposures

    def risk_decomposition(
        self,
        betas: dict[str, float],
        factor_cov: np.ndarray,
        residual_vol: float,
    ) -> dict[str, float]:
        """Decompose total risk into factor and residual components."""
        beta_vec = np.array(list(betas.values()))
        n_factors = len(beta_vec)

        # Factor risk: beta' * Sigma_f * beta
        if factor_cov.shape == (n_factors, n_factors):
            factor_var = float(beta_vec @ factor_cov @ beta_vec)
        else:
            factor_var = float(np.sum(beta_vec ** 2 * np.diag(factor_cov[:n_factors, :n_factors])))

        total_var = factor_var + residual_vol ** 2

        result = {"total_risk": np.sqrt(total_var)}
        result["factor_risk"] = np.sqrt(factor_var)
        result["residual_risk"] = residual_vol

        if total_var > 0:
            result["factor_risk_pct"] = factor_var / total_var
            result["residual_risk_pct"] = residual_vol ** 2 / total_var

        return result

    def tracking_error(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """Compute tracking error vs benchmark."""
        n = min(len(portfolio_returns), len(benchmark_returns))
        active = portfolio_returns[:n] - benchmark_returns[:n]
        return float(np.std(active) * np.sqrt(self.annualization))

    def information_ratio(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
    ) -> float:
        """Compute information ratio."""
        n = min(len(portfolio_returns), len(benchmark_returns))
        active = portfolio_returns[:n] - benchmark_returns[:n]
        te = np.std(active)
        if te < 1e-10:
            return 0.0
        return float(np.mean(active) / te * np.sqrt(self.annualization))
