"""Options Greeks and Derivatives Risk Metrics.

Implements:
- Delta, Gamma, Theta, Vega, Rho (Black-Scholes)
- Implied volatility calculation
- Greeks aggregation for portfolio
- P&L attribution (delta P&L, gamma P&L, theta decay)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptionGreeks:
    """Greeks for a single option position."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0  # Per day
    vega: float = 0.0   # Per 1% vol move
    rho: float = 0.0
    implied_vol: float = 0.0
    option_price: float = 0.0


@dataclass
class PortfolioGreeks:
    """Aggregated Greeks for a portfolio."""
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_theta: float = 0.0
    net_vega: float = 0.0
    net_rho: float = 0.0
    delta_pnl_1pct: float = 0.0  # P&L for 1% move in underlying
    gamma_pnl_1pct: float = 0.0  # Gamma P&L for 1% move
    theta_decay_daily: float = 0.0


class BlackScholesCalculator:
    """Black-Scholes options pricing and Greeks calculator.

    Supports European calls and puts on crypto (no dividend yield).
    """

    def __init__(self, risk_free_rate: float = 0.04):
        self.r = risk_free_rate

    def _norm_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_pdf(self, x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    def d1(self, S: float, K: float, T: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (math.log(S / K) + (self.r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    def d2(self, S: float, K: float, T: float, sigma: float) -> float:
        return self.d1(S, K, T, sigma) - sigma * math.sqrt(T)

    def call_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Black-Scholes call option price."""
        if T <= 0:
            return max(0, S - K)
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        return S * self._norm_cdf(d1) - K * math.exp(-self.r * T) * self._norm_cdf(d2)

    def put_price(self, S: float, K: float, T: float, sigma: float) -> float:
        """Black-Scholes put option price."""
        if T <= 0:
            return max(0, K - S)
        d1 = self.d1(S, K, T, sigma)
        d2 = self.d2(S, K, T, sigma)
        return K * math.exp(-self.r * T) * self._norm_cdf(-d2) - S * self._norm_cdf(-d1)

    def call_greeks(self, S: float, K: float, T: float, sigma: float) -> OptionGreeks:
        """Compute all Greeks for a call option."""
        if T <= 0 or sigma <= 0:
            intrinsic = max(0, S - K)
            return OptionGreeks(
                delta=1.0 if S > K else 0.0,
                option_price=intrinsic,
            )

        d1 = self.d1(S, K, T, sigma)
        d2 = d1 - sigma * math.sqrt(T)

        delta = self._norm_cdf(d1)
        gamma = self._norm_pdf(d1) / (S * sigma * math.sqrt(T))
        theta = (-(S * self._norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
                 - self.r * K * math.exp(-self.r * T) * self._norm_cdf(d2)) / 365
        vega = S * self._norm_pdf(d1) * math.sqrt(T) / 100
        rho = K * T * math.exp(-self.r * T) * self._norm_cdf(d2) / 100

        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            implied_vol=sigma,
            option_price=self.call_price(S, K, T, sigma),
        )

    def put_greeks(self, S: float, K: float, T: float, sigma: float) -> OptionGreeks:
        """Compute all Greeks for a put option."""
        if T <= 0 or sigma <= 0:
            intrinsic = max(0, K - S)
            return OptionGreeks(
                delta=-1.0 if K > S else 0.0,
                option_price=intrinsic,
            )

        d1 = self.d1(S, K, T, sigma)
        d2 = d1 - sigma * math.sqrt(T)

        delta = self._norm_cdf(d1) - 1
        gamma = self._norm_pdf(d1) / (S * sigma * math.sqrt(T))
        theta = (-(S * self._norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
                 + self.r * K * math.exp(-self.r * T) * self._norm_cdf(-d2)) / 365
        vega = S * self._norm_pdf(d1) * math.sqrt(T) / 100
        rho = -K * T * math.exp(-self.r * T) * self._norm_cdf(-d2) / 100

        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
            implied_vol=sigma,
            option_price=self.put_price(S, K, T, sigma),
        )

    def implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        is_call: bool = True,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> float:
        """Compute implied volatility using Newton-Raphson."""
        sigma = 0.3  # Initial guess

        for _ in range(max_iter):
            if is_call:
                price = self.call_price(S, K, T, sigma)
            else:
                price = self.put_price(S, K, T, sigma)

            diff = price - market_price

            # Vega (sensitivity to sigma)
            d1 = self.d1(S, K, T, sigma)
            vega = S * self._norm_pdf(d1) * math.sqrt(T)

            if abs(vega) < 1e-10:
                break

            sigma -= diff / vega
            sigma = max(0.001, min(5.0, sigma))

            if abs(diff) < tol:
                break

        return sigma

    def aggregate_portfolio_greeks(
        self,
        positions: list[dict],
        underlying_price: float,
    ) -> PortfolioGreeks:
        """Aggregate Greeks across multiple option positions.

        Args:
            positions: List of dicts with keys: type (call/put), S, K, T, sigma, quantity
            underlying_price: Current underlying price for P&L calculation
        """
        net = PortfolioGreeks()

        for pos in positions:
            S = pos.get("S", underlying_price)
            K = pos.get("K", 0)
            T = pos.get("T", 0)
            sigma = pos.get("sigma", 0.3)
            qty = pos.get("quantity", 1)
            opt_type = pos.get("type", "call")

            if opt_type == "call":
                greeks = self.call_greeks(S, K, T, sigma)
            else:
                greeks = self.put_greeks(S, K, T, sigma)

            net.net_delta += greeks.delta * qty
            net.net_gamma += greeks.gamma * qty
            net.net_theta += greeks.theta * qty
            net.net_vega += greeks.vega * qty
            net.net_rho += greeks.rho * qty

        # P&L attribution for 1% move
        move_pct = 0.01
        move_abs = underlying_price * move_pct
        net.delta_pnl_1pct = net.net_delta * move_abs
        net.gamma_pnl_1pct = 0.5 * net.net_gamma * move_abs ** 2
        net.theta_decay_daily = net.net_theta

        return net
