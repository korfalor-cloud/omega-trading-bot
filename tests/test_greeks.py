"""Tests for options Greeks calculator."""
from __future__ import annotations

import pytest
import math

from tradingbot.risk.greeks import BlackScholesCalculator, OptionGreeks, PortfolioGreeks


class TestBlackScholesCalculator:
    @pytest.fixture
    def bs(self):
        return BlackScholesCalculator(risk_free_rate=0.04)

    def test_call_price(self, bs):
        price = bs.call_price(S=100, K=100, T=1.0, sigma=0.2)
        assert price > 0
        assert price < 100  # Can't be worth more than underlying

    def test_put_price(self, bs):
        price = bs.put_price(S=100, K=100, T=1.0, sigma=0.2)
        assert price > 0
        assert price < 100

    def test_put_call_parity(self, bs):
        S, K, T, sigma = 100, 100, 1.0, 0.2
        call = bs.call_price(S, K, T, sigma)
        put = bs.put_price(S, K, T, sigma)
        # C - P = S - K * exp(-rT)
        lhs = call - put
        rhs = S - K * math.exp(-bs.r * T)
        assert abs(lhs - rhs) < 0.01

    def test_call_deep_itm(self, bs):
        price = bs.call_price(S=200, K=100, T=1.0, sigma=0.2)
        assert price > 99  # Should be approximately S - K

    def test_call_deep_otm(self, bs):
        price = bs.call_price(S=50, K=100, T=0.1, sigma=0.2)
        assert price < 1  # Should be near zero

    def test_call_greeks(self, bs):
        greeks = bs.call_greeks(S=100, K=100, T=1.0, sigma=0.2)
        assert isinstance(greeks, OptionGreeks)
        assert 0 <= greeks.delta <= 1  # Call delta is 0 to 1
        assert greeks.gamma > 0  # Gamma is always positive
        assert greeks.vega > 0  # Vega is always positive
        assert greeks.theta < 0  # Time decay is negative

    def test_put_greeks(self, bs):
        greeks = bs.put_greeks(S=100, K=100, T=1.0, sigma=0.2)
        assert -1 <= greeks.delta <= 0  # Put delta is -1 to 0
        assert greeks.gamma > 0
        assert greeks.vega > 0

    def test_implied_volatility(self, bs):
        # Price from known vol, then recover it
        true_vol = 0.25
        price = bs.call_price(S=100, K=100, T=1.0, sigma=true_vol)
        recovered_vol = bs.implied_volatility(price, S=100, K=100, T=1.0)
        assert abs(recovered_vol - true_vol) < 0.01

    def test_implied_vol_put(self, bs):
        true_vol = 0.3
        price = bs.put_price(S=100, K=100, T=1.0, sigma=true_vol)
        recovered_vol = bs.implied_volatility(price, S=100, K=100, T=1.0, is_call=False)
        assert abs(recovered_vol - true_vol) < 0.01

    def test_expired_option(self, bs):
        # At expiry, call = max(0, S - K)
        assert bs.call_price(S=110, K=100, T=0, sigma=0.2) == 10
        assert bs.call_price(S=90, K=100, T=0, sigma=0.2) == 0

    def test_portfolio_greeks(self, bs):
        positions = [
            {"type": "call", "S": 100, "K": 100, "T": 0.5, "sigma": 0.2, "quantity": 10},
            {"type": "put", "S": 100, "K": 95, "T": 0.5, "sigma": 0.25, "quantity": -5},
        ]
        portfolio = bs.aggregate_portfolio_greeks(positions, underlying_price=100)
        assert isinstance(portfolio, PortfolioGreeks)
        assert portfolio.net_delta != 0
        assert portfolio.theta_decay_daily != 0
