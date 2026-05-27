"""Tests for volatility models."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.risk.volatility import VolatilityModel, VolForecast


class TestVolatilityModel:
    @pytest.fixture
    def model(self):
        return VolatilityModel()

    @pytest.fixture
    def returns(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal(300) * 0.02

    @pytest.fixture
    def ohlc_data(self):
        rng = np.random.default_rng(42)
        n = 300
        closes = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.02))
        opens = closes * (1 + rng.standard_normal(n) * 0.005)
        highs = np.maximum(opens, closes) * (1 + abs(rng.standard_normal(n)) * 0.01)
        lows = np.minimum(opens, closes) * (1 - abs(rng.standard_normal(n)) * 0.01)
        return opens, highs, lows, closes

    def test_ewma_shape(self, model, returns):
        vols = model.ewma(returns)
        assert len(vols) == len(returns)
        assert vols[0] >= 0

    def test_ewma_positive(self, model, returns):
        vols = model.ewma(returns)
        assert all(v >= 0 for v in vols)

    def test_garch_shape(self, model, returns):
        vols = model.garch(returns)
        assert len(vols) == len(returns)
        assert all(v >= 0 for v in vols)

    def test_garch_forecast(self, model, returns):
        forecast = model.garch_forecast(returns, horizon=1)
        assert isinstance(forecast, VolForecast)
        assert forecast.current_vol > 0
        assert forecast.forecast_vol > 0
        assert forecast.model == "garch_1_1"

    def test_garch_multi_step(self, model, returns):
        f1 = model.garch_forecast(returns, horizon=1)
        f5 = model.garch_forecast(returns, horizon=5)
        # Both should be positive
        assert f1.forecast_vol > 0
        assert f5.forecast_vol > 0

    def test_realized_volatility(self, model, returns):
        rv = model.realized_volatility(returns, window=20)
        assert len(rv) == len(returns)
        assert np.isnan(rv[0])
        assert not np.isnan(rv[20])
        assert rv[20] > 0

    def test_parkinson_volatility(self, model, ohlc_data):
        opens, highs, lows, closes = ohlc_data
        pv = model.parkinson_volatility(highs, lows, window=20)
        assert len(pv) == len(highs)
        assert not np.isnan(pv[20])
        assert pv[20] > 0

    def test_garman_klass(self, model, ohlc_data):
        opens, highs, lows, closes = ohlc_data
        gk = model.garman_klass_volatility(opens, highs, lows, closes, window=20)
        assert len(gk) == len(opens)
        assert not np.isnan(gk[20])

    def test_yang_zhang(self, model, ohlc_data):
        opens, highs, lows, closes = ohlc_data
        yz = model.yang_zhang_volatility(opens, highs, lows, closes, window=20)
        assert len(yz) == len(opens)
        assert not np.isnan(yz[20])

    def test_volatility_cone(self, model, returns):
        cone = model.volatility_cone(returns, windows=[10, 20], percentiles=[0.25, 0.5, 0.75])
        assert "10d" in cone
        assert "20d" in cone
        assert 0.5 in cone["10d"]
        assert cone["10d"][0.25] <= cone["10d"][0.5] <= cone["10d"][0.75]

    def test_vol_of_vol(self, model, returns):
        vov = model.vol_of_vol(returns, window=20)
        assert vov >= 0

    def test_vol_regime_thresholds(self, model, returns):
        low, high = model.vol_regime_thresholds(returns, lookback=200)
        assert low < high
        assert low >= 0

    def test_custom_config(self):
        model = VolatilityModel(config={
            "ewma_span": 10,
            "garch_alpha": 0.05,
            "garch_beta": 0.90,
            "annualization": 252,
        })
        assert model.ewma_span == 10
        assert model.annualization == 252
