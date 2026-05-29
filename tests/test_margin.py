"""Tests for margin calculator."""
from __future__ import annotations

import pytest

from tradingbot.risk.margin import MarginCalculator, MarginResult


class TestMarginCalculator:
    @pytest.fixture
    def calc(self):
        return MarginCalculator(config={"initial_margin_rate": 0.10, "maintenance_margin_rate": 0.05, "max_leverage": 10})

    def test_calculate_margin(self, calc):
        positions = {"BTC": 0.5}
        prices = {"BTC": 60000}
        result = calc.calculate_margin(positions, prices, 100000)
        assert isinstance(result, MarginResult)
        assert result.initial_margin == 3000  # 30000 * 0.10
        assert result.leverage == 0.3

    def test_available_margin(self, calc):
        positions = {"BTC": 0.5}
        prices = {"BTC": 60000}
        result = calc.calculate_margin(positions, prices, 100000)
        assert result.available_margin == 97000

    def test_margin_utilization(self, calc):
        positions = {"BTC": 0.5}
        prices = {"BTC": 60000}
        result = calc.calculate_margin(positions, prices, 100000)
        assert result.margin_utilization == pytest.approx(0.03, abs=0.01)

    def test_liquidation_price_long(self, calc):
        # Leveraged: notional=600000, equity=50000
        liq = calc.liquidation_price(60000, 10, "buy", 50000)
        assert liq < 60000
        assert liq > 0

    def test_liquidation_price_short(self, calc):
        liq = calc.liquidation_price(60000, 10, "sell", 50000)
        assert liq > 60000

    def test_margin_call(self, calc):
        positions = {"BTC": 1.0}
        prices = {"BTC": 60000}
        # equity=1000, maintenance=60000*0.05=3000
        assert calc.check_margin_call(positions, prices, 1000) is True
        assert calc.check_margin_call(positions, prices, 100000) is False

    def test_max_position(self, calc):
        max_qty = calc.max_position_size(60000, 100000)
        assert max_qty > 0
        assert max_qty == pytest.approx(100000 * 10 / 60000, abs=0.01)

    def test_empty_positions(self, calc):
        result = calc.calculate_margin({}, {}, 100000)
        assert result.initial_margin == 0
        assert result.leverage == 0
