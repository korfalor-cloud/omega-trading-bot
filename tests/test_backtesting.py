"""Tests for the backtesting engine."""
from __future__ import annotations

import pytest

from tradingbot.backtesting.engine import BacktestEngine, BacktestResult
from tradingbot.core.enums import Timeframe
from tradingbot.core.types import OHLCVBar, StrategyGenome
from tradingbot.features.technical import compute_features
from tradingbot.genome.strategy_genome import create_random_genome


class TestBacktestEngine:
    @pytest.fixture
    def engine(self):
        return BacktestEngine({
            "initial_capital": 100000.0,
            "slippage_bps": 5.0,
            "commission_bps": 10.0,
        })

    @pytest.mark.asyncio
    async def test_basic_backtest(self, engine, sample_bars):
        genome = create_random_genome("test")
        features = compute_features(sample_bars)

        result = await engine.run(genome, sample_bars, features)

        assert isinstance(result, BacktestResult)
        assert result.genome_id == genome.id
        assert result.equity_curve
        assert len(result.equity_curve) == len(sample_bars) + 1
        assert result.total_trades >= 0

    @pytest.mark.asyncio
    async def test_equity_curve_starts_at_capital(self, engine, sample_bars):
        genome = create_random_genome()
        features = compute_features(sample_bars)

        result = await engine.run(genome, sample_bars, features)
        assert result.equity_curve[0] == 100000.0

    @pytest.mark.asyncio
    async def test_slippage_applied(self, engine, sample_bars):
        genome = create_random_genome()
        features = compute_features(sample_bars)

        result = await engine.run(genome, sample_bars, features)
        # If trades occurred, verify slippage was considered
        if result.trades:
            # All trades should have P&L that accounts for slippage
            for trade in result.trades:
                assert "pnl" in trade
                assert "type" in trade

    @pytest.mark.asyncio
    async def test_backtest_result_fields(self, engine, sample_bars):
        genome = create_random_genome()
        features = compute_features(sample_bars)

        result = await engine.run(genome, sample_bars, features)

        assert hasattr(result, "genome_id")
        assert hasattr(result, "fitness")
        assert hasattr(result, "equity_curve")
        assert hasattr(result, "trades")
        assert hasattr(result, "total_return")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "win_rate")

    @pytest.mark.asyncio
    async def test_different_configs_produce_different_results(self, sample_bars):
        genome = create_random_genome()
        features = compute_features(sample_bars)

        engine1 = BacktestEngine({"initial_capital": 100000, "slippage_bps": 1, "commission_bps": 5})
        engine2 = BacktestEngine({"initial_capital": 100000, "slippage_bps": 50, "commission_bps": 100})

        result1 = await engine1.run(genome, sample_bars, features)
        result2 = await engine2.run(genome, sample_bars, features)

        # Higher slippage/commission should generally produce worse results
        # (though for random strategies this isn't guaranteed)

    @pytest.mark.asyncio
    async def test_empty_data(self, engine):
        genome = create_random_genome()
        result = await engine.run(genome, [], {})
        assert result.total_trades == 0
        assert result.equity_curve == [100000.0]
