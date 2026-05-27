"""Tests for trade journal."""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone

from tradingbot.monitoring.trade_journal import (
    JournalStats,
    TradeEntry,
    TradeJournal,
)


def make_trade(
    symbol="BTC/USDT",
    side="buy",
    pnl=100.0,
    strategy_id="trend",
    entry_hour=10,
    exit_hour=14,
    base_date=None,
):
    if base_date is None:
        base_date = datetime(2024, 6, 1, tzinfo=timezone.utc)
    entry = base_date.replace(hour=entry_hour)
    exit_time = base_date.replace(hour=exit_hour)
    return TradeEntry(
        id=f"trade_{symbol}_{entry_hour}",
        symbol=symbol,
        side=side,
        entry_price=50000.0,
        exit_price=51000.0 if pnl > 0 else 49000.0,
        quantity=0.1,
        entry_time=entry,
        exit_time=exit_time,
        pnl=pnl,
        pnl_pct=pnl / 5000,
        fees=5.0,
        strategy_id=strategy_id,
    )


class TestTradeJournal:
    @pytest.fixture
    def journal(self):
        return TradeJournal()

    @pytest.fixture
    def sample_trades(self):
        return [
            make_trade(pnl=100, strategy_id="trend", symbol="BTC/USDT", base_date=datetime(2024, 6, 1, tzinfo=timezone.utc)),
            make_trade(pnl=-50, strategy_id="mean_rev", symbol="ETH/USDT", base_date=datetime(2024, 6, 1, tzinfo=timezone.utc)),
            make_trade(pnl=200, strategy_id="trend", symbol="BTC/USDT", base_date=datetime(2024, 6, 2, tzinfo=timezone.utc)),
            make_trade(pnl=-80, strategy_id="trend", symbol="BTC/USDT", base_date=datetime(2024, 6, 2, tzinfo=timezone.utc)),
            make_trade(pnl=0, strategy_id="scalp", symbol="SOL/USDT", base_date=datetime(2024, 6, 3, tzinfo=timezone.utc)),
        ]

    def test_add_trade(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        assert len(journal._trades) == 5

    def test_daily_pnl_tracking(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        daily = journal.get_daily_pnl()
        assert "2024-06-01" in daily
        assert daily["2024-06-01"] == 50  # 100 + (-50)
        assert daily["2024-06-02"] == 120  # 200 + (-80)

    def test_add_tag(self, journal):
        t = make_trade()
        journal.add_trade(t)
        journal.add_tag(t.id, "breakout")
        journal.add_tag(t.id, "high_vol")
        journal.add_tag(t.id, "breakout")  # Duplicate
        assert journal._trades[0].tags == ["breakout", "high_vol"]

    def test_get_stats_basic(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        stats = journal.get_stats()
        assert isinstance(stats, JournalStats)
        assert stats.total_trades == 5
        assert stats.winners == 2
        assert stats.losers == 2
        assert stats.breakeven == 1
        assert stats.win_rate == pytest.approx(2 / 5, abs=0.01)

    def test_stats_pnl(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        stats = journal.get_stats()
        assert stats.largest_win == 200
        assert stats.largest_loss == -80
        assert stats.avg_win == pytest.approx(150, abs=1)  # (100+200)/2
        assert stats.avg_loss == pytest.approx(-65, abs=1)  # (-50+-80)/2

    def test_profit_factor(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        stats = journal.get_stats()
        # Gross profit: 300, Gross loss: 130
        assert stats.profit_factor == pytest.approx(300 / 130, abs=0.1)

    def test_streaks(self, journal):
        # Win, Win, Loss, Loss, Loss, Win
        trades = [
            make_trade(pnl=100, base_date=datetime(2024, 6, 1, tzinfo=timezone.utc)),
            make_trade(pnl=50, base_date=datetime(2024, 6, 2, tzinfo=timezone.utc)),
            make_trade(pnl=-30, base_date=datetime(2024, 6, 3, tzinfo=timezone.utc)),
            make_trade(pnl=-20, base_date=datetime(2024, 6, 4, tzinfo=timezone.utc)),
            make_trade(pnl=-10, base_date=datetime(2024, 6, 5, tzinfo=timezone.utc)),
            make_trade(pnl=40, base_date=datetime(2024, 6, 6, tzinfo=timezone.utc)),
        ]
        for t in trades:
            journal.add_trade(t)
        stats = journal.get_stats()
        assert stats.max_win_streak == 2
        assert stats.max_loss_streak == 3
        assert stats.current_streak == 1  # Last was a win

    def test_strategy_filter(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        stats = journal.get_stats(strategy_id="trend")
        assert stats.total_trades == 3

    def test_symbol_filter(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        stats = journal.get_stats(symbol="BTC/USDT")
        assert stats.total_trades == 3

    def test_session_analysis(self, journal):
        # Asian session (hour 0-7)
        asian_trade = make_trade(entry_hour=3, exit_hour=5, pnl=50)
        # European session (hour 8-15)
        eu_trade = make_trade(entry_hour=10, exit_hour=12, pnl=100)
        # US session (hour 16-23)
        us_trade = make_trade(entry_hour=18, exit_hour=20, pnl=-30)

        for t in [asian_trade, eu_trade, us_trade]:
            journal.add_trade(t)

        analysis = journal.get_session_analysis()
        assert "asian" in analysis
        assert "european" in analysis
        assert "us" in analysis
        assert analysis["asian"].total_trades == 1
        assert analysis["european"].total_trades == 1
        assert analysis["us"].total_trades == 1

    def test_strategy_breakdown(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        breakdown = journal.get_strategy_breakdown()
        assert "trend" in breakdown
        assert "mean_rev" in breakdown
        assert breakdown["trend"].total_trades == 3

    def test_symbol_breakdown(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        breakdown = journal.get_symbol_breakdown()
        assert "BTC/USDT" in breakdown
        assert "ETH/USDT" in breakdown
        assert breakdown["BTC/USDT"].total_trades == 3

    def test_calendar_view(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        cal = journal.get_calendar_view("2024-06")
        assert "2024-06-01" in cal
        assert "+" in cal or "-" in cal

    def test_generate_report(self, journal, sample_trades):
        for t in sample_trades:
            journal.add_trade(t)
        report = journal.generate_report()
        assert "TRADE JOURNAL REPORT" in report
        assert "Win Rate" in report
        assert "Profit Factor" in report

    def test_empty_journal(self, journal):
        stats = journal.get_stats()
        assert stats.total_trades == 0
        report = journal.generate_report()
        assert "TRADE JOURNAL REPORT" in report

    def test_duration_property(self):
        t = make_trade(entry_hour=10, exit_hour=14)
        assert t.duration == timedelta(hours=4)

    def test_is_winner_property(self):
        assert make_trade(pnl=100).is_winner is True
        assert make_trade(pnl=-100).is_winner is False
        assert make_trade(pnl=0).is_winner is False

    def test_session_property(self):
        assert make_trade(entry_hour=3).session == "asian"
        assert make_trade(entry_hour=10).session == "european"
        assert make_trade(entry_hour=18).session == "us"
