"""Tests for CalendarView — monthly P&L calendar."""
from __future__ import annotations

import pytest

from tradingbot.monitoring.calendar_view import CalendarView


class TestCalendarView:
    @pytest.fixture
    def calendar(self):
        c = CalendarView()
        c.add_trade("2024-01-01", 100.0)
        c.add_trade("2024-01-01", 50.0)
        c.add_trade("2024-01-02", -200.0)
        c.add_trade("2024-01-03", 150.0)
        c.add_trade("2024-01-04", -30.0)
        c.add_trade("2024-01-05", 80.0)
        c.add_trade("2024-02-01", 500.0)
        return c

    def test_add_trade(self, calendar):
        calendar.add_trade("2024-03-01", 100.0)
        assert calendar._daily_pnl["2024-03-01"] == 100.0

    def test_add_trade_accumulates(self, calendar):
        # Two trades on same day should sum
        assert calendar._daily_pnl["2024-01-01"] == 150.0
        assert calendar._daily_trades["2024-01-01"] == 2

    def test_get_monthly_pnl(self, calendar):
        monthly = calendar.get_monthly_pnl(2024, 1)
        assert "2024-01-01" in monthly
        assert "2024-01-02" in monthly
        assert "2024-02-01" not in monthly

    def test_get_monthly_total(self, calendar):
        total = calendar.get_monthly_total(2024, 1)
        assert total == 150.0 + (-200.0) + 150.0 + (-30.0) + 80.0

    def test_get_monthly_total_empty(self, calendar):
        total = calendar.get_monthly_total(2024, 6)
        assert total == 0.0

    def test_get_yearly_pnl(self, calendar):
        yearly = calendar.get_yearly_pnl(2024)
        assert "2024-01" in yearly
        assert "2024-02" in yearly
        assert len(yearly) == 2

    def test_format_calendar(self, calendar):
        text = calendar.format_calendar(2024, 1)
        assert "2024-01" in text
        assert "Calendar" in text
        assert "Total" in text

    def test_get_streaks(self, calendar):
        streaks = calendar.get_streaks()
        assert "max_win_streak" in streaks
        assert "max_loss_streak" in streaks
        assert "current_streak" in streaks
        assert "best_day" in streaks
        assert "worst_day" in streaks
        assert streaks["best_day"] == 500.0
        assert streaks["worst_day"] == -200.0

    def test_empty_streaks(self):
        c = CalendarView()
        streaks = c.get_streaks()
        assert streaks["max_win_streak"] == 0
        assert streaks["max_loss_streak"] == 0

    def test_streaks_winning_run(self):
        c = CalendarView()
        for i in range(5):
            c.add_trade(f"2024-01-{i+1:02d}", 100.0)
        streaks = c.get_streaks()
        assert streaks["max_win_streak"] == 5
        assert streaks["current_streak"] == 5
