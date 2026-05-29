"""Calendar View — monthly P&L calendar.

Implements:
- Monthly P&L heatmap
- Daily P&L calendar
- Win/loss streaks
- Best/worst periods
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


class CalendarView:
    """P&L calendar visualization."""

    def __init__(self):
        self._daily_pnl: dict[str, float] = defaultdict(float)
        self._daily_trades: dict[str, int] = defaultdict(int)

    def add_trade(self, date_str: str, pnl: float) -> None:
        """Add a trade P&L for a date (YYYY-MM-DD)."""
        self._daily_pnl[date_str] += pnl
        self._daily_trades[date_str] += 1

    def get_monthly_pnl(self, year: int, month: int) -> dict[str, float]:
        """Get daily P&L for a month."""
        prefix = f"{year}-{month:02d}"
        return {k: v for k, v in self._daily_pnl.items() if k.startswith(prefix)}

    def get_monthly_total(self, year: int, month: int) -> float:
        return sum(self.get_monthly_pnl(year, month).values())

    def get_yearly_pnl(self, year: int) -> dict[str, float]:
        """Get monthly P&L for a year."""
        result = {}
        for month in range(1, 13):
            total = self.get_monthly_total(year, month)
            if total != 0:
                result[f"{year}-{month:02d}"] = total
        return result

    def format_calendar(self, year: int, month: int) -> str:
        """Format calendar as text."""
        daily = self.get_monthly_pnl(year, month)
        lines = [f"\n{'='*50}", f"  {year}-{month:02d} Calendar", f"{'='*50}"]

        for day in range(1, 32):
            date = f"{year}-{month:02d}-{day:02d}"
            pnl = daily.get(date, 0)
            trades = self._daily_trades.get(date, 0)
            if pnl != 0:
                emoji = "🟢" if pnl > 0 else "🔴"
                lines.append(f"  {date} {emoji} {pnl:>+10.2f} ({trades} trades)")

        total = sum(daily.values())
        emoji = "🟢" if total > 0 else "🔴"
        lines.append(f"  {'─'*40}")
        lines.append(f"  Total {emoji} {total:>+10.2f}")
        return "\n".join(lines)

    def get_streaks(self) -> dict:
        """Get winning/losing streaks."""
        dates = sorted(self._daily_pnl.keys())
        if not dates:
            return {"max_win_streak": 0, "max_loss_streak": 0, "current_streak": 0}

        win_streak = 0
        loss_streak = 0
        max_win = 0
        max_loss = 0

        for date in dates:
            if self._daily_pnl[date] > 0:
                win_streak += 1
                loss_streak = 0
                max_win = max(max_win, win_streak)
            elif self._daily_pnl[date] < 0:
                loss_streak += 1
                win_streak = 0
                max_loss = max(max_loss, loss_streak)
            else:
                win_streak = 0
                loss_streak = 0

        current = win_streak if win_streak > 0 else -loss_streak

        return {
            "max_win_streak": max_win,
            "max_loss_streak": max_loss,
            "current_streak": current,
            "best_day": max(self._daily_pnl.values()) if self._daily_pnl else 0,
            "worst_day": min(self._daily_pnl.values()) if self._daily_pnl else 0,
        }
