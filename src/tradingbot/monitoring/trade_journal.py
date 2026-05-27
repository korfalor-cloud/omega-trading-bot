"""Trade Journal — Detailed trade logging and analytics.

Implements:
- Trade tagging and categorization
- P&L attribution by strategy, symbol, time
- Session analysis (Asian, European, US)
- Calendar view of daily P&L
- Win/loss streaks
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeEntry:
    """Detailed trade journal entry."""
    id: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: datetime = field(default_factory=datetime.utcnow)
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    strategy_id: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    setup_type: str = ""  # e.g., "breakout", "mean_reversion"
    market_regime: str = ""
    confidence_at_entry: float = 0.0
    risk_reward_planned: float = 0.0
    risk_reward_actual: float = 0.0

    @property
    def duration(self) -> timedelta:
        return self.exit_time - self.entry_time

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0

    @property
    def session(self) -> str:
        """Determine trading session (Asian/European/US)."""
        hour = self.entry_time.hour
        if 0 <= hour < 8:
            return "asian"
        elif 8 <= hour < 16:
            return "european"
        else:
            return "us"


@dataclass
class JournalStats:
    """Journal statistics."""
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_duration_hours: float = 0.0
    current_streak: int = 0  # Positive = win streak, negative = loss streak
    max_win_streak: int = 0
    max_loss_streak: int = 0
    best_day: float = 0.0
    worst_day: float = 0.0
    profitable_days: int = 0
    losing_days: int = 0


class TradeJournal:
    """Trade journal and analytics.

    Usage:
        journal = TradeJournal()
        journal.add_trade(trade)
        stats = journal.get_stats()
        report = journal.generate_report()
    """

    def __init__(self):
        self._trades: list[TradeEntry] = []
        self._daily_pnl: dict[str, float] = defaultdict(float)

    def add_trade(self, trade: TradeEntry) -> None:
        self._trades.append(trade)
        day_key = trade.exit_time.strftime("%Y-%m-%d")
        self._daily_pnl[day_key] += trade.pnl

    def add_tag(self, trade_id: str, tag: str) -> None:
        for trade in self._trades:
            if trade.id == trade_id:
                if tag not in trade.tags:
                    trade.tags.append(tag)
                break

    def get_stats(
        self,
        strategy_id: str = "",
        symbol: str = "",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> JournalStats:
        """Compute journal statistics with optional filters."""
        filtered = self._filter_trades(strategy_id, symbol, start_date, end_date)

        if not filtered:
            return JournalStats()

        pnls = [t.pnl for t in filtered]
        winners = [t for t in filtered if t.is_winner]
        losers = [t for t in filtered if not t.is_winner and t.pnl != 0]
        breakeven = [t for t in filtered if t.pnl == 0]

        win_pnls = [t.pnl for t in winners]
        loss_pnls = [t.pnl for t in losers]

        win_rate = len(winners) / len(filtered) if filtered else 0
        avg_win = np.mean(win_pnls) if win_pnls else 0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0

        gross_profit = sum(win_pnls)
        gross_loss = abs(sum(loss_pnls))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        streak = 0
        for t in filtered:
            if t.is_winner:
                if streak > 0:
                    streak += 1
                else:
                    streak = 1
                max_win_streak = max(max_win_streak, streak)
            else:
                if streak < 0:
                    streak -= 1
                else:
                    streak = -1
                max_loss_streak = max(max_loss_streak, abs(streak))
        current_streak = streak

        # Duration
        durations = [t.duration.total_seconds() / 3600 for t in filtered]
        avg_duration = np.mean(durations) if durations else 0

        # Daily stats
        daily_values = list(self._daily_pnl.values())
        profitable_days = sum(1 for d in daily_values if d > 0)
        losing_days = sum(1 for d in daily_values if d < 0)

        return JournalStats(
            total_trades=len(filtered),
            winners=len(winners),
            losers=len(losers),
            breakeven=len(breakeven),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max(pnls) if pnls else 0,
            largest_loss=min(pnls) if pnls else 0,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_duration_hours=avg_duration,
            current_streak=current_streak,
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            best_day=max(daily_values) if daily_values else 0,
            worst_day=min(daily_values) if daily_values else 0,
            profitable_days=profitable_days,
            losing_days=losing_days,
        )

    def get_session_analysis(self) -> dict[str, JournalStats]:
        """Analyze performance by trading session."""
        sessions: dict[str, list[TradeEntry]] = defaultdict(list)
        for trade in self._trades:
            sessions[trade.session].append(trade)

        result = {}
        for session, trades in sessions.items():
            journal = TradeJournal()
            journal._trades = trades
            result[session] = journal.get_stats()

        return result

    def get_strategy_breakdown(self) -> dict[str, JournalStats]:
        """Analyze performance by strategy."""
        strategies: dict[str, list[TradeEntry]] = defaultdict(list)
        for trade in self._trades:
            strategies[trade.strategy_id or "unknown"].append(trade)

        result = {}
        for strategy, trades in strategies.items():
            journal = TradeJournal()
            journal._trades = trades
            result[strategy] = journal.get_stats()

        return result

    def get_symbol_breakdown(self) -> dict[str, JournalStats]:
        """Analyze performance by symbol."""
        symbols: dict[str, list[TradeEntry]] = defaultdict(list)
        for trade in self._trades:
            symbols[trade.symbol].append(trade)

        result = {}
        for symbol, trades in symbols.items():
            journal = TradeJournal()
            journal._trades = trades
            result[symbol] = journal.get_stats()

        return result

    def get_daily_pnl(self) -> dict[str, float]:
        return dict(self._daily_pnl)

    def get_calendar_view(self, month: Optional[str] = None) -> str:
        """Generate a calendar view of daily P&L."""
        if not month:
            month = datetime.utcnow().strftime("%Y-%m")

        lines = [f"=== Trade Calendar: {month} ===", ""]
        for day_key in sorted(self._daily_pnl.keys()):
            if day_key.startswith(month):
                pnl = self._daily_pnl[day_key]
                marker = "+" if pnl > 0 else "-" if pnl < 0 else "="
                lines.append(f"  {day_key}  {marker} ${pnl:>10,.2f}")

        return "\n".join(lines)

    def generate_report(self) -> str:
        """Generate full journal report."""
        stats = self.get_stats()

        lines = [
            "=" * 60,
            "TRADE JOURNAL REPORT",
            "=" * 60,
            "",
            f"Total Trades:      {stats.total_trades}",
            f"Winners:           {stats.winners}",
            f"Losers:            {stats.losers}",
            f"Win Rate:          {stats.win_rate:.1%}",
            "",
            f"Avg Win:           ${stats.avg_win:,.2f}",
            f"Avg Loss:          ${stats.avg_loss:,.2f}",
            f"Largest Win:       ${stats.largest_win:,.2f}",
            f"Largest Loss:      ${stats.largest_loss:,.2f}",
            f"Profit Factor:     {stats.profit_factor:.2f}",
            f"Expectancy:        ${stats.expectancy:,.2f}",
            "",
            f"Current Streak:    {stats.current_streak}",
            f"Max Win Streak:    {stats.max_win_streak}",
            f"Max Loss Streak:   {stats.max_loss_streak}",
            "",
            f"Best Day:          ${stats.best_day:,.2f}",
            f"Worst Day:         ${stats.worst_day:,.2f}",
            f"Profitable Days:   {stats.profitable_days}",
            f"Losing Days:       {stats.losing_days}",
            "",
            f"Avg Duration:      {stats.avg_duration_hours:.1f} hours",
            "=" * 60,
        ]
        return "\n".join(lines)

    def _filter_trades(
        self,
        strategy_id: str = "",
        symbol: str = "",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list[TradeEntry]:
        filtered = self._trades
        if strategy_id:
            filtered = [t for t in filtered if t.strategy_id == strategy_id]
        if symbol:
            filtered = [t for t in filtered if t.symbol == symbol]
        if start_date:
            filtered = [t for t in filtered if t.exit_time >= start_date]
        if end_date:
            filtered = [t for t in filtered if t.exit_time <= end_date]
        return filtered
