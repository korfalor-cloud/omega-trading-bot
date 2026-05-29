"""PDF Tearsheet Generation — HTML-based report printable to PDF.

Implements:
- Performance summary (returns, ratios, costs)
- Drawdown analysis (max DD, duration, recovery)
- Trade statistics (win rate, profit factor, expectancy)
- HTML-based report (can be printed to PDF via browser)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Trading Performance Report"
    include_drawdown: bool = True
    include_trade_stats: bool = True
    include_equity_curve: bool = True
    include_monthly_table: bool = True
    dark_mode: bool = False


@dataclass
class DrawdownInfo:
    """Drawdown analysis data."""
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: float = 0.0
    max_drawdown_start: str = ""
    max_drawdown_end: str = ""
    current_drawdown_pct: float = 0.0
    avg_drawdown_pct: float = 0.0
    drawdown_count: int = 0


@dataclass
class TradeStats:
    """Aggregated trade statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    avg_holding_period_hours: float = 0.0
    total_pnl: float = 0.0
    total_fees: float = 0.0


class PDFTearsheet:
    """Generate an HTML tearsheet report that can be printed to PDF.

    Usage:
        report = PDFTearsheet(config)
        report.load_equity(equity_curve)
        report.load_trades(trades)
        html = report.generate()
    """

    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()
        self._equity_curve: list[tuple[datetime, float]] = []
        self._trades: list[dict] = []
        self._daily_returns: list[float] = []
        self._risk_free_rate = 0.04

    def load_equity(self, equity_curve: list[tuple[datetime, float]]) -> None:
        """Load equity curve as list of (timestamp, equity_value)."""
        self._equity_curve = list(equity_curve)

    def load_trades(self, trades: list[dict]) -> None:
        """Load trade records. Each dict should have keys:
        entry_time, exit_time, pnl, pnl_pct, side, symbol, fees (optional)
        """
        self._trades = list(trades)

    def load_daily_returns(self, returns: list[float]) -> None:
        """Load daily return series."""
        self._daily_returns = list(returns)

    def compute_drawdown(self) -> DrawdownInfo:
        """Compute drawdown analysis from equity curve."""
        if not self._equity_curve:
            return DrawdownInfo()

        equities = [e[1] for e in self._equity_curve]
        timestamps = [e[0] for e in self._equity_curve]
        peak = equities[0]
        max_dd = 0.0
        max_dd_start = timestamps[0]
        max_dd_end = timestamps[0]
        dd_start = timestamps[0]
        dd_durations: list[float] = []
        dd_depths: list[float] = []

        for i, (ts, eq) in enumerate(self._equity_curve):
            if eq > peak:
                if max_dd > 0:
                    dd_durations.append((ts - dd_start).total_seconds() / 86400)
                    dd_depths.append(max_dd)
                peak = eq
                dd_start = ts
                max_dd = 0.0

            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_end = ts
                max_dd_start = dd_start

        # Current drawdown
        if equities:
            current_peak = max(equities)
            current_dd = (current_peak - equities[-1]) / current_peak if current_peak > 0 else 0
        else:
            current_dd = 0.0

        return DrawdownInfo(
            max_drawdown_pct=max_dd * 100,
            max_drawdown_duration_days=(max_dd_end - max_dd_start).total_seconds() / 86400 if max_dd > 0 else 0,
            max_drawdown_start=max_dd_start.strftime("%Y-%m-%d") if isinstance(max_dd_start, datetime) else "",
            max_drawdown_end=max_dd_end.strftime("%Y-%m-%d") if isinstance(max_dd_end, datetime) else "",
            current_drawdown_pct=current_dd * 100,
            avg_drawdown_pct=float(np.mean(dd_depths)) * 100 if dd_depths else 0,
            drawdown_count=len(dd_depths),
        )

    def compute_trade_stats(self) -> TradeStats:
        """Compute trade statistics."""
        if not self._trades:
            return TradeStats()

        pnls = [t.get("pnl", 0) for t in self._trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0

        win_rate = len(winners) / len(pnls) if pnls else 0
        avg_win = float(np.mean(winners)) if winners else 0
        avg_loss = float(np.mean(losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        # Holding periods
        durations = []
        for t in self._trades:
            entry = t.get("entry_time")
            exit_ = t.get("exit_time")
            if isinstance(entry, datetime) and isinstance(exit_, datetime):
                durations.append((exit_ - entry).total_seconds() / 3600)

        total_fees = sum(t.get("fees", 0) for t in self._trades)

        return TradeStats(
            total_trades=len(pnls),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max(pnls) if pnls else 0,
            largest_loss=min(pnls) if pnls else 0,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_holding_period_hours=float(np.mean(durations)) if durations else 0,
            total_pnl=sum(pnls),
            total_fees=total_fees,
        )

    def _compute_performance_summary(self) -> dict:
        """Compute top-level performance metrics."""
        if not self._equity_curve:
            return {}

        equities = [e[1] for e in self._equity_curve]
        total_return = (equities[-1] - equities[0]) / equities[0] if equities[0] > 0 else 0

        # Sharpe / Sortino from daily returns
        sharpe = 0.0
        sortino = 0.0
        if self._daily_returns:
            rets = np.array(self._daily_returns)
            excess = rets - self._risk_free_rate / 365
            std = np.std(excess)
            if std > 0:
                sharpe = float(np.mean(excess) / std * np.sqrt(365))
            downside = rets[rets < 0]
            down_std = np.std(downside) if len(downside) > 0 else 0
            if down_std > 0:
                sortino = float(np.mean(excess) / down_std * np.sqrt(365))

        # Annualised return
        if self._equity_curve:
            days = (self._equity_curve[-1][0] - self._equity_curve[0][0]).total_seconds() / 86400
            ann_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if days > 0 else 0
        else:
            ann_return = 0

        return {
            "total_return_pct": total_return * 100,
            "annualised_return_pct": ann_return * 100,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "start_equity": equities[0],
            "end_equity": equities[-1],
            "peak_equity": max(equities),
            "trading_days": len(self._equity_curve),
        }

    def generate(self) -> str:
        """Generate the full HTML tearsheet report."""
        summary = self._compute_performance_summary()
        dd = self.compute_drawdown()
        ts = self.compute_trade_stats()

        bg = "#1a1a2e" if self.config.dark_mode else "#ffffff"
        fg = "#e0e0e0" if self.config.dark_mode else "#333333"
        card_bg = "#16213e" if self.config.dark_mode else "#f8f9fa"
        accent = "#0f3460" if self.config.dark_mode else "#0066cc"

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='utf-8'>",
            f"<title>{self.config.title}</title>",
            "<style>",
            f"body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; "
            f"background: {bg}; color: {fg}; margin: 20px; font-size: 14px; }}",
            f"h1 {{ color: {accent}; border-bottom: 2px solid {accent}; padding-bottom: 10px; }}",
            f"h2 {{ color: {accent}; margin-top: 30px; }}",
            ".card { background: " + card_bg + "; border-radius: 8px; padding: 16px; "
            "margin: 10px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
            ".grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }",
            ".metric-label { font-size: 12px; opacity: 0.7; text-transform: uppercase; }",
            ".metric-value { font-size: 22px; font-weight: 600; margin-top: 4px; }",
            ".positive { color: #28a745; } .negative { color: #dc3545; }",
            "table { width: 100%; border-collapse: collapse; margin: 10px 0; }",
            "th, td { padding: 8px 12px; text-align: right; border-bottom: 1px solid "
            + ("#333" if self.config.dark_mode else "#dee2e6") + "; }",
            "th { text-align: left; font-weight: 600; }",
            "</style></head><body>",
            f"<h1>{self.config.title}</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
        ]

        # ── Performance Summary ───────────────────────────────────
        html_parts.append("<h2>Performance Summary</h2><div class='grid'>")
        for label, key, fmt in [
            ("Total Return", "total_return_pct", "{:.2f}%"),
            ("Annualised Return", "annualised_return_pct", "{:.2f}%"),
            ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
            ("Sortino Ratio", "sortino_ratio", "{:.2f}"),
            ("Start Equity", "start_equity", "${:,.2f}"),
            ("End Equity", "end_equity", "${:,.2f}"),
            ("Peak Equity", "peak_equity", "${:,.2f}"),
            ("Trading Days", "trading_days", "{:.0f}"),
        ]:
            val = summary.get(key, 0)
            css = ""
            if "Return" in label or "Ratio" in label:
                css = " positive" if val > 0 else " negative"
            html_parts.append(
                f"<div class='card'><div class='metric-label'>{label}</div>"
                f"<div class='metric-value{css}'>{fmt.format(val)}</div></div>"
            )
        html_parts.append("</div>")

        # ── Drawdown Analysis ─────────────────────────────────────
        if self.config.include_drawdown:
            html_parts.append("<h2>Drawdown Analysis</h2><div class='grid'>")
            for label, val in [
                ("Max Drawdown", f"{dd.max_drawdown_pct:.2f}%"),
                ("Max DD Duration", f"{dd.max_drawdown_duration_days:.0f} days"),
                ("Max DD Start", dd.max_drawdown_start),
                ("Max DD End", dd.max_drawdown_end),
                ("Current Drawdown", f"{dd.current_drawdown_pct:.2f}%"),
                ("Avg Drawdown", f"{dd.avg_drawdown_pct:.2f}%"),
                ("DD Count", str(dd.drawdown_count)),
            ]:
                html_parts.append(
                    f"<div class='card'><div class='metric-label'>{label}</div>"
                    f"<div class='metric-value'>{val}</div></div>"
                )
            html_parts.append("</div>")

        # ── Trade Statistics ──────────────────────────────────────
        if self.config.include_trade_stats:
            html_parts.append("<h2>Trade Statistics</h2>")
            html_parts.append("<table>")
            for label, val in [
                ("Total Trades", f"{ts.total_trades}"),
                ("Winning Trades", f"{ts.winning_trades}"),
                ("Losing Trades", f"{ts.losing_trades}"),
                ("Win Rate", f"{ts.win_rate:.1%}"),
                ("Average Win", f"${ts.avg_win:,.2f}"),
                ("Average Loss", f"${ts.avg_loss:,.2f}"),
                ("Largest Win", f"${ts.largest_win:,.2f}"),
                ("Largest Loss", f"${ts.largest_loss:,.2f}"),
                ("Profit Factor", f"{ts.profit_factor:.2f}"),
                ("Expectancy", f"${ts.expectancy:,.2f}"),
                ("Avg Holding Period", f"{ts.avg_holding_period_hours:.1f} hours"),
                ("Total P&L", f"${ts.total_pnl:,.2f}"),
                ("Total Fees", f"${ts.total_fees:,.2f}"),
            ]:
                html_parts.append(f"<tr><th>{label}</th><td>{val}</td></tr>")
            html_parts.append("</table>")

        # ── Equity Curve (ASCII sparkline as placeholder) ──────────
        if self.config.include_equity_curve and len(self._equity_curve) > 1:
            html_parts.append("<h2>Equity Curve</h2>")
            equities = [e[1] for e in self._equity_curve]
            eq_min, eq_max = min(equities), max(equities)
            eq_range = eq_max - eq_min if eq_max > eq_min else 1
            n_bars = min(100, len(equities))
            step = max(1, len(equities) // n_bars)

            html_parts.append("<div style='display:flex;align-items:flex-end;height:120px;gap:1px;'>")
            for i in range(0, len(equities), step):
                val = equities[i]
                h = max(2, int((val - eq_min) / eq_range * 100))
                color = "#28a745" if val >= equities[0] else "#dc3545"
                html_parts.append(
                    f"<div style='width:3px;height:{h}px;background:{color};border-radius:1px;'></div>"
                )
            html_parts.append("</div>")
            html_parts.append(
                f"<p style='font-size:12px;opacity:0.6;'>"
                f"${eq_min:,.0f} to ${eq_max:,.0f} ({n_bars} bars)</p>"
            )

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    def save(self, filepath: str) -> None:
        """Save the HTML report to a file."""
        html = self.generate()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Report saved to %s", filepath)
