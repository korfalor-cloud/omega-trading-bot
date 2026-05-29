"""Backtesting Report Generator — HTML tearsheet.

Implements:
- Performance summary
- Equity curve chart (ASCII)
- Drawdown analysis
- Trade distribution
- Monthly returns
- Strategy comparison
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ReportData:
    """Data for report generation."""
    equity_curve: np.ndarray = None
    returns: np.ndarray = None
    trades: list = None
    strategy_name: str = ""
    start_date: str = ""
    end_date: str = ""

    def __post_init__(self):
        if self.equity_curve is None:
            self.equity_curve = np.array([])
        if self.returns is None:
            self.returns = np.array([])
        if self.trades is None:
            self.trades = []


class ReportGenerator:
    """Generate backtesting performance reports."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.annualization = config.get("annualization", 365)

    def generate_html(self, data: ReportData) -> str:
        """Generate HTML report."""
        metrics = self._compute_metrics(data)
        equity_chart = self._ascii_chart(data.equity_curve)
        dd_chart = self._ascii_drawdown(data.equity_curve)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Omega Trading Bot — Backtest Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #e0e0e0; }}
        h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
        h2 {{ color: #00d4ff; margin-top: 30px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: #16213e; padding: 15px; border-radius: 8px; border: 1px solid #0f3460; }}
        .metric-label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #00d4ff; }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        pre {{ background: #0d1117; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 12px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ color: #00d4ff; }}
    </style>
</head>
<body>
    <h1>📊 Omega Trading Bot — Backtest Report</h1>
    <p>Strategy: <strong>{data.strategy_name}</strong> | {data.start_date} to {data.end_date}</p>

    <h2>Performance Metrics</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'positive' if metrics['total_return'] >= 0 else 'negative'}">{metrics['total_return']:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{metrics['sharpe']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sortino Ratio</div>
            <div class="metric-value">{metrics['sortino']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{metrics['max_drawdown']:.2%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{metrics['win_rate']:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">{metrics['profit_factor']:.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">{metrics['total_trades']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Trade</div>
            <div class="metric-value {'positive' if metrics['avg_trade'] >= 0 else 'negative'}">{metrics['avg_trade']:.2f}</div>
        </div>
    </div>

    <h2>Equity Curve</h2>
    <pre>{equity_chart}</pre>

    <h2>Drawdown</h2>
    <pre>{dd_chart}</pre>

    <h2>Trade Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Avg Win</td><td class="positive">{metrics['avg_win']:.2f}</td></tr>
        <tr><td>Avg Loss</td><td class="negative">{metrics['avg_loss']:.2f}</td></tr>
        <tr><td>Largest Win</td><td class="positive">{metrics['largest_win']:.2f}</td></tr>
        <tr><td>Largest Loss</td><td class="negative">{metrics['largest_loss']:.2f}</td></tr>
        <tr><td>Avg Duration</td><td>{metrics['avg_duration']:.1f} bars</td></tr>
        <tr><td>Calmar Ratio</td><td>{metrics['calmar']:.2f}</td></tr>
    </table>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #333; color: #666;">
        Generated by Omega Trading Bot | {datetime.utcnow().isoformat()}
    </footer>
</body>
</html>"""
        return html

    def _compute_metrics(self, data: ReportData) -> dict:
        """Compute all performance metrics."""
        equity = data.equity_curve
        returns = data.returns
        trades = data.trades

        if len(equity) < 2:
            return {k: 0 for k in [
                "total_return", "sharpe", "sortino", "max_drawdown",
                "win_rate", "profit_factor", "total_trades", "avg_trade",
                "avg_win", "avg_loss", "largest_win", "largest_loss",
                "avg_duration", "calmar",
            ]}

        total_return = (equity[-1] / equity[0]) - 1
        mean_ret = np.mean(returns) if len(returns) > 0 else 0
        std_ret = np.std(returns) if len(returns) > 0 else 1
        down_std = np.std(returns[returns < 0]) if np.any(returns < 0) else std_ret

        sharpe = mean_ret / std_ret * np.sqrt(self.annualization) if std_ret > 0 else 0
        sortino = mean_ret / down_std * np.sqrt(self.annualization) if down_std > 0 else 0

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak
        max_dd = float(np.max(dd))

        calmar = total_return / max_dd if max_dd > 0 else 0

        # Trade stats
        pnls = [t.get("pnl", 0) for t in trades] if trades else []
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        win_rate = len(wins) / len(pnls) if pnls else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0
        avg_trade = np.mean(pnls) if pnls else 0

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": len(pnls),
            "avg_trade": avg_trade,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": min(losses) if losses else 0,
            "avg_duration": np.mean([t.get("duration", 0) for t in trades]) if trades else 0,
            "calmar": calmar,
        }

    def _ascii_chart(self, data: np.ndarray, width: int = 80, height: int = 20) -> str:
        """Generate ASCII chart."""
        if len(data) < 2:
            return "No data"

        # Downsample
        step = max(1, len(data) // width)
        sampled = data[::step][:width]

        mn, mx = np.min(sampled), np.max(sampled)
        rng = mx - mn if mx != mn else 1

        lines = []
        for row in range(height - 1, -1, -1):
            threshold = mn + rng * row / (height - 1)
            line = ""
            for val in sampled:
                if val >= threshold:
                    line += "█"
                else:
                    line += " "
            label = f"{threshold:>10.0f}" if row % 4 == 0 else " " * 10
            lines.append(f"{label} │{line}")

        lines.append(" " * 10 + "└" + "─" * len(sampled))
        return "\n".join(lines)

    def _ascii_drawdown(self, equity: np.ndarray, width: int = 80, height: int = 10) -> str:
        """Generate ASCII drawdown chart."""
        if len(equity) < 2:
            return "No data"

        peak = np.maximum.accumulate(equity)
        dd = (peak - equity) / peak * 100  # Percentage

        step = max(1, len(dd) // width)
        sampled = dd[::step][:width]

        lines = []
        for row in range(height):
            threshold = (row + 1) * np.max(sampled) / height if np.max(sampled) > 0 else 0
            line = ""
            for val in sampled:
                if val >= threshold:
                    line += "▓"
                else:
                    line += " "
            label = f"{threshold:>8.1f}%" if row % 3 == 0 else " " * 8
            lines.append(f"{label} │{line}")

        lines.append(" " * 8 + "└" + "─" * len(sampled))
        return "\n".join(lines)

    def save_report(self, data: ReportData, path: str = "reports/backtest.html") -> str:
        """Generate and save report."""
        html = self.generate_html(data)
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(html)
        return path
