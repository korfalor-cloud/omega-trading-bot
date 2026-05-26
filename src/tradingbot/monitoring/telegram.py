"""Telegram Bot — Real-time alerts, P&L reports, kill switch."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import httpx

from ..core.types import PortfolioState, RiskAlert

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Telegram bot for trading system notifications.

    Features:
    - Trade fill alerts
    - Risk warnings
    - Daily P&L summary
    - Kill switch command
    - Strategy performance reports
    """

    def __init__(self, config: dict):
        self.bot_token = config.get("telegram_bot_token", "")
        self.chat_id = config.get("telegram_chat_id", "")
        self.enabled = bool(self.bot_token and self.chat_id)
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self) -> None:
        if self.enabled:
            self._client = httpx.AsyncClient(timeout=10)
            logger.info("Telegram notifier started")

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    async def send_message(self, text: str) -> bool:
        """Send a message to the configured chat."""
        if not self.enabled or not self._client:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            resp = await self._client.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            })
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    async def notify_trade(self, symbol: str, side: str, quantity: float,
                          price: float, strategy_id: str) -> None:
        """Notify about a trade fill."""
        emoji = "🟢" if side == "buy" else "🔴"
        msg = (
            f"{emoji} *Trade Executed*\n"
            f"Symbol: `{symbol}`\n"
            f"Side: `{side.upper()}`\n"
            f"Qty: `{quantity:.4f}`\n"
            f"Price: `${price:,.2f}`\n"
            f"Strategy: `{strategy_id[:8]}`"
        )
        await self.send_message(msg)

    async def notify_risk_alert(self, alert: RiskAlert) -> None:
        """Notify about a risk alert."""
        emoji = "⚠️" if alert.level == "warning" else "🚨"
        msg = (
            f"{emoji} *Risk Alert: {alert.level.upper()}*\n"
            f"{alert.message}\n"
            f"Metric: `{alert.metric}`\n"
            f"Value: `{alert.current_value:.4f}`\n"
            f"Threshold: `{alert.threshold:.4f}`"
        )
        await self.send_message(msg)

    async def notify_daily_summary(self, portfolio: PortfolioState) -> None:
        """Send daily P&L summary."""
        pnl_emoji = "📈" if portfolio.unrealized_pnl >= 0 else "📉"
        msg = (
            f"📊 *Daily Summary*\n"
            f"Equity: `${portfolio.total_equity:,.2f}`\n"
            f"Unrealized P&L: {pnl_emoji} `${portfolio.unrealized_pnl:,.2f}`\n"
            f"Realized P&L: `${portfolio.realized_pnl:,.2f}`\n"
            f"Drawdown: `{portfolio.current_drawdown:.1%}`\n"
            f"Sharpe: `{portfolio.sharpe_ratio:.2f}`\n"
            f"Positions: `{len(portfolio.positions)}`"
        )
        await self.send_message(msg)

    async def notify_evolution(self, generation: int, best_fitness: float,
                              avg_fitness: float, population_size: int) -> None:
        """Notify about evolution progress."""
        msg = (
            f"🧬 *Evolution Update*\n"
            f"Generation: `{generation}`\n"
            f"Best Fitness: `{best_fitness:.4f}`\n"
            f"Avg Fitness: `{avg_fitness:.4f}`\n"
            f"Population: `{population_size}`"
        )
        await self.send_message(msg)

    async def notify_regime_change(self, old_regime: str, new_regime: str,
                                   confidence: float) -> None:
        """Notify about regime change."""
        msg = (
            f"🔄 *Regime Change*\n"
            f"From: `{old_regime}`\n"
            f"To: `{new_regime}`\n"
            f"Confidence: `{confidence:.1%}`"
        )
        await self.send_message(msg)
