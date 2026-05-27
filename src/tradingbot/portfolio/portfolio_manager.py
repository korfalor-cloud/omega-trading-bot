"""Portfolio Manager — Track positions, P&L, and portfolio analytics.

Maintains the real-time state of all open positions and computes
portfolio-level metrics for risk management and reporting.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from ..core.enums import Side
from ..core.types import Fill, PortfolioState, Position

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Tracks positions and computes portfolio analytics.

    Responsibilities:
    - Update positions on fills
    - Track realized/unrealized P&L
    - Compute portfolio-level metrics (exposure, drawdown, Sharpe)
    - Provide position queries for risk checks
    """

    def __init__(self, initial_cash: float = 100_000.0):
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: dict[str, Position] = {}  # key: "symbol:strategy_id"
        self._closed_trades: list[dict] = []
        self._equity_curve: list[tuple[datetime, float]] = []
        self._peak_equity = initial_cash
        self._realized_pnl = 0.0
        self._total_commission = 0.0

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> list[Position]:
        return list(self._positions.values())

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def total_commission(self) -> float:
        return self._total_commission

    def get_position(self, symbol: str, strategy_id: str) -> Optional[Position]:
        key = f"{symbol}:{strategy_id}"
        return self._positions.get(key)

    def get_positions_for_strategy(self, strategy_id: str) -> list[Position]:
        return [p for p in self._positions.values() if p.strategy_id == strategy_id]

    def get_positions_for_symbol(self, symbol: str) -> list[Position]:
        return [p for p in self._positions.values() if p.symbol == symbol]

    def apply_fill(self, fill: Fill, strategy_id: str = "") -> None:
        """Update positions based on a fill."""
        key = f"{fill.symbol}:{strategy_id}"
        self._total_commission += fill.commission

        if key in self._positions:
            pos = self._positions[key]
            if pos.side == fill.side:
                # Adding to position
                total_qty = pos.quantity + fill.quantity
                pos.avg_entry_price = (
                    (pos.avg_entry_price * pos.quantity + fill.price * fill.quantity) / total_qty
                )
                pos.quantity = total_qty
            else:
                # Reducing or closing position
                if fill.quantity >= pos.quantity:
                    # Closing position
                    pnl = self._calc_pnl(pos, fill.price, pos.quantity)
                    self._realized_pnl += pnl - fill.commission
                    self._cash += pos.quantity * fill.price - fill.commission
                    self._closed_trades.append({
                        "symbol": fill.symbol,
                        "side": pos.side.value,
                        "entry_price": pos.avg_entry_price,
                        "exit_price": fill.price,
                        "quantity": pos.quantity,
                        "pnl": pnl,
                        "commission": fill.commission,
                        "timestamp": fill.timestamp,
                    })
                    del self._positions[key]

                    # If overfill, open reverse position
                    remaining = fill.quantity - pos.quantity
                    if remaining > 0:
                        self._open_position(fill, strategy_id, remaining)
                else:
                    # Partial close
                    pnl = self._calc_pnl(pos, fill.price, fill.quantity)
                    self._realized_pnl += pnl - fill.commission
                    self._cash += fill.quantity * fill.price - fill.commission
                    pos.quantity -= fill.quantity
                    self._closed_trades.append({
                        "symbol": fill.symbol,
                        "side": pos.side.value,
                        "entry_price": pos.avg_entry_price,
                        "exit_price": fill.price,
                        "quantity": fill.quantity,
                        "pnl": pnl,
                        "commission": fill.commission,
                        "timestamp": fill.timestamp,
                    })
        else:
            # New position
            self._open_position(fill, strategy_id, fill.quantity)

    def _open_position(self, fill: Fill, strategy_id: str, quantity: float) -> None:
        key = f"{fill.symbol}:{strategy_id}"
        cost = quantity * fill.price + fill.commission
        self._cash -= cost

        self._positions[key] = Position(
            symbol=fill.symbol,
            strategy_id=strategy_id,
            side=fill.side,
            quantity=quantity,
            avg_entry_price=fill.price,
            current_price=fill.price,
            exchange=fill.exchange,
        )

    def _calc_pnl(self, position: Position, exit_price: float, quantity: float) -> float:
        if position.side == Side.BUY:
            return (exit_price - position.avg_entry_price) * quantity
        else:
            return (position.avg_entry_price - exit_price) * quantity

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all positions."""
        for pos in self._positions.values():
            if pos.symbol in prices:
                pos.update_price(prices[pos.symbol])

    def get_portfolio_state(self) -> PortfolioState:
        """Compute current portfolio state."""
        positions_value = sum(p.notional_value for p in self._positions.values())
        unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
        total_equity = self._cash + positions_value

        # Track peak for drawdown
        if total_equity > self._peak_equity:
            self._peak_equity = total_equity

        current_dd = (self._peak_equity - total_equity) / self._peak_equity if self._peak_equity > 0 else 0

        # Gross/net exposure
        long_value = sum(p.notional_value for p in self._positions.values() if p.side == Side.BUY)
        short_value = sum(p.notional_value for p in self._positions.values() if p.side == Side.SELL)
        gross_exposure = long_value + short_value
        net_exposure = long_value - short_value
        leverage = gross_exposure / total_equity if total_equity > 0 else 0

        # Compute Sharpe from equity curve
        sharpe = self._compute_sharpe()

        # Record equity point
        now = datetime.now(timezone.utc)
        self._equity_curve.append((now, total_equity))

        return PortfolioState(
            timestamp=now,
            total_equity=total_equity,
            cash=self._cash,
            positions_value=positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._realized_pnl,
            positions=list(self._positions.values()),
            max_drawdown=self._max_drawdown(),
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            leverage=leverage,
        )

    def _max_drawdown(self) -> float:
        """Maximum drawdown from equity curve."""
        if len(self._equity_curve) < 2:
            return 0.0
        equities = [e for _, e in self._equity_curve]
        peak = equities[0]
        max_dd = 0.0
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    def _compute_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe ratio from equity curve."""
        if len(self._equity_curve) < 10:
            return 0.0
        equities = [e for _, e in self._equity_curve]
        returns = [
            (equities[i] - equities[i - 1]) / equities[i - 1]
            for i in range(1, len(equities))
            if equities[i - 1] > 0
        ]
        if len(returns) < 2:
            return 0.0
        import math
        mean_ret = sum(returns) / len(returns)
        var = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(var) if var > 0 else 1e-10
        return (mean_ret - risk_free_rate / 252) / std * math.sqrt(252)

    def get_trade_history(self) -> list[dict]:
        """Return all closed trades."""
        return list(self._closed_trades)

    def reset(self, initial_cash: Optional[float] = None) -> None:
        """Reset portfolio to initial state."""
        self._cash = initial_cash or self._initial_cash
        self._initial_cash = self._cash
        self._positions.clear()
        self._closed_trades.clear()
        self._equity_curve.clear()
        self._peak_equity = self._cash
        self._realized_pnl = 0.0
        self._total_commission = 0.0
