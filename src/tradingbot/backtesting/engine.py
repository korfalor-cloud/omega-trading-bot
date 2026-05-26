"""Backtesting Engine — Event-driven strategy evaluation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.types import Fill, OHLCVBar, Signal, StrategyGenome
from ..core.enums import OrderState, OrderType, Side
from ..population.fitness import FitnessEvaluator, FitnessResult

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    genome_id: str
    fitness: FitnessResult
    equity_curve: list[float]
    trades: list[dict]
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int


class BacktestEngine:
    """Event-driven backtesting engine.

    Replays historical OHLCV bars through a strategy genome
    and evaluates performance with realistic fill simulation.
    """

    def __init__(self, config: dict):
        self.initial_capital = config.get("initial_capital", 100_000.0)
        self.slippage_bps = config.get("slippage_bps", 5.0)
        self.commission_bps = config.get("commission_bps", 10.0)
        self.fitness_evaluator = FitnessEvaluator(config)

    async def run(
        self,
        genome: StrategyGenome,
        bars: list[OHLCVBar],
        features: dict[str, list[float]],
    ) -> BacktestResult:
        """Run a backtest for a genome on historical data."""
        from ..genome.genome_encoder import GenomeEvaluator

        evaluator = GenomeEvaluator()
        equity = self.initial_capital
        equity_curve = [equity]
        trades: list[dict] = []
        position: Optional[dict] = None  # {side, entry_price, quantity, entry_idx}

        # Build feature dict per bar
        for i, bar in enumerate(bars):
            bar_features = {k: v[i] for k, v in features.items() if i < len(v)}
            history = bars[max(0, i - 50):i + 1]

            # Evaluate genome
            signal = evaluator.evaluate(genome, bar, history, bar_features)

            # Check stop loss / take profit if in position
            if position is not None:
                pnl_pct = self._calc_pnl_pct(position, bar.close)
                stop_pct = -genome.stop_loss_param / 100 if genome.stop_loss_method == "fixed" else -genome.stop_loss_param * bar_features.get("atr", bar.close * 0.02) / bar.close
                tp_pct = genome.take_profit_ratio * abs(stop_pct)

                if pnl_pct <= stop_pct:
                    # Stop loss hit
                    trade_pnl = self._close_position(position, bar.close, "stop_loss")
                    equity += trade_pnl
                    trades.append({"pnl": trade_pnl, "type": "stop_loss", "bar_idx": i})
                    position = None
                elif pnl_pct >= tp_pct:
                    # Take profit hit
                    trade_pnl = self._close_position(position, bar.close, "take_profit")
                    equity += trade_pnl
                    trades.append({"pnl": trade_pnl, "type": "take_profit", "bar_idx": i})
                    position = None

            # Process signal
            if signal is not None and position is None:
                # Open position
                side = signal.side
                entry_price = bar.close * (1 + self.slippage_bps / 10000 if side == Side.BUY else 1 - self.slippage_bps / 10000)
                quantity = (equity * genome.max_position_pct) / entry_price
                commission = quantity * entry_price * self.commission_bps / 10000

                position = {
                    "side": side,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "commission": commission,
                    "entry_idx": i,
                }
                equity -= commission

            elif signal is not None and position is not None:
                # Check for exit signal (opposite direction)
                if (signal.side == Side.SELL and position["side"] == Side.BUY) or \
                   (signal.side == Side.BUY and position["side"] == Side.SELL):
                    trade_pnl = self._close_position(position, bar.close, "signal_exit")
                    equity += trade_pnl
                    trades.append({"pnl": trade_pnl, "type": "signal_exit", "bar_idx": i})
                    position = None

            # Update equity curve
            if position is not None:
                unrealized = self._calc_unrealized_pnl(position, bar.close)
                equity_curve.append(equity + unrealized)
            else:
                equity_curve.append(equity)

        # Close any remaining position
        if position is not None:
            trade_pnl = self._close_position(position, bars[-1].close, "end_of_backtest")
            equity += trade_pnl
            trades.append({"pnl": trade_pnl, "type": "end_of_backtest", "bar_idx": len(bars) - 1})

        # Calculate fitness
        trade_returns = [t["pnl"] / self.initial_capital for t in trades]
        fitness = self.fitness_evaluator.evaluate(equity_curve, trade_returns)

        return BacktestResult(
            genome_id=genome.id,
            fitness=fitness,
            equity_curve=equity_curve,
            trades=trades,
            total_return=(equity - self.initial_capital) / self.initial_capital,
            max_drawdown=fitness.max_drawdown,
            sharpe_ratio=fitness.sharpe_ratio,
            win_rate=fitness.win_rate,
            total_trades=len(trades),
        )

    def _calc_pnl_pct(self, position: dict, current_price: float) -> float:
        entry = position["entry_price"]
        if position["side"] == Side.BUY:
            return (current_price - entry) / entry
        else:
            return (entry - current_price) / entry

    def _calc_unrealized_pnl(self, position: dict, current_price: float) -> float:
        entry = position["entry_price"]
        qty = position["quantity"]
        if position["side"] == Side.BUY:
            return (current_price - entry) * qty
        else:
            return (entry - current_price) * qty

    def _close_position(self, position: dict, exit_price: float, reason: str) -> float:
        entry = position["entry_price"]
        qty = position["quantity"]
        commission = qty * exit_price * self.commission_bps / 10000

        if position["side"] == Side.BUY:
            pnl = (exit_price - entry) * qty - commission - position["commission"]
        else:
            pnl = (entry - exit_price) * qty - commission - position["commission"]

        return pnl
