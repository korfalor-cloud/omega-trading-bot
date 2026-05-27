"""Adversarial Stress Tester — Test strategies against extreme market scenarios.

Generates synthetic extreme events (flash crashes, regime shifts, correlation
breakdowns, liquidity crises) and evaluates strategy robustness.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from ..core.enums import Timeframe
from ..core.types import OHLCVBar, StrategyGenome

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Definition of a stress test scenario."""
    name: str
    description: str
    duration_bars: int
    severity: float  # 0-1, how extreme the scenario is


@dataclass
class StressResult:
    """Result of a stress test."""
    scenario: str
    max_drawdown: float
    total_return: float
    sharpe_ratio: float
    max_loss_streak: int
    recovery_time_bars: int
    survived: bool  # Did the strategy survive without blowing up?
    equity_curve: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class AdversarialTester:
    """Stress test strategies against extreme market conditions.

    Scenarios include:
    - Flash crash (sudden 20-40% drop with quick recovery)
    - Prolonged bear market (6-12 months of decline)
    - Volatility spike (VIX-like explosion)
    - Correlation breakdown (diversification fails)
    - Liquidity crisis (wide spreads, gaps)
    - Whipsaw (rapid direction changes)
    - Black swan (single extreme event)
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.initial_capital = cfg.get("initial_capital", 100_000.0)
        self.scenarios = self._build_scenarios()

    def _build_scenarios(self) -> list[StressScenario]:
        return [
            StressScenario(
                name="flash_crash",
                description="Sudden 30% drop with partial recovery within 24 bars",
                duration_bars=48,
                severity=0.8,
            ),
            StressScenario(
                name="prolonged_bear",
                description="Steady 50% decline over 500 bars",
                duration_bars=500,
                severity=0.9,
            ),
            StressScenario(
                name="volatility_spike",
                description="Volatility increases 5x with whipsaw moves",
                duration_bars=200,
                severity=0.7,
            ),
            StressScenario(
                name="whipsaw",
                description="Rapid alternating 5% moves every 3-5 bars",
                duration_bars=100,
                severity=0.6,
            ),
            StressScenario(
                name="liquidity_crisis",
                description="Gaps and wide spreads, 3x normal slippage",
                duration_bars=100,
                severity=0.7,
            ),
            StressScenario(
                name="black_swan",
                description="Single 40% drop in one bar",
                duration_bars=50,
                severity=1.0,
            ),
            StressScenario(
                name="slow_grind",
                description="Small daily losses accumulating over time",
                duration_bars=300,
                severity=0.5,
            ),
        ]

    def generate_scenario_data(
        self,
        scenario: StressScenario,
        base_price: float = 50000.0,
        rng_seed: int = 42,
    ) -> list[OHLCVBar]:
        """Generate OHLCV bars for a stress scenario."""
        rng = np.random.RandomState(rng_seed)
        bars = []
        price = base_price
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)

        if scenario.name == "flash_crash":
            bars = self._gen_flash_crash(price, ts, scenario.duration_bars, rng)
        elif scenario.name == "prolonged_bear":
            bars = self._gen_prolonged_bear(price, ts, scenario.duration_bars, rng)
        elif scenario.name == "volatility_spike":
            bars = self._gen_volatility_spike(price, ts, scenario.duration_bars, rng)
        elif scenario.name == "whipsaw":
            bars = self._gen_whipsaw(price, ts, scenario.duration_bars, rng)
        elif scenario.name == "liquidity_crisis":
            bars = self._gen_liquidity_crisis(price, ts, scenario.duration_bars, rng)
        elif scenario.name == "black_swan":
            bars = self._gen_black_swan(price, ts, scenario.duration_bars, rng)
        elif scenario.name == "slow_grind":
            bars = self._gen_slow_grind(price, ts, scenario.duration_bars, rng)
        else:
            bars = self._gen_default(price, ts, scenario.duration_bars, rng)

        return bars

    def _gen_flash_crash(self, base_price, ts, n, rng) -> list[OHLCVBar]:
        bars = []
        price = base_price
        crash_bar = n // 4
        recovery_start = crash_bar + 6

        for i in range(n):
            if i == crash_bar:
                # Flash crash: -30%
                price *= 0.7
            elif crash_bar < i < recovery_start:
                # Stabilization
                price *= (1 + rng.normal(0.005, 0.02))
            elif i >= recovery_start:
                # Recovery: slow climb back
                price *= (1 + rng.normal(0.002, 0.01))
            else:
                # Normal before crash
                price *= (1 + rng.normal(0.0003, 0.012))

            bars.append(self._make_bar(price, ts, i, rng))
        return bars

    def _gen_prolonged_bear(self, base_price, ts, n, rng) -> list[OHLCVBar]:
        bars = []
        price = base_price
        for i in range(n):
            # Steady decline with occasional bounces
            drift = -0.001 + (0.003 if rng.random() < 0.1 else 0)
            price *= (1 + drift + rng.normal(0, 0.015))
            bars.append(self._make_bar(price, ts, i, rng))
        return bars

    def _gen_volatility_spike(self, base_price, ts, n, rng) -> list[OHLCVBar]:
        bars = []
        price = base_price
        spike_start = n // 3
        spike_end = 2 * n // 3

        for i in range(n):
            if spike_start <= i < spike_end:
                # High volatility period
                vol = 0.05
            else:
                vol = 0.012
            price *= (1 + rng.normal(0, vol))
            bars.append(self._make_bar(price, ts, i, rng, vol_mult=vol / 0.012))
        return bars

    def _gen_whipsaw(self, base_price, ts, n, rng) -> list[OHLCVBar]:
        bars = []
        price = base_price
        direction = 1

        for i in range(n):
            if i % 4 == 0:
                direction *= -1
            price *= (1 + direction * 0.01 + rng.normal(0, 0.008))
            bars.append(self._make_bar(price, ts, i, rng))
        return bars

    def _gen_liquidity_crisis(self, base_price, ts, n, rng) -> list[OHLCVBar]:
        bars = []
        price = base_price

        for i in range(n):
            # Random gaps
            if rng.random() < 0.1:
                gap = rng.choice([-1, 1]) * rng.uniform(0.02, 0.05)
                price *= (1 + gap)
            else:
                price *= (1 + rng.normal(0, 0.015))

            bar = self._make_bar(price, ts, i, rng)
            # Wider spreads (simulated via higher high-low range)
            bar = OHLCVBar(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                timeframe=bar.timeframe,
                open=bar.open,
                high=bar.high * 1.01,
                low=bar.low * 0.99,
                close=bar.close,
                volume=bar.volume * 0.3,  # Low volume
                exchange=bar.exchange,
            )
            bars.append(bar)
        return bars

    def _gen_black_swan(self, base_price, ts, n, rng) -> list[OHLCVBar]:
        bars = []
        price = base_price
        swan_bar = n // 2

        for i in range(n):
            if i == swan_bar:
                # Black swan: -40% in one bar
                price *= 0.6
            elif i > swan_bar:
                # Slow partial recovery
                price *= (1 + rng.normal(0.001, 0.02))
            else:
                price *= (1 + rng.normal(0.0003, 0.01))
            bars.append(self._make_bar(price, ts, i, rng))
        return bars

    def _gen_slow_grind(self, base_price, ts, n, rng) -> list[OHLCVBar]:
        bars = []
        price = base_price
        for i in range(n):
            # Small negative bias
            price *= (1 + rng.normal(-0.0005, 0.008))
            bars.append(self._make_bar(price, ts, i, rng))
        return bars

    def _gen_default(self, base_price, ts, n, rng) -> list[OHLCVBar]:
        bars = []
        price = base_price
        for i in range(n):
            price *= (1 + rng.normal(0, 0.015))
            bars.append(self._make_bar(price, ts, i, rng))
        return bars

    def _make_bar(
        self, price: float, ts: datetime, i: int,
        rng: np.random.RandomState, vol_mult: float = 1.0,
    ) -> OHLCVBar:
        noise = abs(rng.normal(0, 0.005 * vol_mult))
        return OHLCVBar(
            timestamp=ts + timedelta(hours=i),
            symbol="BTC/USDT",
            timeframe=Timeframe.H1,
            open=price * (1 + rng.normal(0, 0.002)),
            high=price * (1 + noise),
            low=price * (1 - noise),
            close=price,
            volume=rng.uniform(100, 500),
            exchange="binance",
        )

    async def run_stress_test(
        self,
        genome: StrategyGenome,
        scenario: StressScenario,
        rng_seed: int = 42,
    ) -> StressResult:
        """Run a single stress test scenario against a genome."""
        from ..backtesting.engine import BacktestEngine
        from ..features.technical import compute_features

        bars = self.generate_scenario_data(scenario, rng_seed=rng_seed)
        features = compute_features(bars)

        engine = BacktestEngine({
            "initial_capital": self.initial_capital,
            "slippage_bps": 10.0,  # Higher slippage for stress
            "commission_bps": 10.0,
        })

        result = await engine.run(genome, bars, features)

        # Calculate max consecutive losses
        max_loss_streak = 0
        current_streak = 0
        for trade in result.trades:
            if trade["pnl"] < 0:
                current_streak += 1
                max_loss_streak = max(max_loss_streak, current_streak)
            else:
                current_streak = 0

        # Recovery time: bars from max drawdown to recovery
        recovery_time = self._calc_recovery_time(result.equity_curve)

        # Survived if drawdown < 50% and still has capital
        survived = result.max_drawdown < 0.5 and result.equity_curve[-1] > self.initial_capital * 0.1

        return StressResult(
            scenario=scenario.name,
            max_drawdown=result.max_drawdown,
            total_return=result.total_return,
            sharpe_ratio=result.sharpe_ratio,
            max_loss_streak=max_loss_streak,
            recovery_time_bars=recovery_time,
            survived=survived,
            equity_curve=result.equity_curve,
            metadata={
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "severity": scenario.severity,
            },
        )

    def _calc_recovery_time(self, equity_curve: list[float]) -> int:
        """Calculate bars from peak drawdown to recovery."""
        if len(equity_curve) < 2:
            return 0
        peak = equity_curve[0]
        max_dd_idx = 0
        max_dd = 0.0

        for i, eq in enumerate(equity_curve):
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_idx = i

        # Find recovery point
        for i in range(max_dd_idx, len(equity_curve)):
            if equity_curve[i] >= peak * 0.95:  # Recovered to within 5% of peak
                return i - max_dd_idx

        return len(equity_curve) - max_dd_idx  # Not recovered

    async def run_all_scenarios(
        self,
        genome: StrategyGenome,
    ) -> list[StressResult]:
        """Run all stress test scenarios."""
        results = []
        for scenario in self.scenarios:
            result = await self.run_stress_test(genome, scenario)
            results.append(result)
            logger.info(
                f"Stress test '{scenario.name}': "
                f"drawdown={result.max_drawdown:.1%}, "
                f"return={result.total_return:.1%}, "
                f"survived={result.survived}"
            )
        return results

    def generate_report(self, results: list[StressResult]) -> str:
        """Generate a human-readable stress test report."""
        lines = ["=" * 60]
        lines.append("ADVERSARIAL STRESS TEST REPORT")
        lines.append("=" * 60)

        survived_count = sum(1 for r in results if r.survived)
        lines.append(f"\nScenarios: {len(results)} | Survived: {survived_count} | Failed: {len(results) - survived_count}")
        lines.append("-" * 60)

        for r in results:
            status = "PASS" if r.survived else "FAIL"
            lines.append(f"\n[{status}] {r.scenario}")
            lines.append(f"  Max Drawdown:    {r.max_drawdown:.1%}")
            lines.append(f"  Total Return:    {r.total_return:.1%}")
            lines.append(f"  Sharpe Ratio:    {r.sharpe_ratio:.2f}")
            lines.append(f"  Max Loss Streak: {r.max_loss_streak}")
            lines.append(f"  Recovery Time:   {r.recovery_time_bars} bars")

        # Overall robustness score
        robustness = survived_count / len(results) if results else 0
        lines.append(f"\n{'=' * 60}")
        lines.append(f"ROBUSTNESS SCORE: {robustness:.0%}")
        lines.append(f"{'=' * 60}")

        return "\n".join(lines)
