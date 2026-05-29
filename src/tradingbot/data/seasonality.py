"""Seasonality Analysis — Time-based pattern detection in returns.

Implements:
- Hour-of-day return analysis
- Day-of-week return analysis
- Monthly return analysis
- Statistical significance testing (t-test, bootstrap)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PeriodStats:
    """Return statistics for a single time period bucket."""
    label: str = ""
    mean_return: float = 0.0
    std_return: float = 0.0
    median_return: float = 0.0
    count: int = 0
    win_rate: float = 0.0
    t_stat: float = 0.0
    p_value: float = 0.0
    is_significant: bool = False


@dataclass
class SeasonalityResult:
    """Full seasonality analysis result."""
    period_type: str = ""
    stats: list[PeriodStats] = field(default_factory=list)
    best_period: str = ""
    worst_period: str = ""


class SeasonalityAnalyzer:
    """Analyze time-of-day, day-of-week, and monthly return patterns.

    Accepts arrays of timestamps (unix epoch seconds or datetime objects)
    paired with returns, and computes per-bucket statistics with
    significance testing.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.significance_level = config.get("significance_level", 0.05)
        self.min_samples = config.get("min_samples", 10)
        self.bootstrap_iterations = config.get("bootstrap_iterations", 1000)

    def analyze_hour(
        self,
        timestamps: np.ndarray,
        returns: np.ndarray,
    ) -> SeasonalityResult:
        """Analyze returns by hour of day (0-23)."""
        hours = self._extract_hours(timestamps)
        return self._analyze_by_bucket(hours, returns, "hour", 24)

    def analyze_day_of_week(
        self,
        timestamps: np.ndarray,
        returns: np.ndarray,
    ) -> SeasonalityResult:
        """Analyze returns by day of week (0=Mon, 6=Sun)."""
        dow = self._extract_day_of_week(timestamps)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        result = self._analyze_by_bucket(dow, returns, "day_of_week", 7)
        for s in result.stats:
            idx = int(s.label)
            s.label = day_names[idx] if 0 <= idx < 7 else s.label
        return result

    def analyze_monthly(
        self,
        timestamps: np.ndarray,
        returns: np.ndarray,
    ) -> SeasonalityResult:
        """Analyze returns by month (1-12)."""
        months = self._extract_months(timestamps)
        month_names = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        result = self._analyze_by_bucket(months, returns, "monthly", 12, offset=1)
        for s in result.stats:
            idx = int(s.label) - 1
            s.label = month_names[idx] if 0 <= idx < 12 else s.label
        return result

    def full_analysis(
        self,
        timestamps: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, SeasonalityResult]:
        """Run all three seasonality analyses."""
        return {
            "hourly": self.analyze_hour(timestamps, returns),
            "daily": self.analyze_day_of_week(timestamps, returns),
            "monthly": self.analyze_monthly(timestamps, returns),
        }

    # ── Core analysis ─────────────────────────────────────────────

    def _analyze_by_bucket(
        self,
        buckets: np.ndarray,
        returns: np.ndarray,
        period_type: str,
        n_buckets: int,
        offset: int = 0,
    ) -> SeasonalityResult:
        """Compute per-bucket statistics with t-test significance."""
        # Clip and shift buckets
        buckets = buckets.astype(int)
        stats: list[PeriodStats] = []
        global_mean = float(np.mean(returns)) if len(returns) > 0 else 0.0

        for b in range(n_buckets):
            bucket_val = b + offset
            mask = buckets == bucket_val
            bucket_returns = returns[mask]

            if len(bucket_returns) < self.min_samples:
                stats.append(PeriodStats(label=str(bucket_val), count=len(bucket_returns)))
                continue

            mean_ret = float(np.mean(bucket_returns))
            std_ret = float(np.std(bucket_returns, ddof=1))
            median_ret = float(np.median(bucket_returns))
            win_rate = float(np.mean(bucket_returns > 0))
            n = len(bucket_returns)

            # One-sample t-test: H0: mean = global_mean
            if std_ret > 0 and n > 1:
                se = std_ret / np.sqrt(n)
                t_stat = (mean_ret - global_mean) / se
                p_value = self._t_test_p_value(t_stat, n - 1)
            else:
                t_stat = 0.0
                p_value = 1.0

            is_sig = p_value < self.significance_level

            stats.append(PeriodStats(
                label=str(bucket_val),
                mean_return=mean_ret,
                std_return=std_ret,
                median_return=median_ret,
                count=n,
                win_rate=win_rate,
                t_stat=t_stat,
                p_value=p_value,
                is_significant=is_sig,
            ))

        # Best / worst
        valid_stats = [s for s in stats if s.count >= self.min_samples]
        best = max(valid_stats, key=lambda s: s.mean_return) if valid_stats else PeriodStats()
        worst = min(valid_stats, key=lambda s: s.mean_return) if valid_stats else PeriodStats()

        return SeasonalityResult(
            period_type=period_type,
            stats=stats,
            best_period=best.label,
            worst_period=worst.label,
        )

    # ── Significance testing ──────────────────────────────────────

    def _t_test_p_value(self, t_stat: float, df: int) -> float:
        """Approximate two-tailed p-value for a t-statistic.

        Uses the normal approximation for large df, otherwise a
        simple approximation.
        """
        # For large df, t ~ N(0,1)
        z = abs(t_stat)
        # Approximate standard normal survival function
        # P(|Z| > z) = 2 * (1 - Phi(z))
        # Using rational approximation
        if z > 8:
            return 0.0
        p = 2.0 * self._normal_sf(z)
        return min(1.0, p)

    @staticmethod
    def _normal_sf(z: float) -> float:
        """Survival function (1 - CDF) of the standard normal distribution."""
        # Horner form rational approximation (Abramowitz & Stegun 26.2.17)
        if z < 0:
            return 1.0 - SeasonalityAnalyzer._normal_sf(-z)
        t = 1.0 / (1.0 + 0.2316419 * z)
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        p = d * np.exp(-0.5 * z * z) * (
            t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        )
        return p

    def bootstrap_test(
        self,
        bucket_returns: np.ndarray,
        all_returns: np.ndarray,
        n_iter: int | None = None,
    ) -> tuple[float, float]:
        """Bootstrap test: is the bucket mean significantly different from population?

        Returns (observed_diff, p_value).
        """
        n_iter = n_iter or self.bootstrap_iterations
        observed_mean = float(np.mean(bucket_returns))
        global_mean = float(np.mean(all_returns))
        observed_diff = observed_mean - global_mean

        n = len(bucket_returns)
        count_extreme = 0

        for _ in range(n_iter):
            sample = np.random.choice(all_returns, size=n, replace=True)
            if abs(np.mean(sample) - global_mean) >= abs(observed_diff):
                count_extreme += 1

        p_value = count_extreme / n_iter
        return observed_diff, p_value

    # ── Timestamp extraction ──────────────────────────────────────

    @staticmethod
    def _to_datetime_array(timestamps: np.ndarray) -> np.ndarray:
        """Convert timestamps to datetime objects if needed."""
        if len(timestamps) == 0:
            return np.array([])

        # Check if already datetime objects
        sample = timestamps[0]
        if isinstance(sample, datetime):
            return timestamps

        # Assume unix timestamps
        return np.array([datetime.utcfromtimestamp(float(t)) for t in timestamps])

    def _extract_hours(self, timestamps: np.ndarray) -> np.ndarray:
        dt_arr = self._to_datetime_array(timestamps)
        return np.array([dt.hour for dt in dt_arr])

    def _extract_day_of_week(self, timestamps: np.ndarray) -> np.ndarray:
        dt_arr = self._to_datetime_array(timestamps)
        return np.array([dt.weekday() for dt in dt_arr])

    def _extract_months(self, timestamps: np.ndarray) -> np.ndarray:
        dt_arr = self._to_datetime_array(timestamps)
        return np.array([dt.month for dt in dt_arr])

    # ── Reporting ─────────────────────────────────────────────────

    def format_report(self, results: dict[str, SeasonalityResult]) -> str:
        """Format a human-readable seasonality report."""
        lines: list[str] = []
        for name, result in results.items():
            lines.append(f"\n{'='*50}")
            lines.append(f"  {name.upper()} SEASONALITY")
            lines.append(f"{'='*50}")
            lines.append(f"  Best period:  {result.best_period}")
            lines.append(f"  Worst period: {result.worst_period}")
            lines.append("")
            lines.append(f"  {'Period':<10} {'Mean':>10} {'Std':>10} {'WinRate':>10} {'N':>8} {'p-val':>8} {'Sig':>4}")

            for s in result.stats:
                sig = "*" if s.is_significant else ""
                lines.append(
                    f"  {s.label:<10} {s.mean_return:>10.6f} {s.std_return:>10.6f} "
                    f"{s.win_rate:>10.1%} {s.count:>8} {s.p_value:>8.4f} {sig:>4}"
                )

        return "\n".join(lines)
