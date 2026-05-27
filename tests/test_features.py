"""Tests for feature engineering and technical indicators."""
from __future__ import annotations

import numpy as np
import pytest

from tradingbot.core.enums import Timeframe
from tradingbot.core.types import OHLCVBar
from tradingbot.features.technical import TechnicalIndicators, compute_features


class TestTechnicalIndicators:
    """Test individual indicator computations."""

    def test_sma_basic(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        sma = ti.sma(20)

        # First 18 values should be NaN
        assert np.isnan(sma[0])
        assert np.isnan(sma[18])
        # 19th value should be valid
        assert not np.isnan(sma[19])
        # SMA should be close to mean of first 20 closes
        expected = np.mean([b.close for b in sample_bars[:20]])
        assert abs(sma[19] - expected) < 1e-6

    def test_ema_basic(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        ema = ti.ema(20)

        assert np.isnan(ema[0])
        assert not np.isnan(ema[19])
        # EMA should be a valid number
        assert ema[19] > 0

    def test_rsi_range(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        rsi = ti.rsi(14)

        # RSI should be between 0 and 100
        valid = rsi[~np.isnan(rsi)]
        assert len(valid) > 0
        assert all(0 <= v <= 100 for v in valid)

    def test_macd_components(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        macd_line, signal_line, histogram = ti.macd()

        # All should have valid values
        valid_macd = macd_line[~np.isnan(macd_line)]
        valid_signal = signal_line[~np.isnan(signal_line)]
        valid_hist = histogram[~np.isnan(histogram)]

        assert len(valid_macd) > 0
        assert len(valid_signal) > 0
        assert len(valid_hist) > 0

        # Histogram = macd - signal
        for i in range(len(histogram)):
            if not np.isnan(macd_line[i]) and not np.isnan(signal_line[i]):
                assert abs(histogram[i] - (macd_line[i] - signal_line[i])) < 1e-10

    def test_atr_positive(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        atr = ti.atr(14)

        valid = atr[~np.isnan(atr)]
        assert len(valid) > 0
        assert all(v > 0 for v in valid)

    def test_adx_range(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        adx = ti.adx(14)

        valid = adx[~np.isnan(adx)]
        # ADX needs 2*period bars to produce values
        if len(valid) > 0:
            assert all(0 <= v <= 100 for v in valid)

    def test_bollinger_bands_ordering(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        upper, middle, lower = ti.bollinger_bands(20, 2.0)

        for i in range(len(upper)):
            if not any(np.isnan([upper[i], middle[i], lower[i]])):
                assert upper[i] >= middle[i] >= lower[i]

    def test_stochastic_range(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        k, d = ti.stochastic(14, 3)

        valid_k = k[~np.isnan(k)]
        valid_d = d[~np.isnan(d)]
        assert all(0 <= v <= 100 for v in valid_k)
        assert all(0 <= v <= 100 for v in valid_d)

    def test_obv_monotonic_in_trend(self, trending_bars):
        ti = TechnicalIndicators(trending_bars)
        obv = ti.obv()
        # In an uptrend, OBV should generally increase
        assert obv[-1] > obv[0]

    def test_vwap_reasonable(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        vwap = ti.vwap()
        # VWAP should be within price range
        closes = [b.close for b in sample_bars]
        min_price = min(closes)
        max_price = max(closes)
        assert min_price * 0.9 < vwap[-1] < max_price * 1.1

    def test_roc_sign(self, trending_bars):
        ti = TechnicalIndicators(trending_bars)
        roc = ti.roc(10)
        # In an uptrend, ROC should generally be positive
        valid = roc[~np.isnan(roc)]
        positive_count = sum(1 for v in valid if v > 0)
        assert positive_count > len(valid) * 0.5  # Majority positive

    def test_williams_r_range(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        wr = ti.williams_r(14)
        valid = wr[~np.isnan(wr)]
        assert all(-100 <= v <= 0 for v in valid)

    def test_cci_no_crash(self, sample_bars):
        ti = TechnicalIndicators(sample_bars)
        cci = ti.cci(20)
        valid = cci[~np.isnan(cci)]
        assert len(valid) > 0


class TestComputeFeatures:
    def test_returns_dict(self, sample_bars):
        features = compute_features(sample_bars)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_feature_length_matches_bars(self, sample_bars):
        features = compute_features(sample_bars)
        for name, values in features.items():
            assert len(values) == len(sample_bars), f"Feature {name} has wrong length"

    def test_no_nans_in_output(self, sample_bars):
        features = compute_features(sample_bars)
        for name, values in features.items():
            nan_count = sum(1 for v in values if v != v)  # NaN != NaN
            assert nan_count == 0, f"Feature {name} contains {nan_count} NaN values"

    def test_expected_features_present(self, sample_bars):
        features = compute_features(sample_bars)
        expected = ["rsi_14", "ema_8", "ema_21", "atr_14", "adx_14",
                     "macd", "bb_upper", "volatility", "signal_strength"]
        for name in expected:
            assert name in features, f"Missing feature: {name}"

    def test_rsi_in_range(self, sample_bars):
        features = compute_features(sample_bars)
        rsi = features["rsi_14"]
        # Forward-filled values may include 0 for leading NaN
        assert all(0 <= v <= 100 for v in rsi)

    def test_signal_strength_in_range(self, sample_bars):
        features = compute_features(sample_bars)
        strength = features["signal_strength"]
        assert all(-1 <= v <= 1 for v in strength)

    def test_custom_config(self, sample_bars):
        config = {
            "sma_periods": [10, 30],
            "ema_periods": [5, 15],
            "rsi_periods": [7, 21],
        }
        features = compute_features(sample_bars, config)
        assert "sma_10" in features
        assert "sma_30" in features
        assert "ema_5" in features
        assert "rsi_7" in features
        assert "rsi_21" in features

    def test_empty_bars(self):
        features = compute_features([])
        assert features == {}
