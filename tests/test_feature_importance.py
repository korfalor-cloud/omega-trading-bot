"""Tests for feature importance analysis."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.ml.features.feature_importance import (
    FeatureImportanceAnalyzer,
    FeatureImportanceResult,
)


class TestFeatureImportanceAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return FeatureImportanceAnalyzer(["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"])

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5))
        # y is mostly determined by features 0 and 2
        y = 2 * X[:, 0] + 0.5 * X[:, 2] + rng.standard_normal(200) * 0.1
        return X, y

    def test_correlation_importance(self, analyzer, data):
        X, y = data
        result = analyzer.correlation_importance(X, y)
        assert isinstance(result, FeatureImportanceResult)
        assert len(result.importance_scores) == 5
        # Feature 0 should be most important
        assert result.importance_scores[0] > result.importance_scores[1]

    def test_mutual_information(self, analyzer, data):
        X, y = data
        result = analyzer.mutual_information(X, y)
        assert len(result.importance_scores) == 5
        assert all(v >= 0 for v in result.importance_scores)

    def test_variance_importance(self, analyzer, data):
        X, _ = data
        result = analyzer.variance_importance(X)
        assert len(result.importance_scores) == 5
        assert all(0 <= v <= 1 for v in result.importance_scores)

    def test_permutation_importance(self, analyzer, data):
        X, y = data

        class MockModel:
            def predict(self, X):
                return 2 * X[:, 0] + 0.5 * X[:, 2]

        model = MockModel()
        result = analyzer.permutation_importance(model, X, y, n_repeats=3)
        assert len(result.importance_scores) == 5
        # Feature 0 should be most important (by magnitude)
        assert abs(result.importance_scores[0]) > abs(result.importance_scores[1])

    def test_stability_importance(self, analyzer):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((100, 5))
        y_train = 2 * X_train[:, 0] + rng.standard_normal(100) * 0.1
        X_test = rng.standard_normal((100, 5))
        y_test = 2 * X_test[:, 0] + rng.standard_normal(100) * 0.1

        result = analyzer.stability_importance(X_train, y_train, X_test, y_test)
        assert len(result.importance_scores) == 5

    def test_select_features(self, analyzer, data):
        X, y = data
        result = analyzer.correlation_importance(X, y)
        selected = analyzer.select_features(result, threshold=0.3)
        assert len(selected) > 0
        assert 0 in selected  # Feature 0 should be selected

    def test_get_top_n(self, analyzer, data):
        X, y = data
        result = analyzer.correlation_importance(X, y)
        top = result.get_top_n(3)
        assert len(top) == 3
        assert top[0][0] == "feat_a"  # Most correlated

    def test_set_feature_names(self):
        analyzer = FeatureImportanceAnalyzer()
        analyzer.set_feature_names(["a", "b", "c"])
        assert analyzer.feature_names == ["a", "b", "c"]
