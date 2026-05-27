"""XGBoost Model — Gradient boosting for trading signal prediction.

Handles training, inference, persistence, and feature importance
for XGBoost-based trading models.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class XGBSignalModel:
    """XGBoost model for predicting trading signals.

    Wraps sklearn's GradientBoostingClassifier (or XGBoost if available)
    with trading-specific functionality.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        self.n_estimators = cfg.get("n_estimators", 200)
        self.max_depth = cfg.get("max_depth", 5)
        self.learning_rate = cfg.get("learning_rate", 0.05)
        self.subsample = cfg.get("subsample", 0.8)
        self.min_samples_leaf = cfg.get("min_samples_leaf", 20)
        self.model = None
        self._feature_names: list[str] = []
        self._is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list[str]] = None) -> dict:
        """Train the model and return training metrics."""
        if len(X) < 100:
            logger.warning(f"Insufficient training data: {len(X)} samples")
            return {"error": "insufficient_data"}

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Try XGBoost first, fall back to sklearn
        try:
            import xgboost as xgb
            self.model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                min_child_weight=self.min_samples_leaf,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42,
            )

        # Train with time-series split validation
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            self.model.fit(X_train, y_train)
            scores.append(self.model.score(X_val, y_val))

        # Final fit on all data
        self.model.fit(X, y)
        self._is_trained = True

        train_score = self.model.score(X, y)

        metrics = {
            "train_accuracy": train_score,
            "cv_mean_accuracy": np.mean(scores),
            "cv_std_accuracy": np.std(scores),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "class_balance": {
                "class_0": int(np.sum(y == 0)),
                "class_1": int(np.sum(y == 1)),
            },
        }

        logger.info(
            f"Model trained: train_acc={train_score:.3f}, "
            f"cv_acc={np.mean(scores):.3f}±{np.std(scores):.3f}"
        )

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(X)

    def get_signal(self, X: np.ndarray, min_confidence: float = 0.6) -> tuple[int, float]:
        """Get trading signal with confidence.

        Returns:
            direction: 1 (long), -1 (short), 0 (no signal)
            confidence: probability of predicted class
        """
        proba = self.predict_proba(X.reshape(1, -1))[0]
        pred_class = np.argmax(proba)
        confidence = max(proba)

        if confidence < min_confidence:
            return 0, confidence

        direction = 1 if pred_class == 1 else -1
        return direction, confidence

    def get_feature_importance(self, top_n: int = 15) -> dict[str, float]:
        """Get feature importance scores."""
        if not self._is_trained:
            return {}

        importance = {}
        if hasattr(self.model, "feature_importances_"):
            for name, imp in zip(self._feature_names, self.model.feature_importances_):
                importance[name] = float(imp)

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n])

    def save(self, path: str) -> None:
        """Save model to disk."""
        data = {
            "model": self.model,
            "feature_names": self._feature_names,
            "is_trained": self._is_trained,
            "config": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self._feature_names = data["feature_names"]
        self._is_trained = data["is_trained"]
        logger.info(f"Model loaded from {path}")

    @property
    def is_trained(self) -> bool:
        return self._is_trained
