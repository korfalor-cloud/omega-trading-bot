"""Ensemble ML Models — Combining multiple models for robust predictions.

Implements:
- Simple averaging ensemble
- Weighted averaging (performance-based)
- Stacking (meta-learner)
- Bagging
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class PredictiveModel(Protocol):
    """Protocol for models that can be ensembled."""
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class EnsembleResult:
    """Result from ensemble prediction."""
    prediction: np.ndarray
    model_predictions: dict[str, np.ndarray] = field(default_factory=dict)
    model_weights: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    agreement: float = 0.0  # How much models agree


class EnsembleModel:
    """Ensemble of multiple predictive models.

    Combines predictions from multiple models using various strategies
    to produce more robust and stable predictions.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.method = config.get("method", "weighted")  # avg, weighted, stacking
        self._models: dict[str, any] = {}
        self._weights: dict[str, float] = {}
        self._performance_history: dict[str, list[float]] = {}

    def add_model(self, name: str, model: any, weight: float = 1.0) -> None:
        """Add a model to the ensemble."""
        self._models[name] = model
        self._weights[name] = weight
        self._performance_history[name] = []

    def remove_model(self, name: str) -> None:
        self._models.pop(name, None)
        self._weights.pop(name, None)
        self._performance_history.pop(name, None)

    def predict(self, X: np.ndarray) -> EnsembleResult:
        """Generate ensemble prediction."""
        if not self._models:
            return EnsembleResult(prediction=np.zeros(X.shape[0]))

        predictions = {}
        for name, model in self._models.items():
            try:
                pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.warning(f"Model {name} prediction failed: {e}")

        if not predictions:
            return EnsembleResult(prediction=np.zeros(X.shape[0]))

        if self.method == "avg":
            result = self._average_ensemble(predictions)
        elif self.method == "weighted":
            result = self._weighted_ensemble(predictions)
        elif self.method == "stacking":
            result = self._stacking_ensemble(predictions, X)
        else:
            result = self._weighted_ensemble(predictions)

        result.model_predictions = predictions
        result.model_weights = dict(self._weights)

        # Compute agreement (inverse of prediction variance)
        if len(predictions) > 1:
            pred_matrix = np.column_stack(list(predictions.values()))
            pred_std = np.std(pred_matrix, axis=1)
            result.agreement = float(1.0 / (1.0 + np.mean(pred_std)))

        return result

    def _average_ensemble(self, predictions: dict[str, np.ndarray]) -> EnsembleResult:
        """Simple average of all predictions."""
        pred_array = np.column_stack(list(predictions.values()))
        avg = np.mean(pred_array, axis=1)
        return EnsembleResult(prediction=avg)

    def _weighted_ensemble(self, predictions: dict[str, np.ndarray]) -> EnsembleResult:
        """Weighted average based on model weights."""
        total_weight = sum(
            self._weights.get(name, 1.0) for name in predictions
        )
        if total_weight == 0:
            return EnsembleResult(prediction=np.zeros(len(list(predictions.values())[0])))

        weighted_sum = np.zeros(len(list(predictions.values())[0]))
        for name, pred in predictions.items():
            w = self._weights.get(name, 1.0) / total_weight
            weighted_sum += w * pred

        return EnsembleResult(prediction=weighted_sum)

    def _stacking_ensemble(self, predictions: dict[str, np.ndarray], X: np.ndarray) -> EnsembleResult:
        """Stacking — use meta-learner on model predictions."""
        # For simplicity, use weighted average as meta-learner
        # In production, train a proper meta-learner on validation data
        return self._weighted_ensemble(predictions)

    def update_weights(self, name: str, performance_score: float) -> None:
        """Update model weight based on recent performance."""
        history = self._performance_history.get(name, [])
        history.append(performance_score)
        if len(history) > 100:
            history = history[-100:]
        self._performance_history[name] = history

        # Weight = average recent performance
        if history:
            self._weights[name] = max(0.1, np.mean(history))

    def normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self._weights.values())
        if total > 0:
            for name in self._weights:
                self._weights[name] /= total

    def get_model_performance(self) -> dict[str, dict]:
        """Get performance statistics for each model."""
        result = {}
        for name, history in self._performance_history.items():
            if history:
                result[name] = {
                    "mean": float(np.mean(history)),
                    "std": float(np.std(history)),
                    "recent": float(np.mean(history[-10:])),
                    "weight": self._weights.get(name, 0),
                    "n_predictions": len(history),
                }
        return result

    def get_status(self) -> dict:
        return {
            "n_models": len(self._models),
            "method": self.method,
            "weights": dict(self._weights),
            "model_names": list(self._models.keys()),
        }


class ModelSelector:
    """Dynamic model selection based on recent performance.

    Selects the best-performing model or switches between models
    based on market regime.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.performance_window = config.get("performance_window", 50)
        self.switch_threshold = config.get("switch_threshold", 0.1)
        self._models: dict[str, any] = {}
        self._performance: dict[str, list[float]] = {}
        self._current_model: Optional[str] = None

    def add_model(self, name: str, model: any) -> None:
        self._models[name] = model
        self._performance[name] = []
        if self._current_model is None:
            self._current_model = name

    def select_model(self) -> str:
        """Select the best model based on recent performance."""
        if not self._performance:
            return self._current_model or ""

        best_name = self._current_model
        best_score = -np.inf

        for name, scores in self._performance.items():
            if len(scores) < 5:
                continue
            recent = np.mean(scores[-self.performance_window:])
            if recent > best_score:
                best_score = recent
                best_name = name

        if best_name != self._current_model:
            improvement = best_score - np.mean(
                self._performance.get(self._current_model, [0])[-self.performance_window:]
            )
            if improvement > self.switch_threshold:
                logger.info(f"Switching model: {self._current_model} -> {best_name} (improvement={improvement:.3f})")
                self._current_model = best_name

        return best_name or ""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the currently selected model."""
        model_name = self.select_model()
        if model_name and model_name in self._models:
            return self._models[model_name].predict(X)
        return np.zeros(X.shape[0])

    def update_performance(self, name: str, score: float) -> None:
        if name not in self._performance:
            self._performance[name] = []
        self._performance[name].append(score)
        if len(self._performance[name]) > self.performance_window * 2:
            self._performance[name] = self._performance[name][-self.performance_window:]

    def get_status(self) -> dict:
        return {
            "current_model": self._current_model,
            "n_models": len(self._models),
            "performance": {
                name: {
                    "mean": float(np.mean(scores[-50:])) if scores else 0,
                    "n": len(scores),
                }
                for name, scores in self._performance.items()
            },
        }
