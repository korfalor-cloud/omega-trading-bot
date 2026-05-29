"""Hyperparameter Optimization — Search for optimal model configurations.

Implements:
- Grid search over parameter space
- Random search with configurable budget
- Simplified Bayesian optimization (surrogate model)
- K-fold cross-validation for evaluation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

logger = logging.getLogger(__name__)


class TrainableModel(Protocol):
    """Protocol for models compatible with the optimiser."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class TrialResult:
    """Result of a single hyperparameter evaluation."""
    params: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    fold_scores: list[float] = field(default_factory=list)
    duration: float = 0.0


@dataclass
class SearchResults:
    """Aggregated results from a hyperparameter search."""
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = -np.inf
    all_trials: list[TrialResult] = field(default_factory=list)
    n_trials: int = 0


# ── Cross-Validation ──────────────────────────────────────────────


def kfold_cv(
    model_factory: Callable[[], TrainableModel],
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    metric: str = "mse",
) -> list[float]:
    """K-fold cross-validation returning per-fold scores.

    metric: 'mse' (lower is better, negated) or 'accuracy'
    """
    n = len(X)
    if n < k:
        k = max(2, n)

    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)

    scores: list[float] = []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        if metric == "mse":
            score = -float(np.mean((preds - y_test) ** 2))
        elif metric == "accuracy":
            score = float(np.mean(np.round(preds) == y_test))
        elif metric == "mae":
            score = -float(np.mean(np.abs(preds - y_test)))
        else:
            score = -float(np.mean((preds - y_test) ** 2))

        scores.append(score)

    return scores


# ── Grid Search ───────────────────────────────────────────────────


class GridSearch:
    """Exhaustive search over a discrete parameter grid."""

    def __init__(self, param_grid: dict[str, list], k_folds: int = 5, metric: str = "mse"):
        self.param_grid = param_grid
        self.k_folds = k_folds
        self.metric = metric
        self._results: SearchResults = SearchResults()

    def search(
        self,
        model_factory: Callable[[dict[str, Any]], TrainableModel],
        X: np.ndarray,
        y: np.ndarray,
    ) -> SearchResults:
        """Run grid search and return best parameters."""
        param_combos = self._expand_grid()
        logger.info("Grid search: %d combinations x %d folds", len(param_combos), self.k_folds)

        best_score = -np.inf
        best_params: dict[str, Any] = {}
        trials: list[TrialResult] = []

        for combo in param_combos:
            factory = lambda p=combo: model_factory(p)
            fold_scores = kfold_cv(factory, X, y, k=self.k_folds, metric=self.metric)
            score = float(np.mean(fold_scores))

            trial = TrialResult(params=combo, score=score, fold_scores=fold_scores)
            trials.append(trial)

            if score > best_score:
                best_score = score
                best_params = combo

        self._results = SearchResults(
            best_params=best_params,
            best_score=best_score,
            all_trials=trials,
            n_trials=len(trials),
        )
        logger.info("Grid search done. Best score: %.4f", best_score)
        return self._results

    def _expand_grid(self) -> list[dict[str, Any]]:
        """Expand parameter grid into list of dicts."""
        keys = list(self.param_grid.keys())
        values = [self.param_grid[k] for k in keys]

        combos: list[dict[str, Any]] = [{}]
        for key, vals in zip(keys, values):
            new_combos: list[dict[str, Any]] = []
            for combo in combos:
                for v in vals:
                    new_combo = dict(combo)
                    new_combo[key] = v
                    new_combos.append(new_combo)
            combos = new_combos

        return combos


# ── Random Search ─────────────────────────────────────────────────


class RandomSearch:
    """Random search over parameter distributions.

    Supports continuous uniform, log-uniform, and discrete integer/choice
    distributions via param_spec dicts:
        {"lr": ("log_uniform", 1e-5, 1e-1),
         "n_layers": ("int", 1, 5),
         "activation": ("choice", ["relu", "tanh"])}
    """

    def __init__(self, param_spec: dict[str, tuple], n_trials: int = 50, k_folds: int = 5, metric: str = "mse", seed: int = 42):
        self.param_spec = param_spec
        self.n_trials = n_trials
        self.k_folds = k_folds
        self.metric = metric
        self._rng = np.random.RandomState(seed)
        self._results: SearchResults = SearchResults()

    def search(
        self,
        model_factory: Callable[[dict[str, Any]], TrainableModel],
        X: np.ndarray,
        y: np.ndarray,
    ) -> SearchResults:
        """Run random search."""
        logger.info("Random search: %d trials x %d folds", self.n_trials, self.k_folds)

        best_score = -np.inf
        best_params: dict[str, Any] = {}
        trials: list[TrialResult] = []

        for t in range(self.n_trials):
            params = self._sample_params()
            factory = lambda p=params: model_factory(p)
            fold_scores = kfold_cv(factory, X, y, k=self.k_folds, metric=self.metric)
            score = float(np.mean(fold_scores))

            trial = TrialResult(params=params, score=score, fold_scores=fold_scores)
            trials.append(trial)

            if score > best_score:
                best_score = score
                best_params = params

        self._results = SearchResults(
            best_params=best_params,
            best_score=best_score,
            all_trials=trials,
            n_trials=len(trials),
        )
        logger.info("Random search done. Best score: %.4f", best_score)
        return self._results

    def _sample_params(self) -> dict[str, Any]:
        """Sample one parameter combination."""
        params: dict[str, Any] = {}
        for key, spec in self.param_spec.items():
            dist_type = spec[0]
            if dist_type == "uniform":
                params[key] = float(self._rng.uniform(spec[1], spec[2]))
            elif dist_type == "log_uniform":
                log_val = self._rng.uniform(np.log(spec[1]), np.log(spec[2]))
                params[key] = float(np.exp(log_val))
            elif dist_type == "int":
                params[key] = int(self._rng.randint(spec[1], spec[2] + 1))
            elif dist_type == "choice":
                idx = self._rng.randint(len(spec[1]))
                params[key] = spec[1][idx]
            else:
                params[key] = spec[1]
        return params


# ── Bayesian Optimization (Simplified) ────────────────────────────


class BayesianOptimizer:
    """Simplified Bayesian optimization using a Gaussian process surrogate.

    Fits a GP with RBF kernel to observed (params -> score) pairs and
    uses Expected Improvement (EI) as the acquisition function.
    """

    def __init__(
        self,
        param_spec: dict[str, tuple],
        n_trials: int = 30,
        k_folds: int = 5,
        metric: str = "mse",
        seed: int = 42,
        n_initial: int = 5,
    ):
        self.param_spec = param_spec
        self.n_trials = n_trials
        self.k_folds = k_folds
        self.metric = metric
        self.n_initial = n_initial
        self._rng = np.random.RandomState(seed)
        self._results: SearchResults = SearchResults()

        # GP hyperparameters
        self._length_scale = 1.0
        self._signal_var = 1.0
        self._noise_var = 0.1

    def search(
        self,
        model_factory: Callable[[dict[str, Any]], TrainableModel],
        X: np.ndarray,
        y: np.ndarray,
    ) -> SearchResults:
        """Run Bayesian optimization."""
        logger.info("Bayesian optimization: %d trials x %d folds", self.n_trials, self.k_folds)

        # Observed points (normalised param vectors and scores)
        X_obs: list[np.ndarray] = []
        y_obs: list[float] = []
        trials: list[TrialResult] = []
        best_score = -np.inf
        best_params: dict[str, Any] = {}

        for t in range(self.n_trials):
            if t < self.n_initial:
                # Random exploration phase
                params = self._sample_params()
            else:
                # Acquisition-guided phase
                params = self._next_candidate(X_obs, y_obs)

            # Evaluate
            factory = lambda p=params: model_factory(p)
            fold_scores = kfold_cv(factory, X, y, k=self.k_folds, metric=self.metric)
            score = float(np.mean(fold_scores))

            # Record
            x_vec = self._params_to_vector(params)
            X_obs.append(x_vec)
            y_obs.append(score)

            trial = TrialResult(params=params, score=score, fold_scores=fold_scores)
            trials.append(trial)

            if score > best_score:
                best_score = score
                best_params = params

            # Update GP length scale based on observations
            if len(y_obs) > self.n_initial:
                self._update_gp(X_obs, y_obs)

        self._results = SearchResults(
            best_params=best_params,
            best_score=best_score,
            all_trials=trials,
            n_trials=len(trials),
        )
        logger.info("Bayesian optimisation done. Best score: %.4f", best_score)
        return self._results

    # ── GP surrogate ──────────────────────────────────────────────

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """RBF kernel matrix between X1 (m, d) and X2 (n, d)."""
        sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
        sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist = sq1 - 2 * X1 @ X2.T + sq2.T
        return self._signal_var * np.exp(-0.5 * dist / (self._length_scale ** 2))

    def _gp_predict(self, X_train: list[np.ndarray], y_train: list[float], X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """GP posterior mean and std at X_test."""
        Xt = np.array(X_train)
        yt = np.array(y_train)

        K = self._rbf_kernel(Xt, Xt) + self._noise_var * np.eye(len(Xt))
        K_s = self._rbf_kernel(Xt, X_test)
        K_ss = self._rbf_kernel(X_test, X_test) + self._noise_var * np.eye(len(X_test))

        # Cholesky solve
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yt))
            v = np.linalg.solve(L, K_s)
        except np.linalg.LinAlgError:
            alpha = np.linalg.solve(K + 1e-6 * np.eye(len(K)), yt)
            v = np.linalg.solve(K + 1e-6 * np.eye(len(K)), K_s)

        mu = K_s.T @ alpha
        cov = K_ss - v.T @ v
        std = np.sqrt(np.maximum(np.diag(cov), 0.0))

        return mu, std

    def _expected_improvement(self, mu: np.ndarray, std: np.ndarray, best: float) -> np.ndarray:
        """Expected improvement acquisition function."""
        from scipy.stats import norm  # type: ignore

        # If scipy unavailable, use approximation
        try:
            z = (mu - best) / (std + 1e-10)
            ei = (mu - best) * norm.cdf(z) + std * norm.pdf(z)
        except ImportError:
            # Simple approximation
            z = (mu - best) / (std + 1e-10)
            # Approximate standard normal CDF and PDF
            phi = 0.5 * (1 + np.tanh(z * 0.7978))  # approx CDF
            pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
            ei = (mu - best) * phi + std * pdf

        ei[std < 1e-10] = 0.0
        return ei

    def _next_candidate(self, X_obs: list[np.ndarray], y_obs: list[float]) -> dict[str, Any]:
        """Pick the next parameter set by maximising acquisition function."""
        # Generate random candidates
        n_candidates = 200
        candidates = [self._sample_params() for _ in range(n_candidates)]
        X_cand = np.array([self._params_to_vector(p) for p in candidates])

        mu, std = self._gp_predict(X_obs, y_obs, X_cand)
        best = max(y_obs)
        ei = self._expected_improvement(mu, std, best)

        best_idx = int(np.argmax(ei))
        return candidates[best_idx]

    def _update_gp(self, X_obs: list[np.ndarray], y_obs: list[float]) -> None:
        """Update GP hyperparameters based on data."""
        # Simple heuristic: length scale = median pairwise distance
        X_arr = np.array(X_obs)
        if len(X_arr) > 1:
            dists = []
            for i in range(len(X_arr)):
                for j in range(i + 1, len(X_arr)):
                    dists.append(np.linalg.norm(X_arr[i] - X_arr[j]))
            if dists:
                self._length_scale = max(0.1, np.median(dists))

    # ── helpers ────────────────────────────────────────────────────

    def _sample_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for key, spec in self.param_spec.items():
            dist_type = spec[0]
            if dist_type == "uniform":
                params[key] = float(self._rng.uniform(spec[1], spec[2]))
            elif dist_type == "log_uniform":
                log_val = self._rng.uniform(np.log(spec[1]), np.log(spec[2]))
                params[key] = float(np.exp(log_val))
            elif dist_type == "int":
                params[key] = int(self._rng.randint(spec[1], spec[2] + 1))
            elif dist_type == "choice":
                idx = self._rng.randint(len(spec[1]))
                params[key] = spec[1][idx]
            else:
                params[key] = spec[1]
        return params

    def _params_to_vector(self, params: dict[str, Any]) -> np.ndarray:
        """Convert param dict to normalised numeric vector."""
        vec: list[float] = []
        for key in sorted(self.param_spec.keys()):
            spec = self.param_spec[key]
            val = params.get(key, 0)
            dist_type = spec[0]
            if dist_type in ("uniform", "log_uniform"):
                lo, hi = spec[1], spec[2]
                vec.append((float(val) - lo) / (hi - lo + 1e-10))
            elif dist_type == "int":
                lo, hi = spec[1], spec[2]
                vec.append((int(val) - lo) / (hi - lo + 1e-10))
            elif dist_type == "choice":
                choices = spec[1]
                if val in choices:
                    vec.append(choices.index(val) / max(1, len(choices) - 1))
                else:
                    vec.append(0.0)
            else:
                vec.append(float(val))
        return np.array(vec)
