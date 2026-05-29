"""Automated Feature Engineering — Generate and select features from OHLCV data.

Implements:
- Rolling statistics generation (mean, std, skew, kurtosis, min, max)
- Interaction features (ratios, products, differences of base features)
- Lag features (past N values of a base feature)
- Feature selection by variance and mutual-information proxy
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for a named feature matrix."""
    names: list[str] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    selected_mask: np.ndarray = field(default_factory=lambda: np.array([]))


class RollingFeatureGenerator:
    """Generate rolling-window statistics from a 1-D signal.

    Each generator method returns an (n, window_stat_count) array aligned
    with the input signal, with NaN where the window is not yet filled.
    """

    def __init__(self, windows: list[int] | None = None):
        self.windows = windows or [5, 10, 20, 50]

    def generate(self, signal: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Generate all rolling features for a single signal.

        Returns:
            matrix: (n, n_features) array
            names: feature name list
        """
        all_cols: list[np.ndarray] = []
        names: list[str] = []

        for w in self.windows:
            roll_mean = self._rolling_mean(signal, w)
            roll_std = self._rolling_std(signal, w)
            roll_min = self._rolling_min(signal, w)
            roll_max = self._rolling_max(signal, w)
            roll_skew = self._rolling_skew(signal, w)
            roll_kurt = self._rolling_kurtosis(signal, w)

            all_cols.extend([roll_mean, roll_std, roll_min, roll_max, roll_skew, roll_kurt])
            names.extend([
                f"roll_mean_{w}", f"roll_std_{w}",
                f"roll_min_{w}", f"roll_max_{w}",
                f"roll_skew_{w}", f"roll_kurt_{w}",
            ])

        matrix = np.column_stack(all_cols) if all_cols else np.array([]).reshape(len(signal), 0)
        return matrix, names

    # ── rolling helpers ────────────────────────────────────────────

    @staticmethod
    def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(x), np.nan)
        if len(x) < w:
            return out
        cs = np.cumsum(x)
        cs[w:] = cs[w:] - cs[:-w]
        out[w - 1:] = cs[w - 1:] / w
        return out

    @staticmethod
    def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(x), np.nan)
        if len(x) < w:
            return out
        for i in range(w - 1, len(x)):
            out[i] = np.std(x[i - w + 1 : i + 1], ddof=0)
        return out

    @staticmethod
    def _rolling_min(x: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(x), np.nan)
        if len(x) < w:
            return out
        for i in range(w - 1, len(x)):
            out[i] = np.min(x[i - w + 1 : i + 1])
        return out

    @staticmethod
    def _rolling_max(x: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(x), np.nan)
        if len(x) < w:
            return out
        for i in range(w - 1, len(x)):
            out[i] = np.max(x[i - w + 1 : i + 1])
        return out

    @staticmethod
    def _rolling_skew(x: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(x), np.nan)
        if len(x) < w:
            return out
        for i in range(w - 1, len(x)):
            window = x[i - w + 1 : i + 1]
            m = np.mean(window)
            s = np.std(window, ddof=0)
            if s > 0:
                out[i] = float(np.mean(((window - m) / s) ** 3))
            else:
                out[i] = 0.0
        return out

    @staticmethod
    def _rolling_kurtosis(x: np.ndarray, w: int) -> np.ndarray:
        out = np.full(len(x), np.nan)
        if len(x) < w:
            return out
        for i in range(w - 1, len(x)):
            window = x[i - w + 1 : i + 1]
            m = np.mean(window)
            s = np.std(window, ddof=0)
            if s > 0:
                out[i] = float(np.mean(((window - m) / s) ** 4) - 3.0)
            else:
                out[i] = 0.0
        return out


class InteractionFeatureGenerator:
    """Generate pairwise interaction features from a base feature matrix.

    Produces ratio, product, and difference features for selected pairs.
    """

    def __init__(self, max_base_features: int = 10):
        self.max_base_features = max_base_features

    def generate(self, X: np.ndarray, base_names: list[str]) -> tuple[np.ndarray, list[str]]:
        """Generate interaction features.

        Args:
            X: (n, d) base feature matrix
            base_names: names of the d base features

        Returns:
            matrix: (n, n_interactions) array
            names: interaction feature name list
        """
        n_cols = min(X.shape[1], self.max_base_features)
        X_trim = X[:, :n_cols]
        trim_names = base_names[:n_cols]

        cols: list[np.ndarray] = []
        names: list[str] = []

        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                fi = X_trim[:, i]
                fj = X_trim[:, j]

                # Product
                cols.append(fi * fj)
                names.append(f"{trim_names[i]}_x_{trim_names[j]}")

                # Ratio (safe division)
                denom = np.where(np.abs(fj) < 1e-10, 1.0, fj)
                cols.append(fi / denom)
                names.append(f"{trim_names[i]}_div_{trim_names[j]}")

                # Difference
                cols.append(fi - fj)
                names.append(f"{trim_names[i]}_minus_{trim_names[j]}")

        matrix = np.column_stack(cols) if cols else np.array([]).reshape(X.shape[0], 0)
        return matrix, names


class LagFeatureGenerator:
    """Generate lagged versions of features (values N steps in the past)."""

    def __init__(self, lags: list[int] | None = None):
        self.lags = lags or [1, 2, 3, 5, 10]

    def generate(self, signal: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Generate lag features for a 1-D signal.

        Returns:
            matrix: (n, n_lags) array
            names: lag feature name list
        """
        cols: list[np.ndarray] = []
        names: list[str] = []

        for lag in self.lags:
            lagged = np.full(len(signal), np.nan)
            if lag < len(signal):
                lagged[lag:] = signal[:-lag]
            cols.append(lagged)
            names.append(f"lag_{lag}")

        matrix = np.column_stack(cols) if cols else np.array([]).reshape(len(signal), 0)
        return matrix, names


class FeatureSelector:
    """Select the most informative features using variance and correlation filters."""

    def __init__(self, max_features: int = 50, variance_threshold: float = 1e-6, correlation_threshold: float = 0.95):
        self.max_features = max_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

    def select(self, X: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Select features by variance then by correlation.

        Returns:
            X_selected: filtered feature matrix
            selected_names: names of kept features
            mask: boolean mask of selected columns
        """
        if X.shape[1] == 0:
            return X, names, np.array([], dtype=bool)

        # Replace NaN with 0 for computation
        X_clean = np.nan_to_num(X, nan=0.0)

        # Step 1: Variance filter
        variances = np.var(X_clean, axis=0)
        var_mask = variances > self.variance_threshold

        if var_mask.sum() == 0:
            # Keep top-N by variance
            top_idx = np.argsort(variances)[-self.max_features:]
            var_mask = np.zeros(X.shape[1], dtype=bool)
            var_mask[top_idx] = True

        X_filtered = X_clean[:, var_mask]
        filtered_names = [names[i] for i in range(len(names)) if var_mask[i]]

        # Step 2: Correlation filter
        if X_filtered.shape[1] > 1:
            corr = np.corrcoef(X_filtered.T)
            to_remove: set[int] = set()
            for i in range(corr.shape[0]):
                if i in to_remove:
                    continue
                for j in range(i + 1, corr.shape[1]):
                    if j in to_remove:
                        continue
                    if abs(corr[i, j]) > self.correlation_threshold:
                        # Remove the feature with lower variance
                        if variances[list(range(len(names)))[i] if i < len(names) else i] < \
                           variances[list(range(len(names)))[j] if j < len(names) else j]:
                            to_remove.add(i)
                        else:
                            to_remove.add(j)

            keep = [i for i in range(X_filtered.shape[1]) if i not in to_remove]
            if keep:
                X_filtered = X_filtered[:, keep]
                filtered_names = [filtered_names[i] for i in keep]
            else:
                keep = list(range(min(self.max_features, X_filtered.shape[1])))
                X_filtered = X_filtered[:, keep]
                filtered_names = [filtered_names[i] for i in keep]

        # Step 3: Cap at max_features
        if X_filtered.shape[1] > self.max_features:
            # Rank by variance
            sub_var = np.var(X_filtered, axis=0)
            top_idx = np.argsort(sub_var)[-self.max_features:]
            X_filtered = X_filtered[:, top_idx]
            filtered_names = [filtered_names[i] for i in top_idx]

        # Build the final mask
        selected_set = set(filtered_names)
        mask = np.array([n in selected_set for n in names], dtype=bool)

        logger.info(
            "Feature selection: %d -> %d features",
            X.shape[1], X_filtered.shape[1],
        )
        return X_filtered, filtered_names, mask


class AutoFeatureEngine:
    """End-to-end automated feature engineering from OHLCV arrays.

    Combines rolling statistics, interaction features, and lag features,
    then applies feature selection to produce a compact feature matrix.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.rolling_windows = config.get("rolling_windows", [5, 10, 20, 50])
        self.lags = config.get("lags", [1, 2, 3, 5, 10])
        self.max_interaction_base = config.get("max_interaction_base", 6)
        self.max_features = config.get("max_features", 50)

        self._roller = RollingFeatureGenerator(self.rolling_windows)
        self._interactor = InteractionFeatureGenerator(self.max_interaction_base)
        self._lagger = LagFeatureGenerator(self.lags)
        self._selector = FeatureSelector(max_features=self.max_features)
        self._selected_mask: np.ndarray = np.array([])
        self._feature_names: list[str] = []

    def build_features(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
    ) -> FeatureSet:
        """Build a complete feature set from raw OHLCV arrays.

        Returns a FeatureSet with the selected feature matrix and names.
        """
        all_cols: list[np.ndarray] = []
        all_names: list[str] = []

        # ── base signals ───────────────────────────────────────────
        log_returns = np.diff(np.log(np.maximum(closes, 1e-10)))
        log_returns = np.concatenate([[0.0], log_returns])  # align length

        signals = {
            "close": closes,
            "log_ret": log_returns,
            "volume": volumes,
        }
        if highs is not None:
            signals["high"] = highs
        if lows is not None:
            signals["low"] = lows
            if highs is not None:
                signals["range"] = highs - lows

        # ── rolling features ───────────────────────────────────────
        for sig_name, sig in signals.items():
            roll_mat, roll_names = self._roller.generate(sig)
            renamed = [f"{sig_name}_{n}" for n in roll_names]
            all_cols.append(roll_mat)
            all_names.extend(renamed)

        # ── lag features for close and volume ──────────────────────
        for sig_name in ["close", "volume"]:
            lag_mat, lag_names = self._lagger.generate(signals[sig_name])
            renamed = [f"{sig_name}_{n}" for n in lag_names]
            all_cols.append(lag_mat)
            all_names.extend(renamed)

        # ── combine ────────────────────────────────────────────────
        if all_cols:
            X = np.column_stack(all_cols)
        else:
            X = np.array([]).reshape(len(closes), 0)

        # ── interaction features (on subset of base columns) ───────
        n_base = min(X.shape[1], self.max_interaction_base * 6)
        if n_base > 1:
            inter_mat, inter_names = self._interactor.generate(X[:, :n_base], all_names[:n_base])
            X = np.column_stack([X, inter_mat])
            all_names.extend(inter_names)

        # ── feature selection ──────────────────────────────────────
        # Drop rows with NaN for selection, keep mask
        valid_rows = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_rows]

        if len(X_valid) > 0 and X_valid.shape[1] > 0:
            X_sel, sel_names, self._selected_mask = self._selector.select(X_valid, all_names)
        else:
            X_sel = X_valid
            sel_names = all_names
            self._selected_mask = np.ones(X.shape[1], dtype=bool)

        self._feature_names = sel_names

        return FeatureSet(names=sel_names, matrix=X_sel, selected_mask=self._selected_mask)

    def transform(self, closes: np.ndarray, volumes: np.ndarray,
                  highs: np.ndarray | None = None, lows: np.ndarray | None = None) -> np.ndarray:
        """Transform new data using the previously fitted feature selection mask."""
        fs = self.build_features(closes, volumes, highs, lows)

        if self._selected_mask.size > 0 and fs.matrix.shape[1] == self._selected_mask.size:
            return fs.matrix[:, self._selected_mask]
        return fs.matrix

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def get_status(self) -> dict:
        return {
            "n_features": len(self._feature_names),
            "rolling_windows": self.rolling_windows,
            "lags": self.lags,
            "max_features": self.max_features,
        }
