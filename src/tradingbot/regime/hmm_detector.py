"""Hidden Markov Model Regime Detection."""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np

from ..core.types import RegimeState

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """Detects market regimes using Hidden Markov Models.

    Identifies 4 states:
    - Bull + Low Vol: Trending up, calm (best for trend following)
    - Bull + High Vol: Trending up, volatile (best for momentum)
    - Bear + Low Vol: Drifting down, calm (best for mean reversion)
    - Bear + High Vol: Crashing, panic (best for arbitrage/hedging)
    """

    def __init__(self, config: dict):
        self.n_states = config.get("hmm_states", 4)
        self.lookback = config.get("lookback_days", 90)
        self.vol_lookback = config.get("vol_lookback", 20)
        self._model = None
        self._is_fitted = False
        self._last_regime: Optional[RegimeState] = None

    async def fit(self, returns: np.ndarray, volatilities: np.ndarray) -> None:
        """Fit the HMM on historical data."""
        try:
            from hmmlearn.hmm import GaussianHMM

            # Combine returns and volatilities as observations
            X = np.column_stack([returns, volatilities])
            X = X[~np.isnan(X).any(axis=1)]

            if len(X) < self.n_states * 10:
                logger.warning("Insufficient data for HMM fitting")
                return

            self._model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
                tol=0.01,
            )
            self._model.fit(X)
            self._is_fitted = True
            logger.info(f"HMM fitted with {self.n_states} states on {len(X)} observations")

        except ImportError:
            logger.warning("hmmlearn not installed, using simple regime detection")
            self._is_fitted = False

    async def detect(self, returns: np.ndarray, volatilities: np.ndarray) -> RegimeState:
        """Detect current regime from recent data."""
        if not self._is_fitted or self._model is None:
            return self._simple_detect(returns, volatilities)

        X = np.column_stack([returns[-1:], volatilities[-1:]])
        X = X[~np.isnan(X).any(axis=1)]

        if len(X) == 0:
            return self._simple_detect(returns, volatilities)

        try:
            state = self._model.predict(X)[0]
            probs = self._model.predict_proba(X)[0]

            # Map HMM states to regime labels
            means = self._model.means_
            regime = self._map_state_to_regime(state, means, volatilities[-1])

            # Calculate transition probabilities
            trans_mat = self._model.transmat_
            transition_prob = 1.0 - trans_mat[state, state]

            self._last_regime = RegimeState(
                regime=regime,
                confidence=float(probs[state]),
                volatility_percentile=self._vol_percentile(volatilities[-1], volatilities),
                trend_strength=float(np.mean(returns[-20:])),
                correlation_regime="normal",
                transition_probability=transition_prob,
            )
            return self._last_regime

        except Exception as e:
            logger.error(f"HMM prediction failed: {e}")
            return self._simple_detect(returns, volatilities)

    def _simple_detect(self, returns: np.ndarray, volatilities: np.ndarray) -> RegimeState:
        """Simple regime detection without HMM (fallback)."""
        if len(returns) < 20:
            return RegimeState(
                regime="unknown",
                confidence=0.0,
                volatility_percentile=0.5,
                trend_strength=0.0,
                correlation_regime="normal",
                transition_probability=0.0,
            )

        recent_return = float(np.mean(returns[-20:]))
        recent_vol = float(volatilities[-1]) if len(volatilities) > 0 else 0.0
        vol_pctl = self._vol_percentile(recent_vol, volatilities)

        # Simple classification
        if recent_return > 0 and vol_pctl < 0.5:
            regime = "bull_low_vol"
        elif recent_return > 0 and vol_pctl >= 0.5:
            regime = "bull_high_vol"
        elif recent_return <= 0 and vol_pctl < 0.5:
            regime = "bear_low_vol"
        else:
            regime = "bear_high_vol"

        self._last_regime = RegimeState(
            regime=regime,
            confidence=0.6,
            volatility_percentile=vol_pctl,
            trend_strength=recent_return,
            correlation_regime="normal",
            transition_probability=0.05,
        )
        return self._last_regime

    def _map_state_to_regime(self, state: int, means: np.ndarray, current_vol: float) -> str:
        """Map HMM state index to regime label."""
        state_mean_return = means[state, 0]
        state_mean_vol = means[state, 1]
        vol_pctl = self._vol_percentile(state_mean_vol, np.array([m[1] for m in means]))

        if state_mean_return > 0 and vol_pctl < 0.5:
            return "bull_low_vol"
        elif state_mean_return > 0:
            return "bull_high_vol"
        elif vol_pctl < 0.5:
            return "bear_low_vol"
        else:
            return "bear_high_vol"

    def _vol_percentile(self, vol: float, all_vols: np.ndarray) -> float:
        """Calculate volatility percentile."""
        if len(all_vols) == 0:
            return 0.5
        return float(np.mean(all_vols < vol))

    async def get_transition_probabilities(self) -> dict:
        """Get regime transition probabilities."""
        if self._model is None:
            return {}
        return {
            f"state_{i}_to_{j}": float(self._model.transmat_[i, j])
            for i in range(self.n_states)
            for j in range(self.n_states)
        }
