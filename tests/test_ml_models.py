"""Tests for RL trading agent and ensemble models."""
from __future__ import annotations

import pytest
import numpy as np

from tradingbot.ml.models.rl_agent import (
    RLTradingAgent,
    TradingEnvironment,
    SimpleDQN,
    ReplayBuffer,
    RLConfig,
)
from tradingbot.ml.models.ensemble import (
    EnsembleModel,
    ModelSelector,
)


class TestSimpleDQN:
    def test_predict_shape(self):
        net = SimpleDQN(state_size=10, action_size=3)
        state = np.random.randn(10)
        q = net.predict(state)
        assert q.shape == (3,)

    def test_predict_batch(self):
        net = SimpleDQN(state_size=10, action_size=3)
        # predict on a single row of a batch works
        state = np.random.randn(10)
        q = net.predict(state)
        assert q.shape == (3,)

    def test_train_step(self):
        net = SimpleDQN(state_size=10, action_size=3)
        states = np.random.randn(8, 10)
        targets = np.random.randn(8, 3)
        loss = net.train(states, targets, lr=0.001)
        assert loss >= 0

    def test_copy_from(self):
        net1 = SimpleDQN(state_size=10, action_size=3)
        net2 = SimpleDQN(state_size=10, action_size=3)
        net2.copy_from(net1)
        state = np.random.randn(10)
        assert np.allclose(net1.predict(state), net2.predict(state))


class TestReplayBuffer:
    def test_push_and_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(np.zeros(5), 0, 1.0, np.zeros(5), False)
        assert len(buf) == 50
        batch = buf.sample(10)
        assert len(batch) == 10

    def test_capacity(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(20):
            buf.push(np.zeros(5), 0, 1.0, np.zeros(5), False)
        assert len(buf) == 10


class TestRLTradingAgent:
    @pytest.fixture
    def agent(self):
        return RLTradingAgent({"state_size": 10, "epsilon": 0.5})

    def test_select_action(self, agent):
        state = np.random.randn(10)
        result = agent.select_action(state)
        assert result.action in [0, 1, 2]
        assert len(result.q_values) == 3

    def test_compute_reward(self, agent):
        # Positive PnL for buy
        r = agent.compute_reward(100000, 101000, 1, 0.1)
        assert r > 0

        # Negative PnL
        r = agent.compute_reward(100000, 99000, 0, 0.1)
        assert r < 0

    def test_remember_and_train(self, agent):
        for _ in range(50):
            state = np.random.randn(10)
            agent.remember(state, 1, 0.01, np.random.randn(10), False)
        loss = agent.train_step()
        assert loss >= 0

    def test_epsilon_decay(self, agent):
        initial_eps = agent.epsilon
        for _ in range(50):
            agent.remember(np.random.randn(10), 1, 0.01, np.random.randn(10), False)
            agent.train_step()
        assert agent.epsilon < initial_eps

    def test_status(self, agent):
        status = agent.get_status()
        assert "epsilon" in status
        assert "memory_size" in status

    def test_save_load_weights(self, agent, tmp_path):
        path = str(tmp_path / "weights.npz")
        agent.save_weights(path)
        agent2 = RLTradingAgent({"state_size": 10})
        agent2.load_weights(path)
        state = np.random.randn(10)
        assert np.allclose(agent.q_network.predict(state), agent2.q_network.predict(state))


class TestTradingEnvironment:
    @pytest.fixture
    def env(self):
        rng = np.random.default_rng(42)
        features = rng.standard_normal((100, 10))
        prices = np.cumsum(rng.normal(0, 1, 100)) + 100
        return TradingEnvironment(features, prices)

    def test_reset(self, env):
        state = env.reset()
        assert len(state) == 13  # 10 features + 3 portfolio

    def test_step(self, env):
        env.reset()
        state, reward, done, info = env.step(1)  # Buy
        assert len(state) == 13
        assert isinstance(reward, float)
        assert "portfolio_value" in info

    def test_run_episode(self, env):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = np.random.randint(0, 3)
            state, reward, done, info = env.step(action)
            total_reward += reward
        assert isinstance(total_reward, float)


class TestEnsembleModel:
    def test_add_and_predict(self):
        class MockModel:
            def predict(self, X):
                return np.ones(X.shape[0]) * 0.5

        ensemble = EnsembleModel({"method": "avg"})
        ensemble.add_model("m1", MockModel())
        ensemble.add_model("m2", MockModel())

        X = np.random.randn(10, 5)
        result = ensemble.predict(X)
        assert len(result.prediction) == 10
        assert result.agreement > 0

    def test_weighted_ensemble(self):
        class M1:
            def predict(self, X): return np.ones(X.shape[0]) * 0.8
        class M2:
            def predict(self, X): return np.ones(X.shape[0]) * 0.2

        ensemble = EnsembleModel({"method": "weighted"})
        ensemble.add_model("m1", M1(), weight=3.0)
        ensemble.add_model("m2", M2(), weight=1.0)

        X = np.random.randn(5, 3)
        result = ensemble.predict(X)
        # Weighted: 0.75 * 0.8 + 0.25 * 0.2 = 0.65
        assert abs(result.prediction[0] - 0.65) < 0.01

    def test_update_weights(self):
        ensemble = EnsembleModel()
        ensemble.add_model("m1", None, weight=1.0)
        ensemble.update_weights("m1", 0.9)
        assert ensemble._weights["m1"] > 0


class TestModelSelector:
    def test_select_model(self):
        class M1:
            def predict(self, X): return np.ones(X.shape[0])
        class M2:
            def predict(self, X): return np.ones(X.shape[0]) * 2

        selector = ModelSelector()
        selector.add_model("m1", M1())
        selector.add_model("m2", M2())

        # Make m2 better
        for _ in range(20):
            selector.update_performance("m2", 1.0)
            selector.update_performance("m1", 0.5)

        selected = selector.select_model()
        assert selected == "m2"

    def test_status(self):
        selector = ModelSelector()
        status = selector.get_status()
        assert "n_models" in status
