"""Reinforcement Learning Trading Agent.

Implements a DQN (Deep Q-Network) agent for trading decisions.
The agent learns to take actions (buy/sell/hold) based on market state.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradingState:
    """State representation for the RL agent."""
    features: np.ndarray  # Technical indicators, price features
    portfolio_value: float = 0.0
    position: float = 0.0  # Current position size
    unrealized_pnl: float = 0.0
    timestamp: float = 0.0


@dataclass
class RLConfig:
    """Configuration for the RL agent."""
    state_size: int = 20  # Number of features in state
    action_size: int = 3  # buy, sell, hold
    learning_rate: float = 0.001
    gamma: float = 0.95  # Discount factor
    epsilon: float = 1.0  # Exploration rate
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    target_update_freq: int = 100


@dataclass
class RLTradeResult:
    """Result of an RL agent's trading decision."""
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: float = 0.0
    q_values: list[float] = field(default_factory=list)
    exploration: bool = False


class ReplayBuffer:
    """Experience replay buffer for DQN training."""

    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class SimpleDQN:
    """Simple Q-network without deep learning framework dependency.

    Uses a basic neural network with numpy for portability.
    For production, swap with PyTorch/TensorFlow implementation.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        self.state_size = state_size
        self.action_size = action_size

        # Xavier initialization
        scale1 = np.sqrt(2.0 / (state_size + hidden_size))
        scale2 = np.sqrt(2.0 / (hidden_size + hidden_size))
        scale3 = np.sqrt(2.0 / (hidden_size + action_size))

        self.w1 = np.random.randn(state_size, hidden_size) * scale1
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, hidden_size) * scale2
        self.b2 = np.zeros(hidden_size)
        self.w3 = np.random.randn(hidden_size, action_size) * scale3
        self.b3 = np.zeros(action_size)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Forward pass — returns Q-values for each action."""
        x = np.atleast_2d(state)
        h1 = np.maximum(0, x @ self.w1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)  # ReLU
        q = h2 @ self.w3 + self.b3
        return q.flatten()

    def train(self, states: np.ndarray, targets: np.ndarray, lr: float = 0.001) -> float:
        """Simple gradient descent training step."""
        batch_size = states.shape[0]

        # Forward pass
        h1 = np.maximum(0, states @ self.w1 + self.b1)
        h2 = np.maximum(0, h1 @ self.w2 + self.b2)
        q = h2 @ self.w3 + self.b3

        # Loss (MSE)
        loss = np.mean((q - targets) ** 2)

        # Backward pass
        dq = 2 * (q - targets) / batch_size

        dw3 = h2.T @ dq
        db3 = np.sum(dq, axis=0)

        dh2 = dq @ self.w3.T
        dh2[h2 <= 0] = 0  # ReLU gradient

        dw2 = h1.T @ dh2
        db2 = np.sum(dh2, axis=0)

        dh1 = dh2 @ self.w2.T
        dh1[h1 <= 0] = 0  # ReLU gradient

        dw1 = states.T @ dh1
        db1 = np.sum(dh1, axis=0)

        # Update weights
        self.w3 -= lr * dw3
        self.b3 -= lr * db3
        self.w2 -= lr * dw2
        self.b2 -= lr * db2
        self.w1 -= lr * dw1
        self.b1 -= lr * db1

        return loss

    def copy_from(self, other: "SimpleDQN") -> None:
        """Copy weights from another network."""
        self.w1 = other.w1.copy()
        self.b1 = other.b1.copy()
        self.w2 = other.w2.copy()
        self.b2 = other.b2.copy()
        self.w3 = other.w3.copy()
        self.b3 = other.b3.copy()


class RLTradingAgent:
    """DQN-based trading agent.

    Actions:
        0: Hold (do nothing)
        1: Buy (open/increase long position)
        2: Sell (open/increase short position / close long)

    Reward function:
        PnL-based with risk penalty for drawdowns.
    """

    def __init__(self, config: dict | None = None):
        cfg = RLConfig(**(config or {}))
        self.config = cfg

        self.q_network = SimpleDQN(cfg.state_size, cfg.action_size, hidden_size=64)
        self.target_network = SimpleDQN(cfg.state_size, cfg.action_size, hidden_size=64)
        self.target_network.copy_from(self.q_network)

        self.memory = ReplayBuffer(cfg.memory_size)
        self.epsilon = cfg.epsilon
        self._step_count = 0
        self._episode_rewards: list[float] = []
        self._training_losses: list[float] = []

    def select_action(self, state: np.ndarray) -> RLTradeResult:
        """Select action using epsilon-greedy policy."""
        exploration = random.random() < self.epsilon

        if exploration:
            action = random.randint(0, self.config.action_size - 1)
            q_values = [0.0] * self.config.action_size
        else:
            q_values = self.q_network.predict(state).tolist()
            action = int(np.argmax(q_values))

        confidence = max(q_values) / (sum(abs(q) for q in q_values) + 1e-8) if q_values else 0.0

        return RLTradeResult(
            action=action,
            confidence=confidence,
            q_values=q_values,
            exploration=exploration,
        )

    def compute_reward(
        self,
        prev_value: float,
        curr_value: float,
        action: int,
        position: float,
    ) -> float:
        """Compute reward for the agent.

        Reward = PnL - risk_penalty
        """
        if prev_value == 0:
            return 0.0

        pnl_return = (curr_value - prev_value) / prev_value

        # Penalize holding during drawdown
        if pnl_return < -0.01 and action == 0:
            return pnl_return * 2  # Extra penalty for inaction during loss

        # Reward for correct directional trades
        if action == 1 and pnl_return > 0:
            return pnl_return * 1.5
        if action == 2 and pnl_return < 0:
            return abs(pnl_return) * 1.5

        # Small penalty for trading (transaction costs)
        if action != 0:
            return pnl_return - 0.001

        return pnl_return

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """Train on a batch from replay buffer."""
        if len(self.memory) < self.config.batch_size:
            return 0.0

        batch = self.memory.sample(self.config.batch_size)

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Current Q-values
        current_q = np.array([self.q_network.predict(s) for s in states])

        # Target Q-values
        next_q = np.array([self.target_network.predict(s) for s in next_states])
        target_q = current_q.copy()

        for i in range(len(batch)):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.config.gamma * np.max(next_q[i])

        loss = self.q_network.train(states, target_q, self.config.learning_rate)
        self._training_losses.append(loss)

        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_min,
            self.epsilon * self.config.epsilon_decay,
        )

        # Update target network periodically
        self._step_count += 1
        if self._step_count % self.config.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)

        return loss

    def get_action_name(self, action: int) -> str:
        return {0: "hold", 1: "buy", 2: "sell"}.get(action, "unknown")

    def get_status(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "step_count": self._step_count,
            "avg_loss": np.mean(self._training_losses[-100:]) if self._training_losses else 0,
            "avg_reward": np.mean(self._episode_rewards[-100:]) if self._episode_rewards else 0,
        }

    def save_weights(self, path: str) -> None:
        """Save network weights to file."""
        np.savez(
            path,
            w1=self.q_network.w1, b1=self.q_network.b1,
            w2=self.q_network.w2, b2=self.q_network.b2,
            w3=self.q_network.w3, b3=self.q_network.b3,
            epsilon=self.epsilon, step_count=self._step_count,
        )

    def load_weights(self, path: str) -> None:
        """Load network weights from file."""
        data = np.load(path)
        self.q_network.w1 = data["w1"]
        self.q_network.b1 = data["b1"]
        self.q_network.w2 = data["w2"]
        self.q_network.b2 = data["b2"]
        self.q_network.w3 = data["w3"]
        self.q_network.b3 = data["b3"]
        self.epsilon = float(data["epsilon"])
        self._step_count = int(data["step_count"])
        self.target_network.copy_from(self.q_network)


class TradingEnvironment:
    """Simple trading environment for RL training.

    Wraps historical data into an OpenAI Gym-like interface.
    """

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        max_position: float = 1.0,
    ):
        self.features = features
        self.prices = prices
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position

        self._reset()

    def _reset(self) -> np.ndarray:
        self._step = 0
        self._position = 0.0
        self._cash = self.initial_capital
        self._portfolio_value = self.initial_capital
        return self._get_state()

    def reset(self) -> np.ndarray:
        return self._reset()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Take action and return (next_state, reward, done, info)."""
        if self._step >= len(self.prices) - 1:
            return self._get_state(), 0.0, True, {}

        prev_value = self._portfolio_value
        price = self.prices[self._step]

        # Execute action
        if action == 1 and self._position < self.max_position:
            # Buy
            cost = price * self.transaction_cost
            self._position = min(self.max_position, self._position + 0.1)
            self._cash -= price * 0.1 + cost
        elif action == 2 and self._position > -self.max_position:
            # Sell
            cost = price * self.transaction_cost
            self._position = max(-self.max_position, self._position - 0.1)
            self._cash += price * 0.1 - cost

        self._step += 1
        new_price = self.prices[self._step]

        # Update portfolio value
        self._portfolio_value = self._cash + self._position * new_price

        # Reward
        reward = (self._portfolio_value - prev_value) / prev_value if prev_value > 0 else 0

        done = self._step >= len(self.prices) - 1

        info = {
            "portfolio_value": self._portfolio_value,
            "position": self._position,
            "price": new_price,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        if self._step >= len(self.features):
            return np.zeros(self.features.shape[1] + 3)

        market_state = self.features[self._step]
        portfolio_state = np.array([
            self._position / self.max_position,  # Normalized position
            (self._portfolio_value - self.initial_capital) / self.initial_capital,  # PnL %
            self._cash / self.initial_capital,  # Cash ratio
        ])
        return np.concatenate([market_state, portfolio_state])
