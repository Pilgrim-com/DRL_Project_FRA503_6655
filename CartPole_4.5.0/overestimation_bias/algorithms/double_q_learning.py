"""
Tabular Double Q-Learning for IsaacLab CartPole.

Maintains TWO independent Q-tables (Q_A and Q_B).  On each update step,
one table is chosen at random to be updated:

    If updating Q_A:
        a* = argmax_a Q_A(s', a)          ← Q_A selects the best action
        target = r + γ * Q_B(s', a*)      ← Q_B evaluates that action
        Q_A(s, a) ← Q_A(s, a) + α * [target - Q_A(s, a)]

    If updating Q_B:  (symmetric)
        a* = argmax_a Q_B(s', a)
        target = r + γ * Q_A(s', a*)
        Q_B(s, a) ← Q_B(s, a) + α * [target - Q_B(s, a)]

By decoupling action *selection* (which table picked a*) from action
*evaluation* (which table estimated the value of a*), the systematic
upward bias of the max operator is substantially reduced.

Reference:
    Hasselt, H. van (2010). "Double Q-learning." NeurIPS.
"""

from __future__ import annotations

import numpy as np
import pickle
import os
from collections import defaultdict
from typing import List, Optional


class DoubleQLearning:
    """
    Tabular Double Q-Learning agent with epsilon-greedy exploration.

    Action selection during play uses the *sum* Q_A + Q_B for the
    epsilon-greedy policy, so both tables contribute to exploration.

    Args:
        num_actions: Number of discrete actions.
        action_values: Continuous action values per discrete index.
        learning_rate: Step size α.
        discount_factor: Discount factor γ.
        epsilon: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Multiplicative per-episode decay.
    """

    def __init__(
        self,
        num_actions: int = 5,
        action_values: Optional[List[float]] = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
    ) -> None:
        self.num_actions = num_actions
        self.action_values = action_values or [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Two independent Q-tables
        self.q_table_a = defaultdict(lambda: np.zeros(self.num_actions))
        self.q_table_b = defaultdict(lambda: np.zeros(self.num_actions))

        # Training metrics
        self.training_errors = []

    # ------------------------------------------------------------------ #
    # Combined Q-table (for action selection and diagnostics)
    # ------------------------------------------------------------------ #

    @property
    def q_table(self) -> defaultdict:
        """
        Return a *virtual* combined Q-table (Q_A + Q_B) for compatibility
        with evaluation code.  Values are computed on-the-fly.
        """
        combined = defaultdict(lambda: np.zeros(self.num_actions))
        all_keys = set(self.q_table_a.keys()) | set(self.q_table_b.keys())
        for key in all_keys:
            combined[key] = self.q_table_a[key] + self.q_table_b[key]
        return combined

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #

    def get_action_index(self, state: tuple) -> int:
        """
        Select action using epsilon-greedy on Q_A + Q_B.

        Using the sum of both tables for action selection gives a better
        estimate than either table alone and is standard practice.

        Args:
            state: Discretized state tuple.

        Returns:
            Action index.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        else:
            # Greedy w.r.t. combined Q-values
            q_combined = self.q_table_a[state] + self.q_table_b[state]
            max_q = np.max(q_combined)
            best_actions = np.where(q_combined == max_q)[0]
            return int(np.random.choice(best_actions))

    def get_q_value(self, state: tuple, action: int) -> float:
        """
        Get the combined Q-value (Q_A + Q_B) / 2 for a state-action pair.

        We average the two tables to get a single Q-estimate for bias
        measurement purposes.

        Args:
            state: Discretized state tuple.
            action: Action index.

        Returns:
            Average Q-value from both tables.
        """
        q_a = float(self.q_table_a[state][action])
        q_b = float(self.q_table_b[state][action])
        return (q_a + q_b) / 2.0

    def get_max_q_value(self, state: tuple) -> float:
        """Get max_a (Q_A(s,a) + Q_B(s,a)) / 2."""
        q_combined = (self.q_table_a[state] + self.q_table_b[state]) / 2.0
        return float(np.max(q_combined))

    # ------------------------------------------------------------------ #
    # Learning update
    # ------------------------------------------------------------------ #

    def update(
        self,
        state: tuple,
        action: int,
        reward: float,
        next_state: tuple,
        done: bool,
    ) -> float:
        """
        Perform one Double Q-Learning update step.

        With 50% probability, update Q_A using Q_B for evaluation,
        otherwise update Q_B using Q_A for evaluation.

        The key difference from Q-Learning:
        - Q-Learning:  target = r + γ * max_a' Q(s', a')
                        (same table selects AND evaluates → bias)
        - Double Q:    target = r + γ * Q_other(s', argmax_a' Q_self(s', a'))
                        (one table selects, other evaluates → less bias)

        Args:
            state: Current discretized state.
            action: Action index taken.
            reward: Reward received.
            next_state: Next discretized state.
            done: Whether the episode terminated.

        Returns:
            TD-error (float) for monitoring.
        """
        if np.random.random() < 0.5:
            # Update Q_A
            current_q = self.q_table_a[state][action]
            if done:
                target = reward
            else:
                # Q_A selects the best action in next state
                a_star = int(np.argmax(self.q_table_a[next_state]))
                # Q_B evaluates that action (decoupled!)
                target = reward + self.gamma * self.q_table_b[next_state][a_star]

            td_error = target - current_q
            self.q_table_a[state][action] += self.lr * td_error
        else:
            # Update Q_B (symmetric)
            current_q = self.q_table_b[state][action]
            if done:
                target = reward
            else:
                # Q_B selects the best action
                a_star = int(np.argmax(self.q_table_b[next_state]))
                # Q_A evaluates that action
                target = reward + self.gamma * self.q_table_a[next_state][a_star]

            td_error = target - current_q
            self.q_table_b[state][action] += self.lr * td_error

        self.training_errors.append(abs(td_error))
        return td_error

    # ------------------------------------------------------------------ #
    # Exploration decay
    # ------------------------------------------------------------------ #

    def decay_epsilon(self) -> None:
        """Decay epsilon multiplicatively, floored at epsilon_end."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str, filename: str = "double_q_learning.pkl") -> None:
        """Save both Q-tables and metadata."""
        os.makedirs(path, exist_ok=True)
        data = {
            "q_table_a": dict(self.q_table_a),
            "q_table_b": dict(self.q_table_b),
            "epsilon": self.epsilon,
            "num_actions": self.num_actions,
            "action_values": self.action_values,
            "lr": self.lr,
            "gamma": self.gamma,
        }
        with open(os.path.join(path, filename), "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str, filename: str = "double_q_learning.pkl") -> None:
        """Load both Q-tables and metadata."""
        with open(os.path.join(path, filename), "rb") as f:
            data = pickle.load(f)

        self.q_table_a = defaultdict(lambda: np.zeros(self.num_actions))
        self.q_table_b = defaultdict(lambda: np.zeros(self.num_actions))
        for k, v in data["q_table_a"].items():
            self.q_table_a[k] = np.array(v)
        for k, v in data["q_table_b"].items():
            self.q_table_b[k] = np.array(v)

        self.epsilon = data.get("epsilon", self.epsilon)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def get_avg_q_value(self) -> float:
        """Average Q-value across all visited state-action pairs (mean of both tables)."""
        all_keys = set(self.q_table_a.keys()) | set(self.q_table_b.keys())
        if len(all_keys) == 0:
            return 0.0
        all_q = []
        for key in all_keys:
            avg = (self.q_table_a[key] + self.q_table_b[key]) / 2.0
            all_q.append(avg)
        return float(np.mean(np.concatenate(all_q)))

    def num_visited_states(self) -> int:
        """Number of unique states visited in either table."""
        return len(set(self.q_table_a.keys()) | set(self.q_table_b.keys()))

    @property
    def name(self) -> str:
        return "Double Q-Learning"

    def __repr__(self) -> str:
        return (
            f"DoubleQLearning(num_actions={self.num_actions}, "
            f"lr={self.lr}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.4f}, "
            f"visited_states={self.num_visited_states()})"
        )
