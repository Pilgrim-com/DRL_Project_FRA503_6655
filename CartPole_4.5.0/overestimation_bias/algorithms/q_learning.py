"""
Tabular Q-Learning for IsaacLab CartPole.

Uses the standard Q-Learning update with the max operator:

    Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

The max operator is known to cause overestimation bias because it uses the
same Q-table to both *select* the best next action and *evaluate* its value.
When Q-values contain noise/error, max picks the action with the highest
noise, systematically inflating the target.

This implementation serves as the baseline for comparison against
Double Q-Learning, which addresses this bias.
"""

from __future__ import annotations

import numpy as np
import pickle
import os
import json
from collections import defaultdict
from typing import List, Optional


class QLearning:
    """
    Tabular Q-Learning agent with epsilon-greedy exploration.

    Args:
        num_actions: Number of discrete actions.
        action_values: List of continuous action values corresponding to
                       each discrete action index.
        learning_rate: Step size α for Q-value updates.
        discount_factor: Discount factor γ.
        epsilon: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Multiplicative decay applied each episode.
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

        # Q-table: maps state tuple → numpy array of Q-values per action
        # defaultdict auto-initializes unseen states to zeros
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))

        # Training metrics (accumulated during training)
        self.training_errors = []

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #

    def get_action_index(self, state: tuple) -> int:
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Discretized state tuple (e.g., (2, 7, 4, 3)).

        Returns:
            Action index in [0, num_actions-1].
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, self.num_actions)
        else:
            # Exploit: greedy action (break ties randomly)
            q_vals = self.q_table[state]
            max_q = np.max(q_vals)
            # Get all actions with max Q-value (tie-breaking)
            best_actions = np.where(q_vals == max_q)[0]
            return int(np.random.choice(best_actions))

    def get_q_value(self, state: tuple, action: int) -> float:
        """
        Get Q(s, a) from the Q-table.

        Args:
            state: Discretized state tuple.
            action: Action index.

        Returns:
            Q-value for the state-action pair.
        """
        return float(self.q_table[state][action])

    def get_max_q_value(self, state: tuple) -> float:
        """Get max_a Q(s, a) for a given state."""
        return float(np.max(self.q_table[state]))

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
        Perform one Q-Learning update step.

        Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        If done=True, the target is just r (no future reward).

        Args:
            state: Current discretized state.
            action: Action index taken.
            reward: Reward received.
            next_state: Next discretized state.
            done: Whether the episode terminated.

        Returns:
            TD-error (float) for monitoring.
        """
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            # Q-Learning: use max operator on the SAME Q-table
            # This is the source of overestimation bias!
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD-error
        td_error = target - current_q

        # Update Q-value
        self.q_table[state][action] += self.lr * td_error

        self.training_errors.append(abs(td_error))
        return td_error

    # ------------------------------------------------------------------ #
    # Exploration decay
    # ------------------------------------------------------------------ #

    def decay_epsilon(self) -> None:
        """
        Decay epsilon multiplicatively, floored at epsilon_end.
        Call once per episode.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str, filename: str = "q_learning.pkl") -> None:
        """
        Save Q-table and metadata to a pickle file.

        Args:
            path: Directory to save in.
            filename: Filename for the pickle file.
        """
        os.makedirs(path, exist_ok=True)
        data = {
            "q_table": dict(self.q_table),  # convert defaultdict
            "epsilon": self.epsilon,
            "num_actions": self.num_actions,
            "action_values": self.action_values,
            "lr": self.lr,
            "gamma": self.gamma,
        }
        with open(os.path.join(path, filename), "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str, filename: str = "q_learning.pkl") -> None:
        """
        Load Q-table and metadata from a pickle file.

        Args:
            path: Directory containing the file.
            filename: Filename of the pickle file.
        """
        with open(os.path.join(path, filename), "rb") as f:
            data = pickle.load(f)

        # Restore Q-table as defaultdict
        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
        for k, v in data["q_table"].items():
            self.q_table[k] = np.array(v)

        self.epsilon = data.get("epsilon", self.epsilon)

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #

    def get_avg_q_value(self) -> float:
        """Compute the average Q-value across all visited state-action pairs."""
        if len(self.q_table) == 0:
            return 0.0
        all_q = np.concatenate([v for v in self.q_table.values()])
        return float(np.mean(all_q))

    def num_visited_states(self) -> int:
        """Number of unique states visited (present in Q-table)."""
        return len(self.q_table)

    @property
    def name(self) -> str:
        return "Q-Learning"

    def __repr__(self) -> str:
        return (
            f"QLearning(num_actions={self.num_actions}, "
            f"lr={self.lr}, gamma={self.gamma}, "
            f"epsilon={self.epsilon:.4f}, "
            f"visited_states={self.num_visited_states()})"
        )
