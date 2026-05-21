"""
Experience replay buffer for DQN training.

Stores (state, action, reward, next_state, done) transitions and
supports uniform random sampling for mini-batch updates.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-size ring buffer for experience replay.

    Args:
        capacity: Maximum number of transitions stored.
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample a random mini-batch of transitions.

        Returns:
            Tuple of numpy arrays:
                states      (batch, obs_dim)
                actions     (batch,)           int64
                rewards     (batch,)
                next_states (batch, obs_dim)
                dones       (batch,)
        """
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        batch = [self.buffer[i] for i in indices]

        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)
