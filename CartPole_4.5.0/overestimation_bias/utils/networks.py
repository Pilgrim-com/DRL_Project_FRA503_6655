"""
Neural network architectures for DQN-based agents.

QNetwork: Simple feedforward network that maps a 4D continuous
observation to Q-values for each of 5 discrete actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q-value network for CartPole.

    Architecture:
        Linear(obs_dim, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, n_actions)

    Input:  (batch, obs_dim)  — continuous CartPole observation
    Output: (batch, n_actions) — Q-value for each discrete action

    Args:
        obs_dim: Observation space dimension (4 for CartPole).
        n_actions: Number of discrete actions (5).
    """

    def __init__(self, obs_dim: int = 4, n_actions: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Observation tensor, shape (batch, obs_dim).

        Returns:
            Q-values, shape (batch, n_actions).
        """
        return self.net(x)
