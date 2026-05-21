"""
DQN (Deep Q-Network) agent for IsaacLab CartPole.

Uses a neural network to approximate Q-values and a separate target
network for stable TD targets:

    target = r + gamma * max_a' Q_target(s', a')

The max operator applied to the target network is the source of
overestimation bias in DQN — analogous to tabular Q-Learning.
"""

from __future__ import annotations

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional

from utils.networks import QNetwork
from utils.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN agent with experience replay and target network.

    Args:
        obs_dim: Observation dimension.
        num_actions: Number of discrete actions.
        action_values: Continuous values mapped to each action index.
        lr: Learning rate for Adam optimizer.
        gamma: Discount factor.
        buffer_size: Replay buffer capacity.
        batch_size: Mini-batch size for updates.
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay_steps: Number of steps for linear epsilon decay.
        target_update_interval: Steps between target network syncs.
        gradient_clip: Max gradient norm for clipping.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        num_actions: int = 5,
        action_values: Optional[List[float]] = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        target_update_interval: int = 1_000,
        gradient_clip: float = 10.0,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.action_values = action_values or [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.target_update_interval = target_update_interval
        self.gradient_clip = gradient_clip
        self.device = torch.device(device)

        # Networks
        self.online_net = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Step counter for epsilon decay and target updates
        self.total_steps = 0
        self.training_losses = []

    @property
    def name(self) -> str:
        return "DQN"

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #

    def get_action_index(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            q_values = self.online_net(s)
            return int(q_values.argmax(dim=1).item())

    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """Get Q(s, a) from the online network."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            q_values = self.online_net(s)
            return float(q_values[0, action].item())

    def get_max_q_value(self, state: np.ndarray) -> float:
        """Get max_a Q(s, a) from the online network."""
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device)
            if s.dim() == 1:
                s = s.unsqueeze(0)
            q_values = self.online_net(s)
            return float(q_values.max(dim=1)[0].item())

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, min_buffer_size: int = 1_000) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient step.

        DQN target:
            target = r + gamma * max_a' Q_target(s', a')

        Returns:
            Loss value, or None if buffer too small.
        """
        if len(self.replay_buffer) < min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q-values for taken actions
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # DQN target: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        loss_val = float(loss.item())
        self.training_losses.append(loss_val)
        return loss_val

    # ------------------------------------------------------------------ #
    # Epsilon schedule and target update
    # ------------------------------------------------------------------ #

    def step_epsilon(self) -> None:
        """Linear epsilon decay based on total_steps."""
        self.total_steps += 1
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def maybe_update_target(self) -> bool:
        """Sync target network if interval reached. Returns True if updated."""
        if self.total_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            return True
        return False

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str, filename: str = "dqn.pt") -> None:
        """Save model weights and training state."""
        os.makedirs(path, exist_ok=True)
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
        }, os.path.join(path, filename))

    def load(self, path: str, filename: str = "dqn.pt") -> None:
        """Load model weights and training state."""
        ckpt = torch.load(os.path.join(path, filename), map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt.get("epsilon", self.epsilon)
        self.total_steps = ckpt.get("total_steps", 0)

    def __repr__(self) -> str:
        return (
            f"DQNAgent(obs_dim={self.obs_dim}, num_actions={self.num_actions}, "
            f"epsilon={self.epsilon:.4f}, steps={self.total_steps})"
        )
