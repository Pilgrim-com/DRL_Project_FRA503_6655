"""
Double DQN agent for IsaacLab CartPole.

Identical to DQN except for the target computation:

    DQN:        target = r + gamma * max_a' Q_target(s', a')
    Double DQN: a* = argmax_a' Q_online(s', a')
                target = r + gamma * Q_target(s', a*)

By using the online network to *select* the best action and the target
network to *evaluate* it, Double DQN breaks the correlation that causes
overestimation bias in standard DQN.

Reference:
    van Hasselt, Guez, Silver (2016). "Deep Reinforcement Learning with
    Double Q-learning." AAAI.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from algorithms.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent — inherits everything from DQN, overrides only
    the update method's target computation.
    """

    @property
    def name(self) -> str:
        return "Double DQN"

    def update(self, min_buffer_size: int = 1_000) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient step.

        Double DQN target:
            a* = argmax_a' Q_online(s', a')
            target = r + gamma * Q_target(s', a*)

        The ONLY difference from DQN is this target calculation.
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

        # Double DQN target:
        #   1) Online network SELECTS the best action
        #   2) Target network EVALUATES that action
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), self.gradient_clip)
        self.optimizer.step()

        loss_val = float(loss.item())
        self.training_losses.append(loss_val)
        return loss_val

    def __repr__(self) -> str:
        return (
            f"DoubleDQNAgent(obs_dim={self.obs_dim}, num_actions={self.num_actions}, "
            f"epsilon={self.epsilon:.4f}, steps={self.total_steps})"
        )
