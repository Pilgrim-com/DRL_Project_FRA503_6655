"""
Behavior logger for CartPole evaluation episodes.

Captures per-step data during greedy evaluation:
    step, cart_position, cart_velocity, pole_angle, pole_angular_velocity,
    action_idx, action_value, q_estimated, mc_return, bias

This data is used to understand HOW overestimation bias affects CartPole
control behavior — not just that it exists, but what it does to the pole,
cart, and action selection.
"""

from __future__ import annotations

import csv
import json
import os
import numpy as np
import torch
from typing import List, Dict, Optional


def _compute_mc_returns(rewards: List[float], gamma: float) -> List[float]:
    """Compute discounted Monte Carlo returns (local copy to avoid circular import)."""
    T = len(rewards)
    if T == 0:
        return []
    returns = [0.0] * T
    returns[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns


class BehaviorLogger:
    """
    Collects per-step CartPole state and decision data during evaluation.

    Usage:
        logger = BehaviorLogger()
        # During each evaluation episode:
        logger.begin_episode(episode_id)
        for each step:
            logger.log_step(step, obs, action_idx, action_val, q_est)
        logger.end_episode(gamma)

        # After all episodes:
        logger.save_csv(path)
        logger.get_action_distribution()
    """

    def __init__(self) -> None:
        self.episodes: List[Dict] = []         # completed episodes
        self._current_steps: List[Dict] = []   # steps in current episode
        self._current_rewards: List[float] = []
        self._current_episode_id: int = 0

    def begin_episode(self, episode_id: int = 0) -> None:
        """Start recording a new episode."""
        self._current_steps = []
        self._current_rewards = []
        self._current_episode_id = episode_id

    def log_step(
        self,
        step: int,
        raw_obs,
        action_idx: int,
        action_value: float,
        q_estimated: float,
        reward: float,
    ) -> None:
        """
        Record one step of the episode.

        Args:
            step: Step number within the episode.
            raw_obs: Raw IsaacLab observation (dict or tensor).
                     Expected shape: obs["policy"] → (1, 4) or (4,).
            action_idx: Discrete action index chosen.
            action_value: Continuous action value sent to env.
            q_estimated: Q(s, a) from the agent.
            reward: Reward received after taking this action.
        """
        # Extract continuous observation values
        if isinstance(raw_obs, dict):
            obs_tensor = raw_obs.get("policy", next(iter(raw_obs.values())))
        else:
            obs_tensor = raw_obs

        if isinstance(obs_tensor, torch.Tensor):
            obs_arr = obs_tensor.detach().cpu().numpy().flatten()
        else:
            obs_arr = np.asarray(obs_tensor).flatten()

        self._current_steps.append({
            "step": step,
            "cart_position": float(obs_arr[0]),
            "pole_angle": float(obs_arr[1]),
            "cart_velocity": float(obs_arr[2]),
            "pole_angular_velocity": float(obs_arr[3]),
            "action_idx": action_idx,
            "action_value": action_value,
            "q_estimated": q_estimated,
            "reward": reward,
            # mc_return and bias filled in end_episode()
            "mc_return": 0.0,
            "bias": 0.0,
        })
        self._current_rewards.append(reward)

    def end_episode(self, gamma: float) -> Dict:
        """
        Finalize the current episode: compute MC returns and bias.

        Args:
            gamma: Discount factor for MC return computation.

        Returns:
            Dict with episode summary stats.
        """
        if not self._current_steps:
            return {}

        # Compute MC returns for each step
        mc_returns = _compute_mc_returns(self._current_rewards, gamma)

        # Fill in mc_return and bias for each step
        for i, step_data in enumerate(self._current_steps):
            step_data["mc_return"] = mc_returns[i]
            step_data["bias"] = step_data["q_estimated"] - mc_returns[i]

        # Episode summary
        episode_data = {
            "episode_id": self._current_episode_id,
            "duration": len(self._current_steps),
            "total_reward": sum(self._current_rewards),
            "steps": list(self._current_steps),  # copy
        }
        self.episodes.append(episode_data)

        # Reset
        self._current_steps = []
        self._current_rewards = []

        return episode_data

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #

    def save_csv(self, filepath: str) -> None:
        """
        Save all logged episodes as a flat CSV file.

        Each row = one step. Columns:
            episode, step, cart_position, pole_angle, cart_velocity,
            pole_angular_velocity, action_idx, action_value,
            q_estimated, reward, mc_return, bias
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        fieldnames = [
            "episode", "step", "cart_position", "pole_angle",
            "cart_velocity", "pole_angular_velocity",
            "action_idx", "action_value",
            "q_estimated", "reward", "mc_return", "bias",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep in self.episodes:
                for step_data in ep["steps"]:
                    row = {"episode": ep["episode_id"]}
                    row.update({k: step_data[k] for k in fieldnames if k != "episode"})
                    writer.writerow(row)

    def save_json(self, filepath: str) -> None:
        """Save all logged episodes as a JSON file (richer structure)."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.episodes, f, indent=2)

    # ------------------------------------------------------------------ #
    # Analysis helpers
    # ------------------------------------------------------------------ #

    def get_action_distribution(self, num_actions: int = 5) -> Dict[int, int]:
        """
        Count how often each discrete action was selected across all episodes.

        Returns:
            Dict mapping action_idx → count.
        """
        counts = {i: 0 for i in range(num_actions)}
        for ep in self.episodes:
            for step_data in ep["steps"]:
                idx = step_data["action_idx"]
                counts[idx] = counts.get(idx, 0) + 1
        return counts

    def get_all_steps_flat(self) -> List[Dict]:
        """Return all step data across all episodes as a flat list."""
        flat = []
        for ep in self.episodes:
            for step_data in ep["steps"]:
                row = dict(step_data)
                row["episode"] = ep["episode_id"]
                flat.append(row)
        return flat

    def get_episode_summaries(self) -> List[Dict]:
        """
        Get per-episode summary statistics.

        Returns list of dicts with keys:
            episode_id, duration, total_reward,
            avg_q_estimated, avg_mc_return, avg_bias,
            avg_abs_pole_angle, avg_abs_cart_position
        """
        summaries = []
        for ep in self.episodes:
            steps = ep["steps"]
            if not steps:
                continue
            summaries.append({
                "episode_id": ep["episode_id"],
                "duration": ep["duration"],
                "total_reward": ep["total_reward"],
                "avg_q_estimated": np.mean([s["q_estimated"] for s in steps]),
                "avg_mc_return": np.mean([s["mc_return"] for s in steps]),
                "avg_bias": np.mean([s["bias"] for s in steps]),
                "avg_abs_pole_angle": np.mean([abs(s["pole_angle"]) for s in steps]),
                "avg_abs_cart_position": np.mean([abs(s["cart_position"]) for s in steps]),
                "max_abs_pole_angle": max(abs(s["pole_angle"]) for s in steps),
                "max_abs_cart_position": max(abs(s["cart_position"]) for s in steps),
            })
        return summaries

    def clear(self) -> None:
        """Clear all logged data."""
        self.episodes = []
        self._current_steps = []
        self._current_rewards = []
