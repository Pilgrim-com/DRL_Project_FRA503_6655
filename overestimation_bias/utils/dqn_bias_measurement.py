"""
Bias measurement for DQN / Double DQN agents.

Runs greedy evaluation episodes, records per-step Q_taken and max_Q,
computes Monte Carlo returns, and aggregates both bias metrics:

    TakenAction_Bias = Q(s, a_taken) - MC_Return
    MaxQ_Bias        = max_a Q(s, a)  - MC_Return
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List


def compute_mc_returns(rewards: List[float], gamma: float) -> List[float]:
    """Backward-pass discounted MC returns."""
    T = len(rewards)
    if T == 0:
        return []
    G = [0.0] * T
    G[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    return G


def evaluate_dqn_agent(
    agent,
    env,
    n_episodes: int,
    gamma: float,
    action_values: List[float],
    max_steps: int = 500,
) -> Dict:
    """
    Run greedy evaluation and compute bias metrics.

    Args:
        agent: DQN or DoubleDQN agent with get_q_value / get_max_q_value.
        env: IsaacLab gym env.
        n_episodes: Number of evaluation episodes.
        gamma: Discount factor.
        action_values: Continuous action values per index.
        max_steps: Safety limit.

    Returns:
        Dict with aggregated bias stats and per-episode data.
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    all_taken_bias = []
    all_maxq_bias = []
    all_mc = []
    all_q_taken = []
    all_max_q = []
    episode_rewards = []
    episode_durations = []
    all_episode_steps = []  # for trajectory plots

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        # Per-step records
        step_records = []
        rewards_list = []

        while not done and step < max_steps:
            # Extract flat observation
            if isinstance(obs, dict):
                obs_tensor = obs.get("policy", next(iter(obs.values())))
            else:
                obs_tensor = obs
            if isinstance(obs_tensor, torch.Tensor):
                obs_np = obs_tensor.detach().cpu().numpy().flatten()
            else:
                obs_np = np.asarray(obs_tensor).flatten()

            # Greedy action
            action_idx = agent.get_action_index(obs_np)
            q_taken = agent.get_q_value(obs_np, action_idx)
            max_q = agent.get_max_q_value(obs_np)

            # Step environment
            action_val = action_values[action_idx]
            action_tensor = torch.tensor([[action_val]], dtype=torch.float32)
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'device'):
                action_tensor = action_tensor.to(env.unwrapped.device)

            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)

            if isinstance(reward, torch.Tensor):
                rew = reward.squeeze().item()
            else:
                rew = float(np.squeeze(reward))

            if isinstance(terminated, torch.Tensor):
                term = terminated.squeeze().item()
                trunc = truncated.squeeze().item()
            else:
                term = bool(np.squeeze(terminated))
                trunc = bool(np.squeeze(truncated))

            step_records.append({
                "step": step,
                "cart_position": float(obs_np[0]),
                "pole_angle": float(obs_np[1]),
                "cart_velocity": float(obs_np[2]),
                "pole_angular_velocity": float(obs_np[3]),
                "action_idx": action_idx,
                "action_value": action_val,
                "q_taken": q_taken,
                "max_q": max_q,
                "reward": rew,
            })
            rewards_list.append(rew)

            done = term or trunc
            obs = next_obs
            step += 1

        # Compute MC returns and bias
        mc_returns = compute_mc_returns(rewards_list, gamma)
        for i, rec in enumerate(step_records):
            rec["mc_return"] = mc_returns[i]
            rec["taken_bias"] = rec["q_taken"] - mc_returns[i]
            rec["maxq_bias"] = rec["max_q"] - mc_returns[i]

            all_taken_bias.append(rec["taken_bias"])
            all_maxq_bias.append(rec["maxq_bias"])
            all_mc.append(rec["mc_return"])
            all_q_taken.append(rec["q_taken"])
            all_max_q.append(rec["max_q"])

        episode_rewards.append(sum(rewards_list))
        episode_durations.append(step)
        all_episode_steps.append(step_records)

    agent.epsilon = original_epsilon

    n = max(len(all_taken_bias), 1)
    return {
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_duration": float(np.mean(episode_durations)),
        "avg_mc_return": float(np.mean(all_mc)) if all_mc else 0.0,
        "avg_q_taken": float(np.mean(all_q_taken)) if all_q_taken else 0.0,
        "avg_max_q": float(np.mean(all_max_q)) if all_max_q else 0.0,
        "avg_taken_bias": float(np.mean(all_taken_bias)) if all_taken_bias else 0.0,
        "std_taken_bias": float(np.std(all_taken_bias)) if all_taken_bias else 0.0,
        "avg_maxq_bias": float(np.mean(all_maxq_bias)) if all_maxq_bias else 0.0,
        "std_maxq_bias": float(np.std(all_maxq_bias)) if all_maxq_bias else 0.0,
        "episode_steps": all_episode_steps,
    }
