"""
Bias measurement utilities using Monte Carlo returns.

Core idea: compare the Q-value the agent *thinks* a state-action pair is
worth (Q_estimated) against the *actual* discounted return observed when
following the greedy policy from that state (Monte Carlo return G_t).

    Bias(s, a) = Q_estimated(s, a) - G_t

Positive bias = overestimation.  This is the key metric for studying the
max-operator bias in Q-Learning vs Double Q-Learning.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


def compute_mc_returns(rewards: List[float], gamma: float) -> List[float]:
    """
    Compute discounted Monte Carlo returns for each timestep in an episode.

    G_t = r_{t+1} + gamma * r_{t+2} + gamma^2 * r_{t+3} + ...

    Args:
        rewards: List of rewards [r_0, r_1, ..., r_{T-1}] received at each step.
        gamma: Discount factor.

    Returns:
        List of returns [G_0, G_1, ..., G_{T-1}], same length as rewards.
    """
    T = len(rewards)
    if T == 0:
        return []

    returns = [0.0] * T
    # Backward pass: G_t = r_t + gamma * G_{t+1}
    returns[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    return returns


def collect_episode_data(
    states: List[tuple],
    actions: List[int],
    rewards: List[float],
    q_estimates: List[float],
    gamma: float,
) -> Dict[str, list]:
    """
    For a single evaluation episode, compute MC returns and per-step bias.

    Args:
        states: Discretized state tuples at each step.
        actions: Action indices at each step.
        rewards: Rewards received at each step.
        q_estimates: Q(s_t, a_t) estimated by the agent at each step.
        gamma: Discount factor.

    Returns:
        Dict with keys:
            "mc_returns": List[float]  — G_t for each step
            "q_estimates": List[float] — Q(s_t, a_t) for each step
            "biases": List[float]      — Q_est - G_t for each step
    """
    mc_returns = compute_mc_returns(rewards, gamma)
    biases = [q - g for q, g in zip(q_estimates, mc_returns)]

    return {
        "mc_returns": mc_returns,
        "q_estimates": q_estimates,
        "biases": biases,
    }


def aggregate_bias_stats(
    all_episode_data: List[Dict[str, list]],
) -> Dict[str, float]:
    """
    Aggregate bias statistics across multiple evaluation episodes.

    Args:
        all_episode_data: List of dicts from collect_episode_data().

    Returns:
        Dict with aggregated metrics:
            "avg_mc_return": average MC return across all steps/episodes
            "avg_q_estimate": average Q estimate across all steps/episodes
            "avg_bias": average overestimation bias
            "std_bias": standard deviation of bias
            "max_bias": maximum bias observed
            "min_bias": minimum bias observed
            "num_samples": total number of (s,a) samples
    """
    all_mc = []
    all_q = []
    all_bias = []

    for ep_data in all_episode_data:
        all_mc.extend(ep_data["mc_returns"])
        all_q.extend(ep_data["q_estimates"])
        all_bias.extend(ep_data["biases"])

    if len(all_bias) == 0:
        return {
            "avg_mc_return": 0.0,
            "avg_q_estimate": 0.0,
            "avg_bias": 0.0,
            "std_bias": 0.0,
            "max_bias": 0.0,
            "min_bias": 0.0,
            "num_samples": 0,
        }

    return {
        "avg_mc_return": float(np.mean(all_mc)),
        "avg_q_estimate": float(np.mean(all_q)),
        "avg_bias": float(np.mean(all_bias)),
        "std_bias": float(np.std(all_bias)),
        "max_bias": float(np.max(all_bias)),
        "min_bias": float(np.min(all_bias)),
        "num_samples": len(all_bias),
    }


def evaluate_tabular_agent(
    agent,
    env,
    discretizer,
    n_episodes: int,
    gamma: float,
    max_steps: int = 500,
    eval_epsilon: float = 0.0,
) -> Dict[str, object]:
    """
    Run greedy evaluation episodes and compute bias statistics.

    This function temporarily overrides the agent's epsilon to eval_epsilon
    (default 0.0 = fully greedy), runs episodes, computes Monte Carlo returns,
    and measures overestimation bias.

    Args:
        agent: A tabular Q-Learning or Double Q-Learning agent.
        env: IsaacLab CartPole gym environment.
        discretizer: StateDiscretizer instance.
        n_episodes: Number of evaluation episodes to run.
        gamma: Discount factor for MC return computation.
        max_steps: Max steps per episode (safety limit).
        eval_epsilon: Epsilon during evaluation (0.0 = greedy).

    Returns:
        Dict with:
            "bias_stats": aggregated bias statistics (dict)
            "avg_reward": average episode reward
            "avg_duration": average episode length
            "episode_data": list of per-episode data dicts
    """
    import torch

    # Save and override epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = eval_epsilon

    all_episode_data = []
    total_reward = 0.0
    total_duration = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        states = []
        actions = []
        rewards = []
        q_estimates = []
        step = 0

        while not done and step < max_steps:
            # Discretize state
            state_key = discretizer.discretize(obs)

            # Get action and Q-estimate from agent
            action_idx = agent.get_action_index(state_key)
            q_est = agent.get_q_value(state_key, action_idx)

            # Map discrete action to continuous and create tensor
            action_val = agent.action_values[action_idx]
            action_tensor = torch.tensor([[action_val]], dtype=torch.float32)
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'device'):
                action_tensor = action_tensor.to(env.unwrapped.device)

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action_tensor)

            # Extract scalar reward
            if isinstance(reward, torch.Tensor):
                rew_val = reward.squeeze().item()
            else:
                rew_val = float(np.squeeze(reward))

            # Check done
            if isinstance(terminated, torch.Tensor):
                term = terminated.squeeze().item()
                trunc = truncated.squeeze().item()
            else:
                term = bool(np.squeeze(terminated))
                trunc = bool(np.squeeze(truncated))

            # Record step data
            states.append(state_key)
            actions.append(action_idx)
            rewards.append(rew_val)
            q_estimates.append(q_est)

            done = term or trunc
            obs = next_obs
            step += 1

        # Compute MC returns and bias for this episode
        ep_data = collect_episode_data(states, actions, rewards, q_estimates, gamma)
        all_episode_data.append(ep_data)

        total_reward += sum(rewards)
        total_duration += step

    # Restore epsilon
    agent.epsilon = original_epsilon

    # Aggregate
    bias_stats = aggregate_bias_stats(all_episode_data)

    return {
        "bias_stats": bias_stats,
        "avg_reward": total_reward / max(n_episodes, 1),
        "avg_duration": total_duration / max(n_episodes, 1),
        "episode_data": all_episode_data,
    }
