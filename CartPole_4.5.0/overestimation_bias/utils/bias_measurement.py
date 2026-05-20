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
from typing import List, Dict
from utils.behavior_logger import BehaviorLogger


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


def aggregate_bias_stats(
    all_episode_data: List[Dict[str, list]],
) -> Dict[str, float]:
    """
    Aggregate bias statistics across multiple evaluation episodes.

    Args:
        all_episode_data: List of dicts, each with keys
            "mc_returns", "q_estimates", "biases".

    Returns:
        Dict with aggregated metrics.
    """
    all_mc = []
    all_q = []
    all_bias = []

    for ep_data in all_episode_data:
        steps = ep_data.get("steps", [])
        if steps:
            all_mc.extend([s["mc_return"] for s in steps])
            all_q.extend([s["q_estimated"] for s in steps])
            all_bias.extend([s["bias"] for s in steps])

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
    save_behavior_csv: str = None,
) -> Dict[str, object]:
    """
    Run greedy evaluation episodes and compute bias + behavior data.

    Records full per-step CartPole state (cart_position, pole_angle,
    velocities, action, Q-estimate, MC return, bias) using BehaviorLogger.

    Args:
        agent: A tabular Q-Learning or Double Q-Learning agent.
        env: IsaacLab CartPole gym environment.
        discretizer: StateDiscretizer instance.
        n_episodes: Number of evaluation episodes to run.
        gamma: Discount factor for MC return computation.
        max_steps: Max steps per episode (safety limit).
        eval_epsilon: Epsilon during evaluation (0.0 = greedy).
        save_behavior_csv: If provided, save per-step behavior to this CSV path.

    Returns:
        Dict with:
            "bias_stats": aggregated bias statistics
            "avg_reward": average episode reward
            "avg_duration": average episode length
            "behavior_logger": BehaviorLogger with full per-step data
    """
    import torch

    # Save and override epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = eval_epsilon

    logger = BehaviorLogger()
    total_reward = 0.0
    total_duration = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0

        logger.begin_episode(episode_id=ep)

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

            # Log step with raw observation (for behavior analysis)
            logger.log_step(
                step=step,
                raw_obs=obs,
                action_idx=action_idx,
                action_value=action_val,
                q_estimated=q_est,
                reward=rew_val,
            )

            done = term or trunc
            obs = next_obs
            step += 1

        # End episode: compute MC returns and per-step bias
        ep_summary = logger.end_episode(gamma)
        total_reward += ep_summary.get("total_reward", 0.0)
        total_duration += ep_summary.get("duration", 0)

    # Restore epsilon
    agent.epsilon = original_epsilon

    # Aggregate bias stats from logger data
    bias_stats = aggregate_bias_stats(logger.episodes)

    # Save behavior CSV if requested
    if save_behavior_csv:
        logger.save_csv(save_behavior_csv)

    return {
        "bias_stats": bias_stats,
        "avg_reward": total_reward / max(n_episodes, 1),
        "avg_duration": total_duration / max(n_episodes, 1),
        "behavior_logger": logger,
    }
