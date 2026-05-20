"""
Bias measurement utilities using Monte Carlo returns.

Two bias metrics:

    TakenAction_Bias = Q(s, a_taken) - MC_Return(s, a_taken)
    MaxQ_Bias        = max_a Q(s, a) - MC_Return(s, a_taken)

MaxQ_Bias isolates the effect of the max operator. Positive MaxQ_Bias
indicates that the agent's highest-valued action in a state exceeds the
actual return achieved, which is the theoretical signature of
overestimation bias.

For Double Q-Learning, max_a Q is computed on the averaged table:
    Q_combined(s, a) = (Q_A(s, a) + Q_B(s, a)) / 2
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
    returns[-1] = rewards[-1]
    for t in range(T - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    return returns


def aggregate_bias_stats(
    all_episode_data: List[Dict[str, list]],
) -> Dict[str, float]:
    """
    Aggregate bias statistics across multiple evaluation episodes.

    Returns both TakenAction_Bias and MaxQ_Bias statistics.
    """
    all_mc = []
    all_q_taken = []
    all_max_q = []
    all_taken_bias = []
    all_maxq_bias = []

    for ep_data in all_episode_data:
        steps = ep_data.get("steps", [])
        if steps:
            all_mc.extend([s["mc_return"] for s in steps])
            all_q_taken.extend([s["q_estimated"] for s in steps])
            all_max_q.extend([s["max_q_estimated"] for s in steps])
            all_taken_bias.extend([s["taken_bias"] for s in steps])
            all_maxq_bias.extend([s["maxq_bias"] for s in steps])

    if len(all_taken_bias) == 0:
        return {
            "avg_mc_return": 0.0,
            "avg_q_taken": 0.0,
            "avg_max_q": 0.0,
            "avg_taken_bias": 0.0,
            "std_taken_bias": 0.0,
            "avg_maxq_bias": 0.0,
            "std_maxq_bias": 0.0,
            "num_samples": 0,
        }

    return {
        "avg_mc_return": float(np.mean(all_mc)),
        "avg_q_taken": float(np.mean(all_q_taken)),
        "avg_max_q": float(np.mean(all_max_q)),
        "avg_taken_bias": float(np.mean(all_taken_bias)),
        "std_taken_bias": float(np.std(all_taken_bias)),
        "avg_maxq_bias": float(np.mean(all_maxq_bias)),
        "std_maxq_bias": float(np.std(all_maxq_bias)),
        "num_samples": len(all_taken_bias),
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
    Run greedy evaluation episodes and compute both bias metrics + behavior.

    Records per-step: Q(s, a_taken), max_a Q(s, a), MC return,
    TakenAction_Bias, and MaxQ_Bias.

    Args:
        agent: A tabular Q-Learning or Double Q-Learning agent.
                Must have get_q_value(s, a) and get_max_q_value(s).
        env: IsaacLab CartPole gym environment.
        discretizer: StateDiscretizer instance.
        n_episodes: Number of evaluation episodes to run.
        gamma: Discount factor for MC return computation.
        max_steps: Max steps per episode (safety limit).
        eval_epsilon: Epsilon during evaluation (0.0 = greedy).
        save_behavior_csv: If provided, save per-step behavior to this CSV.

    Returns:
        Dict with:
            "bias_stats": aggregated bias statistics (both metrics)
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

            # Get action, Q(s, a_taken), and max_a Q(s, a)
            action_idx = agent.get_action_index(state_key)
            q_taken = agent.get_q_value(state_key, action_idx)
            max_q = agent.get_max_q_value(state_key)

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

            # Log step with both Q metrics
            logger.log_step(
                step=step,
                raw_obs=obs,
                action_idx=action_idx,
                action_value=action_val,
                q_estimated=q_taken,
                max_q_estimated=max_q,
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
