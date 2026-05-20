"""
Standalone evaluation script for trained tabular agents.

Loads a trained Q-table, runs greedy evaluation episodes, and produces
detailed behavior data for analysis.

This script can be run AFTER training to collect more evaluation episodes
or to evaluate at a different checkpoint.

Usage:
    python scripts/evaluate_behavior.py --task Stabilize-Isaac-Cartpole-v0 \\
        --algorithm q_learning \\
        --model results/q_learning_TIMESTAMP/models/q_learning_final.pkl \\
        --output results/q_learning_TIMESTAMP/behavior/final_eval.csv \\
        --eval_episodes 50
"""

import argparse
import sys
import os
import json

# ===== IsaacLab launch ===== #
from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

parser = argparse.ArgumentParser(description="Evaluate trained tabular RL agent behavior.")
parser.add_argument("--task", type=str, default="Stabilize-Isaac-Cartpole-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--algorithm", type=str, default="q_learning",
                    choices=["q_learning", "double_q_learning"],
                    help="Which algorithm to evaluate.")
parser.add_argument("--model", type=str, required=True,
                    help="Path to trained model (.pkl file).")
parser.add_argument("--output", type=str, default=None,
                    help="Output CSV path for behavior data.")
parser.add_argument("--eval_episodes", type=int, default=50,
                    help="Number of greedy evaluation episodes.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===== Now safe to import everything else ===== #
import gymnasium as gym
import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

import CartPole.tasks  # noqa: F401

from algorithms.q_learning import QLearning
from algorithms.double_q_learning import DoubleQLearning
from utils.discretizer import StateDiscretizer
from utils.bias_measurement import evaluate_tabular_agent
from configs.hyperparams import TabularConfig


def main():
    """Evaluate a trained agent and save behavior data."""

    cfg = TabularConfig()
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    # ------------------------------------------------------------------ #
    # Create environment
    # ------------------------------------------------------------------ #
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # ------------------------------------------------------------------ #
    # Create agent and load trained model
    # ------------------------------------------------------------------ #
    common_kwargs = dict(
        num_actions=cfg.num_actions,
        action_values=cfg.action_values,
        learning_rate=cfg.learning_rate,
        discount_factor=cfg.discount_factor,
        epsilon=0.0,       # greedy
        epsilon_end=0.0,
        epsilon_decay=1.0,
    )
    if args_cli.algorithm == "q_learning":
        agent = QLearning(**common_kwargs)
    else:
        agent = DoubleQLearning(**common_kwargs)

    model_dir = os.path.dirname(args_cli.model)
    model_file = os.path.basename(args_cli.model)
    agent.load(model_dir, model_file)
    agent.epsilon = 0.0  # ensure greedy

    discretizer = StateDiscretizer(bins=cfg.bins, obs_bounds=cfg.obs_bounds)

    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {args_cli.algorithm}")
    print(f"  Model:      {args_cli.model}")
    print(f"  Episodes:   {args_cli.eval_episodes}")
    print(f"  States in Q-table: {agent.num_visited_states()}")
    print(f"{'=' * 70}\n")

    # ------------------------------------------------------------------ #
    # Output path
    # ------------------------------------------------------------------ #
    if args_cli.output is None:
        out_dir = os.path.dirname(args_cli.model).replace("models", "behavior")
        os.makedirs(out_dir, exist_ok=True)
        args_cli.output = os.path.join(out_dir, "final_eval.csv")

    # ------------------------------------------------------------------ #
    # Run evaluation
    # ------------------------------------------------------------------ #
    eval_result = evaluate_tabular_agent(
        agent=agent,
        env=env,
        discretizer=discretizer,
        n_episodes=args_cli.eval_episodes,
        gamma=cfg.discount_factor,
        max_steps=cfg.max_steps_per_episode,
        eval_epsilon=0.0,
        save_behavior_csv=args_cli.output,
    )

    stats = eval_result["bias_stats"]
    logger = eval_result["behavior_logger"]

    # ------------------------------------------------------------------ #
    # Print results
    # ------------------------------------------------------------------ #
    print(f"\n  Evaluation Results:")
    print(f"  {'─' * 50}")
    print(f"  Avg Episode Reward:   {eval_result['avg_reward']:8.2f}")
    print(f"  Avg Episode Duration: {eval_result['avg_duration']:8.1f} steps")
    print(f"  {'─' * 50}")
    print(f"  Avg Q-estimate:       {stats['avg_q_estimate']:8.4f}")
    print(f"  Avg MC Return:        {stats['avg_mc_return']:8.4f}")
    print(f"  Avg Bias:             {stats['avg_bias']:+8.4f}")
    print(f"  Std Bias:             {stats['std_bias']:8.4f}")
    print(f"  Max Bias:             {stats['max_bias']:+8.4f}")
    print(f"  Min Bias:             {stats['min_bias']:+8.4f}")
    print(f"  Total samples:        {stats['num_samples']}")

    # Action distribution
    action_dist = logger.get_action_distribution(cfg.num_actions)
    total_actions = sum(action_dist.values())
    print(f"\n  Action Distribution:")
    for idx, count in action_dist.items():
        pct = count / max(total_actions, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"    Action {idx} ({cfg.action_values[idx]:+.1f}): "
              f"{count:5d} ({pct:5.1f}%) {bar}")

    # Episode summaries
    summaries = logger.get_episode_summaries()
    if summaries:
        avg_pole = np.mean([s["avg_abs_pole_angle"] for s in summaries])
        avg_cart = np.mean([s["avg_abs_cart_position"] for s in summaries])
        max_pole = max(s["max_abs_pole_angle"] for s in summaries)
        max_cart = max(s["max_abs_cart_position"] for s in summaries)
        print(f"\n  CartPole Behavior:")
        print(f"    Avg |pole_angle|:      {avg_pole:.4f} rad ({np.degrees(avg_pole):.2f}°)")
        print(f"    Max |pole_angle|:      {max_pole:.4f} rad ({np.degrees(max_pole):.2f}°)")
        print(f"    Avg |cart_position|:   {avg_cart:.4f}")
        print(f"    Max |cart_position|:   {max_cart:.4f}")

    # Also save as JSON summary
    json_path = args_cli.output.replace(".csv", "_summary.json")
    summary_data = {
        "algorithm": args_cli.algorithm,
        "model": args_cli.model,
        "eval_episodes": args_cli.eval_episodes,
        "bias_stats": stats,
        "avg_reward": eval_result["avg_reward"],
        "avg_duration": eval_result["avg_duration"],
        "action_distribution": action_dist,
        "episode_summaries": summaries,
    }
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=float)

    print(f"\n  Behavior CSV:  {args_cli.output}")
    print(f"  Summary JSON:  {json_path}")
    print(f"{'=' * 70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
