"""
Standalone evaluation script for trained tabular agents.

Loads a trained Q-table, runs greedy evaluation episodes, and produces
detailed behavior data with both bias metrics (TakenAction_Bias, MaxQ_Bias).

Usage:
    python scripts/evaluate_behavior.py --task Stabilize-Isaac-Cartpole-v0 \\
        --algorithm q_learning \\
        --model results/q_learning_TIMESTAMP/seed_42/models/q_learning_final.pkl \\
        --eval_episodes 50
"""

import argparse, sys, os, json

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

parser = argparse.ArgumentParser(description="Evaluate trained tabular RL agent.")
parser.add_argument("--task", type=str, default="Stabilize-Isaac-Cartpole-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--algorithm", type=str, default="q_learning",
                    choices=["q_learning", "double_q_learning"])
parser.add_argument("--model", type=str, required=True, help="Path to .pkl model.")
parser.add_argument("--output", type=str, default=None, help="Output CSV path.")
parser.add_argument("--eval_episodes", type=int, default=50)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
    cfg = TabularConfig()
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    common_kwargs = dict(
        num_actions=cfg.num_actions, action_values=cfg.action_values,
        learning_rate=cfg.learning_rate, discount_factor=cfg.discount_factor,
        epsilon=0.0, epsilon_end=0.0, epsilon_decay=1.0,
    )
    agent = QLearning(**common_kwargs) if args_cli.algorithm == "q_learning" else DoubleQLearning(**common_kwargs)
    agent.load(os.path.dirname(args_cli.model), os.path.basename(args_cli.model))
    agent.epsilon = 0.0

    discretizer = StateDiscretizer(bins=cfg.bins, obs_bounds=cfg.obs_bounds)

    print(f"\n{'='*70}")
    print(f"  Evaluating: {args_cli.algorithm} | Episodes: {args_cli.eval_episodes}")
    print(f"  Model: {args_cli.model} | States: {agent.num_visited_states()}")
    print(f"{'='*70}\n")

    if args_cli.output is None:
        out_dir = os.path.dirname(args_cli.model).replace("models", "behavior")
        os.makedirs(out_dir, exist_ok=True)
        args_cli.output = os.path.join(out_dir, "final_eval.csv")

    eval_result = evaluate_tabular_agent(
        agent=agent, env=env, discretizer=discretizer,
        n_episodes=args_cli.eval_episodes, gamma=cfg.discount_factor,
        max_steps=cfg.max_steps_per_episode, eval_epsilon=0.0,
        save_behavior_csv=args_cli.output,
    )

    stats = eval_result["bias_stats"]
    logger = eval_result["behavior_logger"]

    print(f"  Avg Reward:        {eval_result['avg_reward']:8.2f}")
    print(f"  Avg Duration:      {eval_result['avg_duration']:8.1f}")
    print(f"  Avg Q_taken:       {stats['avg_q_taken']:8.4f}")
    print(f"  Avg max_a Q:       {stats['avg_max_q']:8.4f}")
    print(f"  Avg MC Return:     {stats['avg_mc_return']:8.4f}")
    print(f"  TakenAction Bias:  {stats['avg_taken_bias']:+8.4f}")
    print(f"  MaxQ Bias:         {stats['avg_maxq_bias']:+8.4f}")

    action_dist = logger.get_action_distribution(cfg.num_actions)
    total = max(sum(action_dist.values()), 1)
    print(f"\n  Action Distribution:")
    for idx, count in action_dist.items():
        pct = count / total * 100
        print(f"    Action {idx} ({cfg.action_values[idx]:+.1f}): {count:5d} ({pct:5.1f}%)")

    json_path = args_cli.output.replace(".csv", "_summary.json")
    with open(json_path, "w") as f:
        json.dump({"algorithm": args_cli.algorithm, "bias_stats": stats,
                    "avg_reward": eval_result["avg_reward"],
                    "avg_duration": eval_result["avg_duration"],
                    "action_distribution": action_dist}, f, indent=2, default=float)

    print(f"\n  CSV: {args_cli.output}")
    print(f"  JSON: {json_path}\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
