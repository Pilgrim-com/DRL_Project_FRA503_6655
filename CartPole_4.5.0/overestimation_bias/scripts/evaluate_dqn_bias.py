"""
Standalone evaluation for trained DQN / Double DQN models.

Usage:
    python scripts/evaluate_dqn_bias.py \\
        --algorithm dqn \\
        --model results/dqn_TIMESTAMP/seed_42/models/dqn_final.pt \\
        --eval_episodes 50
"""

import argparse, sys, os, json

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

parser = argparse.ArgumentParser(description="Evaluate trained DQN agent.")
parser.add_argument("--task", type=str, default="Stabilize-Isaac-Cartpole-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--algorithm", type=str, default="dqn",
                    choices=["dqn", "double_dqn"])
parser.add_argument("--model", type=str, required=True)
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
import csv

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
import CartPole.tasks  # noqa: F401

from algorithms.dqn import DQNAgent
from algorithms.double_dqn import DoubleDQNAgent
from utils.dqn_bias_measurement import evaluate_dqn_agent
from configs.dqn_hyperparams import DQNConfig


def main():
    cfg = DQNConfig()
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    kwargs = dict(
        obs_dim=cfg.obs_dim, num_actions=cfg.num_actions,
        action_values=cfg.action_values, lr=cfg.learning_rate,
        gamma=cfg.gamma, buffer_size=1000, batch_size=cfg.batch_size,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_decay_steps=1,
        target_update_interval=cfg.target_update_interval,
        gradient_clip=cfg.gradient_clip, device="cpu",
    )
    if args_cli.algorithm == "dqn":
        agent = DQNAgent(**kwargs)
    else:
        agent = DoubleDQNAgent(**kwargs)

    agent.load(os.path.dirname(args_cli.model), os.path.basename(args_cli.model))
    agent.epsilon = 0.0

    print("\n" + "=" * 70)
    print("  Evaluating: %s | Episodes: %d" % (args_cli.algorithm, args_cli.eval_episodes))
    print("  Model: %s" % args_cli.model)
    print("=" * 70 + "\n")

    result = evaluate_dqn_agent(
        agent=agent, env=env, n_episodes=args_cli.eval_episodes,
        gamma=cfg.gamma, action_values=cfg.action_values,
        max_steps=cfg.max_steps_per_episode,
    )

    print("  Avg Reward:        %8.2f" % result["avg_reward"])
    print("  Avg Duration:      %8.1f" % result["avg_duration"])
    print("  Avg Q_taken:       %8.4f" % result["avg_q_taken"])
    print("  Avg max_a Q:       %8.4f" % result["avg_max_q"])
    print("  Avg MC Return:     %8.4f" % result["avg_mc_return"])
    print("  TakenAction Bias:  %+8.4f" % result["avg_taken_bias"])
    print("  MaxQ Bias:         %+8.4f" % result["avg_maxq_bias"])

    # Action distribution
    action_counts = [0] * cfg.num_actions
    for steps in result["episode_steps"]:
        for rec in steps:
            action_counts[rec["action_idx"]] += 1
    total = max(sum(action_counts), 1)
    print("\n  Action Distribution:")
    for idx, count in enumerate(action_counts):
        pct = count / total * 100
        print("    Action %d (%+.1f): %5d (%5.1f%%)" % (idx, cfg.action_values[idx], count, pct))

    # Save results
    out_dir = os.path.dirname(args_cli.model).replace("models", "behavior")
    os.makedirs(out_dir, exist_ok=True)

    summary = {k: v for k, v in result.items() if k != "episode_steps"}
    summary["algorithm"] = args_cli.algorithm
    summary["action_counts"] = action_counts
    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n  Results saved to %s\n" % out_dir)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
