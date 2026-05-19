"""
Training script for tabular Q-Learning and Double Q-Learning
on IsaacLab CartPole (Stabilize-Isaac-Cartpole-v0).

Features:
- Supports both Q-Learning and Double Q-Learning via --algorithm flag
- Periodic bias evaluation using Monte Carlo returns
- Saves training logs as JSON
- Saves Q-table checkpoints as pickle files
- Progress bar with tqdm

Usage:
    # Q-Learning (1000 episodes for debugging)
    python scripts/train_tabular.py --task Stabilize-Isaac-Cartpole-v0 \\
        --algorithm q_learning --episodes 1000

    # Double Q-Learning
    python scripts/train_tabular.py --task Stabilize-Isaac-Cartpole-v0 \\
        --algorithm double_q_learning --episodes 1000

    # Full experiment (3000 episodes)
    python scripts/train_tabular.py --task Stabilize-Isaac-Cartpole-v0 \\
        --algorithm q_learning --episodes 3000
"""

import argparse
import sys
import os
import json
import time

# ===== IsaacLab launch (must come before other imports) ===== #
from isaaclab.app import AppLauncher

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

parser = argparse.ArgumentParser(description="Train tabular RL agents on CartPole.")
parser.add_argument("--task", type=str, default="Stabilize-Isaac-Cartpole-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--algorithm", type=str, default="q_learning",
                    choices=["q_learning", "double_q_learning"],
                    help="Which algorithm to train.")
parser.add_argument("--episodes", type=int, default=None,
                    help="Number of training episodes (overrides config).")
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
from datetime import datetime
from tqdm import tqdm

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg

# Register CartPole environments
import CartPole.tasks  # noqa: F401

from algorithms.q_learning import QLearning
from algorithms.double_q_learning import DoubleQLearning
from utils.discretizer import StateDiscretizer
from utils.bias_measurement import evaluate_tabular_agent
from configs.hyperparams import TabularConfig


def create_agent(algorithm: str, cfg: TabularConfig):
    """Create the appropriate agent based on algorithm name."""
    common_kwargs = dict(
        num_actions=cfg.num_actions,
        action_values=cfg.action_values,
        learning_rate=cfg.learning_rate,
        discount_factor=cfg.discount_factor,
        epsilon=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay=cfg.epsilon_decay,
    )
    if algorithm == "q_learning":
        return QLearning(**common_kwargs)
    elif algorithm == "double_q_learning":
        return DoubleQLearning(**common_kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    """Train a tabular RL agent on IsaacLab CartPole."""

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    cfg = TabularConfig()
    if args_cli.episodes is not None:
        cfg.n_episodes = args_cli.episodes

    # Set random seeds
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args_cli.algorithm}_{timestamp}"
    run_dir = os.path.join(
        os.path.dirname(__file__), "..", cfg.results_dir, run_name
    )
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  Training: {args_cli.algorithm}")
    print(f"  Task:     {args_cli.task}")
    print(f"  Episodes: {cfg.n_episodes}")
    print(f"  Seed:     {args_cli.seed}")
    print(f"  Output:   {run_dir}")
    print("=" * 70 + "\n")

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
    # Create agent and discretizer
    # ------------------------------------------------------------------ #
    agent = create_agent(args_cli.algorithm, cfg)
    discretizer = StateDiscretizer(bins=cfg.bins, obs_bounds=cfg.obs_bounds)

    print(f"  Agent:       {agent}")
    print(f"  Discretizer: {discretizer}")
    print(f"  Actions:     {cfg.action_values}\n")

    # ------------------------------------------------------------------ #
    # Training metrics storage
    # ------------------------------------------------------------------ #
    training_log = {
        "algorithm": args_cli.algorithm,
        "config": {
            "bins": cfg.bins,
            "obs_bounds": cfg.obs_bounds,
            "num_actions": cfg.num_actions,
            "action_values": cfg.action_values,
            "learning_rate": cfg.learning_rate,
            "discount_factor": cfg.discount_factor,
            "epsilon_start": cfg.epsilon_start,
            "epsilon_end": cfg.epsilon_end,
            "epsilon_decay": cfg.epsilon_decay,
            "n_episodes": cfg.n_episodes,
            "seed": args_cli.seed,
        },
        "episode_rewards": [],
        "episode_durations": [],
        "episode_avg_q": [],
        "epsilon_history": [],
        # Bias evaluation data (recorded periodically)
        "bias_eval_episodes": [],       # episode numbers where bias was evaluated
        "bias_avg_mc_return": [],
        "bias_avg_q_estimate": [],
        "bias_avg_bias": [],
        "bias_std_bias": [],
        "bias_avg_reward": [],
        "bias_avg_duration": [],
    }

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    start_time = time.time()
    running_reward = 0.0
    running_duration = 0

    with torch.inference_mode():
        for episode in tqdm(range(cfg.n_episodes), desc=f"Training {agent.name}"):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            step = 0

            while not done and step < cfg.max_steps_per_episode:
                # 1. Discretize current observation
                state = discretizer.discretize(obs)

                # 2. Select action (epsilon-greedy)
                action_idx = agent.get_action_index(state)

                # 3. Map to continuous action and create tensor for IsaacLab
                action_val = cfg.action_values[action_idx]
                action_tensor = torch.tensor(
                    [[action_val]], dtype=torch.float32
                )
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'device'):
                    action_tensor = action_tensor.to(env.unwrapped.device)

                # 4. Step the environment
                next_obs, reward, terminated, truncated, info = env.step(action_tensor)

                # 5. Extract scalars
                if isinstance(reward, torch.Tensor):
                    rew_val = reward.squeeze().item()
                else:
                    rew_val = float(np.squeeze(reward))

                if isinstance(terminated, torch.Tensor):
                    term = terminated.squeeze().item()
                    trunc = truncated.squeeze().item()
                else:
                    term = bool(np.squeeze(terminated))
                    trunc = bool(np.squeeze(truncated))

                # 6. Discretize next state
                next_state = discretizer.discretize(next_obs)

                # 7. Update Q-table(s)
                agent.update(state, action_idx, rew_val, next_state, term or trunc)

                # 8. Advance
                episode_reward += rew_val
                done = term or trunc
                obs = next_obs
                step += 1

            # ---- End of episode ---- #
            agent.decay_epsilon()

            # Record training metrics
            training_log["episode_rewards"].append(episode_reward)
            training_log["episode_durations"].append(step)
            training_log["episode_avg_q"].append(agent.get_avg_q_value())
            training_log["epsilon_history"].append(agent.epsilon)

            running_reward += episode_reward
            running_duration += step

            # ---- Periodic logging ---- #
            if (episode + 1) % cfg.log_interval == 0:
                avg_rew = running_reward / cfg.log_interval
                avg_dur = running_duration / cfg.log_interval
                tqdm.write(
                    f"  Episode {episode + 1:5d} | "
                    f"Avg Reward: {avg_rew:7.2f} | "
                    f"Avg Duration: {avg_dur:6.1f} | "
                    f"Epsilon: {agent.epsilon:.4f} | "
                    f"States visited: {agent.num_visited_states()}"
                )
                running_reward = 0.0
                running_duration = 0

            # ---- Periodic bias evaluation ---- #
            if (episode + 1) % cfg.bias_eval_interval == 0:
                tqdm.write(f"  [Bias Eval] Running {cfg.bias_eval_episodes} "
                           f"greedy episodes...")
                eval_result = evaluate_tabular_agent(
                    agent=agent,
                    env=env,
                    discretizer=discretizer,
                    n_episodes=cfg.bias_eval_episodes,
                    gamma=cfg.discount_factor,
                    max_steps=cfg.max_steps_per_episode,
                    eval_epsilon=cfg.eval_epsilon,
                )
                stats = eval_result["bias_stats"]
                tqdm.write(
                    f"  [Bias Eval] "
                    f"Avg MC Return: {stats['avg_mc_return']:7.3f} | "
                    f"Avg Q-est: {stats['avg_q_estimate']:7.3f} | "
                    f"Avg Bias: {stats['avg_bias']:+7.3f} | "
                    f"Eval Reward: {eval_result['avg_reward']:7.2f}"
                )

                training_log["bias_eval_episodes"].append(episode + 1)
                training_log["bias_avg_mc_return"].append(stats["avg_mc_return"])
                training_log["bias_avg_q_estimate"].append(stats["avg_q_estimate"])
                training_log["bias_avg_bias"].append(stats["avg_bias"])
                training_log["bias_std_bias"].append(stats["std_bias"])
                training_log["bias_avg_reward"].append(eval_result["avg_reward"])
                training_log["bias_avg_duration"].append(eval_result["avg_duration"])

            # ---- Periodic checkpoint ---- #
            if (episode + 1) % cfg.save_interval == 0:
                ckpt_name = f"{args_cli.algorithm}_ep{episode + 1}.pkl"
                agent.save(models_dir, ckpt_name)

    # ------------------------------------------------------------------ #
    # Final save
    # ------------------------------------------------------------------ #
    elapsed = time.time() - start_time
    training_log["training_time_seconds"] = elapsed

    # Save final model
    agent.save(models_dir, f"{args_cli.algorithm}_final.pkl")

    # Save training log as JSON
    log_path = os.path.join(run_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Training complete!")
    print(f"  Time:    {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Model:   {models_dir}/{args_cli.algorithm}_final.pkl")
    print(f"  Log:     {log_path}")
    print(f"  States:  {agent.num_visited_states()} unique states visited")
    print(f"{'=' * 70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
