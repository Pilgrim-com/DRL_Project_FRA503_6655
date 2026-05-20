"""
Training script for tabular Q-Learning and Double Q-Learning
on IsaacLab CartPole (Stabilize-Isaac-Cartpole-v0).

Features:
- Multi-seed experiment: runs over multiple random seeds and aggregates
- Supports both Q-Learning and Double Q-Learning via --algorithm flag
- Dual bias metrics: TakenAction_Bias and MaxQ_Bias
- Saves per-step behavior data as CSV at each eval checkpoint
- Saves per-seed training logs as JSON
- Saves aggregated results across seeds

Usage:
    # Debug: single seed, 1000 episodes
    python scripts/train_tabular.py --task Stabilize-Isaac-Cartpole-v0 \\
        --algorithm q_learning --mode debug

    # Full: 5 seeds, 5000 episodes each
    python scripts/train_tabular.py --task Stabilize-Isaac-Cartpole-v0 \\
        --algorithm q_learning --mode full

    # Custom: specific seeds and episodes
    python scripts/train_tabular.py --task Stabilize-Isaac-Cartpole-v0 \\
        --algorithm q_learning --episodes 3000 --seeds 0 42 123
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
parser.add_argument("--mode", type=str, default=None, choices=["debug", "full"],
                    help="Preset mode: 'debug' (1 seed, 1k ep) or 'full' (5 seeds, 5k ep).")
parser.add_argument("--episodes", type=int, default=None,
                    help="Number of training episodes (overrides config/mode).")
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Random seeds to run (overrides config/mode).")
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
from configs.hyperparams import TabularConfig, debug_config, full_config


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


def train_single_seed(
    seed: int,
    algorithm: str,
    cfg: TabularConfig,
    env,
    run_dir: str,
) -> dict:
    """
    Train a single agent with a specific seed and return the training log.
    """
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create per-seed output directories
    seed_dir = os.path.join(run_dir, f"seed_{seed}")
    models_dir = os.path.join(seed_dir, "models")
    behavior_dir = os.path.join(seed_dir, "behavior")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(behavior_dir, exist_ok=True)

    # Create agent and discretizer
    agent = create_agent(algorithm, cfg)
    discretizer = StateDiscretizer(bins=cfg.bins, obs_bounds=cfg.obs_bounds)

    print(f"\n  --- Seed {seed} ---")
    print(f"  Agent: {agent}")

    # Training metrics storage
    training_log = {
        "algorithm": algorithm,
        "seed": seed,
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
            "seed": seed,
        },
        "episode_rewards": [],
        "episode_durations": [],
        "episode_avg_q": [],
        "epsilon_history": [],
        # Bias evaluation data (recorded periodically)
        "bias_eval_episodes": [],
        "bias_avg_mc_return": [],
        "bias_avg_q_taken": [],
        "bias_avg_max_q": [],
        "bias_avg_taken_bias": [],
        "bias_std_taken_bias": [],
        "bias_avg_maxq_bias": [],
        "bias_std_maxq_bias": [],
        "bias_avg_reward": [],
        "bias_avg_duration": [],
        # Behavior CSV file paths
        "behavior_csv_files": [],
    }

    # Training loop
    start_time = time.time()
    running_reward = 0.0
    running_duration = 0

    with torch.inference_mode():
        for episode in tqdm(range(cfg.n_episodes), desc=f"[seed={seed}] {agent.name}"):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            step = 0

            while not done and step < cfg.max_steps_per_episode:
                state = discretizer.discretize(obs)
                action_idx = agent.get_action_index(state)
                action_val = cfg.action_values[action_idx]
                action_tensor = torch.tensor(
                    [[action_val]], dtype=torch.float32
                )
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'device'):
                    action_tensor = action_tensor.to(env.unwrapped.device)

                next_obs, reward, terminated, truncated, info = env.step(action_tensor)

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

                next_state = discretizer.discretize(next_obs)
                agent.update(state, action_idx, rew_val, next_state, term or trunc)

                episode_reward += rew_val
                done = term or trunc
                obs = next_obs
                step += 1

            agent.decay_epsilon()

            training_log["episode_rewards"].append(episode_reward)
            training_log["episode_durations"].append(step)
            training_log["episode_avg_q"].append(agent.get_avg_q_value())
            training_log["epsilon_history"].append(agent.epsilon)

            running_reward += episode_reward
            running_duration += step

            # Periodic logging
            if (episode + 1) % cfg.log_interval == 0:
                avg_rew = running_reward / cfg.log_interval
                avg_dur = running_duration / cfg.log_interval
                tqdm.write(
                    f"  [seed={seed}] Ep {episode + 1:5d} | "
                    f"Rew: {avg_rew:7.2f} | "
                    f"Dur: {avg_dur:6.1f} | "
                    f"ε: {agent.epsilon:.4f} | "
                    f"States: {agent.num_visited_states()}"
                )
                running_reward = 0.0
                running_duration = 0

            # Periodic bias evaluation + behavior logging
            if (episode + 1) % cfg.bias_eval_interval == 0:
                ep_num = episode + 1
                behavior_csv = os.path.join(
                    behavior_dir, f"behavior_ep{ep_num}.csv"
                )

                tqdm.write(f"  [seed={seed}] [Eval] {cfg.bias_eval_episodes} "
                           f"greedy episodes at ep {ep_num}...")

                eval_result = evaluate_tabular_agent(
                    agent=agent,
                    env=env,
                    discretizer=discretizer,
                    n_episodes=cfg.bias_eval_episodes,
                    gamma=cfg.discount_factor,
                    max_steps=cfg.max_steps_per_episode,
                    eval_epsilon=cfg.eval_epsilon,
                    save_behavior_csv=behavior_csv,
                )

                stats = eval_result["bias_stats"]

                tqdm.write(
                    f"  [seed={seed}] [Eval] "
                    f"MC: {stats['avg_mc_return']:7.3f} | "
                    f"Q_taken: {stats['avg_q_taken']:7.3f} | "
                    f"maxQ: {stats['avg_max_q']:7.3f} | "
                    f"TakenBias: {stats['avg_taken_bias']:+7.3f} | "
                    f"MaxQBias: {stats['avg_maxq_bias']:+7.3f} | "
                    f"Rew: {eval_result['avg_reward']:7.2f}"
                )

                training_log["bias_eval_episodes"].append(ep_num)
                training_log["bias_avg_mc_return"].append(stats["avg_mc_return"])
                training_log["bias_avg_q_taken"].append(stats["avg_q_taken"])
                training_log["bias_avg_max_q"].append(stats["avg_max_q"])
                training_log["bias_avg_taken_bias"].append(stats["avg_taken_bias"])
                training_log["bias_std_taken_bias"].append(stats["std_taken_bias"])
                training_log["bias_avg_maxq_bias"].append(stats["avg_maxq_bias"])
                training_log["bias_std_maxq_bias"].append(stats["std_maxq_bias"])
                training_log["bias_avg_reward"].append(eval_result["avg_reward"])
                training_log["bias_avg_duration"].append(eval_result["avg_duration"])
                training_log["behavior_csv_files"].append(behavior_csv)

            # Periodic checkpoint
            if (episode + 1) % cfg.save_interval == 0:
                ckpt_name = f"{algorithm}_ep{episode + 1}.pkl"
                agent.save(models_dir, ckpt_name)

    # Final save
    elapsed = time.time() - start_time
    training_log["training_time_seconds"] = elapsed

    agent.save(models_dir, f"{algorithm}_final.pkl")

    log_path = os.path.join(seed_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print(f"  [seed={seed}] Done in {elapsed:.1f}s — "
          f"States: {agent.num_visited_states()}")

    return training_log


def aggregate_across_seeds(all_logs: list, run_dir: str) -> dict:
    """
    Aggregate training metrics across multiple seeds.
    Computes mean and std for all time-series metrics.
    """
    if not all_logs:
        return {}

    # Get the common eval episodes from the first log
    eval_eps = all_logs[0].get("bias_eval_episodes", [])
    n_episodes = all_logs[0]["config"]["n_episodes"]

    # Aggregate episode-level metrics
    all_rewards = np.array([log["episode_rewards"] for log in all_logs])
    all_durations = np.array([log["episode_durations"] for log in all_logs])

    # Aggregate bias eval metrics
    keys_to_aggregate = [
        "bias_avg_mc_return", "bias_avg_q_taken", "bias_avg_max_q",
        "bias_avg_taken_bias", "bias_avg_maxq_bias",
        "bias_avg_reward", "bias_avg_duration",
    ]

    agg = {
        "algorithm": all_logs[0]["algorithm"],
        "n_seeds": len(all_logs),
        "seeds": [log["seed"] for log in all_logs],
        "config": all_logs[0]["config"],
        "n_episodes": n_episodes,
        "bias_eval_episodes": eval_eps,
        # Episode-level: mean and std
        "episode_rewards_mean": all_rewards.mean(axis=0).tolist(),
        "episode_rewards_std": all_rewards.std(axis=0).tolist(),
        "episode_durations_mean": all_durations.mean(axis=0).tolist(),
        "episode_durations_std": all_durations.std(axis=0).tolist(),
    }

    for key in keys_to_aggregate:
        vals = np.array([log[key] for log in all_logs])
        agg[f"{key}_mean"] = vals.mean(axis=0).tolist()
        agg[f"{key}_std"] = vals.std(axis=0).tolist()

    # Save
    agg_path = os.path.join(run_dir, "aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    print(f"\n  Aggregated results saved to {agg_path}")
    return agg


def main():
    """Train a tabular RL agent across multiple seeds."""

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #
    if args_cli.mode == "debug":
        cfg = debug_config()
    elif args_cli.mode == "full":
        cfg = full_config()
    else:
        cfg = TabularConfig()

    # CLI overrides
    if args_cli.episodes is not None:
        cfg.n_episodes = args_cli.episodes
    if args_cli.seeds is not None:
        cfg.seeds = args_cli.seeds

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args_cli.algorithm}_{timestamp}"
    run_dir = os.path.join(
        os.path.dirname(__file__), "..", cfg.results_dir, run_name
    )
    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"  Training: {args_cli.algorithm}")
    print(f"  Task:     {args_cli.task}")
    print(f"  Episodes: {cfg.n_episodes}")
    print(f"  Seeds:    {cfg.seeds} ({len(cfg.seeds)} runs)")
    print(f"  Output:   {run_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Create environment (shared across seeds)
    # ------------------------------------------------------------------ #
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    # ------------------------------------------------------------------ #
    # Run training for each seed
    # ------------------------------------------------------------------ #
    all_logs = []
    total_start = time.time()

    for i, seed in enumerate(cfg.seeds):
        print(f"\n{'─' * 70}")
        print(f"  Run {i + 1}/{len(cfg.seeds)} — Seed {seed}")
        print(f"{'─' * 70}")

        log = train_single_seed(
            seed=seed,
            algorithm=args_cli.algorithm,
            cfg=cfg,
            env=env,
            run_dir=run_dir,
        )
        all_logs.append(log)

    # ------------------------------------------------------------------ #
    # Aggregate across seeds
    # ------------------------------------------------------------------ #
    if len(all_logs) > 1:
        agg = aggregate_across_seeds(all_logs, run_dir)
    else:
        # Single seed: just copy the log as aggregated
        agg_path = os.path.join(run_dir, "aggregated_results.json")
        with open(agg_path, "w") as f:
            json.dump(all_logs[0], f, indent=2)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  All training complete!")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Seeds:      {cfg.seeds}")
    print(f"  Results:    {run_dir}")
    print(f"{'=' * 70}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
