"""
Training script for DQN and Double DQN on IsaacLab CartPole.

Usage:
    # Debug: single seed, 500 episodes
    python scripts/train_dqn.py --algorithm dqn --mode debug

    # Full: 5 seeds, 3000 episodes each
    python scripts/train_dqn.py --algorithm double_dqn --mode full

    # Custom
    python scripts/train_dqn.py --algorithm dqn --episodes 1000 --seeds 0 42
"""

import argparse, sys, os, json, time, csv

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

parser = argparse.ArgumentParser(description="Train DQN agents on CartPole.")
parser.add_argument("--task", type=str, default="Stabilize-Isaac-Cartpole-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--algorithm", type=str, default="dqn",
                    choices=["dqn", "double_dqn"])
parser.add_argument("--mode", type=str, default=None, choices=["debug", "full"])
parser.add_argument("--episodes", type=int, default=None)
parser.add_argument("--seeds", type=int, nargs="+", default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg
from isaaclab_tasks.utils import parse_env_cfg
import CartPole.tasks  # noqa: F401

from algorithms.dqn import DQNAgent
from algorithms.double_dqn import DoubleDQNAgent
from utils.dqn_bias_measurement import evaluate_dqn_agent
from configs.dqn_hyperparams import DQNConfig, dqn_debug_config, dqn_full_config


def extract_obs(obs) -> np.ndarray:
    """Extract flat numpy observation from IsaacLab obs dict/tensor."""
    if isinstance(obs, dict):
        t = obs.get("policy", next(iter(obs.values())))
    else:
        t = obs
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy().flatten()
    return np.asarray(t).flatten()


def create_agent(algorithm: str, cfg: DQNConfig, device: str):
    """Create DQN or DoubleDQN agent."""
    kwargs = dict(
        obs_dim=cfg.obs_dim, num_actions=cfg.num_actions,
        action_values=cfg.action_values, lr=cfg.learning_rate,
        gamma=cfg.gamma, buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size, epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end, epsilon_decay_steps=cfg.epsilon_decay_steps,
        target_update_interval=cfg.target_update_interval,
        gradient_clip=cfg.gradient_clip, device=device,
    )
    if algorithm == "dqn":
        return DQNAgent(**kwargs)
    else:
        return DoubleDQNAgent(**kwargs)


def save_behavior_csv(episode_steps, filepath):
    """Save per-step behavior data from the last evaluation."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    fields = [
        "episode", "step", "cart_position", "pole_angle",
        "cart_velocity", "pole_angular_velocity",
        "action_idx", "action_value", "q_taken", "max_q",
        "reward", "mc_return", "taken_bias", "maxq_bias",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ep_idx, steps in enumerate(episode_steps):
            for rec in steps:
                row = {"episode": ep_idx}
                row.update({k: rec.get(k, 0) for k in fields if k != "episode"})
                writer.writerow(row)


def train_single_seed(seed, algorithm, cfg, env, run_dir):
    """Train one seed and return training log."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    seed_dir = os.path.join(run_dir, "seed_" + str(seed))
    models_dir = os.path.join(seed_dir, "models")
    behavior_dir = os.path.join(seed_dir, "behavior")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(behavior_dir, exist_ok=True)

    device = "cpu"
    agent = create_agent(algorithm, cfg, device)

    print("\n  --- Seed", seed, "---")
    print("  Agent:", agent)

    log = {
        "algorithm": algorithm, "seed": seed,
        "config": {
            "obs_dim": cfg.obs_dim, "num_actions": cfg.num_actions,
            "action_values": cfg.action_values, "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma, "n_episodes": cfg.n_episodes,
            "buffer_size": cfg.buffer_size, "batch_size": cfg.batch_size,
            "epsilon_start": cfg.epsilon_start, "epsilon_end": cfg.epsilon_end,
            "epsilon_decay_steps": cfg.epsilon_decay_steps,
            "target_update_interval": cfg.target_update_interval,
        },
        "episode_rewards": [], "episode_durations": [], "episode_losses": [],
        "epsilon_history": [],
        "bias_eval_episodes": [],
        "bias_avg_mc_return": [], "bias_avg_q_taken": [], "bias_avg_max_q": [],
        "bias_avg_taken_bias": [], "bias_std_taken_bias": [],
        "bias_avg_maxq_bias": [], "bias_std_maxq_bias": [],
        "bias_avg_reward": [], "bias_avg_duration": [],
    }

    start_time = time.time()
    running_reward = 0.0
    running_duration = 0

    for episode in tqdm(range(cfg.n_episodes), desc="[seed=" + str(seed) + "] " + agent.name):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_losses = []
        step = 0

        while not done and step < cfg.max_steps_per_episode:
            obs_np = extract_obs(obs)
            action_idx = agent.get_action_index(obs_np)

            action_val = cfg.action_values[action_idx]
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

            next_obs_np = extract_obs(next_obs)
            agent.store_transition(obs_np, action_idx, rew, next_obs_np, term or trunc)

            loss = agent.update(min_buffer_size=cfg.min_buffer_size)
            if loss is not None:
                ep_losses.append(loss)

            agent.step_epsilon()
            agent.maybe_update_target()

            ep_reward += rew
            done = term or trunc
            obs = next_obs
            step += 1

        log["episode_rewards"].append(ep_reward)
        log["episode_durations"].append(step)
        log["episode_losses"].append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        log["epsilon_history"].append(agent.epsilon)

        running_reward += ep_reward
        running_duration += step

        if (episode + 1) % cfg.log_interval == 0:
            avg_r = running_reward / cfg.log_interval
            avg_d = running_duration / cfg.log_interval
            tqdm.write(
                "  [seed=%d] Ep %5d | Rew: %7.2f | Dur: %6.1f | eps: %.4f | steps: %d"
                % (seed, episode + 1, avg_r, avg_d, agent.epsilon, agent.total_steps)
            )
            running_reward = 0.0
            running_duration = 0

        # Periodic bias evaluation
        if (episode + 1) % cfg.eval_interval == 0:
            ep_num = episode + 1
            tqdm.write("  [seed=%d] [Eval] %d greedy episodes at ep %d..."
                        % (seed, cfg.eval_episodes, ep_num))

            eval_result = evaluate_dqn_agent(
                agent=agent, env=env, n_episodes=cfg.eval_episodes,
                gamma=cfg.gamma, action_values=cfg.action_values,
                max_steps=cfg.max_steps_per_episode,
            )

            s = eval_result
            tqdm.write(
                "  [seed=%d] [Eval] MC: %7.3f | Q_taken: %7.3f | maxQ: %7.3f | "
                "TakenBias: %+7.3f | MaxQBias: %+7.3f | Rew: %7.2f"
                % (seed, s["avg_mc_return"], s["avg_q_taken"], s["avg_max_q"],
                   s["avg_taken_bias"], s["avg_maxq_bias"], s["avg_reward"])
            )

            log["bias_eval_episodes"].append(ep_num)
            log["bias_avg_mc_return"].append(s["avg_mc_return"])
            log["bias_avg_q_taken"].append(s["avg_q_taken"])
            log["bias_avg_max_q"].append(s["avg_max_q"])
            log["bias_avg_taken_bias"].append(s["avg_taken_bias"])
            log["bias_std_taken_bias"].append(s["std_taken_bias"])
            log["bias_avg_maxq_bias"].append(s["avg_maxq_bias"])
            log["bias_std_maxq_bias"].append(s["std_maxq_bias"])
            log["bias_avg_reward"].append(s["avg_reward"])
            log["bias_avg_duration"].append(s["avg_duration"])

            # Save behavior CSV
            csv_path = os.path.join(behavior_dir, "behavior_ep%d.csv" % ep_num)
            save_behavior_csv(eval_result["episode_steps"], csv_path)

        # Periodic checkpoint
        if (episode + 1) % cfg.save_interval == 0:
            agent.save(models_dir, "%s_ep%d.pt" % (algorithm, episode + 1))

    elapsed = time.time() - start_time
    log["training_time_seconds"] = elapsed
    agent.save(models_dir, "%s_final.pt" % algorithm)

    with open(os.path.join(seed_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print("  [seed=%d] Done in %.1fs — total_steps: %d" % (seed, elapsed, agent.total_steps))
    return log


def aggregate_across_seeds(all_logs, run_dir):
    """Compute mean and std across seeds."""
    if not all_logs:
        return {}

    import numpy as _np

    agg = {
        "algorithm": all_logs[0]["algorithm"],
        "n_seeds": len(all_logs),
        "seeds": [l["seed"] for l in all_logs],
        "config": all_logs[0]["config"],
        "n_episodes": all_logs[0]["config"]["n_episodes"],
        "bias_eval_episodes": all_logs[0].get("bias_eval_episodes", []),
    }

    for key in ["episode_rewards", "episode_durations", "episode_losses"]:
        vals = _np.array([l[key] for l in all_logs])
        agg[key + "_mean"] = vals.mean(axis=0).tolist()
        agg[key + "_std"] = vals.std(axis=0).tolist()

    for key in ["bias_avg_mc_return", "bias_avg_q_taken", "bias_avg_max_q",
                 "bias_avg_taken_bias", "bias_avg_maxq_bias",
                 "bias_avg_reward", "bias_avg_duration"]:
        vals = _np.array([l[key] for l in all_logs])
        agg[key + "_mean"] = vals.mean(axis=0).tolist()
        agg[key + "_std"] = vals.std(axis=0).tolist()

    path = os.path.join(run_dir, "aggregated_results.json")
    with open(path, "w") as f:
        json.dump(agg, f, indent=2)
    print("\n  Aggregated results saved to", path)
    return agg


def main():
    if args_cli.mode == "debug":
        cfg = dqn_debug_config()
    elif args_cli.mode == "full":
        cfg = dqn_full_config()
    else:
        cfg = DQNConfig()

    if args_cli.episodes is not None:
        cfg.n_episodes = args_cli.episodes
    if args_cli.seeds is not None:
        cfg.seeds = args_cli.seeds

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args_cli.algorithm + "_" + timestamp
    run_dir = os.path.join(os.path.dirname(__file__), "..", cfg.results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Training: %s" % args_cli.algorithm)
    print("  Task:     %s" % args_cli.task)
    print("  Episodes: %d" % cfg.n_episodes)
    print("  Seeds:    %s (%d runs)" % (cfg.seeds, len(cfg.seeds)))
    print("  Output:   %s" % run_dir)
    print("=" * 70)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    all_logs = []
    total_start = time.time()

    for i, seed in enumerate(cfg.seeds):
        print("\n" + "-" * 70)
        print("  Run %d/%d - Seed %d" % (i + 1, len(cfg.seeds), seed))
        print("-" * 70)

        log = train_single_seed(seed, args_cli.algorithm, cfg, env, run_dir)
        all_logs.append(log)

    if len(all_logs) > 1:
        aggregate_across_seeds(all_logs, run_dir)
    else:
        with open(os.path.join(run_dir, "aggregated_results.json"), "w") as f:
            json.dump(all_logs[0], f, indent=2)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("  All training complete!")
    print("  Total time: %.1fs (%.1f min)" % (total_elapsed, total_elapsed / 60))
    print("  Results: %s" % run_dir)
    print("=" * 70 + "\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()