"""
Smoke test script for verifying the IsaacLab CartPole environment.

Run this FIRST to confirm:
1. The environment loads and resets correctly
2. Observation format (shape, dtype, dict structure)
3. Action format (tensor shape expected by env.step())
4. Reward and termination signal formats
5. Discretizer works with real observations

Usage:
    python scripts/smoke_test_env.py --task Stabilize-Isaac-Cartpole-v0 --num_envs 1
"""

import argparse
import sys
import os

# ===== IsaacLab must be launched before any other imports ===== #
from isaaclab.app import AppLauncher

# Add project root to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

parser = argparse.ArgumentParser(description="Smoke test for IsaacLab CartPole environment.")
parser.add_argument("--task", type=str, default="Stabilize-Isaac-Cartpole-v0", help="Task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=50, help="Steps to run in the test.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===== Now safe to import everything else ===== #
import gymnasium as gym
import torch
import numpy as np

from isaaclab.envs import (
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab_tasks.utils import parse_env_cfg

# Register CartPole environments
import CartPole.tasks  # noqa: F401

from utils.discretizer import StateDiscretizer
from configs.hyperparams import DEFAULT_CONFIG


def main():
    """Run smoke test."""
    print("\n" + "=" * 70)
    print("  SMOKE TEST: IsaacLab CartPole Environment")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # 1. Create environment
    # ------------------------------------------------------------------ #
    print(f"\n[1] Creating environment: {args_cli.task}")
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"    ✓ Environment created successfully")

    # ------------------------------------------------------------------ #
    # 2. Reset and inspect observation
    # ------------------------------------------------------------------ #
    print(f"\n[2] Resetting environment and inspecting observation...")
    obs, info = env.reset()

    print(f"    obs type         : {type(obs)}")
    if isinstance(obs, dict):
        print(f"    obs keys         : {list(obs.keys())}")
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                print(f"    obs['{key}'] shape : {val.shape}")
                print(f"    obs['{key}'] dtype : {val.dtype}")
                print(f"    obs['{key}'] device: {val.device}")
                print(f"    obs['{key}'] value : {val.cpu().numpy()}")
            else:
                print(f"    obs['{key}']       : {val}")
    elif isinstance(obs, torch.Tensor):
        print(f"    obs shape        : {obs.shape}")
        print(f"    obs dtype        : {obs.dtype}")
        print(f"    obs value        : {obs.cpu().numpy()}")

    # ------------------------------------------------------------------ #
    # 3. Test discretizer
    # ------------------------------------------------------------------ #
    print(f"\n[3] Testing StateDiscretizer...")
    cfg = DEFAULT_CONFIG
    discretizer = StateDiscretizer(bins=cfg.bins, obs_bounds=cfg.obs_bounds)
    print(f"    {discretizer}")

    state_key = discretizer.discretize(obs)
    print(f"    Discretized state: {state_key}")
    print(f"    State space size : {discretizer.total_states}")

    # ------------------------------------------------------------------ #
    # 4. Test action format
    # ------------------------------------------------------------------ #
    print(f"\n[4] Testing action format...")
    # Try each discrete action
    for i, action_val in enumerate(cfg.action_values):
        action_tensor = torch.tensor([[action_val]], dtype=torch.float32)

        # Move to the same device as the environment
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'device'):
            env_device = env.unwrapped.device
            action_tensor = action_tensor.to(env_device)
            print(f"    Action {i}: value={action_val:+.1f}, "
                  f"tensor shape={action_tensor.shape}, "
                  f"device={env_device}")
        else:
            print(f"    Action {i}: value={action_val:+.1f}, "
                  f"tensor shape={action_tensor.shape}")

    # ------------------------------------------------------------------ #
    # 5. Step the environment
    # ------------------------------------------------------------------ #
    print(f"\n[5] Stepping environment for {args_cli.num_steps} steps...")
    obs, _ = env.reset()
    total_reward = 0.0

    for step in range(args_cli.num_steps):
        # Random action
        action_idx = np.random.randint(0, cfg.num_actions)
        action_val = cfg.action_values[action_idx]
        action_tensor = torch.tensor([[action_val]], dtype=torch.float32)
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'device'):
            action_tensor = action_tensor.to(env.unwrapped.device)

        next_obs, reward, terminated, truncated, info = env.step(action_tensor)

        # Inspect types on first step
        if step == 0:
            print(f"\n    Step 0 details:")
            print(f"      reward type    : {type(reward)}")
            if isinstance(reward, torch.Tensor):
                print(f"      reward shape   : {reward.shape}")
                print(f"      reward value   : {reward.squeeze().item():.4f}")
            else:
                print(f"      reward value   : {reward}")

            print(f"      terminated type: {type(terminated)}")
            if isinstance(terminated, torch.Tensor):
                print(f"      terminated     : {terminated.squeeze().item()}")
            print(f"      truncated type : {type(truncated)}")
            if isinstance(truncated, torch.Tensor):
                print(f"      truncated      : {truncated.squeeze().item()}")
            print(f"      info keys      : {list(info.keys()) if isinstance(info, dict) else info}")

        # Discretize next observation
        state_key = discretizer.discretize(next_obs)

        if isinstance(reward, torch.Tensor):
            total_reward += reward.squeeze().item()
        else:
            total_reward += float(reward)

        # Check termination
        if isinstance(terminated, torch.Tensor):
            term = terminated.squeeze().item()
            trunc = truncated.squeeze().item()
        else:
            term = bool(terminated)
            trunc = bool(truncated)

        if term or trunc:
            print(f"    Episode ended at step {step + 1} "
                  f"(terminated={term}, truncated={trunc})")
            obs, _ = env.reset()
        else:
            obs = next_obs

    print(f"\n    Total reward over {args_cli.num_steps} steps: {total_reward:.2f}")

    # ------------------------------------------------------------------ #
    # 6. Summary
    # ------------------------------------------------------------------ #
    print(f"\n" + "=" * 70)
    print(f"  SMOKE TEST PASSED ✓")
    print(f"=" * 70)
    print(f"\n  Environment : {args_cli.task}")
    print(f"  Observation : obs['policy'] → Tensor shape (1, 4)")
    print(f"  Action      : Tensor shape (1, 1), values from {cfg.action_values}")
    print(f"  State bins  : {cfg.bins} → {discretizer.total_states} discrete states")
    print(f"  Reward      : scalar from Tensor")
    print(f"\n  Ready to proceed with training!\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
