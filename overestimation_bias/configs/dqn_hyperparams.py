"""
DQN hyperparameter configuration.

Shared defaults for DQN and Double DQN experiments.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DQNConfig:
    """Configuration for DQN and Double DQN on IsaacLab CartPole."""

    # Environment
    task: str = "Stabilize-Isaac-Cartpole-v0"

    # Action discretization
    num_actions: int = 5
    action_values: List[float] = field(
        default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0]
    )

    # Observation
    obs_dim: int = 4

    # Training
    n_episodes: int = 500
    max_steps_per_episode: int = 500

    # Network
    learning_rate: float = 1e-4
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer_size: int = 5_000  # Increased for diverse warmup batches
    gradient_clip: float = 1.0    # Reduced to prevent destabilizing large updates
    gamma: float = 0.95           # Reduced to prevent scale inflation in short horizons

    # Epsilon schedule (linear decay over steps)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000

    # Target network
    target_update_interval: int = 100  # steps. Reduced to correct bootstrap errors early

    # Evaluation
    eval_interval: int = 100    # episodes
    eval_episodes: int = 20

    # Multi-seed
    seeds: List[int] = field(
        default_factory=lambda: [0, 42, 123, 256, 999]
    )

    # Output
    results_dir: str = "results"

    # Logging
    log_interval: int = 50
    save_interval: int = 500


def dqn_debug_config() -> DQNConfig:
    """Quick test: 500 episodes, single seed."""
    cfg = DQNConfig()
    cfg.n_episodes = 500
    cfg.seeds = [42]
    return cfg


def dqn_full_config() -> DQNConfig:
    """Full experiment: 1000 episodes, 5 seeds."""
    cfg = DQNConfig()
    cfg.n_episodes = 1000  # Reduced to iterate faster during debugging
    cfg.seeds = [0, 42, 123, 256, 999]
    return cfg


def dqn_stable_config() -> DQNConfig:
    """Stable multi-seed experiment using tuned hyperparams."""
    cfg = DQNConfig()
    cfg.n_episodes = 1000
    cfg.seeds = [0, 42, 123, 256, 999]
    return cfg


DQN_DEFAULT_CONFIG = DQNConfig()
