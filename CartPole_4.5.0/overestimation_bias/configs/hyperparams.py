"""
Centralized hyperparameter configuration for tabular RL experiments.

All algorithms share the same hyperparameters for fair comparison.
Modify this file to tune experiments — no need to edit algorithm code.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TabularConfig:
    """Configuration for tabular Q-Learning and Double Q-Learning."""

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    task: str = "Stabilize-Isaac-Cartpole-v0"

    # ------------------------------------------------------------------ #
    # Action discretization
    # ------------------------------------------------------------------ #
    # 5 discrete actions mapped to continuous joint velocity commands
    num_actions: int = 5
    action_values: List[float] = field(
        default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0]
    )

    # ------------------------------------------------------------------ #
    # State discretization
    # ------------------------------------------------------------------ #
    # Number of bins per observation dimension:
    #   [cart_position, pole_angle, cart_velocity, pole_angular_velocity]
    bins: List[int] = field(default_factory=lambda: [8, 16, 8, 8])

    # Observation bounds per dimension (values outside are clipped)
    obs_bounds: List[Tuple[float, float]] = field(
        default_factory=lambda: [
            (-3.0, 3.0),      # cart position  (env terminates at ±3.0)
            (-0.42, 0.42),    # pole angle     (env terminates at ±24° ≈ ±0.42 rad)
            (-5.0, 5.0),      # cart velocity
            (-5.0, 5.0),      # pole angular velocity
        ]
    )

    # ------------------------------------------------------------------ #
    # Learning hyperparameters
    # ------------------------------------------------------------------ #
    learning_rate: float = 0.1
    discount_factor: float = 0.99

    # Epsilon-greedy exploration (multiplicative decay per episode)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995  # epsilon *= epsilon_decay each episode

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    n_episodes: int = 1000          # Start small for debugging; scale to 3000-5000
    max_steps_per_episode: int = 500  # env also terminates at episode_length_s=10

    # ------------------------------------------------------------------ #
    # Logging & checkpoints
    # ------------------------------------------------------------------ #
    log_interval: int = 50          # Print progress every N episodes
    save_interval: int = 200        # Save Q-table checkpoint every N episodes

    # ------------------------------------------------------------------ #
    # Bias evaluation
    # ------------------------------------------------------------------ #
    bias_eval_interval: int = 100   # Run bias evaluation every N episodes
    bias_eval_episodes: int = 20    # Number of greedy episodes for bias eval
    eval_epsilon: float = 0.0       # Greedy during evaluation (no exploration)

    # ------------------------------------------------------------------ #
    # Output directories (relative to overestimation_bias/)
    # ------------------------------------------------------------------ #
    results_dir: str = "results"
    models_dir: str = "results/models"
    logs_dir: str = "results/logs"
    plots_dir: str = "results/plots"


# ====================================================================== #
# Default singleton for easy import
# ====================================================================== #
DEFAULT_CONFIG = TabularConfig()
