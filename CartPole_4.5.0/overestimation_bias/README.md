# Overestimation Bias in Value-Based Reinforcement Learning

## Project Overview

This project studies **overestimation bias** in value-based RL using the **CartPole** environment from **IsaacLab**.

### Research Questions

1. **Does overestimation bias actually occur** in IsaacLab CartPole?
2. **How does it affect CartPole behavior** — pole stability, cart drift, action selection?

### The Problem

Standard Q-Learning uses the **max operator** which uses the same Q-table to both **select** the best action and **evaluate** its value. When Q-values contain estimation errors, `max` systematically picks overestimated values:

```
Q-Learning target:  r + γ * max_a' Q(s', a')    ← same table selects AND evaluates
```

### The Solution: Double Q-Learning

**Double Q-Learning** maintains **two independent Q-tables** and decouples selection from evaluation:

```
Double Q target:  r + γ * Q_B(s', argmax_a' Q_A(s', a'))   ← A selects, B evaluates
```

### How Bias Affects CartPole Control

Overestimation bias doesn't just inflate numbers — it changes **what the agent does**:

- **Over-correction**: Q-Learning may select overestimated actions, causing jerky control
- **Pole oscillation**: Biased action values lead to over-shooting corrections
- **Cart drift**: Systematic action bias can push the cart toward boundaries
- **Shorter episodes**: Overconfident actions can cause earlier termination
- **Q-value inflation**: Q-estimates grow high even when actual returns are lower

---

## Phase 1: Tabular Methods (Current)

### Algorithms
1. **Tabular Q-Learning** — Baseline with max-operator bias
2. **Tabular Double Q-Learning** — Reduced bias via decoupled evaluation

### Environment
- **IsaacLab CartPole** (`Stabilize-Isaac-Cartpole-v0`)
- 4D continuous observations: cart position, pole angle, cart velocity, pole angular velocity
- Action discretized to 5 values: `[-1.0, -0.5, 0.0, 0.5, 1.0]`
- State discretized into bins: `[8, 16, 8, 8]` → 8,192 discrete states

### Bias Measurement

```
Bias(s, a) = Q_estimated(s, a) − MC_Return(s, a)

MC_Return: G_t = r_{t+1} + γ·r_{t+2} + γ²·r_{t+3} + ...
```

Positive bias = overestimation. Measured periodically during training using greedy evaluation episodes.

---

## File Structure

```
overestimation_bias/
├── algorithms/
│   ├── q_learning.py                   # Tabular Q-Learning
│   └── double_q_learning.py            # Tabular Double Q-Learning
├── utils/
│   ├── discretizer.py                  # Continuous → discrete state mapping
│   ├── bias_measurement.py             # MC returns, bias computation, evaluation
│   └── behavior_logger.py              # Per-step CartPole state logging
├── configs/
│   └── hyperparams.py                  # Centralized hyperparameters
├── scripts/
│   ├── smoke_test_env.py               # Verify environment format
│   ├── train_tabular.py                # Train with periodic bias + behavior eval
│   ├── evaluate_behavior.py            # Standalone evaluation with behavior logging
│   └── plot_tabular_bias_behavior.py   # Generate all comparison plots
└── README.md
```

---

## How to Run

### Step 1: Smoke Test
```bash
cd CartPole_4.5.0/overestimation_bias
python scripts/smoke_test_env.py --task Stabilize-Isaac-Cartpole-v0
```

### Step 2: Train Q-Learning
```bash
python scripts/train_tabular.py --task Stabilize-Isaac-Cartpole-v0 \
    --algorithm q_learning --episodes 1000
```

### Step 3: Train Double Q-Learning
```bash
python scripts/train_tabular.py --task Stabilize-Isaac-Cartpole-v0 \
    --algorithm double_q_learning --episodes 1000
```

### Step 4: (Optional) Standalone Evaluation
```bash
python scripts/evaluate_behavior.py --task Stabilize-Isaac-Cartpole-v0 \
    --algorithm q_learning \
    --model results/q_learning_TIMESTAMP/models/q_learning_final.pkl \
    --eval_episodes 50
```

### Step 5: Generate Plots
```bash
python scripts/plot_tabular_bias_behavior.py --auto
```

---

## Plots Generated

| # | Plot | What It Shows |
|---|------|---------------|
| 1 | `01_reward_curves.png` | Episode reward over training |
| 2 | `02_duration_curves.png` | Episode duration (survival time) |
| 3 | `03_q_vs_mc_return.png` | Q-estimate vs actual MC return |
| 4 | `04_bias_curves.png` | Overestimation bias over training |
| 5 | `05_action_distribution.png` | Which actions each algorithm prefers |
| 6 | `06_trajectory_pole_angle.png` | Pole stability comparison |
| 7 | `07_trajectory_cart_position.png` | Cart drift comparison |
| 8 | `08_trajectory_actions.png` | Action selection pattern over time |
| 9 | `09_trajectory_bias.png` | Per-step bias within an episode |
| 10 | `10_combined_summary.png` | 2×3 combined summary figure |

---

## Behavior Data

During each evaluation checkpoint, the training script saves a CSV file with per-step data:

```
episode, step, cart_position, pole_angle, cart_velocity, pole_angular_velocity,
action_idx, action_value, q_estimated, reward, mc_return, bias
```

This enables detailed post-hoc analysis of how overestimation bias affects CartPole control.

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.99 |
| ε start → end | 1.0 → 0.01 (×0.995/episode) |
| State bins | [8, 16, 8, 8] = 8,192 states |
| Actions | [-1.0, -0.5, 0.0, 0.5, 1.0] |
| Debug episodes | 1,000 |
| Full episodes | 3,000–5,000 |
| Eval interval | Every 100 episodes |
| Eval episodes | 20 greedy episodes |

---

## Phase 2 (Planned)
- DQN, Double DQN, Double DQN + Prioritized Experience Replay
