# Overestimation Bias in Value-Based Reinforcement Learning

## Project Overview

This project studies **overestimation bias** in value-based reinforcement learning algorithms using the **CartPole** environment from **IsaacLab**.

### The Problem

In standard Q-Learning, the Q-value update uses the **max operator** to estimate the value of the next state:

```
Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
```

The `max` operator uses the **same** Q-table to both **select** the best action and **evaluate** its value. When Q-values contain estimation errors (which they always do during learning), `max` systematically picks actions whose values are *overestimated* due to noise, causing an **upward bias** in Q-value estimates.

### The Solution: Double Q-Learning

**Double Q-Learning** (van Hasselt, 2010) addresses this by maintaining **two independent Q-tables** (Q_A and Q_B):

```
# When updating Q_A:
a* = argmax_a Q_A(s', a)           # Q_A selects the best action
target = r + γ * Q_B(s', a*)       # Q_B evaluates that action (decoupled!)
Q_A(s, a) ← Q_A(s, a) + α * [target - Q_A(s, a)]
```

By **decoupling** action selection from action evaluation, the systematic upward bias is substantially reduced.

---

## Phase 1: Tabular Methods (Current)

### Algorithms
1. **Tabular Q-Learning** — Baseline with max-operator bias
2. **Tabular Double Q-Learning** — Reduced bias via decoupled evaluation

### Environment
- **IsaacLab CartPole** (`Stabilize-Isaac-Cartpole-v0`)
- 4D continuous observations: cart position, pole angle, cart velocity, pole angular velocity
- Continuous action discretized to 5 values: `[-1.0, -0.5, 0.0, 0.5, 1.0]`
- State space discretized into bins: `[8, 16, 8, 8]` → 8,192 discrete states

### Bias Measurement

For each trained policy, we measure overestimation bias by comparing **Q-value estimates** against **Monte Carlo returns**:

1. Run greedy evaluation episodes (ε = 0)
2. For each step, record Q(s, a) from the agent
3. Compute actual discounted return: `G_t = r_{t+1} + γ*r_{t+2} + γ²*r_{t+3} + ...`
4. Compute bias: `Bias(s, a) = Q_estimated(s, a) - G_t`
5. Average across all visited samples

**Positive bias = overestimation.** Q-Learning should show consistently higher positive bias than Double Q-Learning.

---

## File Structure

```
overestimation_bias/
├── algorithms/
│   ├── q_learning.py              # Tabular Q-Learning
│   └── double_q_learning.py       # Tabular Double Q-Learning
├── utils/
│   ├── discretizer.py             # Continuous → discrete state mapping
│   └── bias_measurement.py        # Monte Carlo returns & bias computation
├── configs/
│   └── hyperparams.py             # Centralized hyperparameter configuration
├── scripts/
│   ├── smoke_test_env.py          # Verify environment format
│   ├── train_tabular.py           # Train Q-Learning or Double Q-Learning
│   └── plot_tabular_results.py    # Generate comparison plots
├── results/                       # Auto-created: logs, models, plots
└── README.md                      # This file
```

---

## How to Run

### Prerequisites
- IsaacLab (Isaac Sim) installed and configured
- CartPole extension installed: `python -m pip install -e ./source/CartPole`
- Python packages: `numpy`, `matplotlib`, `tqdm`

### Step 1: Smoke Test (verify environment)

```bash
cd CartPole_4.5.0/overestimation_bias
python scripts/smoke_test_env.py --task Stabilize-Isaac-Cartpole-v0 --num_envs 1
```

This verifies observation shape, action format, and discretizer functionality.

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

### Step 4: Generate Plots

```bash
python scripts/plot_tabular_results.py --auto
```

Or specify log files directly:

```bash
python scripts/plot_tabular_results.py \
    --q_log results/q_learning_TIMESTAMP/training_log.json \
    --dq_log results/double_q_learning_TIMESTAMP/training_log.json
```

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate (α) | 0.1 | Same for both algorithms |
| Discount factor (γ) | 0.99 | Standard value |
| ε start | 1.0 | Full exploration initially |
| ε end | 0.01 | Minimal exploration at convergence |
| ε decay | 0.995 | Multiplicative per episode |
| State bins | [8, 16, 8, 8] | 8,192 total discrete states |
| Actions | [-1.0, -0.5, 0.0, 0.5, 1.0] | 5 discrete actions |
| Bias eval interval | Every 100 episodes | 20 greedy episodes per eval |

---

## Experiment Design

### Experiment 1: Q-Learning vs Double Q-Learning

**Hypothesis:** Q-Learning will exhibit higher overestimation bias than Double Q-Learning because the max operator in Q-Learning uses the same Q-table for both action selection and evaluation.

**Protocol:**
1. Train both algorithms with identical hyperparameters for fair comparison
2. Every 100 training episodes, pause and run 20 greedy evaluation episodes
3. During evaluation, record Q(s,a) estimates and compute Monte Carlo returns
4. Compare: reward curves, duration curves, Q-estimates vs MC returns, bias curves

**Expected Results:**
- Both algorithms should learn to balance the pole (increasing reward/duration)
- Q-Learning's Q-value estimates should consistently exceed MC returns (positive bias)
- Double Q-Learning's estimates should track MC returns more closely (near-zero bias)
- Q-Learning may show higher variance in Q-value estimates

---

## Phase 2 (Planned)
- DQN (Deep Q-Network)
- Double DQN
- Double DQN + Prioritized Experience Replay
