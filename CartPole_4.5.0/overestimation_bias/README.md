# Overestimation Bias in Value-Based Reinforcement Learning

## Research Questions

1. **Does overestimation bias actually occur** in IsaacLab CartPole?
2. **How does it affect CartPole behavior** — pole stability, cart drift, action selection?
3. **Does function approximation amplify the bias** compared to tabular methods?

## Bias Metrics

| Metric | Formula | What it tests |
|--------|---------|---------------|
| **TakenAction_Bias** | `Q(s, a_taken) − MC_Return` | Bias of the selected action |
| **MaxQ_Bias** | `max_a Q(s,a) − MC_Return` | Bias from the max operator (key metric) |

> **Important:** Overestimation is confirmed only if MaxQ_Bias is consistently **positive**.

---

## Phase 1: Tabular RL

### Algorithms
- **Q-Learning**: `target = r + γ * max_a' Q(s', a')` — same table selects & evaluates → bias
- **Double Q-Learning**: `target = r + γ * Q_B(s', argmax_a' Q_A(s', a'))` — decoupled → less bias

### Result
Q-Learning MaxQ_Bias = **+0.80** (overestimation confirmed).
Double Q-Learning MaxQ_Bias = **−1.24** (conservative).

---

## Phase 2: Deep RL

### Algorithms

**DQN** — uses a neural network to approximate Q-values:
```
target = r + γ * max_a' Q_target(s', a')
```
The max operator on the target network causes overestimation, analogous to tabular Q-Learning.

**Double DQN** — decouples selection from evaluation:
```
a* = argmax_a' Q_online(s', a')
target = r + γ * Q_target(s', a*)
```
The online network selects the action, but the target network evaluates it. This breaks the correlation that inflates Q-values.

### Architecture
```
QNetwork: Linear(4, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 5)
```

---

## Environment
- `Stabilize-Isaac-Cartpole-v0` (IsaacLab)
- Observation: 4D continuous `[cart_pos, pole_angle, cart_vel, pole_ang_vel]`
- Actions: 5 discrete `[-1.0, -0.5, 0.0, 0.5, 1.0]`

## File Structure
```
overestimation_bias/
├── algorithms/
│   ├── q_learning.py           # Tabular Q-Learning
│   ├── double_q_learning.py    # Tabular Double Q-Learning
│   ├── dqn.py                  # DQN agent
│   └── double_dqn.py           # Double DQN agent
├── utils/
│   ├── discretizer.py          # State discretization (tabular)
│   ├── bias_measurement.py     # Tabular bias evaluation
│   ├── behavior_logger.py      # Per-step state logging
│   ├── networks.py             # QNetwork architecture
│   ├── replay_buffer.py        # Experience replay buffer
│   └── dqn_bias_measurement.py # DQN bias evaluation
├── configs/
│   ├── hyperparams.py          # Tabular config
│   └── dqn_hyperparams.py      # DQN config
├── scripts/
│   ├── smoke_test_env.py
│   ├── train_tabular.py        # Tabular multi-seed training
│   ├── evaluate_behavior.py    # Tabular evaluation
│   ├── plot_tabular_bias_behavior.py
│   ├── train_dqn.py            # DQN multi-seed training
│   ├── evaluate_dqn_bias.py    # DQN evaluation
│   └── plot_dqn_results.py     # DQN plots
└── README.md
```

## How to Run

### Phase 1: Tabular
```bash
python scripts/train_tabular.py --algorithm q_learning --mode full
python scripts/train_tabular.py --algorithm double_q_learning --mode full
python scripts/plot_tabular_bias_behavior.py --auto
```

### Phase 2: DQN

#### Debug (1 seed, 500 episodes)
```bash
python scripts/train_dqn.py --algorithm dqn --mode debug
python scripts/train_dqn.py --algorithm double_dqn --mode debug
```

#### Full Experiment (5 seeds, 3000 episodes)
```bash
python scripts/train_dqn.py --algorithm dqn --mode full
python scripts/train_dqn.py --algorithm double_dqn --mode full
```

#### Generate Plots
```bash
python scripts/plot_dqn_results.py --auto
```

## DQN Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| γ (discount) | 0.99 |
| Batch size | 64 |
| Buffer size | 50,000 |
| Min buffer | 1,000 |
| ε: 1.0 → 0.05 | Linear over 50k steps |
| Target update | Every 1,000 steps |
| Gradient clip | 10.0 |
| Debug | 500 ep, 1 seed |
| Full | 3,000 ep, 5 seeds |
| Seeds | [0, 42, 123, 256, 999] |
