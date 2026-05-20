# Overestimation Bias in Value-Based Reinforcement Learning

## Research Questions

1. **Does overestimation bias actually occur** in IsaacLab CartPole?
2. **How does it affect CartPole behavior** — pole stability, cart drift, action selection?

## Bias Metrics

Two metrics are measured:

| Metric | Formula | What it tests |
|--------|---------|---------------|
| **TakenAction_Bias** | `Q(s, a_taken) − MC_Return` | Bias of the action actually selected |
| **MaxQ_Bias** | `max_a Q(s,a) − MC_Return` | Bias from the max operator (theoretical signature) |

For Double Q-Learning: `max_a Q(s,a) = max_a (Q_A + Q_B)/2`

> **Important:** Overestimation bias is only confirmed if MaxQ_Bias becomes consistently **positive**, especially for Q-Learning. If bias remains negative, it indicates underestimation or relative optimism, not overestimation.

## Algorithms
- **Tabular Q-Learning** — uses max operator (source of bias)
- **Tabular Double Q-Learning** — decoupled selection/evaluation

## Environment
- `Stabilize-Isaac-Cartpole-v0` (IsaacLab)
- State: 4D continuous → discretized `[8, 16, 8, 8]` = 8,192 bins
- Actions: `[-1.0, -0.5, 0.0, 0.5, 1.0]`

## File Structure
```
overestimation_bias/
├── algorithms/
│   ├── q_learning.py
│   └── double_q_learning.py
├── utils/
│   ├── discretizer.py
│   ├── bias_measurement.py       # Dual bias metrics + evaluation
│   └── behavior_logger.py        # Per-step CartPole state logging
├── configs/
│   └── hyperparams.py            # debug_config() / full_config() presets
├── scripts/
│   ├── smoke_test_env.py
│   ├── train_tabular.py           # Multi-seed training
│   ├── evaluate_behavior.py       # Standalone evaluation
│   └── plot_tabular_bias_behavior.py  # All plots
└── README.md
```

## How to Run

### Debug (1 seed, 1k episodes)
```bash
python scripts/train_tabular.py --algorithm q_learning --mode debug
python scripts/train_tabular.py --algorithm double_q_learning --mode debug
```

### Full Experiment (5 seeds, 5k episodes)
```bash
python scripts/train_tabular.py --algorithm q_learning --mode full
python scripts/train_tabular.py --algorithm double_q_learning --mode full
```

### Custom
```bash
python scripts/train_tabular.py --algorithm q_learning --episodes 3000 --seeds 0 42 123
```

### Generate Plots
```bash
python scripts/plot_tabular_bias_behavior.py --auto
```

## Plots Generated

| # | Plot | Description |
|---|------|-------------|
| 1 | `01_reward_curves.png` | Episode reward (mean ± std across seeds) |
| 2 | `02_duration_curves.png` | Episode duration |
| 3 | `03_q_taken_vs_mc.png` | Q(s, a_taken) vs MC return |
| 4 | `04_max_q_vs_mc.png` | max_a Q(s,a) vs MC return |
| 5 | `05_taken_bias.png` | TakenAction_Bias over training |
| 6 | `06_maxq_bias.png` | MaxQ_Bias over training |
| 7 | `07_action_distribution.png` | Action selection frequency |
| 8-10 | Trajectory plots | pole_angle, cart_position, per-step MaxQ_Bias |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| α (learning rate) | 0.1 |
| γ (discount) | 0.99 |
| ε: 1.0 → 0.01 | ×0.995/episode |
| Bins | [8, 16, 8, 8] |
| Debug episodes | 1,000 (1 seed) |
| Full episodes | 5,000 (5 seeds) |
| Eval interval | 100 episodes |
| Eval episodes | 20 greedy |
| Seeds | [0, 42, 123, 256, 999] |
