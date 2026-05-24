# Overestimation Bias in Value-Based Reinforcement Learning

This repository contains the code, experiments, and final report for the FRA503 Deep Reinforcement Learning final project: **"Overestimation Bias in Value-Based RL: From Q-Learning to Double DQN"**.

## Project Objectives & Research Questions

The core objective of this project is to empirically observe, measure, and analyze the overestimation bias caused by the $\max$ operator in value-based RL algorithms, transitioning from simple Tabular RL to complex Deep RL using neural network function approximation.

1. **RQ1:** Does Q-Learning exhibit positive overestimation bias in a tabular setting?
2. **RQ2:** Can Double Q-Learning reduce the positive bias caused by the max operator?
3. **RQ3:** After stabilizing Deep RL training, does Double DQN reduce MaxQ Bias compared to DQN?

### Implementation Challenge: The Deadly Triad
During the transition to Deep RL, we encountered massive Q-value divergence caused by the **Deadly Triad** (Function Approximation + Bootstrapping + Off-Policy Learning). Consequently, the scope was revised to focus on hyperparameter tuning to mitigate this instability, allowing us to accurately measure the underlying bias.

---

## Key Findings

### Experiment 1: Tabular RL
* **Q-Learning** develops a positive MaxQ Bias (+0.796), confirming the overestimation effect of the max operator.
* **Double Q-Learning** reduces this bias, remaining largely in the negative domain (-1.241), and achieves a slightly higher average reward.
* *Behavior:* Q-Learning produces a delayed-correction action pattern, whereas Double Q-Learning concentrates on targeted corrective actions.

### Experiment 2: Deep RL (DQN vs Double DQN)
* **Before Tuning (Stability Diagnostic):** Both methods suffered from Deadly Triad divergence. MaxQ Bias collapsed to extreme negatives (DQN: -17,612, DDQN: -13,582).
* **After Tuning:** Once stabilized (via lower $\gamma$, faster target updates, and gradient clipping), the expected bias mechanics re-emerged. DQN showed a higher positive bias (+0.297), while Double DQN maintained a lower bias (+0.202). 
* *Conclusion:* Lower bias does not automatically guarantee better short-term policy performance; DQN achieved a higher final reward than Double DQN under the tested constraints.

---

## Environment

- **Environment:** `Stabilize-Isaac-Cartpole-v0` (IsaacLab)
- **Observation:** 4D continuous `[cart_pos, pole_angle, cart_vel, pole_ang_vel]`
- **Actions:** 5 discrete force levels `[-1.0, -0.5, 0.0, 0.5, 1.0]`

---

## Repository Structure

```
DRL_Project_FRA503_6655/
├── README.md
├── Report/                           # Final LaTeX report and presentation slides (untracked by Git)
├── overestimation_bias/
│   ├── algorithms/                   # Q-Learning, Double Q-Learning, DQN, Double DQN
│   ├── configs/                      # Hyperparameters (Tabular & Deep)
│   ├── scripts/                      # Training and evaluation scripts
│   ├── utils/                        # Replay buffer, neural networks, bias measurement, etc.
│   ├── results/                      # Raw training logs and checkpoints (untracked by Git)
│   └── all_plots/                    # Generated plots for all experiments
│       ├── tabular_plots/
│       ├── dqn_plots_before_tuning/
│       ├── dqn_plots_after_tuning/
│       └── tuning_comparison/
├── source/                           # Custom CartPole environment wrappers
└── scripts/                          # IsaacLab environment launcher scripts
```

---

## How to Run the Experiments

All training and evaluation scripts are located in `overestimation_bias/scripts/`.

### Experiment 1: Tabular RL
```bash
# Train both algorithms (5 seeds, 5000 episodes)
python overestimation_bias/scripts/train_tabular.py --algorithm q_learning --mode full
python overestimation_bias/scripts/train_tabular.py --algorithm double_q_learning --mode full

# Generate comparison plots
python overestimation_bias/scripts/plot_tabular_bias_behavior.py --auto
```

### Experiment 2: Deep RL

You can test the environment and algorithms using the debug mode (1 seed, 500 episodes):
```bash
python overestimation_bias/scripts/train_dqn.py --algorithm dqn --mode debug
python overestimation_bias/scripts/train_dqn.py --algorithm double_dqn --mode debug
```

To run the full stabilized experiment (5 seeds, 1000 episodes, tuned hyperparameters):
```bash
# Ensure configs/dqn_hyperparams.py is set to dqn_stable_config
python overestimation_bias/scripts/train_dqn.py --algorithm dqn --mode full
python overestimation_bias/scripts/train_dqn.py --algorithm double_dqn --mode full

# Generate comparison plots
python overestimation_bias/scripts/plot_dqn_results.py --auto
```

---

## Report & Presentation

The final compiled report and presentation slides can be found in the `Report/` directory. Note that the compiled PDFs and LaTeX source are excluded from version control to keep the repository clean.
