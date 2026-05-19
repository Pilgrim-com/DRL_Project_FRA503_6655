"""
Plotting script for tabular RL experiment results.

Reads training_log.json files produced by train_tabular.py and generates
comparison plots for Q-Learning vs Double Q-Learning.

Plots generated:
1. Average Episode Reward (smoothed)
2. Episode Duration (smoothed)
3. Average Q-estimate vs Average MC Return over training
4. Average Overestimation Bias over training

Usage:
    # Plot results from two training runs
    python scripts/plot_tabular_results.py \\
        --q_log results/q_learning_20260520_020000/training_log.json \\
        --dq_log results/double_q_learning_20260520_023000/training_log.json \\
        --output results/plots

    # Plot a single run
    python scripts/plot_tabular_results.py \\
        --q_log results/q_learning_20260520_020000/training_log.json \\
        --output results/plots

This script does NOT require IsaacLab — it only reads JSON files and
generates matplotlib figures.
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

# Use a clean, publication-friendly style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

# ------------------------------------------------------------------ #
# Color palette
# ------------------------------------------------------------------ #
COLORS = {
    "q_learning": "#E74C3C",           # Red — overestimates
    "double_q_learning": "#2980B9",     # Blue — less bias
    "q_learning_light": "#FADBD8",
    "double_q_learning_light": "#D4E6F1",
}

LABELS = {
    "q_learning": "Q-Learning",
    "double_q_learning": "Double Q-Learning",
}


def smooth(data, window=50):
    """Apply a simple moving average for smoother curves."""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_log(path):
    """Load a training_log.json file."""
    with open(path, "r") as f:
        return json.load(f)


def plot_reward_curves(logs, output_dir):
    """Plot 1: Average episode reward for each algorithm."""
    fig, ax = plt.subplots()

    for algo_name, log in logs.items():
        rewards = log["episode_rewards"]
        color = COLORS[algo_name]
        label = LABELS[algo_name]

        # Raw rewards (faded)
        ax.plot(rewards, alpha=0.15, color=color, linewidth=0.5)
        # Smoothed
        smoothed = smooth(rewards, window=50)
        episodes = np.arange(len(smoothed)) + 25  # center of window
        ax.plot(episodes, smoothed, color=color, label=f"{label} (smoothed)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward: Q-Learning vs Double Q-Learning")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "reward_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_duration_curves(logs, output_dir):
    """Plot 2: Episode duration (survival time)."""
    fig, ax = plt.subplots()

    for algo_name, log in logs.items():
        durations = log["episode_durations"]
        color = COLORS[algo_name]
        label = LABELS[algo_name]

        ax.plot(durations, alpha=0.15, color=color, linewidth=0.5)
        smoothed = smooth(durations, window=50)
        episodes = np.arange(len(smoothed)) + 25
        ax.plot(episodes, smoothed, color=color, label=f"{label} (smoothed)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Duration (steps)")
    ax.set_title("Episode Duration: Q-Learning vs Double Q-Learning")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "duration_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_q_vs_mc(logs, output_dir):
    """Plot 3: Average Q-estimate vs Average MC return over training."""
    fig, ax = plt.subplots()

    for algo_name, log in logs.items():
        eval_episodes = log["bias_eval_episodes"]
        avg_q = log["bias_avg_q_estimate"]
        avg_mc = log["bias_avg_mc_return"]
        color = COLORS[algo_name]
        label = LABELS[algo_name]

        # Q-estimate: solid line
        ax.plot(eval_episodes, avg_q, color=color, linestyle="-",
                marker="o", markersize=4, label=f"{label} — Q estimate")
        # MC return: dashed line
        ax.plot(eval_episodes, avg_mc, color=color, linestyle="--",
                marker="s", markersize=4, label=f"{label} — MC return")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Value")
    ax.set_title("Q-Value Estimate vs Monte Carlo Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "q_vs_mc_return.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_bias_curves(logs, output_dir):
    """Plot 4: Average overestimation bias over training."""
    fig, ax = plt.subplots()

    for algo_name, log in logs.items():
        eval_episodes = log["bias_eval_episodes"]
        avg_bias = log["bias_avg_bias"]
        std_bias = log["bias_std_bias"]
        color = COLORS[algo_name]
        label = LABELS[algo_name]

        avg_bias = np.array(avg_bias)
        std_bias = np.array(std_bias)
        eval_episodes = np.array(eval_episodes)

        ax.plot(eval_episodes, avg_bias, color=color, marker="o",
                markersize=4, label=label)
        # Shaded standard deviation band
        ax.fill_between(
            eval_episodes,
            avg_bias - std_bias,
            avg_bias + std_bias,
            color=color,
            alpha=0.15,
        )

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, label="Zero bias")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Bias (Q_est − MC_return)")
    ax.set_title("Overestimation Bias: Q-Learning vs Double Q-Learning")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "bias_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined_summary(logs, output_dir):
    """Generate a 2x2 summary figure with all four plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ---- (0,0) Reward ---- #
    ax = axes[0, 0]
    for algo_name, log in logs.items():
        rewards = log["episode_rewards"]
        color = COLORS[algo_name]
        label = LABELS[algo_name]
        ax.plot(rewards, alpha=0.15, color=color, linewidth=0.5)
        smoothed = smooth(rewards, window=50)
        ax.plot(np.arange(len(smoothed)) + 25, smoothed, color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("(a) Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- (0,1) Duration ---- #
    ax = axes[0, 1]
    for algo_name, log in logs.items():
        durations = log["episode_durations"]
        color = COLORS[algo_name]
        label = LABELS[algo_name]
        ax.plot(durations, alpha=0.15, color=color, linewidth=0.5)
        smoothed = smooth(durations, window=50)
        ax.plot(np.arange(len(smoothed)) + 25, smoothed, color=color, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Duration (steps)")
    ax.set_title("(b) Episode Duration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- (1,0) Q vs MC ---- #
    ax = axes[1, 0]
    for algo_name, log in logs.items():
        eval_eps = log["bias_eval_episodes"]
        avg_q = log["bias_avg_q_estimate"]
        avg_mc = log["bias_avg_mc_return"]
        color = COLORS[algo_name]
        label = LABELS[algo_name]
        ax.plot(eval_eps, avg_q, color=color, linestyle="-", marker="o",
                markersize=3, label=f"{label} — Q est")
        ax.plot(eval_eps, avg_mc, color=color, linestyle="--", marker="s",
                markersize=3, label=f"{label} — MC ret")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Value")
    ax.set_title("(c) Q Estimate vs MC Return")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- (1,1) Bias ---- #
    ax = axes[1, 1]
    for algo_name, log in logs.items():
        eval_eps = np.array(log["bias_eval_episodes"])
        avg_bias = np.array(log["bias_avg_bias"])
        std_bias = np.array(log["bias_std_bias"])
        color = COLORS[algo_name]
        label = LABELS[algo_name]
        ax.plot(eval_eps, avg_bias, color=color, marker="o", markersize=3, label=label)
        ax.fill_between(eval_eps, avg_bias - std_bias, avg_bias + std_bias,
                        color=color, alpha=0.15)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Bias (Q_est − MC_ret)")
    ax.set_title("(d) Overestimation Bias")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Overestimation Bias Study: Q-Learning vs Double Q-Learning\n"
        "(CartPole — Stabilize-Isaac-Cartpole-v0)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(output_dir, "combined_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(logs):
    """Print a summary table of final metrics."""
    print("\n" + "=" * 75)
    print(f"  {'Metric':<30s} {'Q-Learning':>18s} {'Double Q-Learning':>18s}")
    print("-" * 75)

    for algo_name, log in logs.items():
        label = LABELS.get(algo_name, algo_name)

    # Gather final metrics
    rows = []
    for metric_name, key, fmt in [
        ("Final Avg Reward (last 100)", "episode_rewards", ".2f"),
        ("Final Avg Duration (last 100)", "episode_durations", ".1f"),
        ("Final Avg Q-estimate", "bias_avg_q_estimate", ".4f"),
        ("Final Avg MC Return", "bias_avg_mc_return", ".4f"),
        ("Final Avg Bias", "bias_avg_bias", "+.4f"),
    ]:
        values = {}
        for algo_name, log in logs.items():
            if key.startswith("bias_"):
                val = log[key][-1] if log[key] else float("nan")
            else:
                data = log[key]
                val = np.mean(data[-100:]) if len(data) >= 100 else np.mean(data)
            values[algo_name] = val

        q_val = values.get("q_learning", float("nan"))
        dq_val = values.get("double_q_learning", float("nan"))
        print(f"  {metric_name:<30s} {q_val:>18{fmt}} {dq_val:>18{fmt}}")

    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot tabular RL experiment results.")
    parser.add_argument("--q_log", type=str, default=None,
                        help="Path to Q-Learning training_log.json")
    parser.add_argument("--dq_log", type=str, default=None,
                        help="Path to Double Q-Learning training_log.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect latest runs in results/ directory")
    args = parser.parse_args()

    # ---- Auto-detect mode ---- #
    if args.auto or (args.q_log is None and args.dq_log is None):
        # Look for the latest results in the results directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(script_dir, "..", "results")

        if not os.path.isdir(results_dir):
            print(f"Error: results directory not found at {results_dir}")
            print("Run train_tabular.py first to generate training data.")
            sys.exit(1)

        # Find latest runs for each algorithm
        for algo in ["q_learning", "double_q_learning"]:
            runs = sorted([
                d for d in os.listdir(results_dir)
                if d.startswith(algo) and os.path.isfile(
                    os.path.join(results_dir, d, "training_log.json")
                )
            ])
            if runs:
                log_path = os.path.join(results_dir, runs[-1], "training_log.json")
                if algo == "q_learning" and args.q_log is None:
                    args.q_log = log_path
                elif algo == "double_q_learning" and args.dq_log is None:
                    args.dq_log = log_path

    # ---- Load logs ---- #
    logs = {}
    if args.q_log and os.path.isfile(args.q_log):
        logs["q_learning"] = load_log(args.q_log)
        print(f"  Loaded Q-Learning log: {args.q_log}")
    if args.dq_log and os.path.isfile(args.dq_log):
        logs["double_q_learning"] = load_log(args.dq_log)
        print(f"  Loaded Double Q-Learning log: {args.dq_log}")

    if not logs:
        print("Error: No training logs found. Provide --q_log and/or --dq_log paths,")
        print("       or run with --auto after training.")
        sys.exit(1)

    # ---- Output directory ---- #
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, "..", "results", "plots")
    os.makedirs(args.output, exist_ok=True)

    print(f"\n  Generating plots in: {args.output}\n")

    # ---- Generate plots ---- #
    plot_reward_curves(logs, args.output)
    plot_duration_curves(logs, args.output)

    # Bias plots (only if bias evaluation data exists)
    has_bias = all(len(log.get("bias_eval_episodes", [])) > 0 for log in logs.values())
    if has_bias:
        plot_q_vs_mc(logs, args.output)
        plot_bias_curves(logs, args.output)
        plot_combined_summary(logs, args.output)
    else:
        print("  ⚠ Skipping bias plots — no bias evaluation data found in logs.")
        print("    Make sure bias_eval_interval is set in config.")

    # ---- Summary table ---- #
    if len(logs) >= 2:
        print_summary_table(logs)

    print(f"  ✓ All plots saved to {args.output}\n")


if __name__ == "__main__":
    main()
