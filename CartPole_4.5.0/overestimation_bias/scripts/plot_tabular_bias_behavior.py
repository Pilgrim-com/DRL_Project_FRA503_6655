"""
Comprehensive plotting script for tabular RL bias + behavior analysis.

Reads training_log.json and behavior CSV files, then generates:

1. Episode reward curves (smoothed)
2. Episode duration curves (smoothed)
3. Q-estimate vs MC return over training
4. Overestimation bias over training
5. Action distribution comparison (bar chart)
6. Trajectory comparison: pole_angle over time
7. Trajectory comparison: cart_position over time
8. Trajectory comparison: selected action over time
9. Trajectory comparison: per-step bias over time
10. Combined 2x3 summary figure

Usage:
    # Auto-detect latest runs in results/
    python scripts/plot_tabular_bias_behavior.py --auto

    # Or specify log files directly
    python scripts/plot_tabular_bias_behavior.py \\
        --q_log results/q_learning_TIMESTAMP/training_log.json \\
        --dq_log results/double_q_learning_TIMESTAMP/training_log.json

This script does NOT require IsaacLab — it only reads JSON/CSV files.
"""

import argparse
import json
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Style
# ------------------------------------------------------------------ #
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})

COLORS = {
    "q_learning": "#E74C3C",          # Red — overestimates
    "double_q_learning": "#2980B9",    # Blue — less bias
}
LABELS = {
    "q_learning": "Q-Learning",
    "double_q_learning": "Double Q-Learning",
}


def smooth(data, window=50):
    """Moving average smoothing."""
    if len(data) < window:
        return np.array(data)
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_log(path):
    """Load a training_log.json file."""
    with open(path, "r") as f:
        return json.load(f)


def load_behavior_csv(path):
    """Load a behavior CSV file into a list of dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in row:
                if key in ("episode", "step", "action_idx"):
                    row[key] = int(row[key])
                elif key != "episode":
                    try:
                        row[key] = float(row[key])
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def find_latest_behavior_csv(run_dir):
    """Find the last behavior CSV (highest episode number) in a run dir."""
    behavior_dir = os.path.join(run_dir, "behavior")
    if not os.path.isdir(behavior_dir):
        return None
    csvs = sorted([f for f in os.listdir(behavior_dir) if f.endswith(".csv")])
    if not csvs:
        return None
    return os.path.join(behavior_dir, csvs[-1])


# ====================================================================== #
# Plot functions
# ====================================================================== #

def plot_reward_curves(logs, output_dir):
    """Plot 1: Episode reward curves."""
    fig, ax = plt.subplots()
    for algo, log in logs.items():
        rewards = log["episode_rewards"]
        color = COLORS[algo]
        ax.plot(rewards, alpha=0.15, color=color, linewidth=0.5)
        s = smooth(rewards)
        ax.plot(np.arange(len(s)) + 25, s, color=color, label=LABELS[algo])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "01_reward_curves.png"), bbox_inches="tight")
    plt.close(fig)


def plot_duration_curves(logs, output_dir):
    """Plot 2: Episode duration curves."""
    fig, ax = plt.subplots()
    for algo, log in logs.items():
        durations = log["episode_durations"]
        color = COLORS[algo]
        ax.plot(durations, alpha=0.15, color=color, linewidth=0.5)
        s = smooth(durations)
        ax.plot(np.arange(len(s)) + 25, s, color=color, label=LABELS[algo])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Duration (steps)")
    ax.set_title("Episode Duration (Survival Time)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "02_duration_curves.png"), bbox_inches="tight")
    plt.close(fig)


def plot_q_vs_mc(logs, output_dir):
    """Plot 3: Average Q-estimate vs MC return over training."""
    fig, ax = plt.subplots()
    for algo, log in logs.items():
        eps = log["bias_eval_episodes"]
        q = log["bias_avg_q_estimate"]
        mc = log["bias_avg_mc_return"]
        color = COLORS[algo]
        ax.plot(eps, q, color=color, linestyle="-", marker="o", markersize=4,
                label=f"{LABELS[algo]} — Q estimate")
        ax.plot(eps, mc, color=color, linestyle="--", marker="s", markersize=4,
                label=f"{LABELS[algo]} — MC return")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Value")
    ax.set_title("Q-Value Estimate vs Monte Carlo Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "03_q_vs_mc_return.png"), bbox_inches="tight")
    plt.close(fig)


def plot_bias_curves(logs, output_dir):
    """Plot 4: Overestimation bias over training."""
    fig, ax = plt.subplots()
    for algo, log in logs.items():
        eps = np.array(log["bias_eval_episodes"])
        bias = np.array(log["bias_avg_bias"])
        std = np.array(log["bias_std_bias"])
        color = COLORS[algo]
        ax.plot(eps, bias, color=color, marker="o", markersize=4, label=LABELS[algo])
        ax.fill_between(eps, bias - std, bias + std, color=color, alpha=0.15)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, label="Zero bias")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Bias (Q_est − MC_return)")
    ax.set_title("Overestimation Bias During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "04_bias_curves.png"), bbox_inches="tight")
    plt.close(fig)


def plot_action_distribution(behavior_data, output_dir, action_values=None):
    """Plot 5: Action distribution comparison (bar chart)."""
    if action_values is None:
        action_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    num_actions = len(action_values)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(num_actions)
    width = 0.35
    n_algos = len(behavior_data)

    for i, (algo, rows) in enumerate(behavior_data.items()):
        counts = [0] * num_actions
        for r in rows:
            idx = int(r["action_idx"])
            if 0 <= idx < num_actions:
                counts[idx] += 1
        total = max(sum(counts), 1)
        pcts = [c / total * 100 for c in counts]

        offset = (i - (n_algos - 1) / 2) * width
        bars = ax.bar(x + offset, pcts, width, label=LABELS[algo],
                      color=COLORS[algo], alpha=0.85)
        # Annotate percentages
        for bar, pct in zip(bars, pcts):
            if pct > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Action")
    ax.set_ylabel("Selection Frequency (%)")
    ax.set_title("Action Distribution During Greedy Evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"a{i}\n({v:+.1f})" for i, v in enumerate(action_values)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(output_dir, "05_action_distribution.png"), bbox_inches="tight")
    plt.close(fig)


def plot_trajectory_comparison(behavior_data, output_dir):
    """Plots 6-9: Trajectory comparisons — pole_angle, cart_position, action, bias."""

    # Use the first (or longest) episode from each algorithm
    ep_data = {}
    for algo, rows in behavior_data.items():
        # Group by episode, pick the longest
        episodes = {}
        for r in rows:
            ep_id = int(r["episode"])
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(r)
        if episodes:
            longest_ep = max(episodes.values(), key=len)
            ep_data[algo] = longest_ep

    if not ep_data:
        return

    # --- Plot 6: Pole angle over time ---
    fig, ax = plt.subplots()
    for algo, steps in ep_data.items():
        t = [s["step"] for s in steps]
        pole = [np.degrees(s["pole_angle"]) for s in steps]
        ax.plot(t, pole, color=COLORS[algo], label=LABELS[algo], alpha=0.9)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Pole Angle (degrees)")
    ax.set_title("Pole Angle Trajectory — Q-Learning vs Double Q-Learning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "06_trajectory_pole_angle.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Plot 7: Cart position over time ---
    fig, ax = plt.subplots()
    for algo, steps in ep_data.items():
        t = [s["step"] for s in steps]
        cart = [s["cart_position"] for s in steps]
        ax.plot(t, cart, color=COLORS[algo], label=LABELS[algo], alpha=0.9)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.axhline(y=3.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="Bounds (±3.0)")
    ax.axhline(y=-3.0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cart Position")
    ax.set_title("Cart Position Trajectory — Q-Learning vs Double Q-Learning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "07_trajectory_cart_position.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Plot 8: Selected action over time ---
    fig, ax = plt.subplots()
    for algo, steps in ep_data.items():
        t = [s["step"] for s in steps]
        actions = [s["action_value"] for s in steps]
        ax.step(t, actions, where="mid", color=COLORS[algo],
                label=LABELS[algo], alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Action Value (joint velocity)")
    ax.set_title("Action Selection Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "08_trajectory_actions.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Plot 9: Per-step bias over time ---
    fig, ax = plt.subplots()
    for algo, steps in ep_data.items():
        t = [s["step"] for s in steps]
        bias = [s["bias"] for s in steps]
        ax.plot(t, bias, color=COLORS[algo], label=LABELS[algo], alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, label="Zero bias")
    ax.set_xlabel("Step")
    ax.set_ylabel("Bias (Q_est − MC_return)")
    ax.set_title("Per-Step Overestimation Bias During Episode")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "09_trajectory_bias.png"), bbox_inches="tight")
    plt.close(fig)


def plot_combined_summary(logs, behavior_data, output_dir):
    """Plot 10: Combined 2×3 summary figure."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # (0,0) Reward
    ax = axes[0, 0]
    for algo, log in logs.items():
        r = log["episode_rewards"]
        ax.plot(r, alpha=0.12, color=COLORS[algo], linewidth=0.5)
        s = smooth(r)
        ax.plot(np.arange(len(s)) + 25, s, color=COLORS[algo], label=LABELS[algo])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("(a) Episode Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Q vs MC
    ax = axes[0, 1]
    for algo, log in logs.items():
        eps = log["bias_eval_episodes"]
        ax.plot(eps, log["bias_avg_q_estimate"], color=COLORS[algo],
                linestyle="-", marker="o", markersize=3, label=f"{LABELS[algo]} Q-est")
        ax.plot(eps, log["bias_avg_mc_return"], color=COLORS[algo],
                linestyle="--", marker="s", markersize=3, label=f"{LABELS[algo]} MC-ret")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.set_title("(b) Q Estimate vs MC Return")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (0,2) Bias
    ax = axes[0, 2]
    for algo, log in logs.items():
        eps = np.array(log["bias_eval_episodes"])
        bias = np.array(log["bias_avg_bias"])
        std = np.array(log["bias_std_bias"])
        ax.plot(eps, bias, color=COLORS[algo], marker="o", markersize=3, label=LABELS[algo])
        ax.fill_between(eps, bias - std, bias + std, color=COLORS[algo], alpha=0.15)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Bias")
    ax.set_title("(c) Overestimation Bias")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom row: trajectory plots (from behavior data)
    ep_data = {}
    for algo, rows in behavior_data.items():
        episodes = {}
        for r in rows:
            ep_id = int(r["episode"])
            if ep_id not in episodes:
                episodes[ep_id] = []
            episodes[ep_id].append(r)
        if episodes:
            longest = max(episodes.values(), key=len)
            ep_data[algo] = longest

    # (1,0) Pole angle
    ax = axes[1, 0]
    for algo, steps in ep_data.items():
        t = [s["step"] for s in steps]
        pole = [np.degrees(s["pole_angle"]) for s in steps]
        ax.plot(t, pole, color=COLORS[algo], label=LABELS[algo], alpha=0.9)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Pole Angle (°)")
    ax.set_title("(d) Pole Angle Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Cart position
    ax = axes[1, 1]
    for algo, steps in ep_data.items():
        t = [s["step"] for s in steps]
        cart = [s["cart_position"] for s in steps]
        ax.plot(t, cart, color=COLORS[algo], label=LABELS[algo], alpha=0.9)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cart Position")
    ax.set_title("(e) Cart Position Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,2) Per-step bias
    ax = axes[1, 2]
    for algo, steps in ep_data.items():
        t = [s["step"] for s in steps]
        bias = [s["bias"] for s in steps]
        ax.plot(t, bias, color=COLORS[algo], label=LABELS[algo], alpha=0.8)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Bias")
    ax.set_title("(f) Per-Step Bias")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Overestimation Bias & CartPole Behavior Analysis\n"
        "Q-Learning vs Double Q-Learning (Stabilize-Isaac-Cartpole-v0)",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "10_combined_summary.png"), bbox_inches="tight")
    plt.close(fig)


# ====================================================================== #
# Summary table
# ====================================================================== #

def print_summary_table(logs, behavior_data):
    """Print a comparison table of final metrics."""
    print(f"\n{'=' * 80}")
    print(f"  {'Metric':<35s} {'Q-Learning':>18s} {'Double Q-Learning':>18s}")
    print(f"{'─' * 80}")

    for metric_name, key, fmt in [
        ("Final Avg Reward (last 100)", "episode_rewards", ".2f"),
        ("Final Avg Duration (last 100)", "episode_durations", ".1f"),
        ("Final Avg Q-estimate", "bias_avg_q_estimate", ".4f"),
        ("Final Avg MC Return", "bias_avg_mc_return", ".4f"),
        ("Final Avg Bias", "bias_avg_bias", "+.4f"),
    ]:
        values = {}
        for algo, log in logs.items():
            if key.startswith("bias_"):
                val = log[key][-1] if log[key] else float("nan")
            else:
                data = log[key]
                val = np.mean(data[-100:]) if len(data) >= 100 else np.mean(data)
            values[algo] = val

        q_val = values.get("q_learning", float("nan"))
        dq_val = values.get("double_q_learning", float("nan"))
        print(f"  {metric_name:<35s} {q_val:>18{fmt}} {dq_val:>18{fmt}}")

    # Behavior metrics from final eval
    for algo, rows in behavior_data.items():
        if not rows:
            continue
        avg_pole = np.mean([abs(r["pole_angle"]) for r in rows])
        avg_cart = np.mean([abs(r["cart_position"]) for r in rows])
        print(f"\n  {LABELS[algo]} Behavior:")
        print(f"    Avg |pole_angle|:    {avg_pole:.4f} rad ({np.degrees(avg_pole):.2f}°)")
        print(f"    Avg |cart_position|: {avg_cart:.4f}")

    print(f"{'=' * 80}\n")


# ====================================================================== #
# Main
# ====================================================================== #

def auto_detect_runs(results_dir):
    """Find the latest run directory for each algorithm."""
    found = {}
    for algo in ["q_learning", "double_q_learning"]:
        runs = sorted([
            d for d in os.listdir(results_dir)
            if d.startswith(algo) and os.path.isfile(
                os.path.join(results_dir, d, "training_log.json")
            )
        ])
        if runs:
            found[algo] = os.path.join(results_dir, runs[-1])
    return found


def main():
    parser = argparse.ArgumentParser(description="Plot tabular RL bias + behavior analysis.")
    parser.add_argument("--q_log", type=str, default=None,
                        help="Path to Q-Learning training_log.json")
    parser.add_argument("--dq_log", type=str, default=None,
                        help="Path to Double Q-Learning training_log.json")
    parser.add_argument("--q_behavior", type=str, default=None,
                        help="Path to Q-Learning behavior CSV")
    parser.add_argument("--dq_behavior", type=str, default=None,
                        help="Path to Double Q-Learning behavior CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for plots")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-detect latest runs in results/ directory")
    args = parser.parse_args()

    # ---- Auto-detect mode ---- #
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    if args.auto or (args.q_log is None and args.dq_log is None):
        if not os.path.isdir(results_dir):
            print(f"Error: results/ not found at {results_dir}")
            print("Run train_tabular.py first.")
            sys.exit(1)

        runs = auto_detect_runs(results_dir)
        for algo, run_dir in runs.items():
            log_path = os.path.join(run_dir, "training_log.json")
            beh_path = find_latest_behavior_csv(run_dir)
            if algo == "q_learning":
                args.q_log = args.q_log or log_path
                args.q_behavior = args.q_behavior or beh_path
            elif algo == "double_q_learning":
                args.dq_log = args.dq_log or log_path
                args.dq_behavior = args.dq_behavior or beh_path

    # ---- Load training logs ---- #
    logs = {}
    if args.q_log and os.path.isfile(args.q_log):
        logs["q_learning"] = load_log(args.q_log)
        print(f"  Loaded Q-Learning log: {args.q_log}")
    if args.dq_log and os.path.isfile(args.dq_log):
        logs["double_q_learning"] = load_log(args.dq_log)
        print(f"  Loaded Double Q-Learning log: {args.dq_log}")

    if not logs:
        print("Error: No training logs found.")
        sys.exit(1)

    # ---- Load behavior CSVs ---- #
    behavior_data = {}
    if args.q_behavior and os.path.isfile(args.q_behavior):
        behavior_data["q_learning"] = load_behavior_csv(args.q_behavior)
        print(f"  Loaded Q-Learning behavior: {args.q_behavior} ({len(behavior_data['q_learning'])} steps)")
    if args.dq_behavior and os.path.isfile(args.dq_behavior):
        behavior_data["double_q_learning"] = load_behavior_csv(args.dq_behavior)
        print(f"  Loaded Double Q-Learning behavior: {args.dq_behavior} ({len(behavior_data['double_q_learning'])} steps)")

    # ---- Output directory ---- #
    if args.output is None:
        args.output = os.path.join(results_dir, "plots")
    os.makedirs(args.output, exist_ok=True)

    print(f"\n  Generating plots in: {args.output}\n")

    # ---- Generate training plots ---- #
    plot_reward_curves(logs, args.output)
    print(f"  ✓ 01_reward_curves.png")

    plot_duration_curves(logs, args.output)
    print(f"  ✓ 02_duration_curves.png")

    has_bias = all(len(log.get("bias_eval_episodes", [])) > 0 for log in logs.values())
    if has_bias:
        plot_q_vs_mc(logs, args.output)
        print(f"  ✓ 03_q_vs_mc_return.png")

        plot_bias_curves(logs, args.output)
        print(f"  ✓ 04_bias_curves.png")

    # ---- Generate behavior plots ---- #
    if behavior_data:
        action_values = None
        for log in logs.values():
            if "config" in log and "action_values" in log["config"]:
                action_values = log["config"]["action_values"]
                break

        plot_action_distribution(behavior_data, args.output, action_values)
        print(f"  ✓ 05_action_distribution.png")

        if len(behavior_data) >= 2:
            plot_trajectory_comparison(behavior_data, args.output)
            print(f"  ✓ 06_trajectory_pole_angle.png")
            print(f"  ✓ 07_trajectory_cart_position.png")
            print(f"  ✓ 08_trajectory_actions.png")
            print(f"  ✓ 09_trajectory_bias.png")

    # ---- Combined summary ---- #
    if has_bias and behavior_data:
        plot_combined_summary(logs, behavior_data, args.output)
        print(f"  ✓ 10_combined_summary.png")

    # ---- Summary table ---- #
    if len(logs) >= 2:
        print_summary_table(logs, behavior_data)

    print(f"  All plots saved to {args.output}\n")


if __name__ == "__main__":
    main()
