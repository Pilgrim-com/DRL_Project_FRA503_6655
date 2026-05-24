"""
Plotting script for DQN / Double DQN bias + behavior analysis.

Reads aggregated_results.json and behavior CSVs, then generates
comparison plots with mean ± std across seeds.

Usage:
    python scripts/plot_dqn_results.py --auto
    python scripts/plot_dqn_results.py --dqn_dir results/dqn_... --ddqn_dir results/double_dqn_...
"""

import argparse, json, os, sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (10, 6), "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 12, "legend.fontsize": 10, "lines.linewidth": 1.5,
    "figure.dpi": 150, "savefig.dpi": 150,
})

COLORS = {"dqn": "#E74C3C", "double_dqn": "#2980B9"}
LABELS = {"dqn": "DQN", "double_dqn": "Double DQN"}


def smooth(data, w=50):
    if len(data) < w: return np.array(data)
    return np.convolve(data, np.ones(w)/w, mode="valid")


def load_json(path):
    with open(path) as f: return json.load(f)


def load_agg(run_dir):
    """Load aggregated results or single-seed training_log."""
    agg_path = os.path.join(run_dir, "aggregated_results.json")
    if os.path.isfile(agg_path):
        d = load_json(agg_path)
        if "episode_rewards_mean" not in d:
            # Single seed fallback
            for k in ["episode_rewards", "episode_durations", "episode_losses"]:
                d[k + "_mean"] = d.get(k, [])
                d[k + "_std"] = [0] * len(d.get(k, []))
            for k in ["bias_avg_mc_return", "bias_avg_q_taken", "bias_avg_max_q",
                       "bias_avg_taken_bias", "bias_avg_maxq_bias",
                       "bias_avg_reward", "bias_avg_duration"]:
                d[k + "_mean"] = d.get(k, [])
                d[k + "_std"] = [0] * len(d.get(k, []))
        return d
    # Try single seed
    for dd in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, dd, "training_log.json")
        if os.path.isfile(p):
            d = load_json(p)
            for k in ["episode_rewards", "episode_durations", "episode_losses"]:
                d[k + "_mean"] = d.get(k, [])
                d[k + "_std"] = [0] * len(d.get(k, []))
            for k in ["bias_avg_mc_return", "bias_avg_q_taken", "bias_avg_max_q",
                       "bias_avg_taken_bias", "bias_avg_maxq_bias",
                       "bias_avg_reward", "bias_avg_duration"]:
                d[k + "_mean"] = d.get(k, [])
                d[k + "_std"] = [0] * len(d.get(k, []))
            return d
    return None


def load_behavior_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in row:
                if k in ("episode", "step", "action_idx"):
                    row[k] = int(row[k])
                else:
                    try: row[k] = float(row[k])
                    except: pass
            rows.append(row)
    return rows


def find_latest_csv(run_dir):
    candidates = []
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.startswith("behavior_") and f.endswith(".csv"):
                candidates.append(os.path.join(root, f))
    return sorted(candidates)[-1] if candidates else None


def auto_detect(results_dir):
    found = {}
    for algo in ["dqn", "double_dqn"]:
        runs = sorted([d for d in os.listdir(results_dir)
                       if d.startswith(algo + "_") and os.path.isdir(os.path.join(results_dir, d))])
        if runs:
            d = load_agg(os.path.join(results_dir, runs[-1]))
            if d:
                d["_run_dir"] = os.path.join(results_dir, runs[-1])
                found[algo] = d
    return found


# ====== Plot functions ====== #

def plot_reward(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        m = smooth(np.array(d["episode_rewards_mean"]))
        x = np.arange(len(m)) + 25
        ax.plot(x, m, color=COLORS[algo], label=LABELS[algo])
        if any(v > 0 for v in d.get("episode_rewards_std", [])):
            s = smooth(np.array(d["episode_rewards_std"]))[:len(m)]
            ax.fill_between(x[:len(s)], m[:len(s)]-s, m[:len(s)]+s, color=COLORS[algo], alpha=0.15)
    ax.set(xlabel="Episode", ylabel="Reward", title="Episode Reward (mean ± std across seeds)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "01_dqn_reward.png"), bbox_inches="tight"); plt.close()

def plot_duration(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        m = smooth(np.array(d["episode_durations_mean"]))
        x = np.arange(len(m)) + 25
        ax.plot(x, m, color=COLORS[algo], label=LABELS[algo])
    ax.set(xlabel="Episode", ylabel="Duration", title="Episode Duration (mean across seeds)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "02_dqn_duration.png"), bbox_inches="tight"); plt.close()

def plot_q_taken_vs_mc(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        eps = d.get("bias_eval_episodes", []); c = COLORS[algo]
        if not eps: continue
        ax.plot(eps, d["bias_avg_q_taken_mean"], color=c, marker="o", linestyle="-", ms=4, label=LABELS[algo] + " Q_taken")
        ax.plot(eps, d["bias_avg_mc_return_mean"], color=c, marker="s", linestyle="--", ms=4, label=LABELS[algo] + " MC")
    ax.set(xlabel="Episode", ylabel="Value", title="Q_taken vs MC Return")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "03_dqn_q_taken_vs_mc.png"), bbox_inches="tight"); plt.close()

def plot_max_q_vs_mc(logs, out):
    if not logs: return
    n_algos = len(logs)
    fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos + 2, 5), sharey=True)
    if n_algos == 1: axes = [axes]
    for ax, (algo, d) in zip(axes, logs.items()):
        eps = d.get("bias_eval_episodes", []); c = COLORS[algo]
        if not eps: continue
        ax.plot(eps, d["bias_avg_max_q_mean"], color=c, marker="o", linestyle="-", ms=4, label="max_a Q")
        ax.plot(eps, d["bias_avg_mc_return_mean"], color="gray", marker="s", linestyle="--", ms=4, label="MC Return")
        ax.set_xlabel("Episode")
        if ax == axes[0]: ax.set_ylabel("Value")
        ax.set_title(f"{LABELS[algo]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("max_a Q(s,a) vs MC Return", fontsize=16, y=1.05)
    fig.savefig(os.path.join(out, "04_dqn_max_q_vs_mc.png"), bbox_inches="tight"); plt.close()

def plot_taken_bias(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        eps = d.get("bias_eval_episodes", [])
        if not eps: continue
        m = np.array(d["bias_avg_taken_bias_mean"])
        s = np.array(d.get("bias_avg_taken_bias_std", [0]*len(eps)))
        ax.plot(eps, m, color=COLORS[algo], marker="o", ms=4, label=LABELS[algo])
        ax.fill_between(eps, m-s, m+s, color=COLORS[algo], alpha=0.15)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set(xlabel="Episode", ylabel="TakenAction Bias", title="TakenAction_Bias = Q(s,a_taken) - MC_Return")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "05_dqn_taken_bias.png"), bbox_inches="tight"); plt.close()

def plot_maxq_bias(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        eps = d.get("bias_eval_episodes", [])
        if not eps: continue
        m = np.array(d["bias_avg_maxq_bias_mean"])
        s = np.array(d.get("bias_avg_maxq_bias_std", [0]*len(eps)))
        ax.plot(eps, m, color=COLORS[algo], marker="o", ms=4, label=LABELS[algo])
        ax.fill_between(eps, m-s, m+s, color=COLORS[algo], alpha=0.15)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set(xlabel="Episode", ylabel="MaxQ Bias", title="MaxQ_Bias = max_a Q(s,a) - MC_Return")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "06_dqn_maxq_bias.png"), bbox_inches="tight"); plt.close()

def plot_action_dist(behavior, out, action_values=None):
    if action_values is None: action_values = [-1.0,-0.5,0.0,0.5,1.0]
    n = len(action_values)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n); w = 0.35; na = len(behavior)
    for i, (algo, rows) in enumerate(behavior.items()):
        counts = [0]*n
        for r in rows:
            idx = int(r["action_idx"])
            if 0 <= idx < n: counts[idx] += 1
        total = max(sum(counts), 1)
        pcts = [c/total*100 for c in counts]
        off = (i - (na-1)/2) * w
        bars = ax.bar(x+off, pcts, w, label=LABELS[algo], color=COLORS[algo], alpha=0.85)
        for bar, pct in zip(bars, pcts):
            if pct > 2:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                        "%.0f%%" % pct, ha="center", va="bottom", fontsize=9)
    ax.set(xlabel="Action", ylabel="Frequency (%)", title="Action Distribution (Greedy Evaluation)")
    ax.set_xticks(x)
    ax.set_xticklabels(["a%d\n(%+.1f)" % (i, v) for i, v in enumerate(action_values)])
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(out, "07_dqn_action_dist.png"), bbox_inches="tight"); plt.close()

def plot_trajectories(behavior, out):
    ep_data = {}
    for algo, rows in behavior.items():
        episodes = {}
        for r in rows:
            eid = int(r["episode"])
            episodes.setdefault(eid, []).append(r)
        if episodes:
            ep_data[algo] = max(episodes.values(), key=len)
    if not ep_data: return

    for fname, ylabel, key, title, conv in [
        ("08_dqn_pole_angle.png", "Pole Angle (deg)", "pole_angle", "Pole Angle Trajectory", lambda v: np.degrees(v)),
        ("09_dqn_cart_position.png", "Cart Position", "cart_position", "Cart Position Trajectory", lambda v: v),
        ("10_dqn_maxq_bias_step.png", "MaxQ Bias", "maxq_bias", "Per-Step MaxQ Bias", lambda v: v),
    ]:
        fig, ax = plt.subplots()
        for algo, steps in ep_data.items():
            t = [s["step"] for s in steps]
            vals = [conv(s[key]) for s in steps]
            ax.plot(t, vals, color=COLORS[algo], label=LABELS[algo], alpha=0.9)
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set(xlabel="Step", ylabel=ylabel, title=title)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out, fname), bbox_inches="tight"); plt.close()


def print_summary(logs):
    print("\n" + "=" * 80)
    print("  %-35s %18s %18s" % ("Metric", "DQN", "Double DQN"))
    print("-" * 80)
    for name, key, fmt in [
        ("Final Avg Reward (last 100)", "episode_rewards_mean", ".2f"),
        ("Final Avg Duration (last 100)", "episode_durations_mean", ".1f"),
        ("Final Avg Q_taken", "bias_avg_q_taken_mean", ".4f"),
        ("Final Avg max_a Q", "bias_avg_max_q_mean", ".4f"),
        ("Final Avg MC Return", "bias_avg_mc_return_mean", ".4f"),
        ("Final TakenAction Bias", "bias_avg_taken_bias_mean", "+.4f"),
        ("Final MaxQ Bias", "bias_avg_maxq_bias_mean", "+.4f"),
    ]:
        vals = {}
        for algo, d in logs.items():
            data = d.get(key, [])
            if key.startswith("episode_"):
                vals[algo] = np.mean(data[-100:]) if len(data) >= 100 else (np.mean(data) if data else float("nan"))
            else:
                vals[algo] = data[-1] if data else float("nan")
        q = vals.get("dqn", float("nan"))
        dq = vals.get("double_dqn", float("nan"))
        q_str = f"{q:{fmt}}" if not np.isnan(q) else "nan"
        dq_str = f"{dq:{fmt}}" if not np.isnan(dq) else "nan"
        print(f"  {name:<35s} {q_str:>18s} {dq_str:>18s}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot DQN bias analysis.")
    parser.add_argument("--dqn_dir", type=str, default=None)
    parser.add_argument("--ddqn_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    logs = {}
    if args.auto or (args.dqn_dir is None and args.ddqn_dir is None):
        logs = auto_detect(results_dir)
    else:
        if args.dqn_dir:
            d = load_agg(args.dqn_dir)
            if d: d["_run_dir"] = args.dqn_dir; logs["dqn"] = d
        if args.ddqn_dir:
            d = load_agg(args.ddqn_dir)
            if d: d["_run_dir"] = args.ddqn_dir; logs["double_dqn"] = d

    if not logs:
        print("Error: No results found. Run train_dqn.py first."); sys.exit(1)

    for algo in logs:
        print("  Loaded %s: %s" % (LABELS[algo], logs[algo].get("_run_dir", "?")))

    out = args.output or os.path.join(results_dir, "dqn_plots")
    os.makedirs(out, exist_ok=True)
    print("\n  Generating plots in: %s\n" % out)

    plot_reward(logs, out); print("  + 01_dqn_reward.png")
    plot_duration(logs, out); print("  + 02_dqn_duration.png")

    has_bias = all(len(d.get("bias_eval_episodes", [])) > 0 for d in logs.values())
    if has_bias:
        plot_q_taken_vs_mc(logs, out); print("  + 03_dqn_q_taken_vs_mc.png")
        plot_max_q_vs_mc(logs, out); print("  + 04_dqn_max_q_vs_mc.png")
        plot_taken_bias(logs, out); print("  + 05_dqn_taken_bias.png")
        plot_maxq_bias(logs, out); print("  + 06_dqn_maxq_bias.png")

    behavior = {}
    for algo, d in logs.items():
        csv_path = find_latest_csv(d.get("_run_dir", ""))
        if csv_path:
            behavior[algo] = load_behavior_csv(csv_path)
            print("  Loaded %s behavior: %d steps" % (LABELS[algo], len(behavior[algo])))

    if behavior:
        av = None
        for d in logs.values():
            c = d.get("config", {})
            if "action_values" in c: av = c["action_values"]; break
        plot_action_dist(behavior, out, av); print("  + 07_dqn_action_dist.png")
        if len(behavior) >= 2:
            plot_trajectories(behavior, out); print("  + 08-10 trajectory plots")

    if len(logs) >= 2:
        print_summary(logs)

    print("\n  All plots saved to %s\n" % out)


if __name__ == "__main__":
    main()
