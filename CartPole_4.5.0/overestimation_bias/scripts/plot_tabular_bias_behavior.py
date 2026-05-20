"""
Plotting script for multi-seed tabular RL bias + behavior analysis.

Reads aggregated_results.json (multi-seed) or training_log.json (single seed)
and behavior CSV files, then generates plots comparing Q-Learning vs
Double Q-Learning with both TakenAction_Bias and MaxQ_Bias metrics.

Usage:
    python scripts/plot_tabular_bias_behavior.py --auto
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

COLORS = {"q_learning": "#E74C3C", "double_q_learning": "#2980B9"}
LABELS = {"q_learning": "Q-Learning", "double_q_learning": "Double Q-Learning"}


def smooth(data, window=50):
    if len(data) < window: return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode="valid")


def load_json(path):
    with open(path) as f: return json.load(f)


def load_agg_or_single(run_dir):
    """Load aggregated results or fall back to single-seed training_log."""
    agg_path = os.path.join(run_dir, "aggregated_results.json")
    if os.path.isfile(agg_path):
        data = load_json(agg_path)
        # If single-seed (no _mean keys), wrap it
        if "episode_rewards_mean" not in data:
            data["episode_rewards_mean"] = data.get("episode_rewards", [])
            data["episode_rewards_std"] = [0]*len(data.get("episode_rewards", []))
            data["episode_durations_mean"] = data.get("episode_durations", [])
            data["episode_durations_std"] = [0]*len(data.get("episode_durations", []))
            for k in ["bias_avg_mc_return","bias_avg_q_taken","bias_avg_max_q",
                       "bias_avg_taken_bias","bias_avg_maxq_bias","bias_avg_reward","bias_avg_duration"]:
                data[f"{k}_mean"] = data.get(k, [])
                data[f"{k}_std"] = [0]*len(data.get(k, []))
        return data
    # Try single seed logs
    for d in sorted(os.listdir(run_dir)):
        p = os.path.join(run_dir, d, "training_log.json")
        if os.path.isfile(p):
            data = load_json(p)
            data["episode_rewards_mean"] = data["episode_rewards"]
            data["episode_rewards_std"] = [0]*len(data["episode_rewards"])
            data["episode_durations_mean"] = data["episode_durations"]
            data["episode_durations_std"] = [0]*len(data["episode_durations"])
            for k in ["bias_avg_mc_return","bias_avg_q_taken","bias_avg_max_q",
                       "bias_avg_taken_bias","bias_avg_maxq_bias","bias_avg_reward","bias_avg_duration"]:
                data[f"{k}_mean"] = data.get(k, [])
                data[f"{k}_std"] = [0]*len(data.get(k, []))
            return data
    return None


def load_behavior_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            for k in row:
                if k in ("episode","step","action_idx"): row[k]=int(row[k])
                else:
                    try: row[k]=float(row[k])
                    except: pass
            rows.append(row)
    return rows


def find_latest_behavior_csv(run_dir):
    """Find the last behavior CSV across all seed dirs."""
    candidates = []
    for root, dirs, files in os.walk(run_dir):
        for f in files:
            if f.startswith("behavior_") and f.endswith(".csv"):
                candidates.append(os.path.join(root, f))
    return sorted(candidates)[-1] if candidates else None


def plot_with_std(ax, x, mean, std, color, label, marker=None):
    mean, std = np.array(mean), np.array(std)
    kw = dict(color=color, label=label)
    if marker: kw["marker"], kw["markersize"] = marker, 4
    ax.plot(x, mean, **kw)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.15)


# ====== Plot functions ====== #

def plot_reward(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        m = np.array(d["episode_rewards_mean"])
        s = smooth(m); x = np.arange(len(s))+25
        ax.plot(x, s, color=COLORS[algo], label=LABELS[algo])
        if "episode_rewards_std" in d:
            std = np.array(d["episode_rewards_std"])
            ss = smooth(std)
            ax.fill_between(x, s-ss[:len(s)], s+ss[:len(s)], color=COLORS[algo], alpha=0.15)
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward"); ax.set_title("Episode Reward (mean ± std across seeds)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "01_reward_curves.png"), bbox_inches="tight"); plt.close(fig)

def plot_duration(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        m = np.array(d["episode_durations_mean"])
        s = smooth(m); x = np.arange(len(s))+25
        ax.plot(x, s, color=COLORS[algo], label=LABELS[algo])
    ax.set_xlabel("Episode"); ax.set_ylabel("Duration (steps)"); ax.set_title("Episode Duration (mean across seeds)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "02_duration_curves.png"), bbox_inches="tight"); plt.close(fig)

def plot_q_taken_vs_mc(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        eps = d.get("bias_eval_episodes", [])
        if not eps: continue
        c = COLORS[algo]
        ax.plot(eps, d["bias_avg_q_taken_mean"], color=c, linestyle="-", marker="o", markersize=4, label=f"{LABELS[algo]} Q_taken")
        ax.plot(eps, d["bias_avg_mc_return_mean"], color=c, linestyle="--", marker="s", markersize=4, label=f"{LABELS[algo]} MC return")
    ax.set_xlabel("Episode"); ax.set_ylabel("Value"); ax.set_title("Q_taken vs MC Return")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "03_q_taken_vs_mc.png"), bbox_inches="tight"); plt.close(fig)

def plot_max_q_vs_mc(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        eps = d.get("bias_eval_episodes", [])
        if not eps: continue
        c = COLORS[algo]
        ax.plot(eps, d["bias_avg_max_q_mean"], color=c, linestyle="-", marker="o", markersize=4, label=f"{LABELS[algo]} max_a Q")
        ax.plot(eps, d["bias_avg_mc_return_mean"], color=c, linestyle="--", marker="s", markersize=4, label=f"{LABELS[algo]} MC return")
    ax.set_xlabel("Episode"); ax.set_ylabel("Value"); ax.set_title("max_a Q(s,a) vs MC Return")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "04_max_q_vs_mc.png"), bbox_inches="tight"); plt.close(fig)

def plot_taken_bias(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        eps = d.get("bias_eval_episodes", [])
        if not eps: continue
        plot_with_std(ax, eps, d["bias_avg_taken_bias_mean"], d.get("bias_avg_taken_bias_std",[0]*len(eps)), COLORS[algo], LABELS[algo], "o")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, label="Zero bias")
    ax.set_xlabel("Episode"); ax.set_ylabel("TakenAction Bias"); ax.set_title("TakenAction_Bias = Q(s,a_taken) − MC_Return")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "05_taken_bias.png"), bbox_inches="tight"); plt.close(fig)

def plot_maxq_bias(logs, out):
    fig, ax = plt.subplots()
    for algo, d in logs.items():
        eps = d.get("bias_eval_episodes", [])
        if not eps: continue
        plot_with_std(ax, eps, d["bias_avg_maxq_bias_mean"], d.get("bias_avg_maxq_bias_std",[0]*len(eps)), COLORS[algo], LABELS[algo], "o")
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, label="Zero bias")
    ax.set_xlabel("Episode"); ax.set_ylabel("MaxQ Bias"); ax.set_title("MaxQ_Bias = max_a Q(s,a) − MC_Return")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out, "06_maxq_bias.png"), bbox_inches="tight"); plt.close(fig)

def plot_action_dist(behavior_data, out, action_values=None):
    if action_values is None: action_values = [-1.0,-0.5,0.0,0.5,1.0]
    n = len(action_values); fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(n); w = 0.35; na = len(behavior_data)
    for i, (algo, rows) in enumerate(behavior_data.items()):
        counts = [0]*n
        for r in rows:
            idx = int(r["action_idx"])
            if 0<=idx<n: counts[idx]+=1
        total = max(sum(counts),1); pcts = [c/total*100 for c in counts]
        off = (i-(na-1)/2)*w
        bars = ax.bar(x+off, pcts, w, label=LABELS[algo], color=COLORS[algo], alpha=0.85)
        for bar, pct in zip(bars, pcts):
            if pct>2: ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Action"); ax.set_ylabel("Frequency (%)"); ax.set_title("Action Distribution During Greedy Evaluation")
    ax.set_xticks(x); ax.set_xticklabels([f"a{i}\n({v:+.1f})" for i,v in enumerate(action_values)])
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(os.path.join(out, "07_action_distribution.png"), bbox_inches="tight"); plt.close(fig)

def plot_trajectories(behavior_data, out):
    ep_data = {}
    for algo, rows in behavior_data.items():
        episodes = {}
        for r in rows:
            eid = int(r["episode"])
            episodes.setdefault(eid, []).append(r)
        if episodes: ep_data[algo] = max(episodes.values(), key=len)
    if not ep_data: return

    for fname, ylabel, key, title, conv in [
        ("08_trajectory_pole_angle.png","Pole Angle (°)","pole_angle","Pole Angle Trajectory",lambda v:np.degrees(v)),
        ("09_trajectory_cart_position.png","Cart Position","cart_position","Cart Position Trajectory",lambda v:v),
        ("10_trajectory_maxq_bias.png","MaxQ Bias","maxq_bias","Per-Step MaxQ Bias",lambda v:v),
    ]:
        fig, ax = plt.subplots()
        for algo, steps in ep_data.items():
            t = [s["step"] for s in steps]
            vals = [conv(s[key]) for s in steps]
            ax.plot(t, vals, color=COLORS[algo], label=LABELS[algo], alpha=0.9)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Step"); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out, fname), bbox_inches="tight"); plt.close(fig)

def print_summary(logs):
    print(f"\n{'='*80}")
    print(f"  {'Metric':<35s} {'Q-Learning':>18s} {'Double Q-Learning':>18s}")
    print(f"{'─'*80}")
    for name, key, fmt in [
        ("Final Avg Reward (last 100)","episode_rewards_mean",".2f"),
        ("Final Avg Duration (last 100)","episode_durations_mean",".1f"),
        ("Final Avg Q_taken","bias_avg_q_taken_mean",".4f"),
        ("Final Avg max_a Q","bias_avg_max_q_mean",".4f"),
        ("Final Avg MC Return","bias_avg_mc_return_mean",".4f"),
        ("Final TakenAction Bias","bias_avg_taken_bias_mean","+.4f"),
        ("Final MaxQ Bias","bias_avg_maxq_bias_mean","+.4f"),
    ]:
        vals = {}
        for algo, d in logs.items():
            data = d.get(key, [])
            if key.startswith("episode_"):
                vals[algo] = np.mean(data[-100:]) if len(data)>=100 else (np.mean(data) if data else float("nan"))
            else:
                vals[algo] = data[-1] if data else float("nan")
        q = vals.get("q_learning", float("nan")); dq = vals.get("double_q_learning", float("nan"))
        print(f"  {name:<35s} {q:>18{fmt}} {dq:>18{fmt}}")
    print(f"{'='*80}\n")


# ====== Auto-detect ====== #

def auto_detect_runs(results_dir):
    found = {}
    for algo in ["q_learning", "double_q_learning"]:
        runs = sorted([d for d in os.listdir(results_dir) if d.startswith(algo) and os.path.isdir(os.path.join(results_dir, d))])
        if runs:
            data = load_agg_or_single(os.path.join(results_dir, runs[-1]))
            if data:
                found[algo] = data
                found[algo]["_run_dir"] = os.path.join(results_dir, runs[-1])
    return found


def main():
    parser = argparse.ArgumentParser(description="Plot tabular RL bias + behavior analysis.")
    parser.add_argument("--q_dir", type=str, default=None, help="Q-Learning run directory")
    parser.add_argument("--dq_dir", type=str, default=None, help="Double Q-Learning run directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--auto", action="store_true", help="Auto-detect latest runs")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    logs = {}
    if args.auto or (args.q_dir is None and args.dq_dir is None):
        logs = auto_detect_runs(results_dir)
    else:
        if args.q_dir:
            d = load_agg_or_single(args.q_dir)
            if d: d["_run_dir"]=args.q_dir; logs["q_learning"]=d
        if args.dq_dir:
            d = load_agg_or_single(args.dq_dir)
            if d: d["_run_dir"]=args.dq_dir; logs["double_q_learning"]=d

    if not logs:
        print("Error: No results found. Run train_tabular.py first."); sys.exit(1)

    for algo in logs: print(f"  Loaded {LABELS[algo]}: {logs[algo].get('_run_dir','?')}")

    if args.output is None: args.output = os.path.join(results_dir, "plots")
    os.makedirs(args.output, exist_ok=True)
    print(f"\n  Generating plots in: {args.output}\n")

    plot_reward(logs, args.output); print("  ✓ 01_reward_curves.png")
    plot_duration(logs, args.output); print("  ✓ 02_duration_curves.png")

    has_bias = all(len(d.get("bias_eval_episodes",[])) > 0 for d in logs.values())
    if has_bias:
        plot_q_taken_vs_mc(logs, args.output); print("  ✓ 03_q_taken_vs_mc.png")
        plot_max_q_vs_mc(logs, args.output); print("  ✓ 04_max_q_vs_mc.png")
        plot_taken_bias(logs, args.output); print("  ✓ 05_taken_bias.png")
        plot_maxq_bias(logs, args.output); print("  ✓ 06_maxq_bias.png")

    # Load behavior data
    behavior_data = {}
    for algo, d in logs.items():
        csv_path = find_latest_behavior_csv(d.get("_run_dir",""))
        if csv_path:
            behavior_data[algo] = load_behavior_csv(csv_path)
            print(f"  Loaded {LABELS[algo]} behavior: {len(behavior_data[algo])} steps")

    if behavior_data:
        av = None
        for d in logs.values():
            cfg = d.get("config",{})
            if "action_values" in cfg: av=cfg["action_values"]; break
        plot_action_dist(behavior_data, args.output, av); print("  ✓ 07_action_distribution.png")
        if len(behavior_data) >= 2:
            plot_trajectories(behavior_data, args.output)
            print("  ✓ 08-10 trajectory plots")

    if len(logs) >= 2: print_summary(logs)
    print(f"\n  All plots saved to {args.output}\n")


if __name__ == "__main__":
    main()
