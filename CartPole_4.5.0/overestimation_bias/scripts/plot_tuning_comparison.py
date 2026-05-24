import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_data(run_dir):
    path = os.path.join(run_dir, "aggregated_results.json")
    if not os.path.exists(path):
        for d in sorted(os.listdir(run_dir)):
            p = os.path.join(run_dir, d, "training_log.json")
            if os.path.exists(p): return json.load(open(p))
        return None
    return json.load(open(path))

def plot_comparison():
    dir_before = "results/dqn_before_hyperparameter_tuning"
    dir_after = "results/dqn_after_hyperparameter_tuning"
    
    d_before = load_data(dir_before)
    d_after = load_data(dir_after)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    
    for ax, d, title, c in zip(axes, [d_before, d_after], ["Before Tuning (Deadly Triad)", "After Tuning (Stable)"], ["#E74C3C", "#27AE60"]):
        if not d: continue
        eps = np.array(d.get("bias_eval_episodes", []))
        mask = eps <= 1000
        e = eps[mask]
        
        # Max Q
        q = np.array(d["bias_avg_max_q_mean"])[mask]
        ax.plot(e, q, color=c, marker="o", ms=4, label="max_a Q (Model's Expectation)")
        
        # MC Return
        mc = np.array(d["bias_avg_mc_return_mean"])[mask]
        ax.plot(e, mc, color="gray", marker="s", ls="--", ms=4, label="MC Return (Reality)")
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
    plt.suptitle("First 1,000 Episodes: Unstable vs Stable DQN", fontsize=16)
    plt.tight_layout()
    os.makedirs("results/tuning_comparison", exist_ok=True)
    out_path = "results/tuning_comparison/first_1000_episodes_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    plot_comparison()
