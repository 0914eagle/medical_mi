import json
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
STEERING_RESULTS = f"{BASE_DIR}/results/steering/magnitude_sweep_results.json"
FEATURE_CANDIDATES = f"{BASE_DIR}/results/features/ignorance_feature_candidates.json"
FIGURES_DIR = f"{BASE_DIR}/results/figures"

def plot_steering_summary():
    if not os.path.exists(STEERING_RESULTS):
        print(f"Steering results not found at {STEERING_RESULTS}. Run script 05 first.")
        return

    with open(STEERING_RESULTS) as f:
        data = json.load(f)

    magnitudes = sorted([float(m) for m in data.keys()])
    uncertain_rates = [data[str(m)]["became_uncertain_rate"] for m in magnitudes]
    conf_changes = [data[str(m)]["avg_confidence_change"] for m in magnitudes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Uncertain Rate Plot
    ax1.plot(magnitudes, uncertain_rates, marker='o', color='b', linestyle='-')
    ax1.set_xlabel("Steering Magnitude")
    ax1.set_ylabel("Became Uncertain Rate")
    ax1.set_title("Ignorance Suppression Success Rate")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # 2. Confidence Change Plot
    ax2.plot(magnitudes, conf_changes, marker='s', color='r', linestyle='--')
    ax2.set_xlabel("Steering Magnitude")
    ax2.set_ylabel("Avg Confidence Change")
    ax2.set_title("Reduction in Overconfidence")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/steering_magnitude_sweep.png", dpi=150)
    print(f"Saved: {FIGURES_DIR}/steering_magnitude_sweep.png")

def plot_layer_comparison():
    if not os.path.exists(FEATURE_CANDIDATES):
        print(f"Feature candidates not found. Run script 04 first.")
        return

    with open(FEATURE_CANDIDATES) as f:
        data = json.load(f)

    layers = sorted([int(l) for l in data.keys()])
    max_diffs = [max(data[str(l)]["mean_diff_scores"]) for l in layers]
    min_pvals = [min(data[str(l)]["p_values"]) for l in layers]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Max Mean Difference', color=color)
    ax1.bar(layers, max_diffs, color=color, alpha=0.6, width=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Min P-Value (Log Scale)', color=color)
    ax2.plot(layers, min_pvals, color=color, marker='D', markersize=8)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(0.05, color='gray', linestyle='--', label='p=0.05')

    plt.title("Ignorance Feature Significance across Layers")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/layer_significance_comparison.png", dpi=150)
    print(f"Saved: {FIGURES_DIR}/layer_significance_comparison.png")

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating figures...")
    plot_layer_comparison()
    plot_steering_summary()
