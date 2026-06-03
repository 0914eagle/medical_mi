import json
import matplotlib.pyplot as plt
import os

# --- Config ---
RESULTS_PATH = "results/steering/magnitude_sweep_results.json"
FIGURES_DIR = "results/figures"

def plot_steering_results(results_path=RESULTS_PATH):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)
    
    magnitudes = sorted([float(k) for k in results.keys()])
    uncertain_rates = [results[str(m)]["became_uncertain_rate"] for m in magnitudes]
    conf_changes = [results[str(m)]["avg_confidence_change"] for m in magnitudes]
    
    # Using entropy if available in future versions, currently just using the two core metrics
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Became Uncertain Rate
    axes[0].plot(magnitudes, uncertain_rates, "b-o", label="Uncertain Rate")
    axes[0].set_xlabel("Steering Magnitude")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Ignorance Cases -> Became Uncertain")
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
    axes[0].legend()
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot Confidence Change
    axes[1].plot(magnitudes, conf_changes, "r-o", label="Avg Conf Change")
    axes[1].set_xlabel("Steering Magnitude")
    axes[1].set_ylabel("Confidence Change")
    axes[1].set_title("Average Confidence Change\n(negative = less confident)")
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].legend()
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(f"{FIGURES_DIR}/steering_magnitude_sweep.png", dpi=150)
    print(f"Plot saved to {FIGURES_DIR}/steering_magnitude_sweep.png")
    
    print("\n--- Key Results Summary ---")
    for m in magnitudes:
        ur = results[str(m)]["became_uncertain_rate"]
        cc = results[str(m)]["avg_confidence_change"]
        print(f"  magnitude={m}: uncertain_rate={ur:.2%}, conf_change={cc:+.4f}")

if __name__ == "__main__":
    plot_steering_results()
