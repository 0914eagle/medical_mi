import json
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
SUMMARY_FILE = f"{BASE_DIR}/results/eval/phase0_summary.json"
FIGURES_DIR = f"{BASE_DIR}/results/figures"

def plot_validation_summary():
    if not os.path.exists(SUMMARY_FILE):
        print(f"Summary file not found at {SUMMARY_FILE}. Run Phase 0 validation first.")
        return

    with open(SUMMARY_FILE) as f:
        data = json.load(f)

    models = list(data.keys())
    accuracy = [data[m]["overall_accuracy"] for m in models]
    ignorance_rate = [data[m]["ignorance_rate_in_no"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Overall Accuracy Plot
    ax1.bar(models, accuracy, color='skyblue', alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('PubMedQA Overall Accuracy (with Context)')
    ax1.axhline(0.65, color='red', linestyle='--', label='Threshold (65%)')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Ignorance Rate in 'no' cases Plot
    ax2.bar(models, ignorance_rate, color='salmon', alpha=0.8)
    ax2.set_ylabel('Ignorance Rate')
    ax2.set_title('Ignorance Rate (Confident "Yes" on "No" cases)')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, acc in enumerate(accuracy):
        ax1.text(i, acc + 0.01, f"{acc:.1%}", ha='center', fontweight='bold')
    for i, rate in enumerate(ignorance_rate):
        ax2.text(i, rate + 0.01, f"{rate:.1%}", ha='center', fontweight='bold')

    plt.suptitle("Phase 0: Model Medical Knowledge & Ignorance Baseline Validation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(f"{FIGURES_DIR}/phase0_validation_summary.png", dpi=150)
    print(f"Visualization saved: {FIGURES_DIR}/phase0_validation_summary.png")

if __name__ == "__main__":
    plot_validation_summary()
