import json
import os
import matplotlib.pyplot as plt
import numpy as np

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
SUMMARY_FILE = f"{BASE_DIR}/results/eval/phase0_summary.json"
FIGURES_DIR = f"{BASE_DIR}/results/figures"

def plot_context_gain():
    if not os.path.exists(SUMMARY_FILE):
        print(f"Summary file not found at {SUMMARY_FILE}. Run Phase 0 validation first.")
        return

    with open(SUMMARY_FILE) as f:
        data = json.load(f)

    models = list(data.keys())
    acc_with = [data[m]["accuracy_with_context"] for m in models]
    acc_without = [data[m]["accuracy_without_context"] for m in models]
    gain = [data[m]["context_gain"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Accuracy Comparison
    ax1.bar(x - width/2, acc_without, width, label='Without Context (Prior)', color='gray', alpha=0.6)
    ax1.bar(x + width/2, acc_with, width, label='With Context', color='skyblue')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('PubMedQA Accuracy: Impact of Context')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # 2. Context Gain
    colors = ['green' if g > 0 else 'red' for g in gain]
    ax2.bar(models, gain, color=colors, alpha=0.7)
    ax2.set_ylabel('Accuracy Gain')
    ax2.set_title('Context Gain (%)')
    ax2.axhline(0, color='black', linewidth=1)
    ax2.axhline(0.05, color='blue', linestyle='--', label='Threshold (+5%)')
    ax2.set_ylim(min(min(gain) - 0.05, -0.05), max(max(gain) + 0.1, 0.2))
    ax2.legend()
    
    # Add percentage labels on bars
    for i, g in enumerate(gain):
        ax2.text(i, g + 0.01, f"{g:+.1%}", ha='center', fontweight='bold')

    plt.suptitle("Phase 0: Model Medical Knowledge & Context Utilization Validation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(f"{FIGURES_DIR}/phase0_context_gain_comparison.png", dpi=150)
    print(f"Visualization saved: {FIGURES_DIR}/phase0_context_gain_comparison.png")

if __name__ == "__main__":
    plot_context_gain()
