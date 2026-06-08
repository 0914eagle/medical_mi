#!/bin/bash

# --- Phase 0 Master Execution Script ---
BASE_DIR="/workspace/medical_mi"
export HF_HOME="/workspace/.cache/huggingface"

echo "=========================================================="
echo "   PHASE 0: MODEL MEDICAL KNOWLEDGE VALIDATION"
echo "=========================================================="

# 1. Check for HF_TOKEN (Required for Gemma-2-9B)
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️ Warning: HF_TOKEN is not set. Gemma-2-9B download may fail."
    echo "If you haven't logged in, run 'huggingface-cli login' first."
fi

# 2. Setup: Download Models, SAEs, and Dataset
echo -e "\n[Step 1/3] Downloading models, SAEs, and PubMedQA..."
python3 "$BASE_DIR/scripts/01_setup.py"

if [ $? -ne 0 ]; then
    echo "❌ Error during setup. Aborting."
    exit 1
fi

# 3. Validation: Benchmark on PubMedQA (With vs Without Context)
echo -e "\n[Step 2/3] Benchmarking models on PubMedQA..."
python3 "$BASE_DIR/scripts/15_phase0_validation.py"

if [ $? -ne 0 ]; then
    echo "❌ Error during validation. Aborting."
    exit 1
fi

# 4. Visualization: Generate Analysis Charts
echo -e "\n[Step 3/3] Generating visualization charts..."
python3 "$BASE_DIR/scripts/17_visualize_phase0.py"

echo -e "\n=========================================================="
echo "   PHASE 0 COMPLETED SUCCESSFULLY"
echo "   Results: $BASE_DIR/results/eval/phase0_summary.json"
echo "   Figures: $BASE_DIR/results/figures/phase0_context_gain_comparison.png"
echo "=========================================================="
