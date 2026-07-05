import json
import subprocess
import os

# --- Config ---
BASE_DIR = "/home/eagle0914/medical_mi"
SCRIPTS_DIR = f"{BASE_DIR}/scripts"

def run_cmd(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    model = args.model

    # 1. Phase 1: Labeling
    run_cmd(["python3", f"{SCRIPTS_DIR}/02_phase1_labeling.py", "--model", model])

    # 2. Phase 1: Feature Discovery
    run_cmd(["python3", f"{SCRIPTS_DIR}/03_phase1_sae_features.py", "--model", model])

    # 3. Phase 2: MedQA Negative Control
    run_cmd(["python3", f"{SCRIPTS_DIR}/04_phase2_medqa_control.py", "--model", model])

    # 4. Phase 5: MedAbstain Cross-Validation
    run_cmd(["python3", f"{SCRIPTS_DIR}/07_phase5_medabstain.py", "--model", model])

    # 5. Phase 3 & 4 (Automated for top features)
    features_path = f"{BASE_DIR}/results/features/{model}_phase1_features.json"
    with open(features_path, "r") as f:
        features_data = json.load(f)

    # Pick top features from Layer 20 (as it was proven critical)
    target_layer = "20"
    if target_layer in features_data:
        top_features = features_data[target_layer]["correct_dominant"][:3] # Top 3 features
        for feat_idx in top_features:
            # Phase 3: Interpretation
            run_cmd(["python3", f"{SCRIPTS_DIR}/05_phase3_interpretation.py", 
                     "--model", model, "--layer", target_layer, "--feature_idx", str(feat_idx)])
            
            # Phase 4: Steering
            # Note: alpha might need tuning, starting with 20.0
            run_cmd(["python3", f"{SCRIPTS_DIR}/06_phase4_steering.py", 
                     "--model", model, "--layer", target_layer, "--feature_idx", str(feat_idx), "--alpha", "20.0"])

if __name__ == "__main__":
    main()
