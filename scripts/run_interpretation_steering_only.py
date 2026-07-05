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

    print(f"\n>>> [{model}] Phase 3 & 4: Top Features 분석 시작 <<<")
    features_path = f"{BASE_DIR}/results/features/{model}_phase1_features.json"
    
    if not os.path.exists(features_path):
        print(f"Error: Features file not found at {features_path}")
        return

    with open(features_path, "r") as f:
        features_data = json.load(f)

    target_layer = "20"
    if target_layer in features_data:
        # Get top features
        top_features = features_data[target_layer]["correct_dominant"][:3] 
        print(f"Found {len(top_features)} top features for Layer {target_layer}: {top_features}")
        
        for feat_idx in top_features:
            # Phase 3: Interpretation
            print(f"\n--- [Layer {target_layer} | Feature {feat_idx}] Phase 3 시작 ---")
            run_cmd(["python3", f"{SCRIPTS_DIR}/05_phase3_interpretation.py", 
                     "--model", model, "--layer", target_layer, "--feature_idx", str(feat_idx)])
            
            # Phase 4: Steering (alpha=20.0)
            print(f"--- [Layer {target_layer} | Feature {feat_idx}] Phase 4 시작 ---")
            run_cmd(["python3", f"{SCRIPTS_DIR}/06_phase4_steering.py", 
                     "--model", model, "--layer", target_layer, "--feature_idx", str(feat_idx), "--alpha", "20.0"])
    else:
        print(f"Layer {target_layer} not found in features data.")

if __name__ == "__main__":
    main()
