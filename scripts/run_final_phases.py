import json
import subprocess
import os
import sys

# --- 설정 ---
MODEL_NAME = "qwen3.5-9b"
TARGET_LAYERS = ["18", "20", "22"] # 분석 대상 핵심 레이어
BASE_DIR = "/workspace/medical_mi"
FEATURES_FILE = f"{BASE_DIR}/results/features/{MODEL_NAME}_features.json"
FINAL_REPORT = f"{BASE_DIR}/results/final_analysis_summary.json"
SCRIPTS_DIR = f"{BASE_DIR}/scripts"

def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def main():
    if not os.path.exists(FEATURES_FILE):
        print(f"Features file not found: {FEATURES_FILE}")
        return

    with open(FEATURES_FILE, "r") as f:
        features_data = json.load(f)

    all_results = []

    for layer in TARGET_LAYERS:
        if layer not in features_data:
            continue
        
        candidates = features_data[layer]["intersection"]
        print(f"\n{'='*50}")
        print(f"Layer {layer}: {len(candidates)}개의 후보 Feature 분석 시작")
        print(f{'='*50}")

        # 모든 후보 또는 상위 N개 분석 (효율을 위해 각 레이어당 최대 10개로 제한 가능)
        for idx in candidates:
            print(f"\n>>> [Layer {layer} | Feature {idx}] 분석 중...")

            # Phase D: Interpretation
            print(f"  - Phase D 실행 중...")
            cmd_d = [
                "python3", f"{SCRIPTS_DIR}/05_phaseD_interpretation.py",
                "--model", MODEL_NAME, "--layer", layer, "--feature_idx", str(idx)
            ]
            out_d = run_command(cmd_d)

            # Phase E: Steering
            print(f"  - Phase E 실행 중...")
            cmd_e = [
                "python3", f"{SCRIPTS_DIR}/06_phaseE_steering.py",
                "--model", MODEL_NAME, "--layer", layer, "--feature_idx", str(idx),
                "--alpha", "30.0" # 강도를 약간 높여서 확실한 변위 관찰
            ]
            out_e = run_command(cmd_e)

            # 결과 수집
            all_results.append({
                "layer": layer,
                "feature_idx": idx,
                "interpretation": out_d,
                "steering": out_e
            })

    # 최종 요약 저장
    with open(FINAL_REPORT, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"모든 분석 완료! 결과가 {FINAL_REPORT}에 저장되었습니다.")
    print(f{'='*50}")

if __name__ == "__main__":
    main()
