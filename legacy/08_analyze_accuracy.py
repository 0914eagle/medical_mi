import json
import os

BASE_DIR = "/workspace/medical_mi"
EVAL_RESULTS = f"{BASE_DIR}/data/processed/evaluated_cases.json"
STEERING_RESULTS = f"{BASE_DIR}/results/steering/magnitude_sweep_results.json"

def analyze():
    print("--- 1. Baseline Evaluation (Step 2) ---")
    if os.path.exists(EVAL_RESULTS):
        with open(EVAL_RESULTS) as f:
            data = json.load(f)
        total = len(data)
        correct = sum(1 for c in data if c["is_correct"])
        print(f"Total Cases: {total}")
        print(f"Original Accuracy: {correct/total:.2%} ({correct}/{total})")
    
    print("\n--- 2. Steering Experiment Accuracy (Step 5) ---")
    if os.path.exists(STEERING_RESULTS):
        with open(STEERING_RESULTS) as f:
            steering_data = json.load(f)
        
        for mag, results in steering_data.items():
            cases = results["individual_results"]
            total_test = len(cases)
            # 05번 실험 대상은 원래 'Wrong'인 케이스들임
            orig_correct = sum(1 for c in cases if c["original"]["answer"] == c["correct_answer"])
            steer_correct = sum(1 for c in cases if c["steered"]["answer"] == c["correct_answer"])
            
            # Wrong -> Correct 로 바뀐 케이스
            recovered = sum(1 for c in cases if c["original"]["answer"] != c["correct_answer"] 
                            and c["steered"]["answer"] == c["correct_answer"])
            
            # Wrong -> Still Wrong but Uncertain
            became_uncertain = sum(1 for c in cases if c["became_uncertain"])

            print(f"Magnitude {mag}:")
            print(f"  Recovered (Wrong -> Correct): {recovered}/{total_test}")
            print(f"  Became Uncertain: {became_uncertain}/{total_test}")
            print(f"  Final Accuracy (in this subset): {steer_correct/total_test:.2%}")
            
            # 실제 답변 예시 하나 출력
            if recovered > 0:
                for c in cases:
                    if c["original"]["answer"] != c["correct_answer"] and c["steered"]["answer"] == c["correct_answer"]:
                        print(f"  [Example] Correct Answer: {c['correct_answer']}")
                        print(f"    - Original: {c['original']['answer']} (conf: {c['original']['conf']:.2f})")
                        print(f"    - Steered: {c['steered']['answer']} (conf: {c['steered']['conf']:.2f})")
                        break

if __name__ == "__main__":
    analyze()
