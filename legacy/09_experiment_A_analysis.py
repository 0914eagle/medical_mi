import json
import os

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
STEERING_RESULTS = f"{BASE_DIR}/results/steering/magnitude_sweep_results.json"

def analyze_transitions():
    if not os.path.exists(STEERING_RESULTS):
        print("Steering results not found.")
        return

    with open(STEERING_RESULTS) as f:
        data = json.load(f)

    # Magnitude 0.5 결과를 기준으로 분석 (가장 변화가 컸던 지점)
    if "0.5" not in data:
        print("Magnitude 0.5 results not found.")
        return
        
    cases = data["0.5"]["individual_results"]

    print("=" * 80)
    print("실험 A: MedQA Steering 케이스 분류 분석 (Magnitude 0.5)")
    print("=" * 80)

    changed_to_correct = []
    changed_to_wrong = []
    unchanged_wrong = []
    more_confident = []

    for i, r in enumerate(cases):
        orig = r["original"]
        steer = r["steered"]
        correct = r["correct_answer"]
        
        answer_changed = orig["answer"] != steer["answer"]
        conf_increased = steer["conf"] > orig["conf"]
        
        if answer_changed and steer["answer"] == correct:
            changed_to_correct.append((i, r))
        elif answer_changed and steer["answer"] != correct:
            changed_to_wrong.append((i, r))
        elif not answer_changed and conf_increased:
            more_confident.append((i, r))
        else:
            unchanged_wrong.append((i, r))

    print(f"\n[1. 정답으로 바뀐 케이스: {len(changed_to_correct)}개]")
    for i, r in changed_to_correct:
        orig = r["original"]
        steer = r["steered"]
        print(f"\nCase {i}:")
        # print(f"  질문: {r['question']}") # 너무 길면 생략
        print(f"  정답: {r['correct_answer']}")
        print(f"  변화: {orig['answer']}({orig['conf']:.3f}) → {steer['answer']}({steer['conf']:.3f})")

    print(f"\n[2. 엉뚱한 답으로 바뀐 케이스: {len(changed_to_wrong)}개]")
    for i, r in changed_to_wrong:
        orig = r["original"]
        steer = r["steered"]
        print(f"\nCase {i}:")
        print(f"  정답: {r['correct_answer']}")
        print(f"  변화: {orig['answer']}({orig['conf']:.3f}) → {steer['answer']}({steer['conf']:.3f})")

    print(f"\n[3. 더 확신하게 된 케이스 (반대 방향): {len(more_confident)}개]")
    for i, r in more_confident:
        orig = r["original"]
        steer = r["steered"]
        print(f"Case {i}: {orig['answer']}({orig['conf']:.3f}) → {steer['answer']}({steer['conf']:.3f})")

    print(f"\n[4. 거의 변화 없는 케이스: {len(unchanged_wrong)}개]")
    print(f"총 {len(unchanged_wrong)}개")

if __name__ == "__main__":
    analyze_transitions()
