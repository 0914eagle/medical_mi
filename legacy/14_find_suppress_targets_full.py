import torch
import json
import os

BASE_DIR = "/workspace/medical_mi"
layer = 25
correct_path = f"{BASE_DIR}/results/features/correct_confident_layer{layer}.pt"
wrong_path = f"{BASE_DIR}/results/features/wrong_confident_layer{layer}.pt"

def find_targets():
    print(f"--- Layer {layer} 전수 조사: Suppress 타겟 추출 ---")
    if not os.path.exists(correct_path) or not os.path.exists(wrong_path):
        print("Feature 파일이 없습니다. 스크립트 03을 먼저 실행하세요.")
        return

    c_feat = torch.load(correct_path)
    w_feat = torch.load(wrong_path)

    # 64,536개 전체 차원에 대해 평균 계산
    c_mean = c_feat.mean(0)
    w_mean = w_feat.mean(0)

    # Wrong Dominant (Wrong > Correct) 추출
    diff = w_mean - c_mean
    # 차이가 양수인 것들 중 상위 20개
    wrong_dominant_diff = torch.clamp(diff, min=0)
    top_val, top_idx = torch.topk(wrong_dominant_diff, k=20)

    results = []
    for i in range(len(top_idx)):
        idx = top_idx[i].item()
        results.append({
            "idx": idx,
            "diff": top_val[i].item(),
            "wrong_mean": w_mean[idx].item(),
            "correct_mean": c_mean[idx].item()
        })
        if i < 10:
            print(f"Feature #{idx:6d}: diff={top_val[i].item():.4f} (W:{w_mean[idx].item():.3f} > C:{c_mean[idx].item():.3f})")

    save_data = {
        "layer": layer,
        "targets": results,
        "indices": top_idx.tolist()
    }
    with open(f"{BASE_DIR}/results/features/suppress_targets_full.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n저장 완료: results/features/suppress_targets_full.json")

if __name__ == "__main__":
    find_targets()
