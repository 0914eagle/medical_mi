import torch
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_wrapper import SAEWrapper
from utils import format_pubmedqa, get_sae_path
import gc

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
}
RESULTS_DIR = f"{BASE_DIR}/results/steering_final"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 집중 분석 대상
TARGET_LAYERS = [18, 20, 22]
ALPHAS = [5.0, 7.0, 10.0, 15.0, 20.0] # 5.0은 baseline 비교용

def steer_suppress(model, tokenizer, item, layer, sae, suppress_idxs, alpha):
    """
    오직 wrong_dominant feature들을 억제(suppress)하는 로직
    """
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 뺄(suppress) 벡터들의 합 계산
    steer_vec = torch.zeros(sae.W_dec.shape[1], device=model.device)
    for idx in suppress_idxs:
        steer_vec -= sae.W_dec[idx, :].to(model.device)

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h[0, -1, :] = h[0, -1, :] + alpha * steer_vec.to(device=h.device, dtype=h.dtype)
        return o

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
    handle.remove()

    result = {}
    for word in ["yes", "no", "maybe"]:
        ids = []
        for v in [word, word.capitalize(), " "+word, " "+word.capitalize()]:
            t = tokenizer.encode(v, add_special_tokens=False)
            if t: ids.append(t[0])
        if ids:
            result[word] = max(probs[idx].item() for idx in ids if idx < probs.shape[0])
        else:
            result[word] = 0.0
            
    tot = sum(result.values())
    if tot > 0:
        return {k: v/tot for k, v in result.items()}
    return {"yes": 0.33, "no": 0.33, "maybe": 0.33}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3.5-9b")
    args = parser.parse_args()

    model_name = args.model
    print(f"=== Final Steering Experiment (Suppress Wrong) for {model_name} ===")

    # 1. 가중치 및 데이터 로드
    path = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    # 2. PubMedQA 전체 데이터 및 라벨 로드
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    with open(f"{BASE_DIR}/results/eval/{model_name}_labels.json", "r") as f:
        labels = json.load(f)
    
    item_map = {it["pubid"]: it for it in pubmedqa}
    
    # 전체 Wrong / Correct 케이스 추출 (30개 제한 해제)
    wrong_cases = [item_map[r["item_id"]] for r in labels if not r["is_correct"]]
    correct_cases = [item_map[r["item_id"]] for r in labels if r["is_correct"]]
    print(f"Total wrong cases to test: {len(wrong_cases)}")
    print(f"Total correct cases to test: {len(correct_cases)}")

    # 3. Phase 1 Feature 로드
    feat_path = f"{BASE_DIR}/results/features/{model_name}_phase1_features.json"
    with open(feat_path, "r") as f:
        feat_data = json.load(f)

    final_results = []

    for layer in TARGET_LAYERS:
        print(f"\n--- Layer {layer} 분석 시작 ---")
        if str(layer) not in feat_data:
            print(f"No feature data for Layer {layer}")
            continue
            
        wrong_dom = feat_data[str(layer)]["wrong_dominant"]
        if not wrong_dom:
            print(f"No wrong_dominant features found in Layer {layer}")
            continue
            
        print(f"Found {len(wrong_dom)} wrong_dominant features to suppress.")

        sae_path = get_sae_path(model_name, layer)
        if not sae_path:
            print(f"SAE weights not found for Layer {layer}")
            continue
            
        sae_dict = torch.load(sae_path)
        sae = SAEWrapper(sae_dict, suite="qwen" if "qwen" in model_name else "gemma").to(model.device)

        for alpha in ALPHAS:
            print(f"\n[Layer {layer} | Alpha {alpha}] 테스트 중...")
            
            # Recovery Test (전체 Wrong 케이스)
            recovered = 0
            for item in tqdm(wrong_cases, desc=f"Recovery (Alpha {alpha})", leave=False):
                probs = steer_suppress(model, tokenizer, item, layer, sae, wrong_dom, alpha)
                pred = max(probs, key=probs.get)
                if pred == item["final_decision"]:
                    recovered += 1
            recovery_rate = recovered / len(wrong_cases)

            # Corruption Test (전체 Correct 케이스)
            corrupted = 0
            for item in tqdm(correct_cases, desc=f"Corruption (Alpha {alpha})", leave=False):
                probs = steer_suppress(model, tokenizer, item, layer, sae, wrong_dom, alpha)
                pred = max(probs, key=probs.get)
                if pred != item["final_decision"]:
                    corrupted += 1
            corruption_rate = corrupted / len(correct_cases)

            print(f"결과 -> Recovery: {recovery_rate:5.1%} ({recovered}/{len(wrong_cases)}) | Corruption: {corruption_rate:5.1%} ({corrupted}/{len(correct_cases)})")
            
            final_results.append({
                "layer": layer,
                "alpha": alpha,
                "recovery_rate": recovery_rate,
                "corruption_rate": corruption_rate,
                "recovered_count": recovered,
                "corrupted_count": corrupted
            })

        # 메모리 정리
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    # 최종 결과 저장
    output_path = f"{RESULTS_DIR}/{model_name}_final_steering.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n최종 결과 저장 완료: {output_path}")

if __name__ == "__main__":
    main()
