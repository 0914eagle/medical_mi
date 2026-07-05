import torch
import json
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_wrapper import SAEWrapper
from utils import format_pubmedqa, get_activation_with_hook, get_sae_path
import gc

# --- Config ---
BASE_DIR = "/home/eagle0914/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
}
RESULTS_DIR = f"{BASE_DIR}/results/steering_retry"
os.makedirs(RESULTS_DIR, exist_ok=True)

def steer_multi(model, tokenizer, item, layer, sae, amplify_idxs=None, suppress_idxs=None, alpha=5.0):
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # steering vector 합성 (decoder 방향들의 합)
    # W_dec shape: [d_sae, d_model] -> steer_vec shape should be [d_model]
    steer_vec = torch.zeros(sae.W_dec.shape[1], device=model.device) 
    if amplify_idxs:
        for idx in amplify_idxs:
            steer_vec += sae.W_dec[idx, :].to(model.device)
    if suppress_idxs:
        for idx in suppress_idxs:
            steer_vec -= sae.W_dec[idx, :].to(model.device)

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        # steer_vec을 현재 액티베이션(h)이 있는 GPU와 데이터 타입으로 강제 일치
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
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args()

    model_name = args.model
    layer = args.layer

    print(f"--- Steering Retry for {model_name} Layer {layer} ---")

    # 1. 가중치 및 데이터 로드
    path = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    sae_path = get_sae_path(model_name, layer)
    sae_dict = torch.load(sae_path)
    sae = SAEWrapper(sae_dict, suite="qwen" if "qwen" in model_name else "gemma").to(model.device)

    # 2. Phase 1 Feature 로드
    feat_path = f"{BASE_DIR}/results/features/{model_name}_phase1_features.json"
    with open(feat_path, "r") as f:
        feat_data = json.load(f)
    
    correct_dom = feat_data[str(layer)]["correct_dominant"]
    wrong_dom = feat_data[str(layer)]["wrong_dominant"]
    print(f"Correct-dominant: {len(correct_dom)}, Wrong-dominant: {len(wrong_dom)}")

    # 3. 실험 3: Alpha 스케일 진단
    print("\n--- Phase 3: Alpha Scale Diagnosis ---")
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    sample_item = pubmedqa[0]
    sample_prompt = format_pubmedqa(sample_item, tokenizer, include_context=True)
    sample_act = get_activation_with_hook(model, tokenizer, sample_prompt, layer)
    act_norm = sample_act.norm().item()
    print(f"Original Activation Norm: {act_norm:.3f}")

    # 합성된 steer_vec (both 조건 가정)
    steer_vec_both = torch.zeros(sae.W_dec.shape[1], device=model.device)
    for idx in correct_dom: steer_vec_both += sae.W_dec[idx, :]
    for idx in wrong_dom: steer_vec_both -= sae.W_dec[idx, :]
    steer_norm = steer_vec_both.norm().item()
    print(f"Combined Steer Vec Norm: {steer_norm:.3f}")

    for alpha in [0.1, 0.5, 1, 2, 5, 10]:
        ratio = (alpha * steer_norm) / act_norm
        print(f"alpha={alpha}: ratio={ratio:.3f}")

    # 4. 실험 1+2: Alpha Sweep & Conditions
    print("\n--- Phase 1+2: Multi-feature Steering & Alpha Sweep ---")
    
    # Load labels to find wrong/correct cases
    with open(f"{BASE_DIR}/results/eval/{model_name}_labels.json", "r") as f:
        labels = json.load(f)
    
    item_map = {it["pubid"]: it for it in pubmedqa}
    wrong_cases = [item_map[r["item_id"]] for r in labels if not r["is_correct"]]
    correct_cases = [item_map[r["item_id"]] for r in labels if r["is_correct"]]
    print(f"Total wrong cases available: {len(wrong_cases)}")

    conditions = {
        "amplify_correct": {"amplify_idxs": correct_dom, "suppress_idxs": None},
        "suppress_wrong":  {"amplify_idxs": None, "suppress_idxs": wrong_dom},
        "both":            {"amplify_idxs": correct_dom, "suppress_idxs": wrong_dom},
    }

    # Based on diagnosis, choose alpha range
    alphas = [0.1, 0.5, 1.0, 2.0, 5.0]

    sweep_results = []
    for cond_name, cfg in conditions.items():
        print(f"\nTesting Condition: {cond_name}")
        for alpha in alphas:
            recovered = 0
            n_test = min(30, len(wrong_cases))
            for item in tqdm(wrong_cases[:n_test], desc=f"Alpha {alpha} Recovery", leave=False):
                probs = steer_multi(model, tokenizer, item, layer, sae, alpha=alpha, **cfg)
                pred = max(probs, key=probs.get)
                if pred == item["final_decision"]:
                    recovered += 1
            recovery_rate = recovered / n_test

            corrupted = 0
            n_test_c = min(30, len(correct_cases))
            for item in tqdm(correct_cases[:n_test_c], desc=f"Alpha {alpha} Corruption", leave=False):
                probs = steer_multi(model, tokenizer, item, layer, sae, alpha=alpha, **cfg)
                pred = max(probs, key=probs.get)
                if pred != item["final_decision"]:
                    corrupted += 1
            corruption_rate = corrupted / n_test_c

            print(f"  alpha={alpha:3.1f} | Recovery: {recovery_rate:5.1%} | Corruption: {corruption_rate:5.1%}")
            sweep_results.append({
                "condition": cond_name,
                "alpha": alpha,
                "recovery_rate": recovery_rate,
                "corruption_rate": corruption_rate
            })

    # 최종 결과 저장
    output_path = f"{RESULTS_DIR}/{model_name}_steering_retry.json"
    with open(output_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSweep results saved to {output_path}")

if __name__ == "__main__":
    main()
