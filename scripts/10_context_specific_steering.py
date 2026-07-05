import torch
import json
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_wrapper import SAEWrapper
from utils import format_pubmedqa, format_medqa, get_activation_with_hook, get_sae_path
import gc

# --- Config ---
BASE_DIR = "/home/eagle0914/medical_mi"
MODEL_NAME = "qwen3.5-9b"
LAYER = 20
MODELS = {
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
}
RESULTS_DIR = f"{BASE_DIR}/results/steering_context_specific"
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPHA = 10.0

def get_ynm_probs(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    result = {}
    for word in ["yes", "no", "maybe"]:
        ids = []
        for variant in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if tok: ids.append(tok[0])
        if ids:
            result[word] = max(probs[i].item() for i in ids if i < probs.shape[0])
        else:
            result[word] = 0.0
    
    total = sum(result.values())
    return {k: v/total for k, v in result.items()} if total > 0 else {"yes": 0.33, "no": 0.33, "maybe": 0.33}

def steer_and_test(model, tokenizer, item, layer, steer_vec, alpha):
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h[0, -1, :] = h[0, -1, :] + alpha * steer_vec.to(device=h.device, dtype=h.dtype)
        return o
    handle = model.model.layers[layer].register_forward_hook(hook)
    probs = get_ynm_probs(model, tokenizer, prompt)
    handle.remove()
    return probs

def main():
    print(f"=== Step 1: Feature Filtering (Layer {LAYER}) ===")
    path = MODELS[MODEL_NAME]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    sae_path = get_sae_path(MODEL_NAME, LAYER)
    sae_dict = torch.load(sae_path)
    sae = SAEWrapper(sae_dict, suite="qwen").to(model.device)

    with open(f"{BASE_DIR}/results/features/{MODEL_NAME}_phase1_features.json", "r") as f:
        feat_data = json.load(f)
    
    correct_candidates = feat_data[str(LAYER)]["correct_dominant"]
    wrong_candidates = feat_data[str(LAYER)]["wrong_dominant"]
    all_candidates = list(set(correct_candidates + wrong_candidates))

    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    pqa_subset = [pubmedqa[i] for i in range(min(50, len(pubmedqa)))]
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options")["test"]
    medqa_subset = [medqa[i] for i in range(min(50, len(medqa)))]

    feature_stats = {}
    for feat_idx in tqdm(all_candidates, desc="Filtering"):
        acts_with, acts_without, acts_medqa = [], [], []
        for item in pqa_subset:
            p_with = format_pubmedqa(item, tokenizer, include_context=True)
            act = get_activation_with_hook(model, tokenizer, p_with, LAYER)
            acts_with.append(sae.encode(act)[0, feat_idx].item())
            p_without = format_pubmedqa(item, tokenizer, include_context=False)
            act = get_activation_with_hook(model, tokenizer, p_without, LAYER)
            acts_without.append(sae.encode(act)[0, feat_idx].item())
        for item in medqa_subset:
            p_med = format_medqa(item, tokenizer)
            act = get_activation_with_hook(model, tokenizer, p_med, LAYER)
            acts_medqa.append(sae.encode(act)[0, feat_idx].item())
            
        m_with, m_without, m_medqa = np.mean(acts_with), np.mean(acts_without), np.mean(acts_medqa)
        is_context_specific = (m_with > 2 * m_without) and (m_medqa < 0.05)
        feature_stats[feat_idx] = {"mean_with": m_with, "mean_medqa": m_medqa, "is_context_specific": is_context_specific}

    # Step 2를 위한 필터링된 리스트 추출
    cs_wrong = [f for f in wrong_candidates if feature_stats[f]["is_context_specific"]]
    cs_correct = [f for f in correct_candidates if feature_stats[f]["is_context_specific"]]
    
    # 리스트 출력 (매우 중요!)
    print(f"\n[Step 2] Filtered Context-Specific Features (Wrong-dominant): {cs_wrong}")
    print(f"[Step 2] Filtered Context-Specific Features (Correct-dominant): {cs_correct}")

    steering_sets = {
        "All Wrong-Dom (22)": {"suppress": wrong_candidates, "amplify": []},
        "Step 2: Context-Specific Wrong only": {"suppress": cs_wrong, "amplify": []},
        "Step 3: Single #28696 (Suppress)": {"suppress": [28696], "amplify": []},
        "Step 3: Single #2392 (Amplify)": {"suppress": [], "amplify": [2392]},
        "Full Context Control (Correct+Wrong)": {"suppress": cs_wrong, "amplify": cs_correct}
    }

    with open(f"{BASE_DIR}/results/eval/{MODEL_NAME}_labels.json", "r") as f:
        labels = json.load(f)
    item_map = {it["pubid"]: it for it in pubmedqa}
    wrong_cases = [item_map[r["item_id"]] for r in labels if not r["is_correct"]]
    correct_cases = [item_map[r["item_id"]] for r in labels if r["is_correct"]]

    print("\n=== Steps 2 & 3: Steering Comparison ===")
    table_results = []
    
    for set_name, config in steering_sets.items():
        if not config["suppress"] and not config["amplify"]: continue
        print(f"Testing: {set_name}")
        
        steer_vec = torch.zeros(sae.W_dec.shape[1], device=model.device)
        for idx in config["amplify"]: steer_vec += sae.W_dec[idx, :].to(model.device)
        for idx in config["suppress"]: steer_vec -= sae.W_dec[idx, :].to(model.device)
            
        # Evaluation
        recovered, corrupted = 0, 0
        n_w, n_c = min(100, len(wrong_cases)), min(100, len(correct_cases))
        for item in tqdm(wrong_cases[:n_w], desc="Recovery", leave=False):
            if max(steer_and_test(model, tokenizer, item, LAYER, steer_vec, ALPHA), key=lambda x: x) == item["final_decision"]: recovered += 1
        # Re-evaluating with correct mapping for steer_and_test output
        recovered = 0
        for item in tqdm(wrong_cases[:n_w], desc="Recovery", leave=False):
            probs = steer_and_test(model, tokenizer, item, LAYER, steer_vec, ALPHA)
            if max(probs, key=probs.get) == item["final_decision"]: recovered += 1
            
        for item in tqdm(correct_cases[:n_c], desc="Corruption", leave=False):
            probs = steer_and_test(model, tokenizer, item, LAYER, steer_vec, ALPHA)
            if max(probs, key=probs.get) != item["final_decision"]: corrupted += 1
                
        table_results.append({
            "Set": set_name, 
            "Recovery": f"{recovered/n_w:.1%}", 
            "Corruption": f"{corrupted/n_c:.1%}",
            "IDs": {"amplify": config["amplify"], "suppress": config["suppress"]} # 번호 리스트 저장
        })

    # 결과 출력
    print("\n" + "="*80)
    print(f"{'Set Name':<40} | {'Recovery':<10} | {'Corruption':<10}")
    print("-" * 80)
    for r in table_results:
        print(f"{r['Set']:<40} | {r['Recovery']:<10} | {r['Corruption']:<10}")
    
    with open(f"{RESULTS_DIR}/final_comparison_detailed.json", "w") as f:
        json.dump(table_results, f, indent=2)

if __name__ == "__main__":
    main()
