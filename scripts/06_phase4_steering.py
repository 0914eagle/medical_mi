import torch
import json
import os
import gc
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_wrapper import SAEWrapper
from utils import format_pubmedqa, get_ynm_probs

# --- Config ---
BASE_DIR = "/home/eagle0914/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
}
RESULTS_DIR = f"{BASE_DIR}/results/steering"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_sae_path(model_name, layer):
    from utils import get_sae_path as get_path
    return get_path(model_name, layer)

def steer_and_test(model, tokenizer, item, layer, steer_vec, alpha=10.0):
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    def steering_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[0, -1, :] = h[0, -1, :] + alpha * steer_vec.to(h.device)
        return output

    handle = model.model.layers[layer].register_forward_hook(steering_hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
    handle.remove()
    
    result = {}
    for word in ["yes", "no", "maybe"]:
        ids = []
        for variant in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if tok: ids.append(tok[0])
        if ids: result[word] = max(probs[i].item() for i in ids)
        else: result[word] = 0.0
    
    total = sum(result.values())
    if total > 0: result = {k: v/total for k, v in result.items()}
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--feature_idx", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=20.0)
    args = parser.parse_args()

    model_name = args.model
    layer = args.layer
    feature_idx = args.feature_idx

    print(f"--- Phase 4: Steering for {model_name} L{layer} F{feature_idx} ---")

    # Load labels to identify 'wrong' cases
    labels_path = f"/home/eagle0914/medical_mi/results/eval/{model_name}_labels.json"
    with open(labels_path, "r") as f:
        labels_data = json.load(f)
    
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    item_map = {item["pubid"]: item for item in pubmedqa}
    
    wrong_cases = [item_map[r["item_id"]] for r in labels_data if not r["is_correct"]]
    correct_cases = [item_map[r["item_id"]] for r in labels_data if r["is_correct"]]

    # 모델 & SAE 로드
    path = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    sae_path = get_sae_path(model_name, layer)
    sae_dict = torch.load(sae_path)
    sae = SAEWrapper(sae_dict, suite="qwen" if "qwen" in model_name else "gemma").to(model.device)
    
    steer_vec = sae.W_dec[feature_idx, :]

    # 1. Wrong -> Correct 교정률
    flips = 0
    total_w = 0
    for item in tqdm(wrong_cases[:30], desc="Testing WRONG cases"):
        res = steer_and_test(model, tokenizer, item, layer, steer_vec, alpha=args.alpha)
        if max(res, key=res.get) == item["final_decision"]:
            flips += 1
        total_w += 1
    
    # 2. Correct -> Wrong 오염률 (Selectivity)
    corrupted = 0
    total_c = 0
    for item in tqdm(correct_cases[:30], desc="Testing CORRECT cases"):
        res = steer_and_test(model, tokenizer, item, layer, steer_vec, alpha=args.alpha)
        if max(res, key=res.get) != item["final_decision"]:
            corrupted += 1
        total_c += 1

    summary = {
        "model": model_name,
        "layer": layer,
        "feature_idx": feature_idx,
        "alpha": args.alpha,
        "recovery_rate": flips / total_w if total_w > 0 else 0,
        "corruption_rate": corrupted / total_c if total_c > 0 else 0
    }
    
    print("\nSteering Summary:")
    print(json.dumps(summary, indent=2))
    
    output_path = f"{RESULTS_DIR}/{model_name}_steering_L{layer}_F{feature_idx}.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
