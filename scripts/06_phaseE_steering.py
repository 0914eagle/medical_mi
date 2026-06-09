import torch
import json
import os
import gc
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_wrapper import SAEWrapper
from utils import format_pubmedqa

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
    "gemma3-12b-it": f"{BASE_DIR}/checkpoints/model/gemma3-12b-it",
}
SAE_BASE = f"{BASE_DIR}/checkpoints/sae"
RESULTS_DIR = f"{BASE_DIR}/results/steering"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_sae_path(model_name, layer):
    path_options = [
        f"{SAE_BASE}/{model_name}/layer_{layer}/res_64k/sae_weights.pt",
        f"{SAE_BASE}/{model_name}/layer{layer}.sae.pt",
        f"{SAE_BASE}/{model_name}/{layer}/sae.pt"
    ]
    for p in path_options:
        if os.path.exists(p): return p
    return None

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
    parser.add_argument("--model", type=str, default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--feature_idx", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=20.0, help="Scale factor. Consider the typical activation range from Phase B.")
    args = parser.parse_args()

    model_name = args.model
    layer = args.layer
    feature_idx = args.feature_idx

    print("Loading data...")
    conflict_set_path = f"/workspace/medical_mi/results/eval/{model_name}_conflict_set.json"
    with open(conflict_set_path, "r") as f:
        full_data = json.load(f)
    
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    item_map = {item["pubid"]: item for item in pubmedqa}
    
    ignored_cases = [item_map[r["item_id"]] for r in full_data if r["classification"] == "IGNORED"]
    no_conflict_cases = [item_map[r["item_id"]] for r in full_data if r["classification"] == "NO_CONFLICT"]

    path = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    sae_path = get_sae_path(model_name, layer)
    if not sae_path:
        print(f"SAE not found for layer {layer}")
        return
        
    sae_dict = torch.load(sae_path)
    suite = "qwen" if "qwen" in model_name else "gemma"
    sae = SAEWrapper(sae_dict, suite=suite).to(model.device)
    
    # Steering vector = decoder weight of the feature (Integrated-dominant)
    steer_vec = sae.W_dec[feature_idx, :]

    print(f"\nSteering with Feature {feature_idx} at Layer {layer} (alpha={args.alpha})...")
    
    # 1. IGNORED cases
    flips = 0
    total_ignored = 0
    for item in tqdm(ignored_cases[:20], desc="Testing IGNORED"):
        res = steer_and_test(model, tokenizer, item, layer, steer_vec, alpha=args.alpha)
        if max(res, key=res.get) == "no":
            flips += 1
        total_ignored += 1
    
    print(f"Flip Rate: {flips/total_ignored if total_ignored else 0:.2%}")

    # 2. NO_CONFLICT cases (Selectivity)
    corrupted = 0
    total_nc = 0
    for item in tqdm(no_conflict_cases[:20], desc="Testing Selectivity"):
        res = steer_and_test(model, tokenizer, item, layer, steer_vec, alpha=args.alpha)
        if max(res, key=res.get) != "no":
            corrupted += 1
        total_nc += 1
    
    print(f"Corruption Rate: {corrupted/total_nc if total_nc else 0:.2%}")

if __name__ == "__main__":
    main()
