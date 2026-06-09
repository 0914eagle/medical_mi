import torch
import json
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_wrapper import SAEWrapper
from utils import format_pubmedqa, get_activation_with_hook
import gc
from scipy.stats import ttest_ind

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
    "gemma3-12b-it": f"{BASE_DIR}/checkpoints/model/gemma3-12b-it",
}
SAE_BASE = f"{BASE_DIR}/checkpoints/sae"
RESULTS_DIR = f"{BASE_DIR}/results/features"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_LAYERS = [10, 15, 18, 20, 22, 24, 26, 28, 30, 32]

def get_sae_path(model_name, layer):
    # Flexible SAE path logic - look for layer{n}.sae.pt or similar
    path_options = [
        f"{SAE_BASE}/{model_name}/layer_{layer}/res_64k/sae_weights.pt",
        f"{SAE_BASE}/{model_name}/layer{layer}.sae.pt",
        f"{SAE_BASE}/{model_name}/{layer}/sae.pt"
    ]
    for p in path_options:
        if os.path.exists(p): return p
    return None

def extract_sae_features(model, tokenizer, sae, items, layer, include_context=True):
    all_features = []
    for item in items:
        prompt = format_pubmedqa(item, tokenizer, include_context=include_context)
        act = get_activation_with_hook(model, tokenizer, prompt, layer)
        with torch.no_grad():
            feats = sae.encode(act)
            all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0) if all_features else np.array([])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3.5-9b")
    args = parser.parse_args()

    model_name = args.model
    conflict_set_path = f"/workspace/medical_mi/results/eval/{model_name}_conflict_set.json"
    
    if not os.path.exists(conflict_set_path):
        print("Conflict set not found.")
        return

    with open(conflict_set_path, "r") as f:
        full_data = json.load(f)

    print("Loading PubMedQA...")
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    item_map = {item["pubid"]: item for item in pubmedqa}

    integrated_cases = [item_map[r["item_id"]] for r in full_data if r["classification"] == "INTEGRATED"]
    ignored_cases = [item_map[r["item_id"]] for r in full_data if r["classification"] == "IGNORED"]

    print(f"INTEGRATED: {len(integrated_cases)}, IGNORED: {len(ignored_cases)}")

    # 모델 로드
    path = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    layer_results = {}
    for layer in TARGET_LAYERS:
        print(f"\nAnalyzing Layer {layer}...")
        sae_path = get_sae_path(model_name, layer)
        if not sae_path:
            print(f"SAE not found for layer {layer}, skipping.")
            continue
            
        sae_dict = torch.load(sae_path)
        suite = "qwen" if "qwen" in model_name else "gemma"
        sae = SAEWrapper(sae_dict, suite=suite).to(model.device)
        
        # B-1: Signal 1 (INTEGRATED vs IGNORED)
        feat_integrated = extract_sae_features(model, tokenizer, sae, integrated_cases[:50], layer, include_context=True)
        feat_ignored = extract_sae_features(model, tokenizer, sae, ignored_cases[:50], layer, include_context=True)
        
        t_stats, p_vals = ttest_ind(feat_integrated, feat_ignored, axis=0)
        sig_indices_1 = np.where((p_vals < 0.01) & (np.nan_to_num(t_stats) > 0))[0]
        
        # B-2: Signal 2 (Within-question)
        conflict_cases = (integrated_cases + ignored_cases)[:50]
        feat_C = extract_sae_features(model, tokenizer, sae, conflict_cases, layer, include_context=True)
        feat_P = extract_sae_features(model, tokenizer, sae, conflict_cases, layer, include_context=False)
        
        t_stats_2, p_vals_2 = ttest_ind(feat_C, feat_P, axis=0)
        sig_indices_2 = np.where((p_vals_2 < 0.01) & (np.nan_to_num(t_stats_2) > 0))[0]
        
        # B-3: Intersection
        intersection = np.intersect1d(sig_indices_1, sig_indices_2)
        print(f"Layer {layer}: Found {len(intersection)} intersection features.")
        
        layer_results[layer] = {
            "sig_1_count": len(sig_indices_1),
            "sig_2_count": len(sig_indices_2),
            "intersection": intersection.tolist()
        }
        
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    output_path = f"{RESULTS_DIR}/{model_name}_features.json"
    with open(output_path, "w") as f:
        json.dump(layer_results, f, indent=2)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
