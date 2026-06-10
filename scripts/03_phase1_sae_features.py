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
from scipy.stats import ttest_ind

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
}
RESULTS_DIR = f"{BASE_DIR}/results/features"
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_LAYERS = [10, 15, 18, 20, 22, 24, 26, 28, 30]

def extract_features(model, tokenizer, sae, items, layer):
    all_features = []
    for item in items:
        prompt = format_pubmedqa(item, tokenizer, include_context=True)
        act = get_activation_with_hook(model, tokenizer, prompt, layer)
        with torch.no_grad():
            feats = sae.encode(act)
            all_features.append(feats.cpu().numpy())
    return np.concatenate(all_features, axis=0) if all_features else np.array([])

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model
    labels_path = f"/workspace/medical_mi/results/eval/{model_name}_labels.json"
    
    if not os.path.exists(labels_path):
        print(f"Labels file not found: {labels_path}")
        return

    with open(labels_path, "r") as f:
        labels_data = json.load(f)

    print("Loading PubMedQA...")
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    item_map = {item["pubid"]: item for item in pubmedqa}

    correct_items = [item_map[r["item_id"]] for r in labels_data if r["is_correct"]]
    wrong_items = [item_map[r["item_id"]] for r in labels_data if not r["is_correct"]]

    print(f"Correct: {len(correct_items)}, Wrong: {len(wrong_items)}")

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
            continue
            
        sae_dict = torch.load(sae_path)
        suite = "qwen" if "qwen" in model_name else "gemma"
        sae = SAEWrapper(sae_dict, suite=suite).to(model.device)
        
        # 샘플링 (불균형 해소 및 속도 향상)
        n_sample = min(len(correct_items), len(wrong_items), 100)
        c_samples = correct_items[:n_sample]
        w_samples = wrong_items[:n_sample]
        
        feat_correct = extract_features(model, tokenizer, sae, c_samples, layer)
        feat_wrong = extract_features(model, tokenizer, sae, w_samples, layer)
        
        t_stats, p_vals = ttest_ind(feat_correct, feat_wrong, axis=0, equal_var=False)
        
        # Correct-dominant (t > 0, p < 0.01)
        correct_dom = np.where((p_vals < 0.01) & (np.nan_to_num(t_stats) > 0))[0]
        # Wrong-dominant (t < 0, p < 0.01)
        wrong_dom = np.where((p_vals < 0.01) & (np.nan_to_num(t_stats) < 0))[0]
        
        print(f"Layer {layer}: Correct-dom={len(correct_dom)}, Wrong-dom={len(wrong_dom)}")
        
        layer_results[layer] = {
            "correct_dominant": correct_dom.tolist(),
            "wrong_dominant": wrong_dom.tolist(),
            "n_correct_dom": len(correct_dom),
            "n_wrong_dom": len(wrong_dom)
        }
        
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    output_path = f"{RESULTS_DIR}/{model_name}_phase1_features.json"
    with open(output_path, "w") as f:
        json.dump(layer_results, f, indent=2)
    print(f"Phase 1 features saved to {output_path}")

if __name__ == "__main__":
    main()
