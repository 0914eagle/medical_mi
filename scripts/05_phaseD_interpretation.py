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

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
    "gemma3-12b-it": f"{BASE_DIR}/checkpoints/model/gemma3-12b-it",
}
SAE_BASE = f"{BASE_DIR}/checkpoints/sae"
RESULTS_DIR = f"{BASE_DIR}/results/features"

def get_sae_path(model_name, layer):
    path_options = [
        f"{SAE_BASE}/{model_name}/layer_{layer}/res_64k/sae_weights.pt",
        f"{SAE_BASE}/{model_name}/layer{layer}.sae.pt",
        f"{SAE_BASE}/{model_name}/{layer}/sae.pt"
    ]
    for p in path_options:
        if os.path.exists(p): return p
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--feature_idx", type=int, required=True)
    args = parser.parse_args()

    model_name = args.model
    layer = args.layer
    feature_idx = args.feature_idx

    corpora = {}
    
    # PubMedQA Conflict
    conflict_set_path = f"/workspace/medical_mi/results/eval/{model_name}_conflict_set.json"
    if os.path.exists(conflict_set_path):
        with open(conflict_set_path, "r") as f:
            full_data = json.load(f)
        pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
        item_map = {item["pubid"]: item for item in pubmedqa}
        
        conflict_items = [item_map[r["item_id"]] for r in full_data if r["classification"] in ["INTEGRATED", "IGNORED"]]
        corpora["pubmedqa_conflict"] = [format_pubmedqa(it, None, True) for it in conflict_items[:50]]
        corpora["medical_no_context"] = [format_pubmedqa(it, None, False) for it in conflict_items[:50]]

    # General Text (Wiki)
    try:
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        corpora["general_text"] = [t for t in wiki["text"] if len(t) > 100][:50]
    except:
        corpora["general_text"] = ["Placeholder text for general knowledge analysis."] * 10

    # 모델 & SAE 로드
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

    results = {}
    for corpus_name, texts in corpora.items():
        activations = []
        for text in tqdm(texts, desc=f"Processing {corpus_name}"):
            if not text.strip(): continue
            act = get_activation_with_hook(model, tokenizer, text, layer)
            with torch.no_grad():
                feats = sae.encode(act)
                activations.append(feats[0, feature_idx].item())
        
        results[corpus_name] = {
            "mean": np.mean(activations) if activations else 0,
            "max": np.max(activations) if activations else 0,
        }

    print("\nInterpretation Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
