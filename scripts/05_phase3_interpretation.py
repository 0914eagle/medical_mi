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
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--feature_idx", type=int, required=True)
    args = parser.parse_args()

    model_name = args.model
    layer = args.layer
    feature_idx = args.feature_idx

    print(f"--- Phase 3: Interpretation for {model_name} L{layer} F{feature_idx} ---")
    
    # 1. Corpora 준비
    corpora = {}
    
    # PubMedQA (with context)
    print("Loading PubMedQA...")
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    pqa_subset = pubmedqa.select(range(min(30, len(pubmedqa))))
    corpora["pubmedqa_with_context"] = [format_pubmedqa(it, None, True) for it in pqa_subset]
    corpora["pubmedqa_no_context"] = [format_pubmedqa(it, None, False) for it in pqa_subset]

    # MedQA (no context)
    print("Loading MedQA...")
    try:
        medqa = load_dataset("GBaker/MedQA-USMLE-4-options")["test"]
        medqa_subset = medqa.select(range(min(30, len(medqa))))
        from 04_phase2_medqa_control import format_medqa
        corpora["medqa"] = [format_medqa(it, None) for it in medqa_subset]
    except:
        corpora["medqa"] = ["Placeholder MedQA question."] * 10

    # General (Wiki)
    print("Loading Wiki...")
    try:
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        corpora["general"] = [t for t in wiki["text"] if len(t) > 100][:30]
    except:
        corpora["general"] = ["Placeholder general text."] * 10

    # 모델 & SAE 로드
    path = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    sae_path = get_sae_path(model_name, layer)
    sae_dict = torch.load(sae_path)
    sae = SAEWrapper(sae_dict, suite="qwen" if "qwen" in model_name else "gemma").to(model.device)

    results = {}
    for corpus_name, texts in corpora.items():
        activations = []
        for text in tqdm(texts, desc=f"Processing {corpus_name}"):
            if not text.strip(): continue
            # If tokenizer was passed as None in format_pubmedqa/format_medqa, it returned raw string.
            # We need to apply chat template here if it's not already applied.
            # For simplicity, we assume texts are already formatted or we format them now.
            # Actually, format_pubmedqa(it, None) returns raw prompt. We need to wrap it in chat template.
            messages = [{"role": "user", "content": text}]
            try:
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except:
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            act = get_activation_with_hook(model, tokenizer, formatted_text, layer)
            with torch.no_grad():
                f_val = sae.encode(act)[0, feature_idx].item()
                activations.append(f_val)
        
        results[corpus_name] = {
            "mean": np.mean(activations) if activations else 0,
            "max": np.max(activations) if activations else 0
        }

    print("\nInterpretation Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
