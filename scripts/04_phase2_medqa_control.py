import torch
import json
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_wrapper import SAEWrapper
from utils import get_activation_with_hook, get_sae_path
import gc

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
}
RESULTS_DIR = f"{BASE_DIR}/results/eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

def format_medqa(item, tokenizer):
    """
    MedQA 포맷: Question + Options A,B,C,D
    """
    question = item["question"]
    options = item["options"]
    opt_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    prompt = f"Question: {question}\n\nOptions:\n{opt_str}\n\nAnswer with one letter (A, B, C, or D):"
    
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def get_medqa_pred(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    
    probs = torch.softmax(logits, dim=-1)
    results = {}
    for letter in ["A", "B", "C", "D"]:
        ids = tokenizer.encode(letter, add_special_tokens=False) + tokenizer.encode(" " + letter, add_special_tokens=False)
        results[letter] = max(probs[i].item() for i in ids)
    
    return max(results, key=results.get)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model
    path = MODELS[model_name]
    
    # Phase 1 결과 로드
    features_path = f"/workspace/medical_mi/results/features/{model_name}_phase1_features.json"
    if not os.path.exists(features_path):
        print("Phase 1 features not found.")
        return
    with open(features_path, "r") as f:
        phase1_data = json.load(f)

    print(f"--- Phase 2: MedQA Control for {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    print("Loading MedQA...")
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options")["test"]

    # 1. MedQA 평가 및 데이터 준비
    print("Evaluating MedQA...")
    correct_cases = []
    wrong_cases = []
    for item in tqdm(medqa[:200]): # 시간 단축을 위해 200개 샘플
        prompt = format_medqa(item, tokenizer)
        pred = get_medqa_pred(model, tokenizer, prompt)
        if pred == item["answer_idx"]:
            correct_cases.append(item)
        else:
            wrong_cases.append(item)

    print(f"MedQA Results: Correct={len(correct_cases)}, Wrong={len(wrong_cases)}")

    # 2. Feature 분석
    control_results = {}
    for layer_str, data in phase1_data.items():
        layer = int(layer_str)
        print(f"Analyzing Layer {layer}...")
        sae_path = get_sae_path(model_name, layer)
        if not sae_path: continue
        
        sae_dict = torch.load(sae_path)
        sae = SAEWrapper(sae_dict, suite="qwen" if "qwen" in model_name else "gemma").to(model.device)
        
        correct_features = data["correct_dominant"][:10] # Top 10만 확인
        
        layer_control = []
        for feat_idx in correct_features:
            # MedQA correct/wrong에서 해당 feature의 활성화도 측정
            acts_c = []
            for item in correct_cases[:50]:
                prompt = format_medqa(item, tokenizer)
                act = get_activation_with_hook(model, tokenizer, prompt, layer)
                with torch.no_grad():
                    f_val = sae.encode(act)[0, feat_idx].item()
                    acts_c.append(f_val)
            
            acts_w = []
            for item in wrong_cases[:50]:
                prompt = format_medqa(item, tokenizer)
                act = get_activation_with_hook(model, tokenizer, prompt, layer)
                with torch.no_grad():
                    f_val = sae.encode(act)[0, feat_idx].item()
                    acts_w.append(f_val)
            
            mean_c = np.mean(acts_c) if acts_c else 0
            mean_w = np.mean(acts_w) if acts_w else 0
            
            layer_control.append({
                "feature_idx": feat_idx,
                "medqa_correct_mean": mean_c,
                "medqa_wrong_mean": mean_w,
                "diff": abs(mean_c - mean_w)
            })
            
        control_results[layer] = layer_control
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    output_path = f"{BASE_DIR}/results/eval/{model_name}_phase2_control.json"
    with open(output_path, "w") as f:
        json.dump(control_results, f, indent=2)
    print(f"Control results saved to {output_path}")

if __name__ == "__main__":
    main()
