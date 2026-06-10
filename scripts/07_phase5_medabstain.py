import torch
import json
import os
import numpy as np
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

def get_medabstain_dir():
    """
    MedAbstain 데이터 폴더를 유연하게 탐색
    """
    possible_paths = [
        f"{BASE_DIR}/data/raw/MedAbstain/data",
        f"{BASE_DIR}/data/raw/MedAbstain/MedAbstain/data", # 중첩된 경우 대비
        "/Users/heejae/Developer/MedAbstain/data" # 로컬 테스트 대비
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

def format_medabstain(item, tokenizer, use_original=False):
    """
    MedAbstain 포맷팅: Question + Choices
    perturbed 파일의 경우 'original_question'과 'question'(perturbed)이 둘 다 있음.
    """
    if use_original and "original_question" in item:
        question = item["original_question"]
    else:
        question = item["question"]
    
    choices = item["choices"]
    choice_str = "\n".join([f"{k}: {v}" for k, v in choices.items()])
    
    prompt = f"Question: {question}\n\nChoices:\n{choice_str}\n\nAnswer with one letter (A, B, C, D, or E):"
    
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    model_name = args.model
    path = MODELS[model_name]
    
    # Phase 1 결과 로드 (Top features 찾기 위해)
    features_path = f"{BASE_DIR}/results/features/{model_name}_phase1_features.json"
    if not os.path.exists(features_path):
        print(f"Phase 1 features not found for {model_name}.")
        return
    with open(features_path, "r") as f:
        phase1_data = json.load(f)

    print(f"--- Phase 5: MedAbstain Cross-Validation for {model_name} ---")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    # 데이터 디렉토리 탐색
    data_dir = get_medabstain_dir()
    if data_dir is None:
        print("MedAbstain data directory not found.")
        return
        
    # 데이터 로드: perturbed 파일 하나에 original과 perturbed 정보가 모두 들어있음
    pert_file = f"{data_dir}/perturbed_medqa_1_test_noabst.json"
    if not os.path.exists(pert_file):
        # 파일명이 다른 경우 체크 (alldiff)
        pert_file = f"{data_dir}/perturbed_medqa_alldiff_test_noabst.json"
        
    if not os.path.exists(pert_file):
        print(f"MedAbstain file not found in {data_dir}")
        return
        
    print(f"Loading data from: {pert_file}")
    with open(pert_file, "r") as f:
        data = json.load(f)

    layer_results = {}
    for layer_str, feat_info in phase1_data.items():
        layer = int(layer_str)
        print(f"Analyzing Layer {layer}...")
        sae_path = get_sae_path(model_name, layer)
        if not sae_path: continue
        
        sae_dict = torch.load(sae_path)
        sae = SAEWrapper(sae_dict, suite="qwen" if "qwen" in model_name else "gemma").to(model.device)
        
        top_features = feat_info["correct_dominant"][:5]
        if not top_features: continue

        feature_stats = []
        for feat_idx in top_features:
            acts_orig = []
            acts_pert = []
            
            for item in tqdm(data[:50], desc=f"Feature {feat_idx}"):
                # 1. Original (Full Information)
                prompt_orig = format_medabstain(item, tokenizer, use_original=True)
                act_orig = get_activation_with_hook(model, tokenizer, prompt_orig, layer)
                
                # 2. Perturbed (Missing Information)
                prompt_pert = format_medabstain(item, tokenizer, use_original=False)
                act_pert = get_activation_with_hook(model, tokenizer, prompt_pert, layer)
                
                with torch.no_grad():
                    acts_orig.append(sae.encode(act_orig)[0, feat_idx].item())
                    acts_pert.append(sae.encode(act_pert)[0, feat_idx].item())
            
            m_orig = np.mean(acts_orig)
            m_pert = np.mean(acts_pert)
            feature_stats.append({
                "feature_idx": feat_idx,
                "original_mean": m_orig,
                "perturbed_mean": m_pert,
                "diff": m_orig - m_pert
            })
            
        layer_results[layer] = feature_stats
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    output_path = f"{BASE_DIR}/results/eval/{model_name}_phase5_medabstain.json"
    with open(output_path, "w") as f:
        json.dump(layer_results, f, indent=2)
    print(f"MedAbstain results saved to {output_path}")

if __name__ == "__main__":
    main()
