import torch
import json
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy import stats
from sae_wrapper import SAEWrapper
import gc

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": {"path": f"{BASE_DIR}/checkpoints/model/qwen3-8b", "suite": "qwen"},
    "qwen3.5-9b": {"path": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b", "suite": "qwen"},
    "gemma3-12b-it": {"path": f"{BASE_DIR}/checkpoints/model/gemma3-12b-it", "suite": "gemma"},
}
# 각 모델별 분석할 레이어 (36개 기준 샘플링)
QWEN_LAYERS = [10, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34]
GEMMA_LAYERS = [10, 15, 20, 25, 30, 35, 40] # Gemma 3 12B 레이어 수에 맞춰 조정 필요

def get_residual_activation(model, tokenizer, prompt, layer_idx):
    store = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        store["act"] = h[0, -1, :].detach().cpu()
    
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return store["act"]

def format_pubmedqa(item, tokenizer, enable_thinking=False):
    context_data = item.get("context", "")
    if isinstance(context_data, dict):
        contexts = context_data.get("contexts", [])
        context_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    else:
        context_text = str(context_data)
    
    context_text = context_text[:1500]
    prompt = f"Context: {context_text}\n\nQuestion: {item['question']}\n\nBased ONLY on the context, answer one word: yes, no, or maybe."
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    except:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Specific models to process")
    args = parser.parse_args()

    os.makedirs(f"{BASE_DIR}/results/features", exist_ok=True)
    data = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]

    target_models = args.models if args.models else MODELS.keys()

    for model_name in target_models:
        if model_name not in MODELS: continue
        config = MODELS[model_name]
        if not os.path.exists(config["path"]): continue
        
        print(f"\n--- Phase 1: Finding Features for {model_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(config["path"])
        model = AutoModelForCausalLM.from_pretrained(config["path"], torch_dtype=torch.float16, device_map="auto")
        model.eval()

        # 1. 고집부리는(Ignorance) 케이스와 잘 맞춘(Correct) 케이스 분류
        print("Classifying cases...")
        correct_cases = []
        ignorance_cases = []
        
        # 전체 1,000개 라벨 데이터를 모두 사용하여 분석
        for item in tqdm(data): 
            prompt = format_pubmedqa(item, tokenizer, enable_thinking=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            
            # 'yes', 'no' 확률 추출 (단순화된 방식)
            yes_id = tokenizer.encode("yes", add_special_tokens=False)[0]
            no_id = tokenizer.encode("no", add_special_tokens=False)[0]
            
            p_yes = probs[yes_id].item()
            p_no = probs[no_id].item()
            conf = max(p_yes, p_no)
            pred = "yes" if p_yes > p_no else "no"
            gt = item["final_decision"]

            if gt == "no" and pred == "no" and conf >= 0.70:
                correct_cases.append(item)
            elif gt == "no" and pred == "yes" and conf >= 0.70:
                ignorance_cases.append(item)

        print(f"Correct: {len(correct_cases)}, Ignorance: {len(ignorance_cases)}")
        
        if len(correct_cases) < 5 or len(ignorance_cases) < 5:
            print("데이터 부족으로 스킵")
            continue

        # 2. 레이어별 SAE Feature 통계 분석
        layers = QWEN_LAYERS if "qwen" in model_name else GEMMA_LAYERS
        model_results = {}

        for layer_idx in layers:
            print(f"Analyzing Layer {layer_idx}...")
            # SAE 로드 및 Wrapper 적용
            sae_path = f"{BASE_DIR}/checkpoints/sae/{model_name}/layer{layer_idx}.sae.pt"
            if not os.path.exists(sae_path): continue
            
            sae_dict = torch.load(sae_path, map_location="cpu")
            sae = SAEWrapper(sae_dict, suite=config["suite"])
            
            correct_acts = []
            for item in correct_cases[:50]:
                act = get_residual_activation(model, tokenizer, format_pubmedqa(item, tokenizer), layer_idx)
                correct_acts.append(sae.encode(act.unsqueeze(0)).squeeze(0))
            
            ignorance_acts = []
            for item in ignorance_cases[:50]:
                act = get_residual_activation(model, tokenizer, format_pubmedqa(item, tokenizer), layer_idx)
                ignorance_acts.append(sae.encode(act.unsqueeze(0)).squeeze(0))

            c_feat = torch.stack(correct_acts)
            i_feat = torch.stack(ignorance_acts)
            
            # T-test
            c_mean = c_feat.mean(0)
            i_mean = i_feat.mean(0)
            diff = c_mean - i_mean
            
            t_stats, p_vals = [], []
            for f in range(c_feat.shape[1]):
                t, p = stats.ttest_ind(c_feat[:, f].numpy(), i_feat[:, f].numpy())
                t_stats.append(0 if np.isnan(t) else t)
                p_vals.append(1.0 if np.isnan(p) else p)
            
            # 유의미한 feature (p < 0.05) 저장
            sig_indices = np.where(np.array(p_vals) < 0.05)[0]
            model_results[layer_idx] = {
                "top_indices": sig_indices.tolist(),
                "diffs": diff[sig_indices].tolist(),
                "p_values": np.array(p_vals)[sig_indices].tolist()
            }
            print(f"  Layer {layer_idx}: Found {len(sig_indices)} significant features.")

        with open(f"{BASE_DIR}/results/features/{model_name}_phase1_results.json", "w") as f:
            json.dump(model_results, f, indent=2)

        del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
