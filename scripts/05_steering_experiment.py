import torch
import json
import os
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODEL_PATH = f"{BASE_DIR}/checkpoints/model/Qwen3-8B"
SAE_DIR = f"{BASE_DIR}/checkpoints/sae/Qwen3-8B-SAE"
STEERING_RESULTS_DIR = f"{BASE_DIR}/results/steering"
# Layer 25에서 가장 유의미한 결과(낮은 p-value)가 나왔으므로 25로 변경
PRIMARY_LAYER = 25
MAGNITUDES = [0.5, 1.0, 2.0, 3.0, 5.0]
N_TEST_CASES = 30

# 환경 변수 강제 설정
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

def load_sae(layer_idx, sae_dir=SAE_DIR):
    sae_path = Path(sae_dir) / f"layer{layer_idx}.sae.pt"
    sae = torch.load(sae_path, map_location="cpu")
    print(f"Layer {layer_idx} SAE loaded.")
    return sae

def format_question_for_qwen3(tokenizer, question_text, options):
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    prompt = f"{question_text}\n\nOptions:\n{options_text}\n\nAnswer with just the letter (A, B, C, or D)."
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False # 네이티브 Non-thinking 모드
    )

def get_answer_probabilities(model, tokenizer, formatted_prompt):
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    
    answer_probs = {}
    for choice in ["A", "B", "C", "D"]:
        token_ids = tokenizer.encode(choice, add_special_tokens=False)
        answer_probs[choice] = probs[token_ids[0]].item()
    
    total = sum(answer_probs.values())
    if total > 0:
        answer_probs = {k: v/total for k, v in answer_probs.items()}
    return answer_probs

def entropy(probs_dict):
    return -sum(p * math.log(p + 1e-10) for p in probs_dict.values())

def steer_and_evaluate(model, tokenizer, case, sae, feature_indices, magnitude, layer_idx):
    # SAE 가중치 로드 및 Shape 보정
    W_enc = sae.get("W_enc", sae.get("encoder.weight")).float()
    b_enc = sae.get("b_enc", sae.get("encoder.bias")).float()
    W_dec = sae.get("W_dec", sae.get("decoder.weight")).float()
    b_dec = sae.get("b_dec", sae.get("decoder.bias", None))

    # Qwen-Scope SAE Shape 보정 ([d_sae, d_model] -> [d_model, d_sae])
    if W_enc.shape[0] != 4096 and W_enc.shape[1] == 4096:
        W_enc = W_enc.T
    if W_dec.shape[1] != 4096 and W_dec.shape[0] == 4096:
        W_dec = W_dec.T

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
            
        device = hidden.device
        h_float = hidden.float()
        
        if b_dec is not None:
            h_float = h_float - b_dec.float().to(device)
            
        # Encode
        pre_act = h_float @ W_enc.to(device) + b_enc.to(device)
        
        # TopK k=50
        k = 50
        topk_vals, topk_idx = torch.topk(pre_act, k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        
        features = torch.zeros_like(pre_act)
        features.scatter_(-1, topk_idx, topk_vals)
        
        # Steering: Ignorance feature 증폭
        f_tensor = torch.tensor(feature_indices, dtype=torch.long, device=device)
        features[:, :, f_tensor] *= (1 + magnitude)
        
        # Decode back
        modified = features @ W_dec.to(device)
        if b_dec is not None:
            modified = modified + b_dec.float().to(device)
            
        modified = modified.to(hidden.dtype)
        if rest is not None:
            return (modified,) + rest
        return modified

    formatted = format_question_for_qwen3(tokenizer, case["question"], case["options"])
    
    # Original (No steering)
    orig_probs = get_answer_probabilities(model, tokenizer, formatted)
    orig_answer = max(orig_probs, key=orig_probs.get)
    orig_conf = orig_probs[orig_answer]
    
    # Steered
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    steer_probs = get_answer_probabilities(model, tokenizer, formatted)
    handle.remove()
    
    steer_answer = max(steer_probs, key=steer_probs.get)
    steer_conf = steer_probs[steer_answer]
    
    return {
        "question": case["question"][:100] + "...",
        "correct_answer": case["correct_answer"],
        "original": {"answer": orig_answer, "conf": orig_conf, "entropy": entropy(orig_probs), "probs": orig_probs},
        "steered": {"answer": steer_answer, "conf": steer_conf, "entropy": entropy(steer_probs), "probs": steer_probs},
        "became_uncertain": steer_conf < 0.5 and orig_conf >= 0.7
    }

def main():
    os.makedirs(STEERING_RESULTS_DIR, exist_ok=True)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    
    with open(f"{BASE_DIR}/data/processed/wrong_confident.json") as f:
        ignorance_cases = json.load(f)
    
    with open(f"{BASE_DIR}/results/features/ignorance_feature_candidates.json") as f:
        candidates = json.load(f)
    
    # 해당 레이어의 상위 20개 후보 사용
    top_features = candidates[str(PRIMARY_LAYER)]["top_feature_indices"][:20]
    sae = load_sae(PRIMARY_LAYER)
    
    all_results = {}
    for mag in MAGNITUDES:
        print(f"\nRunning Steering Magnitude: {mag}")
        results = []
        # 가용한 케이스 내에서 테스트
        n_test = min(len(ignorance_cases), N_TEST_CASES)
        for case in tqdm(ignorance_cases[:n_test]):
            res = steer_and_evaluate(model, tokenizer, case, sae, top_features, mag, PRIMARY_LAYER)
            results.append(res)
            
        became_uncertain = sum(r["became_uncertain"] for r in results)
        avg_conf_change = sum(r["steered"]["conf"] - r["original"]["conf"] for r in results) / len(results)
        
        print(f"  Uncertain cases: {became_uncertain}/{len(results)}")
        print(f"  Avg Conf Change: {avg_conf_change:+.4f}")
        
        all_results[str(mag)] = {
            "became_uncertain_rate": became_uncertain / len(results) if len(results) > 0 else 0,
            "avg_confidence_change": avg_conf_change,
            "individual_results": results
        }
        
    with open(f"{STEERING_RESULTS_DIR}/magnitude_sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSteering results saved to {STEERING_RESULTS_DIR}/magnitude_sweep_results.json")

if __name__ == "__main__":
    main()
