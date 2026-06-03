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
RESULTS_DIR = f"{BASE_DIR}/results/steering"
# Layer 25에서 Ignorance-dominant feature 분석
PRIMARY_LAYER = 25
MAGNITUDES = [0.3, 0.5, 0.7, 1.0]

# 환경 변수 강제 설정
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

def load_sae(layer_idx, sae_dir=SAE_DIR):
    sae_path = Path(sae_dir) / f"layer{layer_idx}.sae.pt"
    sae = torch.load(sae_path, map_location="cpu")
    return sae

def format_question_for_qwen3(tokenizer, question_text, options):
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    prompt = f"{question_text}\n\nOptions:\n{options_text}\n\nAnswer with just the letter (A, B, C, or D)."
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def get_answer_probabilities(model, tokenizer, formatted_prompt):
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    answer_probs = {c: probs[tokenizer.encode(c, add_special_tokens=False)[0]].item() for c in ["A", "B", "C", "D"]}
    total = sum(answer_probs.values())
    return {k: v/total for k, v in answer_probs.items()} if total > 0 else answer_probs

def entropy(probs_dict):
    return -sum(p * math.log(p + 1e-10) for p in probs_dict.values())

def steer_suppress_hook(module, input, output, sae, feature_indices, magnitude):
    if isinstance(output, tuple):
        hidden = output[0]
        rest = output[1:]
    else:
        hidden = output
        rest = None
        
    device = hidden.device
    W_enc = sae.get("W_enc", sae.get("encoder.weight")).float().to(device)
    b_enc = sae.get("b_enc", sae.get("encoder.bias")).float().to(device)
    W_dec = sae.get("W_dec", sae.get("decoder.weight")).float().to(device)
    b_dec = sae.get("b_dec", sae.get("decoder.bias", None))

    if W_enc.shape == (65536, 4096): W_enc = W_enc.t().contiguous()
    if W_dec.shape == (4096, 65536): W_dec = W_dec.t().contiguous()

    h_float = hidden.float()
    if b_dec is not None: h_float = h_float - b_dec.float().to(device)
    
    pre_act = h_float @ W_enc + b_enc
    k = 50
    topk_vals, topk_idx = torch.topk(pre_act, k, dim=-1)
    topk_vals = torch.relu(topk_vals)
    
    features = torch.zeros_like(pre_act)
    features.scatter_(-1, topk_idx, topk_vals)
    
    # SUPPRESS: (1 - magnitude)를 곱함
    f_tensor = torch.tensor(feature_indices, dtype=torch.long, device=device)
    features[:, :, f_tensor] *= max(0, 1 - magnitude)
    
    modified = features @ W_dec
    if b_dec is not None: modified = modified + b_dec.float().to(device)
    
    modified = modified.to(hidden.dtype)
    return (modified,) + rest if rest is not None else modified

def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    # 1. Ignorance-dominant feature 식별 (Wrong > Correct 인 것)
    with open(f"{BASE_DIR}/results/features/ignorance_feature_candidates.json") as f:
        candidates = json.load(f)
    
    layer_data = candidates[str(PRIMARY_LAYER)]
    suppress_features = []
    for idx, c_mean, i_mean in zip(
        layer_data["top_feature_indices"],
        layer_data["correct_mean_activation"],
        layer_data["ignorance_mean_activation"]
    ):
        if i_mean > c_mean:
            suppress_features.append({"idx": idx, "diff": i_mean - c_mean})
    
    suppress_features.sort(key=lambda x: x["diff"], reverse=True)
    suppress_indices = [f["idx"] for f in suppress_features[:20]]
    print(f"식별된 Suppress 타겟 feature 수: {len(suppress_indices)}")

    # 2. WRONG_CONFIDENT 케이스 로드
    with open(f"{BASE_DIR}/data/processed/wrong_confident.json") as f:
        wrong_cases = json.load(f)

    sae = load_sae(PRIMARY_LAYER)
    all_results = {}

    for mag in MAGNITUDES:
        print(f"\nRunning Suppress Steering Magnitude: {mag}")
        results = []
        for case in tqdm(wrong_cases[:30]):
            formatted = format_question_for_qwen3(tokenizer, case["question"], case["options"])
            orig_probs = get_answer_probabilities(model, tokenizer, formatted)
            
            handle = model.model.layers[PRIMARY_LAYER].register_forward_hook(
                lambda mod, inp, out, f=suppress_indices, m=mag: steer_suppress_hook(mod, inp, out, sae, f, m)
            )
            supp_probs = get_answer_probabilities(model, tokenizer, formatted)
            handle.remove()
            
            orig_answer = max(orig_probs, key=orig_probs.get)
            supp_answer = max(supp_probs, key=supp_probs.get)
            
            res = {
                "correct_answer": case["correct_answer"],
                "original_answer": orig_answer,
                "original_conf": orig_probs[orig_answer],
                "suppressed_answer": supp_answer,
                "suppressed_conf": supp_probs[supp_answer],
                "became_uncertain": supp_probs[supp_answer] < 0.5 and orig_probs[orig_answer] >= 0.7,
                "changed_to_correct": supp_answer == case["correct_answer"] and orig_answer != case["correct_answer"]
            }
            results.append(res)
            
        uncertain_count = sum(r["became_uncertain"] for r in results)
        correct_count = sum(r["changed_to_correct"] for r in results)
        print(f"  불확실해진 케이스: {uncertain_count}/{len(results)}")
        print(f"  정답으로 바뀐 케이스: {correct_count}/{len(results)}")
        
        all_results[str(mag)] = results

    with open(f"{RESULTS_DIR}/suppress_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()
