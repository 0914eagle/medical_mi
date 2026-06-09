import torch
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODEL_PATH = f"{BASE_DIR}/checkpoints/model/Qwen3-8B"
SAE_DIR = f"{BASE_DIR}/checkpoints/sae/Qwen3-8B-SAE"
RESULTS_DIR = f"{BASE_DIR}/results/steering"
# Layer 25
PRIMARY_LAYER = 25

# 환경 변수 강제 설정
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

def load_sae(layer_idx, sae_dir=SAE_DIR):
    sae_path = Path(sae_dir) / f"layer{layer_idx}.sae.pt"
    sae = torch.load(sae_path, map_location="cpu")
    return sae

def format_pubmedqa_question(tokenizer, item):
    """
    PubMedQA를 yes/no 질문 형식으로 포맷 (Non-thinking mode)
    """
    context = item["context"]["contexts"]
    context_text = " ".join(context[:3])
    context_text = context_text[:1000] # 길이 제한
    
    question = item["question"]
    
    prompt = f"""Context: {context_text}

Question: {question}

Based on the context above, answer with just "yes" or "no"."""
    
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

def get_yes_no_probabilities(model, tokenizer, formatted_prompt):
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    probs = torch.softmax(logits, dim=-1)
    
    # "yes", "no", "Yes", "No" 토큰 ID 합산
    yes_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["yes", "Yes"]]
    no_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["no", "No"]]
    
    yes_prob = sum(probs[tid].item() for tid in yes_ids)
    no_prob = sum(probs[tid].item() for tid in no_ids)
    
    total = yes_prob + no_prob
    if total > 0:
        return {"yes": yes_prob / total, "no": no_prob / total}
    return {"yes": 0.5, "no": 0.5}

def steer_hook(module, input, output, sae, feature_indices, magnitude):
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
    
    f_tensor = torch.tensor(feature_indices, dtype=torch.long, device=device)
    features[:, :, f_tensor] *= (1 + magnitude)
    
    modified = features @ W_dec
    if b_dec is not None: modified = modified + b_dec.float().to(device)
    
    modified = modified.to(hidden.dtype)
    return (modified,) + rest if rest is not None else modified

def main():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    print("Loading PubMedQA dataset...")
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    test_data = pubmedqa["train"] # PubMedQA는 labeled 데이터가 train에 있음

    no_cases = [item for item in test_data if item["final_decision"] == "no"]
    print(f"'no' 케이스 수: {len(no_cases)}")

    # Step 1: Ignorance 케이스 식별 (no인데 yes라고 강하게 답하는 경우)
    print("\nStep 1: Ignorance 케이스 식별 중...")
    ignorance_results = []
    for item in tqdm(no_cases[:100]):
        formatted = format_pubmedqa_question(tokenizer, item)
        probs = get_yes_no_probabilities(model, tokenizer, formatted)
        if probs["yes"] > 0.70:
            ignorance_results.append({
                "item": item,
                "orig_probs": probs
            })
    
    print(f"발견된 Ignorance 케이스: {len(ignorance_results)}개")

    if not ignorance_results:
        print("Ignorance 케이스가 없습니다. 데이터 범위를 늘리세요.")
        return

    # Step 2: Steering 실험
    sae = load_sae(PRIMARY_LAYER)
    with open(f"{BASE_DIR}/results/features/ignorance_feature_candidates.json") as f:
        candidates = json.load(f)
    top_features = candidates[str(PRIMARY_LAYER)]["top_feature_indices"][:20]

    final_results = {}
    for magnitude in [0.5, 1.0, 2.0]:
        print(f"\nMagnitude: {magnitude}")
        steer_count = 0
        moved_to_no = 0
        
        for entry in tqdm(ignorance_results):
            item = entry["item"]
            formatted = format_pubmedqa_question(tokenizer, item)
            
            handle = model.model.layers[PRIMARY_LAYER].register_forward_hook(
                lambda mod, inp, out, f=top_features, m=magnitude: steer_hook(mod, inp, out, sae, f, m)
            )
            steer_probs = get_yes_no_probabilities(model, tokenizer, formatted)
            handle.remove()
            
            if steer_probs["no"] > entry["orig_probs"]["no"]:
                moved_to_no += 1
            steer_count += 1
            
        print(f"  No 방향 이동: {moved_to_no}/{steer_count} ({moved_to_no/steer_count:.2%})")
        final_results[str(magnitude)] = {"moved_to_no": moved_to_no, "total": steer_count}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/pubmedqa_steering_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()
