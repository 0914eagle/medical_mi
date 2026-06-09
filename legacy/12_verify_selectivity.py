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
PRIMARY_LAYER = 25

def load_sae(layer_idx, sae_dir=SAE_DIR):
    sae_path = Path(sae_dir) / f"layer{layer_idx}.sae.pt"
    return torch.load(sae_path, map_location="cpu")

def format_pubmedqa_question(tokenizer, item):
    context = " ".join(item["context"]["contexts"][:3])[:1000]
    prompt = f"Context: {context}\n\nQuestion: {item['question']}\n\nBased on the context above, answer with just \"yes\" or \"no\"."
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

def get_yes_no_probabilities(model, tokenizer, formatted_prompt):
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits[0, -1, :], dim=-1)
    yes_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["yes", "Yes"]]
    no_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in ["no", "No"]]
    y_p = sum(probs[tid].item() for tid in yes_ids)
    n_p = sum(probs[tid].item() for tid in no_ids)
    return {"yes": y_p / (y_p + n_p), "no": n_p / (y_p + n_p)} if (y_p + n_p) > 0 else {"yes": 0.5, "no": 0.5}

def steer_hook(module, input, output, sae, feature_indices, magnitude):
    hidden = output[0] if isinstance(output, tuple) else output
    device = hidden.device
    W_enc = sae.get("W_enc", sae.get("encoder.weight")).float().to(device)
    b_enc = sae.get("b_enc", sae.get("encoder.bias")).float().to(device)
    W_dec = sae.get("W_dec", sae.get("decoder.weight")).float().to(device)
    b_dec = sae.get("b_dec", sae.get("decoder.bias", None))
    if W_enc.shape == (65536, 4096): W_enc = W_enc.t().contiguous()
    if W_dec.shape == (4096, 65536): W_dec = W_dec.t().contiguous()
    h_f = hidden.float()
    if b_dec is not None: h_f = h_f - b_dec.float().to(device)
    pre_act = h_f @ W_enc + b_enc
    vals, idxs = torch.topk(pre_act, 50, dim=-1)
    features = torch.zeros_like(pre_act).scatter_(-1, idxs, torch.relu(vals))
    features[:, :, torch.tensor(feature_indices, device=device)] *= (1 + magnitude)
    mod = features @ W_dec
    if b_dec is not None: mod = mod + b_dec.float().to(device)
    return (mod.to(hidden.dtype),) + output[1:] if isinstance(output, tuple) else mod.to(hidden.dtype)

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")
    
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    # 정답이 'yes'인 케이스 필터링
    yes_cases = [item for item in pubmedqa if item["final_decision"] == "yes"]
    
    print("\n--- Step: 'yes' 케이스 Selectivity 검증 ---")
    correct_yes = []
    for item in tqdm(yes_cases[:100]):
        probs = get_yes_no_probabilities(model, tokenizer, format_pubmedqa_question(tokenizer, item))
        if probs["yes"] > 0.70:
            correct_yes.append(item)
    
    print(f"검증 대상 (Correct Yes): {len(correct_yes)}개")
    sae = load_sae(PRIMARY_LAYER)
    with open(f"{BASE_DIR}/results/features/ignorance_feature_candidates.json") as f:
        candidates = json.load(f)
    top_features = candidates[str(PRIMARY_LAYER)]["top_feature_indices"][:20]

    for mag in [0.5, 2.0]:
        moved_to_no = 0
        for item in tqdm(correct_yes[:30]):
            handle = model.model.layers[PRIMARY_LAYER].register_forward_hook(lambda m, i, o, f=top_features, mg=mag: steer_hook(m, i, o, sae, f, mg))
            steer_p = get_yes_no_probabilities(model, tokenizer, format_pubmedqa_question(tokenizer, item))
            handle.remove()
            if steer_p["no"] > 0.5: moved_to_no += 1
        print(f"Magnitude {mag}: 정답(Yes)이 No로 오염된 비율: {moved_to_no}/{min(30, len(correct_yes))} ({moved_to_no/min(30, len(correct_yes)):.2%})")

if __name__ == "__main__":
    main()
