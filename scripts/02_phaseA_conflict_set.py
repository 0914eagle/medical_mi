import torch
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import format_pubmedqa, get_ynm_probs
import gc

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
    "gemma3-12b-it": f"{BASE_DIR}/checkpoints/model/gemma3-12b-it",
}
RESULTS_DIR = f"{BASE_DIR}/results/eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 환경 변수 강제 설정
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

def run_both_conditions(model, tokenizer, item):
    # 조건 P: prior only (context 없음)
    prompt_P = format_pubmedqa(item, tokenizer, include_context=False)
    probs_P = get_ynm_probs(model, tokenizer, prompt_P)
    prior_answer = max(probs_P, key=probs_P.get)
    
    # 조건 C: with context
    prompt_C = format_pubmedqa(item, tokenizer, include_context=True)
    probs_C = get_ynm_probs(model, tokenizer, prompt_C)
    context_answer = max(probs_C, key=probs_C.get)
    
    return {
        "item_id": item.get("pubid", "unknown"),
        "ground_truth": item["final_decision"],
        "prior_answer": prior_answer,
        "prior_probs": probs_P,
        "context_answer": context_answer,
        "context_probs": probs_C,
    }

def classify_case(r):
    gt = r["ground_truth"]
    prior = r["prior_answer"]
    ctx = r["context_answer"]
    
    # PubMedQA에서 'no'인 케이스에 집중
    if gt == "no":
        if prior != gt: # prior가 "yes" 또는 "maybe"
            if ctx == gt:
                return "INTEGRATED"   # context 반영 성공 (prior 극복)
            else:
                return "IGNORED"      # context 무시 (prior 유지)
        else:
            return "NO_CONFLICT"
    else:
        if prior != gt:
            if ctx == gt: return "INTEGRATED_OTHER"
            else: return "IGNORED_OTHER"
        else:
            return "NO_CONFLICT_OTHER"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3.5-9b")
    args = parser.parse_args()

    name = args.model
    if name not in MODELS:
        print(f"Unknown model: {name}")
        return

    path = MODELS[name]
    if not os.path.exists(path):
        print(f"Model path not found: {path}")
        return

    print(f"Loading model {name}...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    print("Loading PubMedQA dataset...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]

    results = []
    print("Running Phase A...")
    for item in tqdm(dataset):
        res = run_both_conditions(model, tokenizer, item)
        res["classification"] = classify_case(res)
        results.append(res)

    stats = {}
    for r in results:
        cls = r["classification"]
        stats[cls] = stats.get(cls, 0) + 1

    print("\nPhase A Statistics:")
    print(json.dumps(stats, indent=2))

    output_path = f"{RESULTS_DIR}/{name}_conflict_set.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
