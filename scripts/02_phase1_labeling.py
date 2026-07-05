import torch
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import format_pubmedqa, get_ynm_probs
import gc

# --- Config ---
BASE_DIR = "/home/eagle0914/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
}
RESULTS_DIR = f"{BASE_DIR}/results/eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_evaluation(model, tokenizer, dataset):
    results = []
    print("Running evaluation on PubMedQA...")
    for item in tqdm(dataset):
        # Always include context in simplified version
        prompt = format_pubmedqa(item, tokenizer, include_context=True)
        probs = get_ynm_probs(model, tokenizer, prompt)
        pred = max(probs, key=probs.get)
        
        results.append({
            "item_id": item.get("pubid", "unknown"),
            "ground_truth": item["final_decision"],
            "prediction": pred,
            "probs": probs,
            "is_correct": pred == item["final_decision"]
        })
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    name = args.model
    path = MODELS[name]
    
    print(f"--- Phase 1: Labeling for {name} ---")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    results = run_evaluation(model, tokenizer, dataset)

    # Statistics
    correct_count = sum(1 for r in results if r["is_correct"])
    total = len(results)
    print(f"Accuracy: {correct_count/total:.2%} ({correct_count}/{total})")

    output_path = f"{RESULTS_DIR}/{name}_labels.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Labels saved to {output_path}")

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
