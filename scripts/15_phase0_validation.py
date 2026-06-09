import torch
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import gc

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
    "gemma2-9b": f"{BASE_DIR}/checkpoints/model/gemma2-9b",
}
RESULTS_DIR = f"{BASE_DIR}/results/eval"

# 환경 변수 강제 설정
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

def format_pubmedqa(item, tokenizer, include_context=True):
    """
    PubMedQA를 yes/no/maybe 질문으로 포맷
    """
    context_data = item.get("context", "")
    if isinstance(context_data, dict):
        contexts = context_data.get("contexts", [])
        context_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    else:
        context_text = str(context_data)

    if len(context_text) > 1500:
        context_text = context_text[:1500] + "..."
    
    question = item["question"]
    
    if include_context:
        prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nBased ONLY on the context above, answer with one word: yes, no, or maybe."
    else:
        prompt = f"Question: {question}\n\nAnswer with one word: yes, no, or maybe."
    
    messages = [{"role": "user", "content": prompt}]
    
    # Qwen3: enable_thinking=False / Gemma는 해당 옵션 없음
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        # Gemma 등 enable_thinking 미지원 모델
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

def get_ynm_probabilities(model, tokenizer, prompt):
    """yes/no/maybe 토큰 확률 반환"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    result = {}
    for word in ["yes", "no", "maybe"]:
        # 소문자/대문자/공백 포함 첫 토큰 모두 고려
        ids = []
        for variant in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if tok:
                ids.append(tok[0])
        
        if ids:
            # 여러 토큰 중 최대 확률 사용
            result[word] = max(probs[i].item() for i in ids)
        else:
            result[word] = 0.0
    
    total = sum(result.values())
    if total > 0:
        result = {k: v/total for k, v in result.items()}
    else:
        result = {"yes": 0.33, "no": 0.33, "maybe": 0.33}
    return result

def evaluate_model(model, tokenizer, data, include_context=True):
    results = []
    print(f"Evaluating (Include Context: {include_context})...")
    for item in tqdm(data):
        prompt = format_pubmedqa(item, tokenizer, include_context)
        probs = get_ynm_probabilities(model, tokenizer, prompt)
        pred = max(probs, key=probs.get)
        confidence = probs[pred]
        gt = item["final_decision"]
        
        results.append({
            "prediction": pred,
            "ground_truth": gt,
            "is_correct": pred == gt,
            "confidence": confidence,
            "probs": probs,
        })
    
    accuracy = sum(r["is_correct"] for r in results) / len(results)
    return accuracy, results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Specific models to validate (e.g., qwen3-8b)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading PubMedQA dataset...")
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    data = pubmedqa["train"]
    
    summary = {}
    
    # 선택된 모델이 있으면 그들만, 없으면 전체 수행
    target_models = args.models if args.models else MODELS.keys()
    
    for name in target_models:
        if name not in MODELS:
            print(f"Unknown model: {name}. Skipping.")
            continue
            
        path = MODELS[name]
        if not os.path.exists(path):
            print(f"Skipping {name}: Path not found at {path}")
            continue
            
        print(f"\n{'='*20} Validating Model: {name} {'='*20}")
        
        # 모델 로드
        print(f"Loading {name}...")
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        
        # 1. With Context 평가 수행 (이제 이것만 수행)
        acc_with, res_with = evaluate_model(model, tokenizer, data, include_context=True)
        
        # 2. 'no' 케이스 집중 분석 (Ignorance 분석)
        no_cases = [r for r in res_with if r["ground_truth"] == "no"]
        ignorance_cases = [r for r in no_cases if r["prediction"] == "yes"]
        high_conf_ignorance = [r for r in ignorance_cases if r["confidence"] >= 0.70]
        
        summary[name] = {
            "overall_accuracy": acc_with,
            "no_case_count": len(no_cases),
            "ignorance_rate_in_no": len(ignorance_cases) / len(no_cases) if no_cases else 0,
            "high_conf_ignorance_count": len(high_conf_ignorance),
            "avg_ignorance_confidence": sum(r["confidence"] for r in ignorance_cases)/len(ignorance_cases) if ignorance_cases else 0,
        }
        
        # 결과 저장
        with open(f"{RESULTS_DIR}/{name}_validation_results.json", "w") as f:
            json.dump(res_with, f, indent=2)
            
        print(f"Result for {name}:")
        print(f"  Overall Accuracy: {acc_with:.2%}")
        print(f"  Ignorance Rate (in 'no' cases): {summary[name]['ignorance_rate_in_no']:.2%}")
        print(f"  High-Conf Ignorance Count: {len(high_conf_ignorance)}")
        
        # 메모리 해제
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Memory cleared after {name}")

    # 최종 요약 저장 및 출력
    with open(f"{RESULTS_DIR}/phase0_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*50)
    print("PHASE 0 FINAL SUMMARY")
    print("="*50)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
