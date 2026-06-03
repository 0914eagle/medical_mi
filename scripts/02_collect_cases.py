import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# --- Constants & Config ---
MODEL_PATH = "checkpoints/model/Qwen3-8B"
DATA_DIR = "data/processed"
N_CASES = 200 # Pilot phase
HIGH_CONF_THRESHOLD = 0.70
LOW_CONF_THRESHOLD = 0.40

# --- Helper Functions ---

def format_question_for_qwen3(tokenizer, question_text, options):
    """
    Qwen3 non-thinking mode format with /no_think
    """
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    prompt = f"""{question_text}

Options:
{options_text}

Answer with just the letter (A, B, C, or D). /no_think"""
    
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted

def get_answer_probabilities(model, tokenizer, formatted_prompt):
    """
    Extract probabilities for A, B, C, D tokens
    """
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    
    answer_probs = {}
    for choice in ["A", "B", "C", "D"]:
        # Verification of token IDs as requested by user
        token_ids = tokenizer.encode(choice, add_special_tokens=False)
        # Using the first token if multiple are returned
        answer_probs[choice] = probs[token_ids[0]].item()
    
    # Normalization
    total = sum(answer_probs.values())
    if total > 0:
        answer_probs = {k: v/total for k, v in answer_probs.items()}
    
    return answer_probs

def evaluate_case(model, tokenizer, case):
    formatted = format_question_for_qwen3(tokenizer, case["question"], case["options"])
    probs = get_answer_probabilities(model, tokenizer, formatted)
    
    model_answer = max(probs, key=probs.get)
    confidence = probs[model_answer]
    correct_answer = case["answer"]
    is_correct = (model_answer == correct_answer)
    
    if is_correct and confidence >= HIGH_CONF_THRESHOLD:
        case_type = "CORRECT_CONFIDENT"
    elif not is_correct and confidence >= HIGH_CONF_THRESHOLD:
        case_type = "WRONG_CONFIDENT"
    elif is_correct and confidence < LOW_CONF_THRESHOLD:
        case_type = "CORRECT_UNCERTAIN"
    else:
        case_type = "WRONG_UNCERTAIN"
    
    return {
        "question": case["question"],
        "options": case["options"],
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "confidence": confidence,
        "all_probs": probs,
        "case_type": case_type
    }

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading model and tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print("Loading MedQA dataset...")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
    test_data = dataset["test"]

    results = []
    print(f"Evaluating {N_CASES} cases...")
    for i in tqdm(range(min(N_CASES, len(test_data)))):
        case = test_data[i]
        try:
            result = evaluate_case(model, tokenizer, case)
            results.append(result)
        except Exception as e:
            print(f"Case {i} error: {e}")

    # Save all evaluated cases
    with open(os.path.join(DATA_DIR, "evaluated_cases.json"), "w", encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save by type
    for case_type in ["CORRECT_CONFIDENT", "WRONG_CONFIDENT", "CORRECT_UNCERTAIN", "WRONG_UNCERTAIN"]:
        subset = [r for r in results if r["case_type"] == case_type]
        with open(os.path.join(DATA_DIR, f"{case_type.lower()}.json"), "w", encoding='utf-8') as f:
            json.dump(subset, f, indent=2, ensure_ascii=False)
        print(f"{case_type}: {len(subset)} cases saved")

if __name__ == "__main__":
    main()
