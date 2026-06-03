import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def debug():
    MODEL_PATH = "checkpoints/model/Qwen3-8B"
    
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
    
    # Check the first 3 cases
    for i in range(3):
        case = test_data[i]
        print(f"\n{'='*20} Debugging Case {i} {'='*20}")
        
        # 1. Check original data format
        print(f"Dataset Answer: '{case['answer']}' (Type: {type(case['answer'])})")
        print(f"Dataset Options: {case['options']}")

        # 2. Format prompt
        options_text = "\n".join([f"{k}. {v}" for k, v in case["options"].items()])
        prompt = f"{case['question']}\n\nOptions:\n{options_text}\n\nAnswer with just the letter (A, B, C, or D). /no_think"
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print(f"Formatted Prompt (last 50 chars): ...{formatted[-50:]}")

        # 3. Get logits
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)

        # 4. Check Top 5 overall tokens
        top_k = torch.topk(probs, 5)
        print("\nTop 5 predicted tokens:")
        for idx in range(5):
            token_str = tokenizer.decode([top_k.indices[idx]])
            token_prob = top_k.values[idx].item()
            print(f"  '{token_str}' (ID: {top_k.indices[idx]}): {token_prob:.4f}")

        # 5. Check A, B, C, D specifically
        print("\nProbabilities for A, B, C, D:")
        answer_probs = {}
        for choice in ["A", "B", "C", "D"]:
            # Check multiple ways 'A' might be encoded
            token_ids = tokenizer.encode(choice, add_special_tokens=False)
            token_ids_with_space = tokenizer.encode(f" {choice}", add_special_tokens=False)
            
            prob_direct = probs[token_ids[0]].item()
            
            print(f"  '{choice}' (ID: {token_ids[0]}): {prob_direct:.4f}")
            if len(token_ids_with_space) > 0:
                prob_space = probs[token_ids_with_space[0]].item()
                print(f"    (with space ID: {token_ids_with_space[0]}): {prob_space:.4f}")
            
            answer_probs[choice] = prob_direct

        # 6. Check normalization logic
        total = sum(answer_probs.values())
        print(f"\nSum of A,B,C,D raw probs: {total:.4f}")
        if total > 0:
            norm_probs = {k: v/total for k, v in answer_probs.items()}
            best_choice = max(norm_probs, key=norm_probs.get)
            print(f"Normalized Probs: {norm_probs}")
            print(f"Predicted Choice: {best_choice} (Prob: {norm_probs[best_choice]:.4f})")
            print(f"Match correct answer? {best_choice == case['answer']}")

if __name__ == "__main__":
    debug()
