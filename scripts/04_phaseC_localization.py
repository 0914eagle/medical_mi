import torch
import json
import os
import gc
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import format_pubmedqa, get_activation_with_hook, get_ynm_probs

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
    "gemma3-12b-it": f"{BASE_DIR}/checkpoints/model/gemma3-12b-it",
}
RESULTS_DIR = f"{BASE_DIR}/results/eval"
os.makedirs(RESULTS_DIR, exist_ok=True)

def patch_and_test(model, tokenizer, ignored_item, donor_activation, patch_layer):
    """
    IGNORED 케이스를 실행하다가 특정 레이어에서 donor의 activation을 주입
    """
    prompt = format_pubmedqa(ignored_item, tokenizer, include_context=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    def patch_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h[0, -1, :] = donor_activation.to(h.device)
        return output

    handle = model.model.layers[patch_layer].register_forward_hook(patch_hook)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
    handle.remove()
    
    # Process probs
    result = {}
    for word in ["yes", "no", "maybe"]:
        ids = []
        for variant in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if tok: ids.append(tok[0])
        if ids: result[word] = max(probs[i].item() for i in ids)
        else: result[word] = 0.0
    
    total = sum(result.values())
    if total > 0: result = {k: v/total for k, v in result.items()}
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3.5-9b")
    args = parser.parse_args()

    model_name = args.model
    conflict_set_path = f"{RESULTS_DIR}/{model_name}_conflict_set.json"
    
    if not os.path.exists(conflict_set_path):
        print("Conflict set not found.")
        return

    with open(conflict_set_path, "r") as f:
        full_data = json.load(f)

    print("Loading PubMedQA...")
    pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    item_map = {item["pubid"]: item for item in pubmedqa}

    # "같은 질문의 INTEGRATED 버전"이 donor가 되어야 함. 
    # 하지만 IGNORED와 INTEGRATED는 서로 다른 질문들임.
    # 사용자 힌트: "같은 질문의 context 有/無 patching"
    # -> IGNORED 질문에 대해, '만약 context를 제대로 반영했다면 나왔을 activation'을 주입하는 것.
    # 여기서는 '동일한 질문'에 대해 INTEGRATED activation이 없으므로, 
    # '동일한 질문'의 context 유무 차이를 이용하거나, 
    # 설계안대로 INTEGRATED 케이스들의 activation을 donor로 사용하되, 
    # 사용자 지적대로 '평균'보다는 '개별 매칭'이 좋지만 데이터가 없으므로 
    # 일단 '대표적인 INTEGRATED activation'을 사용하거나, 
    # '동일 질문의 context-only activation'을 시도할 수 있음.
    # MD 설계안 준수: "INTEGRATED 케이스의 activation을 주입"
    
    integrated_cases = [item_map[r["item_id"]] for r in full_data if r["classification"] == "INTEGRATED"]
    ignored_cases = [item_map[r["item_id"]] for r in full_data if r["classification"] == "IGNORED"]

    if not integrated_cases or not ignored_cases:
        print("Insufficient cases.")
        return

    # 모델 로드
    path = MODELS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    num_layers = model.config.num_hidden_layers
    layers_to_patch = range(0, num_layers, 2)
    
    layer_flip_rates = {}

    for layer in tqdm(layers_to_patch, desc="Patching layers"):
        flips = 0
        total = 0
        
        # 각 IGNORED 케이스에 대해 INTEGRATED 케이스 중 하나를 donor로 랜덤/순차 매칭
        # (평균의 함정을 피하기 위해)
        for i, item in enumerate(ignored_cases[:20]):
            donor_item = integrated_cases[i % len(integrated_cases)]
            donor_prompt = format_pubmedqa(donor_item, tokenizer, include_context=True)
            donor_act = get_activation_with_hook(model, tokenizer, donor_prompt, layer)[0] # [d_model]
            
            probs = patch_and_test(model, tokenizer, item, donor_act, layer)
            pred = max(probs, key=probs.get)
            if pred == "no": # Ground truth for these cases
                flips += 1
            total += 1
            
        layer_flip_rates[layer] = flips / total
        print(f"Layer {layer}: Flip Rate = {layer_flip_rates[layer]:.2%}")

    # 결과 저장
    with open(f"{RESULTS_DIR}/{model_name}_localization.json", "w") as f:
        json.dump(layer_flip_rates, f, indent=2)
    print(f"Results saved.")

if __name__ == "__main__":
    main()
