import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODEL_PATH = f"{BASE_DIR}/checkpoints/model/Qwen3-8B"
SAE_DIR = f"{BASE_DIR}/checkpoints/sae/Qwen3-8B-SAE"
RESULTS_DIR = f"{BASE_DIR}/results/features"
TARGET_LAYERS = [15, 20, 25]
MAX_CASES = 100

# 환경 변수 강제 설정
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

def load_sae(layer_idx, sae_dir=SAE_DIR):
    sae_path = Path(sae_dir) / f"layer{layer_idx}.sae.pt"
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")
    
    sae = torch.load(sae_path, map_location="cpu")
    print(f"Layer {layer_idx} SAE loaded. Keys: {list(sae.keys())}")
    return sae

def get_residual_stream_activation(model, tokenizer, text, layer_idx):
    activation_store = {}
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # 마지막 토큰(정답 직전)의 activation 추출
        activation_store["activation"] = hidden[0, -1, :].detach().cpu()
    
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    return activation_store["activation"]

def activation_to_sae_features(activation, sae):
    # SAE 가중치 매핑 및 Shape 맞춤
    W_enc = sae.get("W_enc", sae.get("encoder.weight")).float() # [d_sae, d_model] 또는 [d_model, d_sae]
    b_enc = sae.get("b_enc", sae.get("encoder.bias")).float()
    b_dec = sae.get("b_dec", sae.get("decoder.bias", None))

    # Qwen-Scope SAE는 보통 [d_sae, d_model] 형태일 수 있음. 
    # 행렬 곱셈을 위해 [d_model, d_sae]로 변환
    if W_enc.shape[0] != 4096 and W_enc.shape[1] == 4096:
        W_enc = W_enc.T

    activation = activation.float()
    if b_dec is not None:
        activation = activation - b_dec.float()
    
    # [4096] @ [4096, 65536] + [65536] -> [65536]
    pre_activations = activation @ W_enc + b_enc
    
    # TopK activation (k=50)
    k = 50
    topk_values, topk_indices = torch.topk(pre_activations, k)
    topk_values = torch.relu(topk_values)
    
    features = torch.zeros(W_enc.shape[1])
    features[topk_indices] = topk_values
    return features

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

def extract_features_for_cases(model, tokenizer, cases, sae, layer_idx, max_cases=MAX_CASES):
    features_list = []
    # 데이터가 부족할 경우를 대비해 min 처리
    n_to_extract = min(len(cases), max_cases)
    for case in tqdm(cases[:n_to_extract], desc=f"Extracting features Layer {layer_idx}"):
        formatted = format_question_for_qwen3(tokenizer, case["question"], case["options"])
        activation = get_residual_stream_activation(model, tokenizer, formatted, layer_idx)
        features = activation_to_sae_features(activation, sae)
        features_list.append(features)
    
    if not features_list:
        return torch.empty(0)
    return torch.stack(features_list)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    
    # Load cases
    with open(f"{BASE_DIR}/data/processed/correct_confident.json") as f:
        correct_cases = json.load(f)
    with open(f"{BASE_DIR}/data/processed/wrong_confident.json") as f:
        ignorance_cases = json.load(f)
        
    for layer_idx in TARGET_LAYERS:
        print(f"\nProcessing Layer {layer_idx}...")
        sae = load_sae(layer_idx)
        
        correct_features = extract_features_for_cases(model, tokenizer, correct_cases, sae, layer_idx)
        ignorance_features = extract_features_for_cases(model, tokenizer, ignorance_cases, sae, layer_idx)
        
        if correct_features.nelement() > 0:
            torch.save(correct_features, f"{RESULTS_DIR}/correct_confident_layer{layer_idx}.pt")
        if ignorance_features.nelement() > 0:
            torch.save(ignorance_features, f"{RESULTS_DIR}/wrong_confident_layer{layer_idx}.pt")
            
        print(f"Layer {layer_idx} features saved.")

if __name__ == "__main__":
    main()
