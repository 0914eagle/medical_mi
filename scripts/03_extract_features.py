import torch
import json
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config ---
MODEL_PATH = "checkpoints/model/Qwen3-8B"
SAE_DIR = "checkpoints/sae/Qwen3-8B-SAE"
RESULTS_DIR = "results/features"
TARGET_LAYERS = [15, 20, 25]
MAX_CASES = 100

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
    # Flexible key mapping for SAE weights
    # common keys: W_enc, b_enc, W_dec, b_dec
    # some implementations might use: encoder.weight, encoder.bias, etc.
    W_enc = sae.get("W_enc", sae.get("encoder.weight")).float()
    b_enc = sae.get("b_enc", sae.get("encoder.bias")).float()
    b_dec = sae.get("b_dec", sae.get("decoder.bias", None))

    if b_dec is not None:
        activation = activation.float() - b_dec.float()
    
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
    prompt = f"{question_text}\n\nOptions:\n{options_text}\n\nAnswer with just the letter (A, B, C, or D). /no_think"
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def extract_features_for_cases(model, tokenizer, cases, sae, layer_idx, max_cases=MAX_CASES):
    features_list = []
    for case in tqdm(cases[:max_cases], desc=f"Extracting features Layer {layer_idx}"):
        formatted = format_question_for_qwen3(tokenizer, case["question"], case["options"])
        activation = get_residual_stream_activation(model, tokenizer, formatted, layer_idx)
        features = activation_to_sae_features(activation, sae)
        features_list.append(features)
    return torch.stack(features_list)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    
    # Load cases
    with open("data/processed/correct_confident.json") as f:
        correct_cases = json.load(f)
    with open("data/processed/wrong_confident.json") as f:
        ignorance_cases = json.load(f)
        
    for layer_idx in TARGET_LAYERS:
        print(f"\nProcessing Layer {layer_idx}...")
        sae = load_sae(layer_idx)
        
        correct_features = extract_features_for_cases(model, tokenizer, correct_cases, sae, layer_idx)
        ignorance_features = extract_features_for_cases(model, tokenizer, ignorance_cases, sae, layer_idx)
        
        torch.save(correct_features, f"{RESULTS_DIR}/correct_confident_layer{layer_idx}.pt")
        torch.save(ignorance_features, f"{RESULTS_DIR}/wrong_confident_layer{layer_idx}.pt")
        print(f"Layer {layer_idx} features saved.")

if __name__ == "__main__":
    main()
