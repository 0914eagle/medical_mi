import torch
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_wrapper import SAEWrapper
import gc

# --- Config ---
BASE_DIR = "/workspace/medical_mi"
MODELS = {
    "qwen3-8b": {"path": f"{BASE_DIR}/checkpoints/model/qwen3-8b", "suite": "qwen"},
    # Qwen3.5와 Gemma도 필요시 추가
}

def get_residual_activation(model, tokenizer, prompt, layer_idx):
    store = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        store["act"] = h[0, -1, :].detach().cpu()
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return store["act"]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Specific models to process")
    args = parser.parse_args()

    # PubMedQA를 말뭉치(Corpus)로 사용
    data = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    corpus = [item["question"] + " " + " ".join(item["context"]["contexts"]) for item in data]
    
    target_models = args.models if args.models else MODELS.keys()

    for model_name in target_models:
        if model_name not in MODELS: continue
        config = MODELS[model_name]
        results_path = f"{BASE_DIR}/results/features/{model_name}_phase1_results.json"
        if not os.path.exists(results_path): continue
        
        with open(results_path) as f:
            phase1_data = json.load(f)
            
        print(f"\n--- Phase 2: Interpreting Features for {model_name} ---")
        tokenizer = AutoTokenizer.from_pretrained(config["path"])
        model = AutoModelForCausalLM.from_pretrained(config["path"], torch_dtype=torch.float16, device_map="auto")
        model.eval()

        interpretation_results = {}

        # 각 레이어별 상위 feature (p-value가 가장 낮은 것들) 분석
        for layer_idx, layer_data in phase1_data.items():
            top_indices = layer_data["top_indices"][:5] # 상위 5개만 샘플링 해석
            if not top_indices: continue
            
            print(f"Interpreting Layer {layer_idx}, Top 5 features...")
            sae_path = f"{BASE_DIR}/checkpoints/sae/{model_name}/layer{layer_idx}.sae.pt"
            sae = SAEWrapper(torch.load(sae_path, map_location="cpu"), suite=config["suite"])
            
            layer_interpretations = []
            for feat_idx in top_indices:
                print(f"  Feature #{feat_idx}...")
                scores = []
                # 전체 코퍼스에서 Max-activating examples 찾기
                for text in tqdm(corpus[:200], leave=False): # 시간상 200개만 스캔
                    act = get_residual_activation(model, tokenizer, text[:1000], int(layer_idx))
                    feat_val = sae.encode(act.unsqueeze(0))[0, feat_idx].item()
                    scores.append((text[:200], feat_val))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                layer_interpretations.append({
                    "feature_idx": feat_idx,
                    "max_activating": scores[:10] # Top 10 examples
                })
            
            interpretation_results[layer_idx] = layer_interpretations

        with open(f"{BASE_DIR}/results/features/{model_name}_phase2_interpretations.json", "w") as f:
            json.dump(interpretation_results, f, indent=2)
            
        del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
