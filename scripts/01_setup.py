from huggingface_hub import snapshot_download
import os
import torch
from transformers import AutoTokenizer

def setup():
    os.makedirs("checkpoints/model", exist_ok=True)
    os.makedirs("checkpoints/sae", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("results/features", exist_ok=True)
    os.makedirs("results/steering", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Qwen3-8B 다운로드
    # Note: If Qwen3-8B is not available, this might fail. 
    # The instructions specifically mentioned Qwen3-8B.
    try:
        print("Downloading Qwen3-8B...")
        snapshot_download(
            repo_id="Qwen/Qwen3-8B",
            local_dir="checkpoints/model/Qwen3-8B",
            ignore_patterns=["*.msgpack", "flax_model*"]
        )
        print("Qwen3-8B 다운로드 완료")
    except Exception as e:
        print(f"Qwen3-8B 다운로드 실패: {e}")

    # Qwen-Scope SAE 다운로드
    try:
        print("Downloading Qwen-Scope SAE...")
        snapshot_download(
            repo_id="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
            local_dir="checkpoints/sae/Qwen3-8B-SAE",
        )
        print("Qwen-Scope SAE 다운로드 완료")
    except Exception as e:
        print(f"Qwen-Scope SAE 다운로드 실패: {e}")

def check_keys_and_tokens():
    print("\n--- Verification ---")
    
    # 1. Check SAE keys
    sae_path = "checkpoints/sae/Qwen3-8B-SAE/layer20.sae.pt"
    if os.path.exists(sae_path):
        try:
            sae = torch.load(sae_path, map_location="cpu")
            print(f"SAE keys in {sae_path}: {list(sae.keys())}")
        except Exception as e:
            print(f"Error loading SAE: {e}")
    else:
        print(f"SAE file not found: {sae_path}")

    # 2. Check Token IDs
    model_path = "checkpoints/model/Qwen3-8B"
    if os.path.exists(model_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            for choice in ["A", "B", "C", "D"]:
                token_ids = tokenizer.encode(choice, add_special_tokens=False)
                print(f"Token '{choice}': {token_ids} (length: {len(token_ids)})")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
    else:
        print(f"Model path not found: {model_path}")

if __name__ == "__main__":
    setup()
    check_keys_and_tokens()
