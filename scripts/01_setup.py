import os
# 스크립트 실행 시점에 환경 변수 강제 설정 (가장 확실한 방법)
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/.tmp"

from huggingface_hub import snapshot_download
from datasets import load_dataset
import torch
from transformers import AutoTokenizer

def setup():
    # 모든 경로를 /workspace 기준으로 절대 경로 설정
    base_dir = "/workspace/medical_mi"
    os.makedirs(f"{base_dir}/checkpoints/model", exist_ok=True)
    os.makedirs(f"{base_dir}/checkpoints/sae", exist_ok=True)
    os.makedirs(f"{base_dir}/data/raw", exist_ok=True)
    os.makedirs(f"{base_dir}/data/processed", exist_ok=True)
    os.makedirs(f"{base_dir}/results/features", exist_ok=True)
    os.makedirs(f"{base_dir}/results/steering", exist_ok=True)
    os.makedirs(f"{base_dir}/results/figures", exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    # 1. Qwen3-8B 다운로드
    try:
        print("Downloading Qwen3-8B...")
        snapshot_download(
            repo_id="Qwen/Qwen3-8B",
            local_dir=f"{base_dir}/checkpoints/model/Qwen3-8B",
            local_dir_use_symlinks=False,  # 중복 저장 방지
            ignore_patterns=["*.msgpack", "flax_model*"]
        )
        print("Qwen3-8B 다운로드 완료")
    except Exception as e:
        print(f"Qwen3-8B 다운로드 실패: {e}")

    # 2. Qwen-Scope SAE 다운로드
    try:
        print("Downloading Qwen-Scope SAE...")
        snapshot_download(
            repo_id="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
            local_dir=f"{base_dir}/checkpoints/sae/Qwen3-8B-SAE",
            local_dir_use_symlinks=False,  # 중복 저장 방지
        )
        print("Qwen-Scope SAE 다운로드 완료")
    except Exception as e:
        print(f"Qwen-Scope SAE 다운로드 실패: {e}")

    # 3. MedQA 데이터셋 다운로드
    try:
        print("Downloading MedQA-USMLE-4-options dataset...")
        # cache_dir를 명시적으로 지정하여 /workspace에 저장되도록 함
        dataset = load_dataset(
            "GBaker/MedQA-USMLE-4-options",
            cache_dir=f"{os.environ['HF_HOME']}/datasets"
        )
        print(f"MedQA 데이터셋 다운로드 완료 (Test split size: {len(dataset['test'])})")
    except Exception as e:
        print(f"MedQA 데이터셋 다운로드 실패: {e}")

def check_keys_and_tokens():
    print("\n--- Verification ---")
    base_dir = "/workspace/medical_mi"
    
    # 1. Check SAE keys
    sae_path = f"{base_dir}/checkpoints/sae/Qwen3-8B-SAE/layer20.sae.pt"
    if os.path.exists(sae_path):
        try:
            sae = torch.load(sae_path, map_location="cpu")
            print(f"SAE keys in {sae_path}: {list(sae.keys())}")
        except Exception as e:
            print(f"Error loading SAE: {e}")
    else:
        print(f"SAE file not found: {sae_path}")

    # 2. Check Token IDs
    model_path = f"{base_dir}/checkpoints/model/Qwen3-8B"
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

if __name__ == "__main__":
    setup()
    check_keys_and_tokens()
