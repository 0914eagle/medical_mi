import os
import torch
from huggingface_hub import snapshot_download
from datasets import load_dataset

# 환경 변수 강제 설정
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/.tmp"

def setup():
    base_dir = "/workspace/medical_mi"
    os.makedirs(f"{base_dir}/checkpoints/model", exist_ok=True)
    os.makedirs(f"{base_dir}/checkpoints/sae", exist_ok=True)
    os.makedirs(f"{base_dir}/data/raw", exist_ok=True)
    os.makedirs(f"{base_dir}/results/eval", exist_ok=True)
    os.makedirs(f"{base_dir}/results/features", exist_ok=True)
    os.makedirs(f"{base_dir}/results/steering", exist_ok=True)
    os.makedirs(f"{base_dir}/results/figures", exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    MODELS = {
        "qwen3-8b": "Qwen/Qwen3-8B",
        "qwen3.5-9b": "Qwen/Qwen3.5-9B",
        "gemma3-12b-it": "google/gemma-3-12b-it",
    }

    SAE_REPOS = {
        "qwen3-8b": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
        "qwen3.5-9b": "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",
        "gemma3-12b-it": "google/gemma-scope-2-12b-it",
    }

    # 1. 모델 다운로드
    for name, repo in MODELS.items():
        try:
            print(f"\nDownloading model: {name} ({repo})...")
            snapshot_download(
                repo_id=repo,
                local_dir=f"{base_dir}/checkpoints/model/{name}",
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "flax_model*", "*.h5", "*.tflite", "*.onnx"]
            )
        except Exception as e:
            print(f"Model {name} 다운로드 실패: {e}")

    # 2. SAE 다운로드
    for name, repo in SAE_REPOS.items():
        try:
            print(f"\nDownloading SAE suite: {name} ({repo})...")
            # Gemma Scope 2는 매우 크므로 resid_post 계열만 우선 시도하거나 전체 다운로드
            snapshot_download(
                repo_id=repo,
                local_dir=f"{base_dir}/checkpoints/sae/{name}",
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            print(f"SAE {name} 다운로드 실패: {e}")

    # 3. 데이터셋 다운로드
    try:
        print("\nDownloading PubMedQA...")
        load_dataset("qiaojin/PubMedQA", "pqa_labeled", cache_dir=f"{os.environ['HF_HOME']}/datasets")
        print("PubMedQA 완료")
        
        print("\nDownloading MedAbstain from GitHub...")
        med_abstain_path = f"{base_dir}/data/raw/MedAbstain"
        if not os.path.exists(med_abstain_path):
            os.system(f"git clone https://github.com/sravanthi6m/MedAbstain.git {med_abstain_path}")
            print("MedAbstain 다운로드 완료")
        else:
            print("MedAbstain이 이미 존재합니다.")
            
    except Exception as e:
        print(f"데이터셋 다운로드 실패: {e}")

if __name__ == "__main__":
    setup()
