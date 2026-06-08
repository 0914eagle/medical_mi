import os
# 스크립트 실행 시점에 환경 변수 강제 설정
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["TMPDIR"] = "/workspace/.tmp"

from huggingface_hub import snapshot_download
from datasets import load_dataset
import torch

def setup():
    # 모든 경로를 /workspace 기준으로 절대 경로 설정
    base_dir = "/workspace/medical_mi"
    os.makedirs(f"{base_dir}/checkpoints/model", exist_ok=True)
    os.makedirs(f"{base_dir}/checkpoints/sae", exist_ok=True)
    os.makedirs(f"{base_dir}/data/raw", exist_ok=True)
    os.makedirs(f"{base_dir}/results/eval", exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    MODELS = {
        "qwen3-8b": "Qwen/Qwen3-8B",
        "qwen3.5-9b": "Qwen/Qwen3.5-9B",
        "gemma2-9b": "google/gemma-2-9b",
    }

    SAE_REPOS = {
        "qwen3-8b": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
        "qwen3.5-9b": "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",
        "gemma2-9b": "google/gemma-scope-9b-pt-res",
    }

    # 1. 모델 다운로드
    for name, repo in MODELS.items():
        try:
            print(f"Downloading model: {name} ({repo})...")
            snapshot_download(
                repo_id=repo,
                local_dir=f"{base_dir}/checkpoints/model/{name}",
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "flax_model*", "*.safetensors.index.json"] # 용량 최적화 (필요시 수정)
            )
            print(f"Model {name} 다운로드 완료")
        except Exception as e:
            print(f"Model {name} 다운로드 실패: {e}")

    # 2. SAE 다운로드
    for name, repo in SAE_REPOS.items():
        try:
            print(f"Downloading SAE: {name} ({repo})...")
            snapshot_download(
                repo_id=repo,
                local_dir=f"{base_dir}/checkpoints/sae/{name}",
                local_dir_use_symlinks=False,
            )
            print(f"SAE {name} 다운로드 완료")
        except Exception as e:
            print(f"SAE {name} 다운로드 실패: {e}")

    # 3. PubMedQA 데이터셋 다운로드
    try:
        print("Downloading PubMedQA (pqa_labeled) dataset...")
        dataset = load_dataset(
            "qiaojin/PubMedQA", "pqa_labeled",
            cache_dir=f"{os.environ['HF_HOME']}/datasets"
        )
        print(f"PubMedQA 데이터셋 다운로드 완료 (Labeled set size: {len(dataset['train'])})")
    except Exception as e:
        print(f"PubMedQA 데이터셋 다운로드 실패: {e}")

if __name__ == "__main__":
    setup()
