import os
import torch
import argparse
from huggingface_hub import snapshot_download
from datasets import load_dataset

BASE_DIR = os.environ.get("MEDICAL_MI_BASE_DIR", "/home/eagle0914/medical_mi")
DATA_ROOT = os.environ.get("MEDICAL_MI_DATA_ROOT", "/data/heejae")
os.environ.setdefault("HF_HOME", f"{DATA_ROOT}/.cache/huggingface")
os.environ.setdefault("TMPDIR", f"{DATA_ROOT}/tmp")

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

def setup(models):
    base_dir = BASE_DIR
    checkpoint_dir = f"{DATA_ROOT}/checkpoints"
    os.makedirs(f"{checkpoint_dir}/model", exist_ok=True)
    os.makedirs(f"{checkpoint_dir}/sae", exist_ok=True)
    os.makedirs(f"{base_dir}/data/raw", exist_ok=True)
    os.makedirs(f"{base_dir}/results/eval", exist_ok=True)
    os.makedirs(f"{base_dir}/results/features", exist_ok=True)
    os.makedirs(f"{base_dir}/results/steering", exist_ok=True)
    os.makedirs(f"{base_dir}/results/figures", exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)

    selected_models = models or ["qwen3.5-9b"]

    # 1. 모델 다운로드
    for name in selected_models:
        repo = MODELS[name]
        try:
            print(f"\nDownloading model: {name} ({repo})...")
            snapshot_download(
                repo_id=repo,
                local_dir=f"{checkpoint_dir}/model/{name}",
                local_dir_use_symlinks=False,
                ignore_patterns=["*.msgpack", "flax_model*", "*.h5", "*.tflite", "*.onnx"]
            )
        except Exception as e:
            print(f"Model {name} 다운로드 실패: {e}")

    # 2. SAE 다운로드
    for name in selected_models:
        repo = SAE_REPOS[name]
        try:
            print(f"\nDownloading SAE suite: {name} ({repo})...")
            # Gemma Scope 2는 매우 크므로 resid_post 계열만 우선 시도하거나 전체 다운로드
            snapshot_download(
                repo_id=repo,
                local_dir=f"{checkpoint_dir}/sae/{name}",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["qwen3.5-9b"], choices=sorted(MODELS))
    args = parser.parse_args()
    setup(args.models)
