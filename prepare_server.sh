#!/bin/bash

# 1. 경로 설정
export MEDICAL_MI_BASE_DIR="${MEDICAL_MI_BASE_DIR:-/home/eagle0914/medical_mi}"
export MEDICAL_MI_DATA_ROOT="${MEDICAL_MI_DATA_ROOT:-/data/heejae}"
export UV_CACHE_DIR="$MEDICAL_MI_DATA_ROOT/.cache/uv"
export HF_HOME="$MEDICAL_MI_DATA_ROOT/.cache/huggingface"
export TMPDIR="$MEDICAL_MI_DATA_ROOT/tmp"
export PYTHON_VENV="$MEDICAL_MI_DATA_ROOT/.venv"
export MEDICAL_MI_SETUP_MODELS="${MEDICAL_MI_SETUP_MODELS:-qwen3.5-9b}"

# 디렉토리 생성
mkdir -p $UV_CACHE_DIR
mkdir -p $HF_HOME
mkdir -p $TMPDIR
mkdir -p "$MEDICAL_MI_DATA_ROOT/checkpoints/model"
mkdir -p "$MEDICAL_MI_DATA_ROOT/checkpoints/sae"

echo "--- 환경 설정 완료 ---"
echo "UV Cache: $UV_CACHE_DIR"
echo "HF Home: $HF_HOME"
echo "TMPDIR: $TMPDIR"
echo "Venv: $PYTHON_VENV"
echo "Setup models: $MEDICAL_MI_SETUP_MODELS"

cd "$MEDICAL_MI_BASE_DIR"

# 2. uv를 사용한 가상환경 생성 및 패키지 설치
if [ ! -d "$PYTHON_VENV" ]; then
    echo "가상환경 생성 중..."
    uv venv $PYTHON_VENV
fi

# 가상환경 활성화
source $PYTHON_VENV/bin/activate

echo "패키지 설치 중 (requirements.txt)..."
uv pip install -r requirements.txt

# 3. 모델 및 데이터 다운로드 스크립트 실행
echo "모델 및 데이터셋 다운로드 시작..."
python scripts/01_setup_full.py --models $MEDICAL_MI_SETUP_MODELS

echo "모든 설정이 완료되었습니다."
echo "가상환경을 사용하려면 다음 명령어를 입력하세요: source $PYTHON_VENV/bin/activate"
