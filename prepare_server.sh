#!/bin/bash

# 1. 경로 설정 (모든 데이터와 캐시를 /workspace 폴더로 집중)
export WORKSPACE_DIR="/workspace"
export UV_CACHE_DIR="$WORKSPACE_DIR/.cache/uv"
export HF_HOME="$WORKSPACE_DIR/.cache/huggingface"
export PYTHON_VENV="$WORKSPACE_DIR/.venv"

# 디렉토리 생성
mkdir -p $UV_CACHE_DIR
mkdir -p $HF_HOME

echo "--- 환경 설정 완료 ---"
echo "UV Cache: $UV_CACHE_DIR"
echo "HF Home: $HF_HOME"
echo "Venv: $PYTHON_VENV"

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
python scripts/01_setup.py

echo "모든 설정이 완료되었습니다."
echo "가상환경을 사용하려면 다음 명령어를 입력하세요: source $PYTHON_VENV/bin/activate"
