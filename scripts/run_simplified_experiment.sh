#!/bin/bash

# --- 실험 설정 ---
MODELS=("qwen3.5-9b" "qwen3-8b")
BASE_DIR="/workspace/medical_mi"
SCRIPTS_DIR="$BASE_DIR/scripts"
LOG_DIR="$BASE_DIR/results/logs"

# 로그 폴더 생성
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "Medical MI SIMPLIFIED Experiment Pipeline 시작"
echo "대상 모델: ${MODELS[*]}"
echo "시작 시간: $(date)"
echo "=========================================================="

for MODEL_NAME in "${MODELS[@]}"
do
    echo -e "\n>>> [$MODEL_NAME] 전체 실험 시작 <<<"
    # run_automated_pipeline.py를 통해 Phase 1~4 일괄 실행
    python3 "$SCRIPTS_DIR/run_automated_pipeline.py" --model "$MODEL_NAME" 2>&1 | tee "$LOG_DIR/${MODEL_NAME}_simplified.log"
done

echo -e "\n=========================================================="
echo "모든 실험이 완료되었습니다!"
echo "결과 저장 위치: $BASE_DIR/results/"
echo "로그 확인: $LOG_DIR/"
echo "=========================================================="
