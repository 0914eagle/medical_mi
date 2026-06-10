#!/bin/bash
MODELS=("qwen3.5-9b" "qwen3-8b")
BASE_DIR="/workspace/medical_mi"
SCRIPTS_DIR="$BASE_DIR/scripts"
LOG_DIR="$BASE_DIR/results/logs"

mkdir -p "$LOG_DIR"

echo "=== Phase 3 & 4 전용 실행 시작 ==="
for MODEL_NAME in "${MODELS[@]}"
do
    echo ">>> [$MODEL_NAME] 시작"
    python3 "$SCRIPTS_DIR/run_interpretation_steering_only.py" --model "$MODEL_NAME" 2>&1 | tee "$LOG_DIR/${MODEL_NAME}_phase3_4.log"
done
echo "=== 모든 분석이 완료되었습니다 ==="
