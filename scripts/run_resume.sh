#!/bin/bash
MODELS=("qwen3.5-9b" "qwen3-8b")
BASE_DIR="/home/eagle0914/medical_mi"
SCRIPTS_DIR="$BASE_DIR/scripts"
LOG_DIR="$BASE_DIR/results/logs"

mkdir -p "$LOG_DIR"

echo "=== 실험 재개 (Phase 2 onwards) 시작 ==="
for MODEL_NAME in "${MODELS[@]}"
do
    echo ">>> [$MODEL_NAME] 시작"
    python3 "$SCRIPTS_DIR/run_automated_pipeline_resume.py" --model "$MODEL_NAME" 2>&1 | tee "$LOG_DIR/${MODEL_NAME}_resume.log"
done
echo "=== 모든 실험이 완료되었습니다 ==="
