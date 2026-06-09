#!/bin/bash

# --- 실험 설정 ---
MODEL_NAME="qwen3.5-9b"
BASE_DIR="/workspace/medical_mi"
SCRIPTS_DIR="$BASE_DIR/scripts"
LOG_DIR="$BASE_DIR/results/logs"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "Medical MI Redesigned Experiment Pipeline 시작"
echo "대상 모델: $MODEL_NAME"
echo "시작 시간: $(date)"
echo "=========================================================="

# Phase A: 갈등 데이터셋 구축 (전체 1,000개 데이터 검사)
echo -e "\n[Phase A] Conflict Set Construction 시작..."
python3 "$SCRIPTS_DIR/02_phaseA_conflict_set.py" --model "$MODEL_NAME" 2>&1 | tee "$LOG_DIR/phaseA.log"

if [ $? -ne 0 ]; then
    echo "Phase A 실패. 중단합니다."
    exit 1
fi

# Phase B: SAE Feature 발견
echo -e "\n[Phase B] SAE Feature Discovery 시작..."
python3 "$SCRIPTS_DIR/03_phaseB_find_features.py" --model "$MODEL_NAME" 2>&1 | tee "$LOG_DIR/phaseB.log"

if [ $? -ne 0 ]; then
    echo "Phase B 실패. 중단합니다."
    exit 1
fi

# Phase C: Localization (Activation Patching)
echo -e "\n[Phase C] Localization (Patching) 시작..."
python3 "$SCRIPTS_DIR/04_phaseC_localization.py" --model "$MODEL_NAME" 2>&1 | tee "$LOG_DIR/phaseC.log"

if [ $? -ne 0 ]; then
    echo "Phase C 실패. 중단합니다."
    exit 1
fi

echo -e "\n=========================================================="
echo "실험 파이프라인 (A-C) 완료!"
echo "결과 저장 위치:"
echo " - Phase A (분류): $BASE_DIR/results/eval/${MODEL_NAME}_conflict_set.json"
echo " - Phase B (Feature): $BASE_DIR/results/features/${MODEL_NAME}_features.json"
echo " - Phase C (레이어): $BASE_DIR/results/eval/${MODEL_NAME}_localization.json"
echo " - 모든 로그: $LOG_DIR/"
echo "=========================================================="

echo -e "\n[Next Steps]"
echo "1. Phase B 결과에서 가장 유의미한(Intersection) Feature 번호를 확인하세요."
echo "2. 해당 번호로 Phase D (Interpretation)와 Phase E (Steering)를 실행하세요."
echo "   예: python3 scripts/06_phaseE_steering.py --model $MODEL_NAME --layer 20 --feature_idx [IDX]"
