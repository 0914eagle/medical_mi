#!/bin/bash

# --- 의료 LLM SAE Feature 연구: 통합 실험 실행 스크립트 ---
# (다운로드 과정은 제외하고 실제 분석/실험 단계만 수행)

BASE_DIR="/workspace/medical_mi"
export HF_HOME="/workspace/.cache/huggingface"
# 파이썬이 scripts 폴더 내의 모듈(sae_wrapper 등)을 찾을 수 있게 경로 추가
export PYTHONPATH="$BASE_DIR/scripts:$PYTHONPATH"

# 폴더 생성
mkdir -p "$BASE_DIR/results/eval"
mkdir -p "$BASE_DIR/results/features"
mkdir -p "$BASE_DIR/results/figures"

echo "=========================================================="
echo "   STARTING INTEGRATED RESEARCH PIPELINE"
echo "=========================================================="

# 1. Phase 0: 모델 의료 지식 검증 (PubMedQA Accuracy & Context Gain)
echo -e "\n[Phase 0] Validating Model Medical Knowledge..."
python3 "$BASE_DIR/scripts/15_phase0_validation.py"

if [ $? -ne 0 ]; then
    echo "❌ Phase 0 실패. 로그를 확인하세요."
    exit 1
fi

# 1-1. 시각화 (Context Gain 그래프)
echo -e "\n[Phase 0] Generating Visualization..."
python3 "$BASE_DIR/scripts/17_visualize_phase0.py"


# 2. Phase 1: PubMedQA에서 Feature 발견 (Statistical Analysis)
echo -e "\n[Phase 1] Identifying SAE Features (Ignorance vs Correct)..."
python3 "$BASE_DIR/scripts/18_phase1_find_features.py"

if [ $? -ne 0 ]; then
    echo "❌ Phase 1 실패. 로그를 확인하세요."
    exit 1
fi


# 3. Phase 2: Feature Interpretation (Max-activating Examples)
echo -e "\n[Phase 2] Interpreting Found Features..."
python3 "$BASE_DIR/scripts/19_phase2_interpretation.py"

if [ $? -ne 0 ]; then
    echo "❌ Phase 2 실패. 로그를 확인하세요."
    exit 1
fi

echo -e "\n=========================================================="
echo "   모든 단계(Phase 0~2)가 성공적으로 완료되었습니다."
echo "   - 검증 결과: results/eval/"
# echo "   - Feature 분석: results/features/"
echo "   - 시각화 자료: results/figures/"
echo "=========================================================="
