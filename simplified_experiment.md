# 실험 지침서 (단순화 버전) — Context Processing Feature

## 연구 목적

의료 LLM이 주어진 context를 제대로 처리하지 못하는 현상의 내부 메커니즘을
SAE feature로 규명한다. 핵심 단순화:

```
복잡한 버전 (폐기):
  prior 측정 → conflict 정의 → INTEGRATED vs IGNORED

단순한 버전 (채택):
  context 항상 포함
  correct (정답 맞춤) vs wrong (틀림)
  → PubMedQA는 context 처리 태스크이므로
    wrong = context 처리 실패
```

## 왜 단순화가 정당한가

```
MedQA wrong: context 없는 지식 문제 → 틀린 이유가 지식 부족
PubMedQA wrong: context 있는 추론 문제 → 틀린 이유가 context 처리 실패

→ PubMedQA에서는 correct vs wrong이
  곧 "context 처리 성공 vs 실패"
→ prior 분리 불필요
```

---

## 데이터셋 역할 (중요)

| 데이터셋 | 역할 | context | 기대 |
|---|---|---|---|
| PubMedQA | 주력 (feature 발견) | 있음 | correct vs wrong feature |
| MedQA/MedMCQA | 음성 대조군 | 없음 | feature 작동 안 해야 함 |
| MedAbstain | 교차 검증 | 불충분 | feature 전이되어야 함 |
| Rx-LLM | 임상 검증 | 명확한 conflict | clean 재현 |

```
논리:
  PubMedQA에서 찾은 feature가
  - MedQA(context 없음)에서 작동 안 함 → "지식 feature 아님"
  - MedAbstain(context 부족)에서 작동함 → "context feature 맞음"
  - Rx-LLM(명확한 conflict)에서 작동함 → "임상에서도 유효"
```

---

# Phase 1: PubMedQA correct vs wrong feature 발견

## 케이스 정의 (이미 있는 conflict_set.json 재활용)

```python
import json

with open("results/eval/qwen3.5-9b_conflict_set.json") as f:
    data = json.load(f)

# context_answer 기준으로 단순 분류
correct_cases = [r for r in data if r["context_answer"] == r["ground_truth"]]
wrong_cases   = [r for r in data if r["context_answer"] != r["ground_truth"]]

print(f"Correct: {len(correct_cases)}, Wrong: {len(wrong_cases)}")
# 예상: Correct ~740, Wrong ~260 (정확도 74%)

# 선택: wrong을 방향별로 세분화 (나중에 쓸 수 있음)
wrong_no_to_yes = [r for r in data if r["ground_truth"]=="no" and r["context_answer"]=="yes"]
wrong_yes_to_no = [r for r in data if r["ground_truth"]=="yes" and r["context_answer"]=="no"]
```

## SAE feature 추출 및 비교

```python
# utils.py의 get_activation_with_hook 사용
# sae_wrapper.py의 SAEWrapper 사용

TARGET_LAYERS = [10, 15, 18, 20, 22, 24, 26, 28, 30]

for layer in TARGET_LAYERS:
    sae = load_sae(model_name, layer)  # layer{n}.sae.pt
    
    # context 항상 포함하여 feature 추출
    feat_correct = extract_features(model, tok, sae, correct_cases[:100], layer,
                                     include_context=True)
    feat_wrong   = extract_features(model, tok, sae, wrong_cases[:100], layer,
                                     include_context=True)
    
    # t-test
    t_stats, p_vals = ttest_ind(feat_correct, feat_wrong, axis=0)
    
    # correct에서 더 활성화 = context 처리 성공 시 켜지는 feature
    correct_dominant = np.where((p_vals < 0.01) & (t_stats > 0))[0]
    # wrong에서 더 활성화 = context 처리 실패 시 켜지는 feature
    wrong_dominant = np.where((p_vals < 0.01) & (t_stats < 0))[0]
    
    results[layer] = {
        "correct_dominant": correct_dominant.tolist(),
        "wrong_dominant": wrong_dominant.tolist(),
        "n_correct_dom": len(correct_dominant),
        "n_wrong_dom": len(wrong_dominant),
    }
```

## Decision Gate 1
```
통과: 특정 레이어에서 유의미한 feature 다수 발견
      (이전 결과로 Layer 18-28 예상)
저장: 각 레이어의 correct_dominant, wrong_dominant feature
```

---

# Phase 2: 음성 대조군 (MedQA) — 핵심 검증

## 목적
"우리 feature가 context 처리에 관한 것이지 지식/topic이 아니다"를 증명.

```python
# MedQA는 context가 없음 (순수 지식 4지선다)
from datasets import load_dataset
medqa = load_dataset("GBaker/MedQA-USMLE-4-options")["test"]

# MedQA correct vs wrong
# Phase 1에서 찾은 feature가 MedQA correct/wrong을 구분하는가?

feat_correct_medqa = extract_features(medqa_correct_cases, layer)
feat_wrong_medqa = extract_features(medqa_wrong_cases, layer)

# Phase 1의 correct_dominant feature들이
# MedQA에서도 correct/wrong을 구분하는지 확인
for feat_idx in phase1_correct_dominant:
    c = feat_correct_medqa[:, feat_idx].mean()
    w = feat_wrong_medqa[:, feat_idx].mean()
    print(f"Feature {feat_idx}: MedQA correct={c:.3f}, wrong={w:.3f}")

# 기대:
#   PubMedQA에서는 차이 큼 (context feature)
#   MedQA에서는 차이 작음 (context 없으니 작동 안 함)
#   → 차이가 작아야 "context-specific" 증명
```

## Decision Gate 2
```
통과: PubMedQA feature가 MedQA에서는 약하게 작동
      → context-specific 확인

실패: MedQA에서도 강하게 작동
      → 그냥 correct/wrong 일반 feature (context 무관)
      → framing 재검토
```

---

# Phase 3: Feature Interpretation (대조 corpus)

## 목적
feature가 무엇을 인코딩하는지. (topic인지 context 처리인지)

```python
corpora = {
    "pubmedqa_with_context": PubMedQA context 포함 텍스트,
    "pubmedqa_no_context": 질문만,
    "medqa": context 없는 의학 질문,
    "general": wikipedia 등 비의료,
}

# Phase 1 top feature가 각 corpus에서 얼마나 켜지는가
for corpus_name, texts in corpora.items():
    acts = [get_sae_activation(t, feature_idx, layer) for t in texts]
    print(f"{corpus_name}: mean={np.mean(acts):.3f}")

# 해석:
#   pubmedqa_with_context에서만 강함 → context 처리 feature ✓
#   모든 의학 텍스트에서 강함 → topic feature ✗
#   general에서도 강함 → general feature
```

---

# Phase 4: Steering (Causal Validation)

## 목적
correct_dominant feature를 amplify하면 wrong 케이스가 교정되는가.

```python
# Phase 1에서 찾은 correct_dominant feature
# wrong 케이스에 amplify

steer_vec = sae.W_dec[feature_idx, :]  # decoder direction

# wrong 케이스에 적용
for item in wrong_cases:
    steered_answer = steer_and_test(model, item, layer, steer_vec, alpha)
    # 정답으로 바뀌는가?

# selectivity: correct 케이스는 안 건드려야 함
for item in correct_cases:
    steered_answer = steer_and_test(model, item, layer, steer_vec, alpha)
    # 여전히 correct 유지하는가?

측정:
  wrong → correct 교정률
  correct → wrong 오염률 (selectivity)
```

## alpha 결정
```
Phase 1에서 나온 실제 feature activation 값의 범위를 보고
alpha를 그 스케일에 맞춰 설정 (추측값 금지)
예: feature activation이 보통 0-10 범위면 alpha 5-20 시도
```

---

# Phase 5: MedAbstain 교차 검증

## MedAbstain 구조
```
original: 정보 충분 → 답 가능
perturbed: 핵심 정보 제거 → abstain이 정답
```

## 검증
```python
# github.com/sravanthi6m/MedAbstain clone
# perturbed 케이스: 정보 부족한데 모델이 confident하게 답하는 것
#   = context(정보) 처리 실패의 다른 형태

# PubMedQA에서 찾은 correct_dominant feature가
# MedAbstain에서도 correct(abstain) vs wrong(답함)을 구분하는가?

feat_abstain = extract_features(medabstain_correct_abstain, layer)
feat_answered = extract_features(medabstain_wrong_answered, layer)

# Phase 1 feature로 구분되면 → 전이 성공 → 강한 generalization
```

---

# Phase 6: Rx-LLM 임상 검증 (가능하면)

## 목적
PubMedQA의 ground truth 모호성이 없는 clean 케이스.

```
Rx-LLM renal dosing:
  "eGFR 25 환자에게 Metformin 적절한가?"
  prior: "Metformin은 당뇨 1차 치료" (yes 경향)
  context: "eGFR 25 = 금기" (no가 정답)
  ground truth: 임상 가이드라인 (명확)

→ 가장 깔끔한 context-prior conflict
→ PubMedQA feature가 여기서 작동하면 임상 validity
```

주의: Rx-LLM 데이터 구조를 먼저 확인하고 포맷 맞출 것.

---

# 전체 흐름

```
Phase 1: PubMedQA correct vs wrong → feature 발견 ★단순
   ↓
Phase 2: MedQA 음성 대조 → context-specific 확인 ★핵심
   ↓
Phase 3: Interpretation → topic 아님 확인
   ↓
Phase 4: Steering → 인과성 + selectivity
   ↓
Phase 5: MedAbstain → 전이 검증
   ↓
Phase 6: Rx-LLM → 임상 검증 (가능시)
```

---

# 기존 복잡한 버전에서 바뀐 점

```
폐기:
  - prior 측정 (조건 P)
  - INTEGRATED/IGNORED 분류
  - conflict 정의

채택:
  - context 항상 포함
  - correct vs wrong (단순)

유지:
  - feature interpretation (대조 corpus)
  - steering + selectivity
  - 레이어별 분석

추가:
  - MedQA 음성 대조군 (핵심 — context-specific 증명)
  - 데이터셋별 역할 분담
```

---

# 코딩 에이전트 작업 순서

```
1. conflict_set.json에서 correct/wrong 재라벨링 (Phase 1)
   → 모델 재실행 불필요, 기존 데이터 재활용

2. SAE feature 추출 (Phase 1)
   layer{n}.sae.pt 사용, 키 이름 먼저 확인:
   python -c "import torch; print(torch.load('layer20.sae.pt').keys())"

3. MedQA 음성 대조 (Phase 2)

4. Interpretation, Steering (Phase 3-4)

5. MedAbstain, Rx-LLM (Phase 5-6)
```

# 주의사항

```
1. SAE 키 이름 확인 (W_enc/b_enc/W_dec/b_dec 인지)
2. Gemma는 model.model.layers 경로 다를 수 있음
3. correct/wrong 케이스 수 불균형 (740 vs 260)
   → t-test 시 적절히 샘플링 또는 unequal variance 옵션
4. Phase 2(MedQA 대조)가 가장 중요 — 생략 금지
```
