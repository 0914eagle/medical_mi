# 마스터 실험 지침서 — 데이터 설계 + 전체 실험

## 0. 연구 목적 (고정)

의료 LLM이 주어진 context를 무시하고 parametric prior로 답하는 현상을
SAE feature로 (1) 감지하고 (2) 일부 교정하며 (3) 교정 불가능한 경우를 규명한다.
선행연구 SpARE(arXiv:2410.15999)와의 차별점:
- 의료 특화 + 음성 대조(MedQA)로 context-specific 입증
- "감지 != 교정 가능"의 분리 (28% 천장 원인 규명)
- 교정 불가능 시 abstain (의료 안전)

---

## 1. 데이터 설계 (모든 실험의 전제)

### 1-1. 두 종류의 Leakage 구분

```
Leakage A — 모델 사전학습 contamination:
  PubMedQA(2019 공개)를 Qwen3.5가 학습했을 가능성 높음
  -> 우리가 못 막음. limitation 명시 + 깨끗한 데이터로 보완

Leakage B — 실험 설계 leakage:
  feature/layer/alpha를 "고른" 데이터와
  성능을 "측정한" 데이터가 같으면 과대평가
  -> split으로 막을 수 있고 반드시 막아야 함
```

### 1-2. Split 설계 (Leakage B 대응)

```
PubMedQA pqa_labeled 1000개:
  공식 GitHub(pubmedqa/pubmedqa)에 test 500 ID 존재
  -> HuggingFace 버전엔 split 없음, GitHub ID 가져와야

분할:
  Validation 500개: 모든 "선택"을 여기서
    - feature 발견 (correct vs wrong)
    - context-specific 판정
    - layer 결정 (patching)
    - alpha 튜닝
    - conflict score threshold
  Test 500개: 선택된 것을 한 번만 적용, 측정
    - 이 숫자만 논문에 보고
```

### 1-3. 케이스 수 부족 대응

```
문제: wrong(context 무시) 케이스가 모델당 ~46개
      val/test 나누면 각 ~23개 (steering 평가 불안정)

해결: 5-fold Cross Validation
  pqa_labeled 1000개를 5등분
  각 fold: 4/5로 선택(feature/layer/alpha), 1/5로 측정
  5번 반복 후 평균 ± 표준편차 보고
  -> 적은 데이터 효율적 사용, 공식 PubMedQA 권장 방식
```

### 1-4. Contamination 대응 (Leakage A)

```
1. context 없을 때 정확도 함께 보고
   완전 암기면 context 없어도 높아야 함
   Qwen3.5 context 없을 때 55.3% -> 순수 암기는 아님 (약한 증거)

2. 깨끗한 데이터셋 병행 (핵심)
   Rx-LLM, 최신/비공개 임상 케이스
   "오염 안 된 데이터에서도 같은 메커니즘" -> 강한 방어

3. 논문에 limitation 명시 + 선제 방어
```

### 1-5. 데이터셋별 역할 (고정)

```
PubMedQA  : 주력 발견 (evidence 무시), contamination 위험 명시
MedQA     : 음성 대조 (context 없음), feature 작동 안 해야
MedAbstain: 교차 검증 (정보 부족 무시), 행동 기준 분류
Rx-LLM    : 임상 검증 (환자맥락 무시), contamination 적음, clean
```

---

## 2. 완료된 실험 (재검증 필요)

```
Phase 0-4 (Qwen3.5-9B) 완료
단, 모두 1000개 전체로 수행 -> Leakage B 있음
-> 핵심 수치(28% 등)는 val/test split으로 재측정 필요
```

---

## 3. 앞으로 할 실험 (split 적용)

### 3-0. 선행: PubMedQA 성능 재검증

```
- 공식 500 test split으로 정확도 재측정
- context 유/무 정확도 비교 (contamination 추정)
- 기존 leaderboard와 비교 (74% 합리적인지)
```

---

### 🔴 1순위: Context-Specific Steering (split 적용)

```
[Validation에서 선택]
Step A: context-specific feature 선별
  먼저 디버깅: #2392로 세 corpus 값 재현 (1.346/0.386/0.000)
  그 다음 모든 후보를 판정:
    기준 with > 2x without AND medqa < 0.05
  -> cs_wrong, cs_correct 목록 확정 (val에서만)

Step B: alpha 튜닝도 validation에서
  suppress cs_wrong, alpha sweep (5,7,10,15,20)
  최적 alpha 선택

[Test에서 측정]
  validation에서 정한 cs_wrong + 최적 alpha로
  test set steering 한 번만
  -> recovery / corruption 보고

결과물 (test 기준):
  | feature set        | recovery | corruption |
  | 전체 wrong_dom (22) |    ?     |     ?      |
  | context-specific만  |    ?     |     ?      |
  | 단일 #28696        |    ?     |     ?      |
  (5-fold면 평균 +- std)
```

---

### 🔴 2순위: 28% 천장 원인 분석 (H1-H4)

```
Step 1 (H1): context-specific 재실험으로 커버됨

Step 2 (H3 — 가장 중요): 교정 가능/불가능 비교
  test steering 결과에서:
    recovered_cases (교정됨) vs unrecovered_cases (안 됨)
  4가지 비교:
    1. conflict score
    2. prior 확신도
    3. context feature 활성화
    4. SAE reconstruction error
  두 그룹 갈리면 -> "context 무시 두 종류" (mechanistic 발견)

Step 3 (H2): 다중 레이어 동시 steering
  Layer 20 단일 vs L18+20+22 동시
  (layer 선택은 validation, 측정은 test)
  천장 깨지면 H2, 안 깨지면 H3

Step 4 (H4): SAE reconstruction error
  unrecovered에서 recon error 큰지
  크면 SAE 한계
```

---

### 🟡 3순위: MedAbstain 교차 검증

```
이전 방법(original/perturbed 텍스트 비교) 폐기
행동 기준으로:
  correct_abstain vs wrong_answered

Step 1: MedAbstain 케이스 분류 (모델 행동 기준)
Step 2: PubMedQA wrong_dominant feature가
        wrong_answered에서 더 활성화되나
        + TopK 진입 비율
Step 3 (강): PubMedQA feature suppress가
        MedAbstain에서도 abstain 유도하나 (인과 전이)

주의: feature는 PubMedQA validation에서 찾은 것 사용
      MedAbstain은 전이 검증용 (선택 안 함)
```

---

### 🟡 4순위: Conflict Score 검증 (새 방법론 CAI)

```
conflict_score = wrong_signal / (correct + wrong signal)

[Validation] score 계산 방식, threshold 결정
[Test] AUC-ROC 측정
  기대 AUC > 0.70

threshold도 validation에서 정하고 test에서 적용:
  low/medium/high 구간별 pass/steer/abstain
```

---

### 🟢 5순위: Rx-LLM 임상 검증 (contamination 적음)

```
eGFR + Metformin 같은 명확한 conflict
ground truth = 임상 가이드라인 (모호하지 않음)

PubMedQA와 동일 구조로:
  context-specific feature 확인
  steering 효과
  conflict score 예측력

의미: contamination 적은 데이터에서 재현
  -> Leakage A 방어의 핵심
```

---

### 🟢 6순위: Gemma-3-12B 교차 모델

```
Gemma-3-12B-IT + Gemma Scope 2로 전체 반복
비교: 결정 레이어, 천장 fraction, feature 공유
-> 모델 agnostic 주장
```

---

## 4. 전체 흐름

```
[전제] 데이터 split 설계 (val/test, 5-fold, contamination 대응)
   |
[선행] PubMedQA 성능 재검증 (split + context 유무)
   |
당장 (SpARE 차별화):
  1순위 Context-specific steering (split 적용)
  2순위 28% 천장 원인 (H1-H4)
   |
다음 (generalization):
  3순위 MedAbstain (행동 기준)
  4순위 Conflict score (CAI 방법론)
   |
나중 (완성):
  5순위 Rx-LLM (clean, contamination 방어)
  6순위 Gemma (모델 agnostic)
```

---

## 5. 핵심 원칙 (코딩 에이전트 필독)

```
1. Validation에서 모든 선택 (feature/layer/alpha/threshold)
   Test에서는 한 번만 적용, 측정
   Test 보고 파라미터 바꾸면 그 순간 leakage

2. 케이스 적으면 5-fold CV, 평균 +- std 보고

3. 기존 1000개 전체 수치(28% 등)는
   "leakage 있는 예비 결과"로 표기
   split 재측정 후 정식 수치로 교체

4. Contamination은 split과 별개:
   context 유무 비교 + Rx-LLM 병행으로 대응

5. Step별 결과 json 저장 (그래프용)
   recovered/unrecovered 케이스 id 저장 (H3 분석용)
```

---

## 6. 예상 산출물 (교수님/논문)

```
1. Split 기반 정직한 수치
   "data leakage 없이 recovery X% +- Y"

2. 28% 천장 원인
   recoverable vs unrecoverable 구분 -> 두 종류 발견

3. Cross-dataset (MedAbstain 전이, Rx-LLM 재현)
   -> 공통 메커니즘 + contamination 방어

4. CAI 방법론 (conflict score 기반 pass/steer/abstain)
   -> SpARE 넘는 새 방법론
```
