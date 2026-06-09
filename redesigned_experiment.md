# 실험 재설계 — Context-Knowledge Integration Failure의 메커니즘

## 연구 목적 (다시 못박음)

의료 LLM이 주어진 context(연구 데이터/환자 정보)를 무시하고
parametric prior(가중치에 박힌 사전 지식)로 답하는 현상을
mechanistic하게 규명하고, training-free로 개선한다.

근거: NHS 임상 배포 실패의 86%가 contextual reasoning failure
("A Real-World Evaluation of LLM Medication Safety Reviews
  in NHS Primary Care", Normand et al., arXiv:2512.21127)

---

## 기존 설계의 문제 (왜 다시 짜는가)

```
기존: correct 케이스 vs ignorance 케이스의 feature 비교

문제:
  이 두 그룹을 가르는 요인이 여러 개 섞임
  - context를 봤나 안 봤나
  - 질문이 어려웠나
  - prior 지식이 있었나
  → "context 처리 feature"가 분리되지 않음

결과:
  feature interpretation에서
  "context feature"인지 "의학 질문 feature"인지 구분 불가
```

## 새 설계의 핵심 아이디어: Prior와 Context 분리

```
모든 질문을 두 조건에서 실행:
  조건 P (Prior only): context 없이 질문만
    → 모델의 "사전 지식이 뭐라고 하는가" 측정
  조건 C (with Context): context 포함
    → 모델이 "context를 반영하면 뭐라고 하는가"

이걸로 conflict(갈등) 케이스를 정의:
  Prior가 한 방향, Context(=ground truth)가 반대 방향인 케이스
```

**중요:** 조건 P(context 없음)는 정확도 측정용이 아니다.
**모델의 prior를 측정해서 conflict 케이스를 식별하는 용도다.**
(이전에 "Context Gain은 의미 약하다"고 한 것과 모순 아님 —
 여기선 정확도가 아니라 prior 방향을 알아내려는 것)

---

# Phase A: Conflict Set 구성 (핵심)

## A-1. 두 조건에서 실행

```python
def run_both_conditions(model, tokenizer, item):
    # 조건 P: prior only (context 없음)
    prompt_P = format_question(item, include_context=False)
    probs_P = get_ynm_probs(model, tokenizer, prompt_P)
    prior_answer = max(probs_P, key=probs_P.get)
    
    # 조건 C: with context
    prompt_C = format_question(item, include_context=True)
    probs_C = get_ynm_probs(model, tokenizer, prompt_C)
    context_answer = max(probs_C, key=probs_C.get)
    
    return {
        "ground_truth": item["final_decision"],
        "prior_answer": prior_answer,
        "prior_probs": probs_P,
        "context_answer": context_answer,
        "context_probs": probs_C,
    }
```

## A-2. 케이스 분류 (2x2 + conflict 집중)

```python
# ground truth는 "no"인 케이스에 집중
# (PubMedQA no = 연구 데이터가 효과를 지지하지 않음)

def classify(r):
    gt = r["ground_truth"]
    prior = r["prior_answer"]
    ctx = r["context_answer"]
    
    # CONFLICT 케이스: prior가 ground truth와 반대
    #   예: prior="yes", gt="no"
    #   → context를 봐야만 no라고 답할 수 있음
    if prior != gt:
        if ctx == gt:
            return "INTEGRATED"   # context 반영 성공 (prior 극복)
        else:
            return "IGNORED"      # context 무시 (prior 유지) ← 핵심 타겟
    else:
        # prior가 이미 gt와 같음 → context 불필요
        return "NO_CONFLICT"
```

## A-3. 핵심 대조군

```
우리가 비교할 두 그룹:

INTEGRATED 그룹:
  prior = "yes" (사전지식은 yes라고 함)
  ground truth = "no"
  context 줬더니 = "no" (context를 반영해서 prior 극복)

IGNORED 그룹:
  prior = "yes" (사전지식은 yes라고 함)
  ground truth = "no"
  context 줬더니 = "yes" (context 무시, prior 유지)

두 그룹의 공통점:
  - prior가 모두 "yes" (prior pull이 동일)
  - ground truth가 모두 "no" (정답이 동일)
  - context도 모두 "no"를 지지

유일한 차이:
  context를 반영했는가 (INTEGRATED) vs 무시했는가 (IGNORED)

→ 이 두 그룹의 내부 차이 = 순수하게 "context 통합" 신호
→ 난이도, 지식, prior가 통제됨
```

이게 기존 "correct vs ignorance"보다 결정적으로 나은 점이다.

## Decision Gate A
```
통과 조건:
  INTEGRATED, IGNORED 케이스 각각 최소 30개 이상 확보
  (부족하면 maybe 포함하거나 여러 모델 합산)

케이스 부족 시:
  threshold 완화 또는
  conflict 정의 확장 (prior 확신도 기준 추가)
```

---

# Phase B: Feature 발견 (두 가지 신호)

## B-1. 신호 1 — 그룹 간 차이 (INTEGRATED vs IGNORED)

```python
# 두 그룹의 SAE feature를 레이어별로 비교
# 차이나는 feature = "context 통합과 연관된 feature"

for layer in TARGET_LAYERS:  # [10,15,18,20,22,24,26,28,30,32,34]
    integrated_feats = extract_sae(INTEGRATED_cases, layer)
    ignored_feats = extract_sae(IGNORED_cases, layer)
    
    diff = integrated_feats.mean(0) - ignored_feats.mean(0)
    # INTEGRATED에서 더 활성화 = context 통합 시 켜지는 feature
    # t-test로 유의성 검증
```

## B-2. 신호 2 — 같은 질문 내 context 효과 (within-question)

```python
# 같은 질문에서 context 넣었을 때 vs 뺐을 때
# activation이 어떻게 변하는가 = context가 직접 만드는 변화

for item in conflict_cases:
    act_P = get_activation(model, item, layer, include_context=False)
    act_C = get_activation(model, item, layer, include_context=True)
    context_signal = act_C - act_P
    # 이 diff를 SAE로 분해 → "context가 켜는 feature"
```

## B-3. 두 신호 교차

```
신호 1 (INTEGRATED vs IGNORED)에서 나온 feature
신호 2 (context 有/無 diff)에서 나온 feature

겹치는 feature = 강력한 context-integration feature 후보
  → "context가 들어오면 켜지고(신호2)
     켜졌을 때 context를 반영하는(신호1)" feature
```

## Decision Gate B
```
통과: 두 신호에서 공통으로 나오는 feature가 존재
실패: 두 신호가 전혀 다름 → context 처리가 feature 단위가 아닐 수 있음
       → Phase C(patching)로 직행해서 정보 흐름 차원에서 분석
```

---

# Phase C: Localization — context가 언제 사라지는가 (핵심)

## 목적
"feature가 있다"를 넘어 "context가 어느 레이어에서 prior에게 지는가"를 규명.
(activation patching)

## C-1. Activation Patching

```python
def patch_and_test(model, tokenizer, ignored_item, integrated_donor, patch_layer):
    """
    IGNORED 케이스를 실행하다가
    특정 레이어에서 INTEGRATED 케이스의 activation을 주입
    → 출력이 'no'(정답)로 바뀌는가?
    """
    donor_act = get_activation(model, integrated_donor, patch_layer, include_context=True)
    
    def patch_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h[0, -1, :] = donor_act  # 마지막 토큰 activation 교체
        return o
    
    handle = model.model.layers[patch_layer].register_forward_hook(patch_hook)
    prompt = format_question(ignored_item, include_context=True)
    probs = get_ynm_probs(model, tokenizer, prompt)
    handle.remove()
    return probs

# 레이어별로 patching
for layer in range(0, num_layers, 2):
    flip_rate = 측정(IGNORED 케이스들이 'no'로 바뀐 비율)
    # flip_rate가 급증하는 레이어 = context가 결정되는 지점
```

## C-2. 해석

```
Layer 5에서 patch → 변화 없음: context가 아직 처리 안 됨
Layer 20에서 patch → 50% flip: 이 근처에서 context 통합이 일어남
Layer 30에서 patch → 변화 없음: 이미 출력 결정됨 (너무 늦음)

→ flip_rate가 가장 크게 변하는 레이어
  = "context가 prior를 이기거나 지는 결정 지점"
  = 네가 말한 "context를 잃어버리는 지점"
```

## C-3. 방향성 (정보 흐름)

```
추가 분석: context token → 마지막 token으로의 정보 흐름
  - 어느 레이어까지 context token 정보가 살아있는가
  - IGNORED 케이스에서는 그 흐름이 어디서 끊기는가

방법: attention pattern + residual stream에서
      context token 기여도 추적
```

## Decision Gate C
```
통과: flip이 집중되는 특정 레이어 구간 식별
      (Phase B의 feature 레이어와 일치하면 더 강력)
```

---

# Phase D: Feature Interpretation (대조 corpus 필수)

## 기존 문제
PubMedQA만으로 max-activating 하면 모든 게 "의학 질문"이라
context feature인지 topic feature인지 구분 불가.

## D-1. 대조 corpus 사용

```python
corpora = {
    "pubmedqa_conflict": conflict 케이스 텍스트,
    "pubmedqa_noconflict": non-conflict 텍스트,
    "general_text": 비의료 일반 텍스트 (wikipedia 등),
    "medical_no_context": context 없는 의학 질문,
}

# feature가 각 corpus에서 얼마나 활성화되는가
for corpus_name, texts in corpora.items():
    activations = [get_sae(t, feature_idx, layer) for t in texts]
    print(f"{corpus_name}: mean activation = {mean(activations)}")
```

## D-2. 해석 기준

```
feature가 pubmedqa_conflict에서만 강하게 켜지고
다른 corpus에서 약하면:
  → context-conflict 처리 feature (우리가 원하는 것)

feature가 모든 의학 텍스트에서 비슷하게 켜지면:
  → 단순 의학 topic feature (원하는 게 아님)

feature가 general_text에서도 켜지면:
  → general uncertainty/negation feature
  → context-specific은 아니지만 여전히 흥미로움
```

## Decision Gate D
```
통과: feature가 conflict 케이스에 selective하게 반응
실패: topic feature로 판명 → framing 재검토 (중요한 negative result)
```

---

# Phase E: Causal Validation (Steering)

## 목적
Phase B/C에서 찾은 feature/레이어가 정말 인과적인가.

```python
# Phase C에서 찾은 결정 레이어 + Phase B에서 찾은 feature
# IGNORED 케이스에서 그 feature를 amplify (또는 patching 방향)

측정:
  1. IGNORED → 'no'(정답)로 이동한 비율
  2. NO_CONFLICT 케이스 오염율 (selectivity)
     → prior와 gt가 일치하는 케이스는 안 건드려야 함
  3. INTEGRATED 케이스 영향 (이미 맞은 것 유지하는가)
```

## 기존 pilot과의 연결
```
이전 pilot (88.2% / 3.3%)은
  타겟 정의가 불명확한 상태에서 나온 것

이번엔:
  conflict 케이스 정의가 명확
  feature interpretation 완료
  결정 레이어 식별
  → 같은 88% 수준이 나오면 훨씬 강한 증거
  → selectivity도 NO_CONFLICT 기준으로 제대로 측정
```

---

# Phase F: Cross-dataset 검증 (MedAbstain)

```
PubMedQA에서 찾은 feature/레이어가
MedAbstain에서도 작동하는가

MedAbstain:
  original (정보 충분) vs perturbed (정보 제거)
  perturbed = context가 불충분한데 답하면 안 되는 케이스

검증:
  PubMedQA context-integration feature가
  MedAbstain perturbed 케이스에서도 활성화 패턴 보이는가
```

---

# Phase G: Circuit / Attention Head (선택, depth)

```
Phase C에서 찾은 결정 레이어의
어떤 attention head가 context token을 보는가/무시하는가

IGNORED 케이스:
  context token에 대한 attention이 낮은 head 식별
INTEGRATED 케이스:
  같은 head가 context를 보는가

→ "context를 무시하게 만드는 head" 식별
→ 가장 깊은 mechanistic 설명
```

---

# 전체 흐름 요약

```
A. Conflict Set 구성
   prior 측정 → conflict 케이스 식별
   INTEGRATED vs IGNORED (prior/gt 통제된 대조군) ★핵심 개선
        ↓
B. Feature 발견
   그룹 간 차이 + context 有/無 diff, 두 신호 교차
        ↓
C. Localization (patching)
   context가 어느 레이어에서 결정되는가 ★네 질문
        ↓
D. Interpretation (대조 corpus)
   context feature인가 topic feature인가
        ↓
E. Causal Validation (steering)
   인과성 + selectivity (NO_CONFLICT 기준)
        ↓
F. MedAbstain 교차 검증
        ↓
G. Circuit (선택)
```

---

# 기존 설계 대비 핵심 개선점

```
1. Prior와 Context 분리
   기존: correct vs ignorance (요인 혼재)
   신규: 같은 prior(yes) + 같은 gt(no)인데
         context 반영 여부만 다른 대조군
   → 순수한 context 통합 신호 분리

2. context가 사라지는 지점 규명 (Phase C)
   기존: 없음
   신규: activation patching으로 결정 레이어 식별

3. Interpretation에 대조 corpus
   기존: PubMedQA만 → topic과 구분 불가
   신규: 4개 corpus 대조 → context-specific 확인

4. Selectivity를 NO_CONFLICT 기준으로
   기존: yes 케이스 (모호)
   신규: prior=gt인 케이스 (명확)
```

---

# 코딩 에이전트 주의사항

```
1. Phase A의 prior 측정은 정확도가 아니라
   conflict 케이스 식별 용도임을 명심

2. INTEGRATED/IGNORED 케이스 수가 적을 수 있음
   → 여러 모델 합산 또는 maybe 포함 고려
   → 케이스 수 먼저 확인하고 보고

3. Activation patching은 마지막 토큰만 교체로 시작
   → 효과 있으면 여러 토큰/position으로 확장

4. 대조 corpus용 비의료 텍스트 준비 필요
   (wikipedia, c4 등에서 샘플)

5. Phase 순서 지킬 것. Gate 통과 못 하면 보고.
```
