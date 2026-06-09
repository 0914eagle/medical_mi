# 의료 LLM SAE Feature 연구 — 전체 실험 지침서

## 연구 한 줄 요약

의료 QA에서 LLM이 context(주어진 근거)를 무시하고 parametric prior(사전 지식)로
답하는 현상의 내부 메커니즘을 사전학습된 SAE로 규명하고, 그 feature를 활용한다.

## 전체 구조 (Phase는 순차적 — 앞 단계 통과 후 다음 단계)

```
Phase 0: 모델 의료 지식 검증        → 전제 검증
Phase 1: PubMedQA에서 feature 발견  → 핵심 발견
Phase 2: Feature interpretation     → feature가 뭔지 규명 (필수)
Phase 3: MedAbstain 교차 검증        → generalization
Phase 4: Task 수행 (3택1)           → contribution
Phase 5: Circuit 분석 (선택)         → mechanistic depth
```

각 Phase 끝에 Decision Gate가 있다. 통과 못 하면 다음 Phase로 가지 말고
원인 분석 후 보고할 것.

---

## 공통 설정

### Backbone (SAE 학습 금지 — 사전학습 SAE만 사용)

| 모델 | SAE Suite | 비고 |
|---|---|---|
| Qwen3-8B | Qwen-Scope (W64K, L0=50) | Primary 1 |
| Qwen3.5-9B | Qwen-Scope | Primary 2 |
| Gemma-3-12B-IT | Gemma Scope 2 (IT 버전) | Cross-family |

**사용 금지:** Qwen3.6/3.7 (SAE 없음), Gemma-2 (Gemma-3로 대체)

### SAE 명세 확인 (코딩 전 필수)
- Qwen-Scope: HuggingFace `Qwen/SAE-Res-*` 정확한 repo명 확인
- Gemma Scope 2: `google/gemma-scope-2-12b-it` 구조 확인
  - resid_post, 4개 depth (25%, 50%, 65%, 85%), width/L0 다양
  - **Qwen과 Gemma의 SAE 인터페이스가 다르므로 wrapper 통일 필요**

### 데이터셋
- PubMedQA: `qiaojin/PubMedQA`, `pqa_labeled` (1000개)
- MedAbstain: `github.com/sravanthi6m/MedAbstain`
- **MedQA는 사용하지 않음** (context 없는 순수 지식 테스트, 본 연구 범위 밖)

### 공통 규칙
- Qwen3: `enable_thinking=False` (non-thinking mode)
- Context는 항상 프롬프트에 포함 (본 연구의 핵심)
- 모델은 하나씩 로드/해제 (메모리)

---

# Phase 0: 모델 의료 지식 검증

## 목적
"모델이 context를 활용할 줄 안다"는 전제를 검증. 이게 안 되면 연구 무의미.

## 핵심 지표: Context Gain
```
Context Gain = (context 포함 정확도) - (context 미포함 정확도)

양수(+5% 이상): 모델이 context를 활용함 → 타겟 유효
0 또는 음수: context를 못 쓰거나 prior로 다 맞춤 → 재검토
```

## 구현

```python
def format_pubmedqa(item, tokenizer, include_context=True):
    contexts = item["context"]["contexts"]
    context_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    if len(context_text) > 1500:
        context_text = context_text[:1500] + "..."
    q = item["question"]
    if include_context:
        prompt = f"Context: {context_text}\n\nQuestion: {q}\n\nBased ONLY on the context, answer one word: yes, no, or maybe."
    else:
        prompt = f"Question: {q}\n\nAnswer one word: yes, no, or maybe."
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True)

def get_ynm_probs(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    out = {}
    for w in ["yes", "no", "maybe"]:
        ids = []
        for v in [w, w.capitalize(), " "+w, " "+w.capitalize()]:
            t = tokenizer.encode(v, add_special_tokens=False)
            if t: ids.append(t[0])
        out[w] = max(probs[i].item() for i in ids) if ids else 0.0
    tot = sum(out.values())
    return {k: v/tot for k,v in out.items()} if tot>0 else out
```

## 측정
각 모델 × (context 有/無) = 6회 평가. 전체 1000개.

## Decision Gate 0
```
통과 조건 (모두 만족):
  - Qwen3-8B context 포함 정확도 ≥ 65%
  - Context Gain ≥ +5%
  - 3개 모델에서 일관된 패턴

통과 → Phase 1
실패 → 원인 분석:
  정확도 낮음 → 모델이 의료 지식 부족 → backbone 재선정
  Context Gain 없음 → context 활용 안 함 → 프롬프트/태스크 재검토
```

---

# Phase 1: PubMedQA에서 Feature 발견

## 목적
**PubMedQA 자체에서** feature를 찾는다 (MedQA에서 찾아 가져오는 것 아님).
이전 pilot의 "MedQA→PubMedQA 전이" 문제를 해결.

## 케이스 정의

```python
# context 포함 평가 결과(Phase 0)에서:

# 그룹 A: context 반영 성공 (정답)
#   ground truth = no, 모델 = no, 확신 70%+
correct_context = [r for r in results
                   if r["ground_truth"]=="no" and r["prediction"]=="no"
                   and r["confidence"]>=0.70]

# 그룹 B: context 무시 (오답) ← 핵심 타겟
#   ground truth = no, 모델 = yes, 확신 70%+
ignorance = [r for r in results
             if r["ground_truth"]=="no" and r["prediction"]=="yes"
             and r["confidence"]>=0.70]
```

**왜 no 케이스만?**
PubMedQA no = "이 연구는 효과를 지지하지 않음". context를 읽으면 no가 맞는데
모델이 yes라고 하면 = context를 무시하고 prior로 답한 것 = 우리 타겟.

## SAE Feature 추출 (전 레이어)

이전 pilot은 Layer 15,20,25만 봤음. 이번엔 더 넓게.

```python
# Qwen3-8B는 36개 레이어. 균등 샘플링 + 후반 집중
TARGET_LAYERS = [10, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34]

def get_residual_activation(model, tokenizer, text, layer_idx):
    store = {}
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        store["act"] = h[0, -1, :].detach().cpu()
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return store["act"]

def to_sae_features(act, sae):
    # SAE 인터페이스는 suite마다 다름 — wrapper로 통일
    # Qwen-Scope: W_enc, b_enc, TopK
    # Gemma Scope 2: JumpReLU (threshold 방식) — 별도 처리
    return sae.encode(act)  # wrapper 내부에서 분기
```

## 통계 분석

```python
from scipy import stats

def find_features(correct_feats, ignorance_feats, layer):
    c_mean = correct_feats.mean(0)
    i_mean = ignorance_feats.mean(0)
    diff = c_mean - i_mean  # correct에서 더 활성화 = ignorance 시 억제
    
    # 양방향 모두 기록
    # correct-dominant: context 반영 시 켜지는 feature
    # ignorance-dominant: context 무시 시 켜지는 feature
    
    t_stats, p_values = [], []
    for f in range(correct_feats.shape[1]):
        c, i = correct_feats[:,f].numpy(), ignorance_feats[:,f].numpy()
        if c.sum()==0 and i.sum()==0:
            t_stats.append(0); p_values.append(1.0); continue
        t,p = stats.ttest_ind(c,i)
        t_stats.append(0 if np.isnan(t) else t)
        p_values.append(1.0 if np.isnan(p) else p)
    return diff, np.array(t_stats), np.array(p_values)
```

## Decision Gate 1
```
통과 조건:
  - 어떤 레이어에서 p<0.05 feature가 충분히 존재 (>=10개)
  - 3개 모델에서 유의미한 feature가 나타나는 레이어가 존재
    (꼭 같은 레이어 번호일 필요는 없음 — depth 비율로 비교)

핵심 체크: 케이스 수 확보
  ignorance 케이스, correct 케이스 각각 최소 50개 이상
  부족하면 maybe 케이스 포함 또는 confidence threshold 조정
```

---

# Phase 2: Feature Interpretation (필수 — 생략 금지)

## 목적
찾은 feature가 **실제로 무엇을 인코딩하는지** 규명.
이게 없으면 "context feature"라고 부를 수 없음.
("Falsifying SAE Reasoning Features in Language Models" arXiv:2601.05679 경고)

## 방법 1: Max-activating examples

```python
def find_max_activating(sae, model, tokenizer, feature_idx, layer, corpus, top_k=30):
    """이 feature를 가장 강하게 켜는 텍스트 조각 찾기"""
    scores = []
    for text in corpus:
        act = get_residual_activation(model, tokenizer, text, layer)
        feats = to_sae_features(act, sae)
        scores.append((text, feats[feature_idx].item()))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

# 해석 기준:
# "uncertain", "no evidence", "not significant", "inconclusive" 등
#   → epistemic/negation feature → context-conflict feature 가능성
# "study", "trial", "p-value" 등 연구 방법론 표현
#   → evidence-processing feature
# 특정 질병/약물명에 집중
#   → 단순 topic feature (우리가 원하는 게 아님)
```

## 방법 2: Neuronpedia 자동 해석 (있으면)
- Qwen-Scope, Gemma Scope 2 feature는 Neuronpedia에 등록되어 있을 수 있음
- feature index로 조회해서 auto-interpretation 확인

## 방법 3: 세브란스 임상의 validation
- max-activating 케이스를 의사에게 보여주고
- "이 feature가 켜지는 케이스들이 임상적으로 공통점이 있는가" 확인

## Decision Gate 2
```
통과 조건:
  top feature들이 해석 가능한 일관된 개념과 연결됨
  (epistemic uncertainty, evidence-processing, negation 등)

만약 feature가 단순 topic(특정 질병명)이면:
  → "context-knowledge integration feature"가 아님
  → 연구 framing 재검토 필요
  → 이것도 중요한 (negative) 결과
```

---

# Phase 3: MedAbstain 교차 검증

## 목적
PubMedQA에서 찾은 feature가 다른 데이터셋에서도 작동하는가.
데이터셋 특화 artifact가 아님을 입증.

## MedAbstain 구조 활용
```
original 질문: 정보 충분 → 답 가능
perturbed 질문: 핵심 정보 제거 → abstain이 정답

실험:
  perturbed 케이스에서 모델이 confident하게 답하는 경우 (abstain 실패)
  → PubMedQA ignorance 케이스와 같은 feature가 활성화되는가?
  → 같은 레이어, 같은 feature index인가?
```

## 측정
```python
# PubMedQA에서 찾은 top feature가
# MedAbstain abstain-fail 케이스에서도 높게 활성화되는가
def cross_validate(pubmedqa_features, medabstain_cases, model, sae, layer):
    # PubMedQA top feature indices
    top_idx = pubmedqa_features["top_indices"]
    
    activations_on_medabstain = []
    for case in medabstain_cases:
        act = get_residual_activation(model, tokenizer, case["prompt"], layer)
        feats = to_sae_features(act, sae)
        activations_on_medabstain.append(feats[top_idx])
    
    # PubMedQA에서의 활성화 패턴과 비교
    # 상관관계 / 활성화 강도 비교
```

## Decision Gate 3
```
통과 조건:
  PubMedQA top feature가 MedAbstain abstain-fail 케이스에서도
  유의미하게 활성화됨

통과 → feature가 데이터셋-agnostic → 강한 결과
실패 → feature가 PubMedQA 특화 → 범위 제한해서 보고
```

---

# Phase 4: Task 수행 (3택 1)

Phase 0~3 결과를 보고 Task를 확정한다. 미리 정하지 말 것.

## Task A: Self-Check Classifier
```
목표: 답이 틀릴지 사전 예측
구현:
  Layer X SAE feature → logistic regression → P(correct)
평가:
  AUC-ROC, sensitivity @ 90% specificity
baseline 비교:
  output logit only / hidden state full / SAE feature (우리)
선택 기준:
  feature가 epistemic state를 인코딩 (Phase 2에서 확인)
  + 모델 수정 없이 안전한 접근 원할 때
```

## Task B: Self-Check + 이유 설명
```
목표: P(correct) 예측 + 왜 틀릴 것 같은지 설명
구현:
  classifier + 활성화된 feature의 interpretation 제공
  예: "feature #X (=evidence-negation) 미활성 → context 무시 의심"
선택 기준:
  Phase 2 interpretation이 깔끔하게 나왔을 때
  가장 강한 contribution이지만 공수 큼
```

## Task C: Steering (환각 완화)
```
목표: context 무시 케이스를 교정
구현:
  ignorance-dominant feature suppress 또는
  correct-dominant feature amplify
평가:
  교정률 + selectivity (맞는 케이스 보존율)
선택 기준:
  Phase 1에서 steering 효과가 재현될 때
  (단 magnitude 적정값 탐색 필요, 과도하면 backfire)
```

## Task 선택 가이드
```
feature interpretation 명확 + 실용성 우선 → Task A 또는 B
steering 효과 재현됨 + 개입 데모 원함 → Task C
공수 최소 + 안전 → Task A
임팩트 최대 → Task B
```

---

# Phase 5: Circuit 분석 (선택, 시간 허용 시)

## 목적
feature가 모델 내부에서 어떤 경로로 작동하는지.

## 방법
```
1. Attention 분석
   - context token vs question token에 대한 attention
   - 어떤 head가 context를 무시하는가

2. Activation Patching
   - context 정보가 어느 레이어에서 처리되는가
   - prior가 어디서 context를 override하는가

3. Path Patching / EAP-IG
   - feature → 출력까지의 information flow
```

## 주의
Phase 4까지 완료 후 진행. 논문의 mechanistic depth를 더하는 용도.
시간 부족하면 future work로 남겨도 됨.

---

# 전체 Decision Tree 요약

```
Phase 0 (모델 검증)
  Context Gain > +5% ? ──No──> backbone 재선정
        │Yes
Phase 1 (feature 발견)
  p<0.05 feature >= 10개 ? ──No──> 케이스 수/레이어 재검토
        │Yes
Phase 2 (interpretation) ★ 가장 중요
  feature가 해석 가능 ? ──No──> framing 재검토 (negative result도 의미)
        │Yes
Phase 3 (MedAbstain)
  cross-dataset 작동 ? ──No──> 범위 제한 보고
        │Yes
Phase 4 (Task 수행)
  A / B / C 중 선택
        │
Phase 5 (Circuit, 선택)
```

---

# 인과관계/필요성 체크 (각 Phase의 존재 이유)

```
Phase 0: 전제 검증 — "모델이 context를 쓸 줄 아는가"
  없으면: 모든 후속 실험이 모래 위의 성

Phase 1: 핵심 발견 — "어떤 feature가 관여하는가"
  MedQA가 아닌 PubMedQA에서 직접 찾음 = 전이 문제 해결

Phase 2: 정당화 — "그 feature가 진짜 context feature인가"
  없으면: "ignorance feature 발견"이 과장 주장이 됨
  → 절대 생략 불가

Phase 3: 일반화 — "특정 데이터셋 artifact 아닌가"
  없으면: reviewer가 "PubMedQA에만 통하는 거 아니냐" 반박

Phase 4: contribution — "그래서 뭘 할 수 있는가"
  없으면: 분석만 하고 so what 없음

Phase 5: depth — "왜 그렇게 작동하는가"
  없어도 됨 (future work). 있으면 mechanistic 강화
```

---

# 버린 것 / 보류한 것 (명시)

```
버림:
  - MedQA (context 없는 순수 지식, 본 연구 범위 밖)
  - MedQA에서 찾은 기존 feature (전이 문제, 비교 reference로만)
  - 데이터 큐레이션 방향 (이미 포화)
  - 공정성 feature 방향 (이미 포화)

보류:
  - Task 확정 (Phase 0~3 결과 후 결정)
  - Circuit 분석 (Phase 4 후)
  - Fine-tuning 전후 비교 (별도 연구 또는 section)
  - Overthinking 방향 (thinking mode, 별도 트랙)
```

---

# 코딩 에이전트 최우선 확인사항

```
1. Qwen-Scope, Gemma Scope 2 SAE의 정확한 repo명과 인터페이스 확인
   - Qwen: TopK 방식
   - Gemma Scope 2: JumpReLU 방식 (다름!)
   - 두 suite를 통일된 wrapper로 추상화

2. 각 모델 tokenizer에서 yes/no/maybe 토큰 ID 확인

3. enable_thinking=False가 Qwen3에 실제 적용되는지 확인

4. 메모리: 12B 모델 로드 시 양자화 필요할 수 있음 (확인)

5. Phase는 순서대로. Gate 통과 못 하면 다음 Phase 진행 금지, 보고할 것
```
