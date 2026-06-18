# 실험 지침서 — Confident Wrong(틀림 B) 감지

## 배경 (왜 이 실험인가)

기존 conflict score는 같은 coverage에서 random과 차이 없었음.
근본 원인: 우리 wrong feature는 "context 처리하다 실패(틀림 A)"는 잡지만
"context 아예 회피하고 prior로 확신(틀림 B = confident wrong)"은 못 잡음.

```
틀림 A: context 처리 시도 -> 실패. wrong feature 켜짐. 잡힘. 교정됨.
틀림 B: context 회피. prior 확신. wrong feature 약함. 못 잡음. 가장 위험.
```

목표: 틀림 B를 잡는 신호/feature를 찾는다.
이게 되면 SpARE를 넘는 기여, 안 되면 한계로 정직하게 보고.

기존 결과 참고:
- unrecovered 케이스 prior_confidence 0.781 (recovered 0.669)
- conflict score AUC 0.794, 단 precision 56% (절반은 오분류)
- 같은 coverage에서 CAI = always_steer = no_intervention

---

# 실험 1: 틀림 B 케이스 규모 확인 (먼저, 모델 재실행 불필요)

이미 있는 conflict_set.json으로 분류만.

```python
import json
data = json.load(open("results/eval/qwen3.5-9b_conflict_set.json"))

def classify_fine(r):
    gt = r["ground_truth"]
    prior = r["prior_answer"]
    ctx = r["context_answer"]
    prior_conf = max(r["prior_probs"].values())
    ctx_conf = max(r["context_probs"].values())
    if gt != "no":
        return "other"
    # gt == no 인 케이스만
    if ctx == "no":
        return "correct"
    # 틀린 케이스 (ctx != no)
    if prior == ctx:
        # context 줘도 prior와 같은 답 = 무시
        return "typeB_confident_ignore" if ctx_conf >= 0.7 else "typeB_weak"
    else:
        # context 주니 답이 (틀린 방향으로라도) 바뀜
        return "typeA_processing"

from collections import Counter
counts = Counter(classify_fine(r) for r in data)
print(counts)
```

분기:
```
typeB(confident ignore) >= 30개 -> 실험 2 진행
< 30개 -> 여러 모델(Qwen3-8B 등) 합산 또는 maybe 포함 완화
```

저장: 각 케이스 item_id와 분류 라벨.

---

# 실험 2: 틀림 B 특화 Feature 발견

기존은 correct vs 전체 wrong (틀림 A 편향).
신규는 correct vs 틀림 B.

```python
# validation split에서만 (10-fold)
for layer in [16, 18, 20, 22, 24]:
    feat_correct = extract_sae(correct_cases, layer)      # gt=no, ctx=no
    feat_typeB   = extract_sae(typeB_cases, layer)        # gt=no, prior=ctx=yes, 확신>=0.7
    
    t, p = ttest_ind(feat_correct, feat_typeB, axis=0)
    typeB_dominant = where(p<0.01 & t<0)  # typeB에서 더 활성
    correct_dom_vs_B = where(p<0.01 & t>0)
```

핵심 비교:
```
typeB_dominant feature와
기존 wrong_dominant feature가 겹치나?

겹침 많음 -> 같은 feature (틀림 B도 결국 잡힘, 근데 약함)
겹침 적음 -> 틀림 B는 다른 feature가 담당
  -> 이 새 feature를 conflict score에 추가
```

검증: 이 typeB feature가 MedQA에서 0인가 (context-specific 확인).

---

# 실험 3: Context Attention 직접 측정 (핵심 — "무시=안 봄")

feature 말고 attention으로 "context를 봤나"를 직접 측정.

```python
def context_attention(model, item, layer):
    # 프롬프트에서 context token 위치 파악
    prompt = format_pubmedqa(item, tok, include_context=True)
    # "Context:" 와 "Question:" 사이 토큰 = context tokens
    ctx_token_idxs = find_context_span(prompt, tok)
    
    # attention 추출 (output_attentions=True)
    out = model(**inputs, output_attentions=True)
    attn = out.attentions[layer]  # [batch, heads, seq, seq]
    
    # 마지막 토큰 -> context tokens 로의 attention 합
    last_to_ctx = attn[0, :, -1, ctx_token_idxs].sum(dim=-1)  # head별
    return last_to_ctx.mean().item()  # head 평균

# 세 그룹 비교
for group in [correct, typeA, typeB]:
    attns = [context_attention(model, it, layer) for it in group]
    print(f"{group}: mean context attention = {mean(attns)}")
```

가설 검증:
```
틀림 B의 context attention이 correct/틀림A보다 낮은가?
  낮음 -> "무시 = context 안 봄" 직접 증거
       -> attention이 feature보다 나은 신호
  비슷 -> 무시가 attention 문제 아님 (다른 원인)

레이어별 + head별로 측정해서
"어떤 head가 context를 무시하는가" 식별
```

---

# 실험 4: Prior Strength 보조 신호

```python
# prior_confidence = context 없을 때 확신도
# confident wrong은 prior 매우 강함 (기존: unrecovered 0.781)

# 기존 conflict_score에 prior 결합
combined_score = w1 * conflict_score + w2 * prior_confidence
# 또는 prior_confidence 단독

# Risk-Coverage로 평가:
#   combined가 random/conflict_score 단독보다
#   같은 coverage서 risk 낮은가?
```

핵심 측정 (실험 5와 함께 Risk-Coverage):
```
detection 방법별 같은 coverage(0.7)에서 selective accuracy:
  conflict_score 단독 (기존)
  prior_confidence 단독
  conflict + prior 결합
  + (실험 5) context-output 불일치
```

---

# 실험 5: Context-Output 불일치 신호 (가장 직접적)

```python
# "context 줬는데 prior와 같은 답" = 진짜 무시
# behavioral 신호 (feature/attention 불필요)

def disagreement_signal(r):
    # prior_answer와 context_answer가 같으면 무시 의심
    same = (r["prior_answer"] == r["context_answer"])
    # 확신도 차이도 고려
    conf_shift = max(r["prior_probs"].values()) - max(r["context_probs"].values())
    return same, conf_shift

# 이 신호로 wrong 예측 AUC
# 기존 conflict_score(0.794) 이기나?
```

비교 (가장 중요한 표):
```
| detection 방법              | AUC | coverage 0.7 risk |
| conflict_score (기존)        | 0.79 |      15.6%        |
| prior_confidence            |  ?   |       ?          |
| context-output 불일치        |  ?   |       ?          |
| 결합 (전부)                 |  ?   |       ?          |
| random baseline             | 0.50 |      15.6%        |

목표: 결합이 random을 같은 coverage서 명확히 이김
```

---

# 실험 순서 & 분기

```
1. 실험 1 (틀림 B 규모) — 즉시, 데이터만
   -> 30개 미만이면 모델 합산

2. 실험 3 (attention) + 실험 5 (context-output) 우선
   = 네 직관 직접 검증, feature보다 직접적
   -> 둘 중 하나라도 confident wrong 잘 잡으면 핵심 기여

3. 실험 2 (typeB feature) + 실험 4 (prior)
   = 보조 신호 발굴

4. 종합: 최고 detection으로 Risk-Coverage 재평가
   같은 coverage서 baseline 이기나?
```

---

# Decision Gate (연구 사활)

```
성공: 어떤 신호(attention/context-output/결합)가
      같은 coverage(0.7)에서 baseline보다 risk 유의하게 낮음
  -> confident wrong 감지 성공
  -> CAI 살아남, full paper (옵션 B)

실패: 모든 신호가 같은 coverage서 random과 차이 없음
  -> confident wrong은 감지 어렵다는 한계
  -> negative result로 정직하게 (옵션 A, short paper)
  -> "SAE/attention 모두 confident wrong 못 잡음" 자체가 발견
```

---

# 코딩 에이전트 주의사항

```
1. 실험 1은 모델 재실행 없이 conflict_set.json만으로
   -> 틀림 B 케이스 수 먼저 보고

2. attention 측정 시 output_attentions=True
   context token span 정확히 파악 (Context:~Question: 사이)
   Gemma는 attention 구조 다를 수 있음

3. 모든 detection 평가는 같은 coverage 비교 (Risk-Coverage)
   selective accuracy 절대값 비교 금지 (coverage 착시)

4. validation에서 가중치/threshold 정하고 test에서 측정
   10-fold, 평균 +- std

5. 각 실험 결과 json 저장
```
