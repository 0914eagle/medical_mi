# 실험 지침서 — Steering 재시도 (다중 feature + alpha sweep)

## 배경

단일 feature steering이 거의 작동 안 함 (recovery 0~6.7%).
원인 추정: (1) feature 하나씩이라 약함, (2) alpha=20 고정이 부적절,
(3) steering 방향 미검증.

이전 pilot에서 88.2% 나왔던 건 "여러 feature 동시 steering"이었음.
이번엔 그걸 제대로 재현 + alpha sweep + 방향 양쪽 시도.

## 이미 확립된 것 (재확인)

```
Interpretation에서 context-specific feature 확인됨:
  Feature 2392 (L20): with_context 1.346 / no_context 0.386 / medqa 0.000
  Feature 23783 (qwen3-8b L20): with 0.674 / no 0.165 / medqa 0.000
  → context 있을 때 켜지고, MedQA(context 없음)에서 0

Phase 1에서 Layer 20의 correct_dominant 23개, wrong_dominant 22개 확보됨
  (qwen3.5-9b_phase1_features.json)
```

---

## 실험 1: 다중 feature 동시 steering

## 핵심 변경
```
기존: feature 하나만 amplify
신규: Layer 20의 correct_dominant 전체를 동시에 amplify
      + wrong_dominant 전체를 동시에 suppress (별도 실험)
```

```python
import torch, json
from sae_wrapper import SAEWrapper
from utils import format_pubmedqa

def steer_multi(model, tokenizer, item, layer, sae,
                amplify_idxs=None, suppress_idxs=None, alpha=5.0):
    """
    여러 feature 동시 steering
    amplify_idxs: correct_dominant (더할 feature들)
    suppress_idxs: wrong_dominant (뺄 feature들)
    """
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # steering vector 합성 (decoder 방향들의 합)
    steer_vec = torch.zeros(sae.W_dec.shape[1], device=model.device)
    if amplify_idxs:
        for idx in amplify_idxs:
            steer_vec += sae.W_dec[idx, :].to(model.device)
    if suppress_idxs:
        for idx in suppress_idxs:
            steer_vec -= sae.W_dec[idx, :].to(model.device)

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h[0, -1, :] = h[0, -1, :] + alpha * steer_vec.to(h.dtype)
        return o

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
    handle.remove()

    # yes/no/maybe 처리
    result = {}
    for word in ["yes", "no", "maybe"]:
        ids = []
        for v in [word, word.capitalize(), " "+word, " "+word.capitalize()]:
            t = tokenizer.encode(v, add_special_tokens=False)
            if t: ids.append(t[0])
        result[word] = max(probs[i].item() for i in ids) if ids else 0.0
    tot = sum(result.values())
    return {k: v/tot for k,v in result.items()} if tot>0 else result
```

## 3가지 조건 비교
```python
# 조건 A: correct_dominant amplify
# 조건 B: wrong_dominant suppress
# 조건 C: 둘 다 (amplify + suppress)

conditions = {
    "amplify_correct": {"amplify_idxs": correct_dominant, "suppress_idxs": None},
    "suppress_wrong":  {"amplify_idxs": None, "suppress_idxs": wrong_dominant},
    "both":            {"amplify_idxs": correct_dominant, "suppress_idxs": wrong_dominant},
}
```

---

## 실험 2: Alpha sweep

```python
ALPHAS = [1, 2, 3, 5, 10, 20, 50]

# 각 조건 × 각 alpha에 대해:
#   wrong 케이스 recovery rate
#   correct 케이스 corruption rate

for cond_name, cfg in conditions.items():
    for alpha in ALPHAS:
        recovery = test_on_wrong_cases(cfg, alpha)
        corruption = test_on_correct_cases(cfg, alpha)
        print(f"{cond_name} alpha={alpha}: recovery={recovery:.1%}, corruption={corruption:.1%}")
```

## 측정
```python
def evaluate_steering(model, tokenizer, sae, layer, cfg, alpha,
                       wrong_cases, correct_cases):
    # recovery: wrong → 정답으로
    recovered = 0
    for item in wrong_cases[:30]:
        probs = steer_multi(model, tokenizer, item, layer, sae, alpha=alpha, **cfg)
        pred = max(probs, key=probs.get)
        if pred == item_ground_truth(item):  # 정답으로 바뀜
            recovered += 1
    recovery_rate = recovered / 30

    # corruption: correct → 틀림 (selectivity)
    corrupted = 0
    for item in correct_cases[:30]:
        probs = steer_multi(model, tokenizer, item, layer, sae, alpha=alpha, **cfg)
        pred = max(probs, key=probs.get)
        if pred != item_ground_truth(item):  # 맞던 게 틀림
            corrupted += 1
    corruption_rate = corrupted / 30

    return recovery_rate, corruption_rate
```

## 좋은 결과 기준
```
recovery 높고 corruption 낮은 조합 찾기
예: recovery > 30%, corruption < 10%
→ 그 조건(amplify/suppress/both) + alpha가 최적
```

---

## 실험 3: Alpha 스케일 진단 (먼저 할 것)

steering이 안 먹히는 게 alpha 문제인지 먼저 확인.

```python
# Layer 20에서 실제 feature activation 값의 범위 측정
# (steer_vec * alpha가 원래 activation 대비 얼마나 큰지)

sample_act = get_activation_with_hook(model, tok, sample_prompt, 20)
print(f"Activation norm: {sample_act.norm():.3f}")
print(f"Activation mean abs: {sample_act.abs().mean():.3f}")

steer_vec_norm = steer_vec.norm()
print(f"Steer vec norm: {steer_vec_norm:.3f}")

# alpha * steer_vec_norm 이 activation norm과 비슷한 스케일이어야 함
# 너무 작으면 효과 없고, 너무 크면 모델 망가짐
for alpha in [1, 5, 10, 20, 50]:
    ratio = (alpha * steer_vec_norm) / sample_act.norm()
    print(f"alpha={alpha}: steer/act ratio = {ratio:.3f}")
# ratio가 0.1~1.0 정도가 적절. 0.01이면 너무 약함, 5면 너무 강함
```

---

## 실행 순서

```
1. 실험 3 (alpha 진단) — 먼저, 빠름
   → 적절한 alpha 범위 파악

2. 실험 1+2 (다중 feature + alpha sweep)
   Layer 20, qwen3.5-9b
   3 조건 × 적절한 alpha 범위

3. 최적 조합 나오면 다른 레이어(22, 24)에서도 확인
```

---

## Decision Gate

```
통과: 어떤 조건+alpha에서 recovery > 30%, corruption < 10%
  → steering 살아남 → 주력 결과로

실패: 모든 조합에서 recovery < 15%
  → steering 포기
  → Phase C activation patching을 주력 인과 증거로 전환
     (patching은 Layer 22에서 90% flip 이미 확인됨)
  → steering은 "단일 개입으로는 불충분" 보조 결과
```

---

## 중요: steering이 실패해도 연구는 성립

```
이미 가진 강한 결과:
  1. Interpretation: context-specific feature (MedQA에서 0)
  2. Patching: Layer 18-22 결정 지점 (90% flip)

steering은 "추가 개입 데모"일 뿐
없어도 (1)+(2)로 mechanistic finding 충분

따라서 steering에 과도하게 시간 쓰지 말 것.
3가지 조건 × alpha sweep 한 번 돌리고
안 되면 patching으로 전환.
```

---

## 코딩 에이전트 주의사항

```
1. 실험 3(alpha 진단) 먼저 — 스케일 안 맞으면 나머지 무의미
2. correct_dominant/wrong_dominant는 qwen3.5-9b_phase1_features.json에서 로드
3. steer_vec는 여러 decoder 방향의 합 — 정규화 여부 실험
   (정규화 안 하면 feature 많을수록 vector 커짐)
4. recovery/corruption 둘 다 측정 (selectivity 중요)
5. 안 되면 빨리 포기하고 patching 보고 — 시간 낭비 금지
```
