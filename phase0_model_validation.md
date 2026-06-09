# 실험 지침서 — 모델 의료 지식 검증 (Phase 0)

## 이 실험의 목적

본 연구의 전제는 "Qwen 모델이 의료 지식을 충분히 갖고 있다"는 것이다.
이 전제가 깨지면 (모델이 의료 QA를 너무 못하면) SAE feature 분석 자체가
무의미해진다. 따라서 본 실험 전에 backbone 후보들의 의료 QA 성능을 검증한다.

핵심 질문:
- Qwen3-8B, Qwen3.5-9B가 의료 벤치마크에서 충분한 성능을 보이는가?
- MedGemma 수준의 성능이 나오는가?
- 3개 backbone에서 일관된 결과가 나오는가?

---

## ⚠️ 중요 제약: SAE 가용성

SAE를 직접 학습하지 않는다 (교수님 지침: 공수 최소화).
따라서 사전 학습된 SAE가 있는 모델만 사용한다.

### 사용 가능한 backbone (SAE 있음)

| 모델 | SAE | 용도 |
|---|---|---|
| Qwen3-8B | Qwen-Scope | Primary 1 |
| Qwen3.5-9B | Qwen-Scope | Primary 2 (Qwen 2개째) |
| Gemma-2-9B | Gemma Scope | Cross-family 검증 |

### 사용 불가 (SAE 없음 — 직접 학습해야 함)

- Qwen3.6 계열 (35B-A3B, 27B): Qwen-Scope 미제공
- Qwen3.7 Max: closed-weight, 내부 접근 불가

만약 Qwen3.6를 꼭 써야 한다면 SAE 학습이 선행되어야 하며,
이는 별도 의사결정이 필요하다. 본 지침서는 SAE 있는 모델만 다룬다.

---

## Phase 0: 모델 의료 지식 검증

### Step 0-1: 환경 및 모델 다운로드

```python
from huggingface_hub import snapshot_download
import os

MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "gemma2-9b": "google/gemma-2-9b",
}

SAE_REPOS = {
    "qwen3-8b": "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
    "qwen3.5-9b": "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50",  # 정확한 repo명 확인 필요
    "gemma2-9b": "google/gemma-scope-9b-pt-res",  # Gemma Scope
}

for name, repo in MODELS.items():
    snapshot_download(repo_id=repo, local_dir=f"checkpoints/model/{name}")

# SAE repo명은 HuggingFace에서 실제 확인 후 수정할 것
# Qwen-Scope: https://huggingface.co/Qwen 에서 SAE-Res-* 검색
# Gemma Scope: google/gemma-scope-9b-pt-res
```

**주의:** SAE repo 이름은 반드시 HuggingFace에서 실제 확인 후 사용.
Qwen-Scope의 정확한 명명 규칙을 먼저 확인하고 진행.

### Step 0-2: PubMedQA 데이터 준비

```python
from datasets import load_dataset

# PubMedQA labeled set (1000개, yes/no/maybe + context 포함)
pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
data = pubmedqa["train"]  # 1000개

print(f"전체: {len(data)}")
# final_decision 분포 확인
from collections import Counter
print(Counter(item["final_decision"] for item in data))
# 예상: yes ~552, no ~338, maybe ~110
```

### Step 0-3: 평가 함수 (context 포함이 핵심)

```python
import torch

def format_pubmedqa(item, tokenizer, include_context=True):
    """
    PubMedQA를 yes/no/maybe 질문으로 포맷
    
    include_context=True: 논문 초록(연구 데이터)을 포함
    → 이게 본 연구의 핵심. 모델이 context를 보고 판단하는지 측정
    
    Qwen3는 non-thinking mode (enable_thinking=False) 사용
    """
    contexts = item["context"]["contexts"]
    context_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    if len(context_text) > 1500:
        context_text = context_text[:1500] + "..."
    
    question = item["question"]
    
    if include_context:
        prompt = f"""Context: {context_text}

Question: {question}

Based ONLY on the context above, answer with one word: yes, no, or maybe."""
    else:
        prompt = f"""Question: {question}

Answer with one word: yes, no, or maybe."""
    
    messages = [{"role": "user", "content": prompt}]
    # Qwen3: enable_thinking=False / Gemma는 해당 옵션 없음
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
    except TypeError:
        # Gemma 등 enable_thinking 미지원 모델
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def get_ynm_probabilities(model, tokenizer, prompt):
    """yes/no/maybe 토큰 확률 반환"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    result = {}
    for word in ["yes", "no", "maybe"]:
        # 소문자/대문자 첫 토큰 모두 고려
        ids = []
        for variant in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if tok:
                ids.append(tok[0])
        result[word] = max(probs[i].item() for i in ids) if ids else 0.0
    
    total = sum(result.values())
    if total > 0:
        result = {k: v/total for k, v in result.items()}
    return result
```

### Step 0-4: 3개 모델 성능 측정

```python
from tqdm import tqdm
import json

def evaluate_model(model, tokenizer, data, include_context=True):
    results = []
    for item in tqdm(data):
        prompt = format_pubmedqa(item, tokenizer, include_context)
        probs = get_ynm_probabilities(model, tokenizer, prompt)
        pred = max(probs, key=probs.get)
        confidence = probs[pred]
        gt = item["final_decision"]
        results.append({
            "question": item["question"][:100],
            "prediction": pred,
            "ground_truth": gt,
            "is_correct": pred == gt,
            "confidence": confidence,
            "probs": probs,
        })
    accuracy = sum(r["is_correct"] for r in results) / len(results)
    return accuracy, results

# 각 모델에 대해 context 있을 때와 없을 때 모두 측정
summary = {}
for name in MODELS:
    model, tokenizer = load_model(name)  # 구현 필요
    
    # context 포함 (본 연구 설정)
    acc_with, res_with = evaluate_model(model, tokenizer, data, include_context=True)
    # context 미포함 (비교용 — prior knowledge만으로)
    acc_without, res_without = evaluate_model(model, tokenizer, data, include_context=False)
    
    summary[name] = {
        "accuracy_with_context": acc_with,
        "accuracy_without_context": acc_without,
        "context_gain": acc_with - acc_without,
    }
    json.dump(res_with, open(f"results/eval/{name}_with_context.json", "w"), indent=2)
    json.dump(res_without, open(f"results/eval/{name}_without_context.json", "w"), indent=2)
    
    del model
    torch.cuda.empty_cache()

print(json.dumps(summary, indent=2))
```

### Step 0-5: 핵심 분석 — Context Gain

```python
# 이 분석이 본 연구의 전제를 검증한다

for name, s in summary.items():
    print(f"\n{name}:")
    print(f"  Context 있을 때 정확도: {s['accuracy_with_context']:.1%}")
    print(f"  Context 없을 때 정확도: {s['accuracy_without_context']:.1%}")
    print(f"  Context Gain: {s['context_gain']:+.1%}")
```

**Context Gain의 의미:**

```
Context Gain이 크다 (예: +15%):
  → context가 있으면 성능이 크게 오름
  → 모델이 context를 활용할 줄 안다
  → "context를 무시하는 케이스"가 우리 타겟으로 의미있음

Context Gain이 작다/음수:
  → context가 있어도 성능이 안 오름
  → 모델이 애초에 context를 못 쓰거나
    이미 prior로 다 맞추거나
  → 연구 전제 재검토 필요
```

---

## 검증 결과에 따른 의사결정

```
조건 1: Qwen3-8B가 PubMedQA에서 65% 이상 (context 포함)
  → 의료 지식 충분, 본 실험 진행 가능

조건 2: Context Gain이 양수 (+5% 이상)
  → 모델이 context를 활용함
  → "context 무시 케이스" 타겟이 유효

조건 3: 3개 모델에서 일관된 패턴
  → cross-model generalization 기대 가능

세 조건 모두 만족 → Phase 1 (feature 찾기) 진행
하나라도 불만족 → 원인 분석 후 재설계
```

---

## Phase 1 미리보기 (Phase 0 통과 시)

Phase 0이 통과하면 다음을 진행한다. (지금 코딩하지 말 것, Phase 0 결과 먼저)

```
1. PubMedQA에서 직접 feature 찾기
   - correct 케이스 (context 반영, 정답)
   - ignorance 케이스 (context 무시, ground truth no인데 yes 답변)
   - 두 그룹의 Layer별 SAE feature 차이
   - 3개 모델 각각

2. MedAbstain에서 같은 feature 검증
   - 다른 데이터셋에서도 작동하는가

3. Task 확정 (검증 후 결정)
   - classifier / 설명 / steering 중 선택
```

---

## 체크리스트

```
□ SAE repo 이름 HuggingFace에서 정확히 확인
□ Qwen3-8B, Qwen3.5-9B, Gemma-2-9B 다운로드
□ PubMedQA 로드 + final_decision 분포 확인
□ yes/no/maybe 토큰 ID 확인 (각 tokenizer마다)
□ enable_thinking=False 적용 확인 (Qwen3 계열)
□ 3개 모델 × (context 有/無) = 6개 평가
□ Context Gain 분석
□ 의사결정 조건 충족 여부 판단
```

---

## 코딩 에이전트 주의사항

1. SAE repo 이름을 추측하지 말고 HuggingFace에서 실제 확인할 것
2. 각 모델의 tokenizer에서 yes/no/maybe가 단일 토큰인지 확인
3. Gemma는 enable_thinking 옵션이 없으므로 try/except 처리
4. context를 1500자로 자를 때 결론 부분이 잘리지 않는지 확인
   (PubMedQA context는 이미 결론 제외된 초록이므로 괜찮지만 확인)
5. 메모리 관리: 모델 하나씩 로드하고 해제 (3개 동시 로드 불가)
```
