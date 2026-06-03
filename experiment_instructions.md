# Ignorance Suppression in Medical LLMs — Pilot Experiment Instructions

## 연구 목표

의료 QA에서 Qwen3-8B가 **모르는 케이스에서도 confident하게 틀린 답을 생성하는 현상**의 내부 메커니즘을 Qwen-Scope SAE로 분석한다.

구체적으로:
1. Ignorance suppression 케이스 (모르는데 아는 척하는 케이스)를 수집한다
2. Qwen-Scope SAE로 ignorance와 관련된 feature를 식별한다
3. 그 feature를 steering해서 모델이 모른다고 말하게 만들 수 있는지 확인한다

---

## 환경 요구사항

### 하드웨어
- GPU: 최소 16GB VRAM (Qwen3-8B inference용)
- 권장: A100 40GB 또는 A6000 (SAE activation 저장 시 메모리 필요)
- 스토리지: 50GB 이상 (모델 + SAE checkpoint)

### 소프트웨어

```bash
pip install transformers torch datasets accelerate
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tqdm
pip install huggingface_hub
```

---

## 디렉토리 구조

```
project/
├── data/
│   ├── raw/              # 원본 데이터셋
│   └── processed/        # 전처리된 케이스
├── checkpoints/
│   ├── model/            # Qwen3-8B
│   └── sae/              # Qwen-Scope SAE checkpoints
├── results/
│   ├── features/         # 추출된 SAE feature
│   ├── steering/         # Steering 실험 결과
│   └── figures/          # 시각화
└── scripts/
    ├── 01_setup.py
    ├── 02_collect_cases.py
    ├── 03_extract_features.py
    ├── 04_find_ignorance_features.py
    └── 05_steering_experiment.py
```

---

## Step 0: 모델 및 SAE 다운로드

### 0-1. Qwen3-8B 다운로드

```python
# scripts/01_setup.py

from huggingface_hub import snapshot_download
import os

os.makedirs("checkpoints/model", exist_ok=True)
os.makedirs("checkpoints/sae", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)
os.makedirs("results/features", exist_ok=True)
os.makedirs("results/steering", exist_ok=True)

# Qwen3-8B 다운로드
snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir="checkpoints/model/Qwen3-8B",
    ignore_patterns=["*.msgpack", "flax_model*"]
)
print("Qwen3-8B 다운로드 완료")
```

### 0-2. Qwen-Scope SAE 다운로드

```python
# Qwen3-8B용 SAE (W64K: width 64K, L0_50: TopK k=50)
snapshot_download(
    repo_id="Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50",
    local_dir="checkpoints/sae/Qwen3-8B-SAE",
)
print("Qwen-Scope SAE 다운로드 완료")

# checkpoint 구조 확인
# checkpoints/sae/Qwen3-8B-SAE/
#   layer0.sae.pt
#   layer1.sae.pt
#   ...
#   layer35.sae.pt  (Qwen3-8B는 36개 레이어)
```

### 0-3. 모델 로드 확인

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("checkpoints/model/Qwen3-8B")
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/model/Qwen3-8B",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print(f"모델 로드 완료: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
```

---

## Step 1: MedQA 데이터 준비 및 케이스 수집

### 1-1. MedQA 데이터셋 로드

```python
# scripts/02_collect_cases.py

from datasets import load_dataset
import json

# MedQA USMLE 4-option 데이터셋 로드
dataset = load_dataset("GBaker/MedQA-USMLE-4-options")

# test split 사용 (학습 데이터 오염 방지)
test_data = dataset["test"]
print(f"Test 케이스 수: {len(test_data)}")

# 데이터 구조 확인
print(test_data[0])
# 예시:
# {
#   "question": "A 55-year-old man...",
#   "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
#   "answer": "B",
#   "answer_idx": 1
# }
```

### 1-2. 케이스 분류 핵심 함수

#### Qwen3 Non-thinking Mode 설정

```python
def format_question_for_qwen3(question_text, options):
    """
    Qwen3는 thinking mode가 default.
    Non-thinking mode를 명시적으로 설정해야 일관된 logit 측정 가능.
    
    중요: /no_think 토큰으로 thinking 비활성화
    """
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    
    prompt = f"""{question_text}

Options:
{options_text}

Answer with just the letter (A, B, C, or D). /no_think"""
    
    # Qwen3 chat format 적용
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted
```

#### 답변 선택지 확률 계산

```python
def get_answer_probabilities(model, tokenizer, formatted_prompt):
    """
    A, B, C, D 각 선택지에 대한 확률 반환
    
    방법: next-token logit에서 A/B/C/D 토큰의 확률 추출
    이는 모델이 다음에 A, B, C, D 중 어느 것을 생성할 확률
    """
    inputs = tokenizer(
        formatted_prompt, 
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]
    
    # 마지막 토큰 위치의 logit (다음 토큰 예측)
    last_logits = logits[0, -1, :]  # [vocab_size]
    probs = torch.softmax(last_logits, dim=-1)
    
    # A, B, C, D 토큰 ID 추출
    # 주의: tokenizer마다 토큰 ID 다름, 반드시 확인 필요
    answer_probs = {}
    for choice in ["A", "B", "C", "D"]:
        # 단일 토큰으로 인코딩되는지 확인
        token_ids = tokenizer.encode(choice, add_special_tokens=False)
        
        if len(token_ids) == 1:
            answer_probs[choice] = probs[token_ids[0]].item()
        else:
            # 여러 토큰인 경우 첫 번째 토큰 확률 사용
            # (대부분 A,B,C,D는 단일 토큰)
            answer_probs[choice] = probs[token_ids[0]].item()
    
    # 정규화 (A,B,C,D 중에서의 상대 확률)
    total = sum(answer_probs.values())
    if total > 0:
        answer_probs = {k: v/total for k, v in answer_probs.items()}
    
    return answer_probs
```

#### 케이스 실행 및 분류

```python
def evaluate_case(model, tokenizer, case):
    """
    단일 케이스 평가
    
    Returns:
        dict with keys:
            - model_answer: 모델의 예측 (A/B/C/D)
            - correct_answer: 정답
            - is_correct: 정답 여부
            - confidence: 모델의 예측 확률 (0~1)
            - all_probs: A,B,C,D 전체 확률
            - case_type: 케이스 유형 분류
    """
    formatted = format_question_for_qwen3(
        case["question"], 
        case["options"]
    )
    
    probs = get_answer_probabilities(model, tokenizer, formatted)
    
    model_answer = max(probs, key=probs.get)
    confidence = probs[model_answer]
    correct_answer = case["answer"]
    is_correct = (model_answer == correct_answer)
    
    # 케이스 유형 분류
    # HIGH_CONFIDENCE_THRESHOLD: 모델이 "확신"하는 기준
    HIGH_CONF = 0.70  # A,B,C,D 중 70% 이상이면 높은 확신
    LOW_CONF = 0.40   # 40% 미만이면 낮은 확신
    
    if is_correct and confidence >= HIGH_CONF:
        case_type = "CORRECT_CONFIDENT"      # 알고 맞춤
    elif not is_correct and confidence >= HIGH_CONF:
        case_type = "WRONG_CONFIDENT"        # 모르는데 아는 척 ← 핵심 타겟
    elif is_correct and confidence < LOW_CONF:
        case_type = "CORRECT_UNCERTAIN"      # 알면서 불확실
    else:
        case_type = "WRONG_UNCERTAIN"        # 모르고 불확실
    
    return {
        "question": case["question"],
        "options": case["options"],
        "model_answer": model_answer,
        "correct_answer": correct_answer,
        "is_correct": is_correct,
        "confidence": confidence,
        "all_probs": probs,
        "case_type": case_type
    }
```

### 1-3. 전체 케이스 수집 실행

```python
from tqdm import tqdm
import json

# Pilot: 처음 200개만 (빠른 확인용)
# 본 실험: 전체 test set
N_CASES = 200  # pilot 단계

results = []
for i, case in enumerate(tqdm(test_data.select(range(N_CASES)))):
    try:
        result = evaluate_case(model, tokenizer, case)
        results.append(result)
    except Exception as e:
        print(f"Case {i} 오류: {e}")
        continue

# 결과 저장
with open("data/processed/evaluated_cases.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# 케이스 유형별 분포 확인
from collections import Counter
type_counts = Counter(r["case_type"] for r in results)
print("\n케이스 유형별 분포:")
for t, c in type_counts.items():
    print(f"  {t}: {c}개 ({c/len(results)*100:.1f}%)")

# 케이스 분리 저장
for case_type in ["CORRECT_CONFIDENT", "WRONG_CONFIDENT", 
                   "CORRECT_UNCERTAIN", "WRONG_UNCERTAIN"]:
    subset = [r for r in results if r["case_type"] == case_type]
    with open(f"data/processed/{case_type.lower()}.json", "w") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    print(f"{case_type}: {len(subset)}개 저장")
```

---

## Step 2: SAE Feature 추출

### 2-1. SAE 로드 함수

```python
# scripts/03_extract_features.py

import torch
import json
from pathlib import Path

def load_sae(layer_idx, sae_dir="checkpoints/sae/Qwen3-8B-SAE"):
    """
    특정 레이어의 SAE checkpoint 로드
    
    Qwen-Scope SAE checkpoint 구조:
    {
        "W_enc": Tensor [d_model, d_sae],   # encoder weight
        "b_enc": Tensor [d_sae],             # encoder bias
        "W_dec": Tensor [d_sae, d_model],   # decoder weight
        "b_dec": Tensor [d_model],           # decoder bias (pre-encoder bias)
    }
    
    d_model: 4096 (Qwen3-8B hidden size)
    d_sae: 65536 (64K width)
    """
    sae_path = Path(sae_dir) / f"layer{layer_idx}.sae.pt"
    
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {sae_path}")
    
    sae = torch.load(sae_path, map_location="cpu")
    print(f"Layer {layer_idx} SAE 로드 완료")
    print(f"  W_enc shape: {sae['W_enc'].shape}")
    print(f"  d_model: {sae['W_enc'].shape[0]}, d_sae: {sae['W_enc'].shape[1]}")
    
    return sae
```

### 2-2. Activation 추출 함수

```python
def get_residual_stream_activation(model, tokenizer, text, layer_idx):
    """
    특정 레이어의 residual stream activation 추출
    
    Qwen3-8B 구조:
    - 36개 transformer layer
    - 각 레이어 후 residual stream: [batch, seq_len, 4096]
    - 마지막 토큰의 activation을 사용 (답변 직전)
    
    Args:
        layer_idx: 분석할 레이어 인덱스 (0-35)
    
    Returns:
        activation: Tensor [4096] (마지막 토큰의 residual stream)
    """
    activation_store = {}
    
    def hook_fn(module, input, output):
        # output: (hidden_states, ...) 또는 hidden_states
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # 마지막 토큰 activation 저장
        activation_store["activation"] = hidden[0, -1, :].detach().cpu()
    
    # 해당 레이어에 hook 등록
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        model(**inputs)
    
    handle.remove()
    
    return activation_store["activation"]  # [4096]


def activation_to_sae_features(activation, sae):
    """
    Residual stream activation을 SAE feature로 변환
    
    TopK SAE (k=50): 가장 활성화된 50개 feature만 non-zero
    
    Args:
        activation: Tensor [d_model]
        sae: dict with W_enc, b_enc, W_dec, b_dec
    
    Returns:
        features: Tensor [d_sae] (sparse, 50개만 non-zero)
    """
    W_enc = sae["W_enc"].float()  # [d_model, d_sae]
    b_enc = sae["b_enc"].float()  # [d_sae]
    
    # Pre-encoder bias 적용 (있는 경우)
    if "b_dec" in sae:
        activation = activation.float() - sae["b_dec"].float()
    
    # Encode
    pre_activations = activation @ W_enc + b_enc  # [d_sae]
    
    # TopK activation (k=50)
    k = 50
    topk_values, topk_indices = torch.topk(pre_activations, k)
    
    # ReLU 적용 (음수는 0)
    topk_values = torch.relu(topk_values)
    
    # Sparse feature vector 생성
    features = torch.zeros(W_enc.shape[1])
    features[topk_indices] = topk_values
    
    return features  # [d_sae], 50개만 non-zero
```

### 2-3. 케이스 그룹별 Feature 추출

```python
def extract_features_for_cases(model, tokenizer, cases, sae, layer_idx, 
                                 max_cases=100):
    """
    케이스 리스트에서 SAE feature 추출
    
    Args:
        cases: 케이스 리스트 (evaluate_case 결과)
        max_cases: 추출할 최대 케이스 수 (메모리 관리)
    
    Returns:
        feature_matrix: Tensor [n_cases, d_sae]
    """
    features_list = []
    
    for case in tqdm(cases[:max_cases]):
        # 질문을 다시 포맷
        formatted = format_question_for_qwen3(
            case["question"],
            case["options"]
        )
        
        # Activation 추출
        activation = get_residual_stream_activation(
            model, tokenizer, formatted, layer_idx
        )
        
        # SAE feature 변환
        features = activation_to_sae_features(activation, sae)
        features_list.append(features)
    
    return torch.stack(features_list)  # [n_cases, d_sae]


# 분석할 레이어 선택
# Qwen3-8B: 36개 레이어 (0-35)
# 전략: 중간 레이어부터 시작 (layer 15, 20, 25)
# 이유: 지식 표현은 중간-후반 레이어에 집중되는 경향

TARGET_LAYERS = [15, 20, 25]  # pilot에서 3개 레이어 비교

# 케이스 로드
with open("data/processed/correct_confident.json") as f:
    correct_cases = json.load(f)
with open("data/processed/wrong_confident.json") as f:
    ignorance_cases = json.load(f)  # 핵심 타겟

print(f"Correct confident: {len(correct_cases)}개")
print(f"Ignorance (wrong confident): {len(ignorance_cases)}개")

# 각 레이어에서 feature 추출
for layer_idx in TARGET_LAYERS:
    print(f"\nLayer {layer_idx} 분석 중...")
    
    sae = load_sae(layer_idx)
    
    correct_features = extract_features_for_cases(
        model, tokenizer, correct_cases, sae, layer_idx, max_cases=100
    )
    ignorance_features = extract_features_for_cases(
        model, tokenizer, ignorance_cases, sae, layer_idx, max_cases=100
    )
    
    torch.save(correct_features, 
               f"results/features/correct_confident_layer{layer_idx}.pt")
    torch.save(ignorance_features, 
               f"results/features/wrong_confident_layer{layer_idx}.pt")
    
    print(f"  correct_confident features: {correct_features.shape}")
    print(f"  wrong_confident features: {ignorance_features.shape}")
```

---

## Step 3: Ignorance Feature 후보 찾기

```python
# scripts/04_find_ignorance_features.py

import torch
import numpy as np
from scipy import stats

def find_ignorance_features(correct_features, ignorance_features, 
                             layer_idx, top_k=50):
    """
    Ignorance feature 후보 식별
    
    방법 1: Mean activation difference
      - correct_confident에서 더 활성화되는 feature
      = ignorance 시 suppressed되는 feature
    
    방법 2: T-test
      - 통계적으로 유의미하게 다른 feature
    
    Args:
        correct_features: Tensor [n, d_sae] (correct confident 케이스)
        ignorance_features: Tensor [n, d_sae] (wrong confident 케이스)
    
    Returns:
        top_features: 상위 feature indices
        scores: 각 feature의 차이 점수
    """
    correct_mean = correct_features.mean(0)      # [d_sae]
    ignorance_mean = ignorance_features.mean(0)  # [d_sae]
    
    # 방법 1: Mean difference
    # correct에서 더 활성화 = ignorance 시 suppressed
    mean_diff = correct_mean - ignorance_mean    # [d_sae]
    
    # 방법 2: T-test (통계적 유의성)
    t_stats = []
    p_values = []
    
    for feature_idx in range(correct_features.shape[1]):
        c_vals = correct_features[:, feature_idx].numpy()
        i_vals = ignorance_features[:, feature_idx].numpy()
        
        # 둘 다 0인 feature는 skip
        if c_vals.sum() == 0 and i_vals.sum() == 0:
            t_stats.append(0)
            p_values.append(1.0)
            continue
        
        try:
            t, p = stats.ttest_ind(c_vals, i_vals)
            t_stats.append(t if not np.isnan(t) else 0)
            p_values.append(p if not np.isnan(p) else 1.0)
        except:
            t_stats.append(0)
            p_values.append(1.0)
    
    t_stats = torch.tensor(t_stats)
    p_values = torch.tensor(p_values)
    
    # Top-K feature (mean_diff 기준)
    top_indices = torch.topk(mean_diff, k=top_k).indices
    
    # 결과 정리
    results = {
        "top_feature_indices": top_indices.tolist(),
        "mean_diff_scores": mean_diff[top_indices].tolist(),
        "t_stats": t_stats[top_indices].tolist(),
        "p_values": p_values[top_indices].tolist(),
        "correct_mean_activation": correct_mean[top_indices].tolist(),
        "ignorance_mean_activation": ignorance_mean[top_indices].tolist()
    }
    
    print(f"\nLayer {layer_idx} Top-10 Ignorance Feature 후보:")
    for i in range(min(10, top_k)):
        idx = top_indices[i].item()
        diff = mean_diff[idx].item()
        t = t_stats[idx].item()
        p = p_values[idx].item()
        print(f"  Feature #{idx:6d}: diff={diff:.4f}, t={t:.2f}, p={p:.4f}")
    
    return results


# 레이어별 분석 실행
all_layer_results = {}

for layer_idx in TARGET_LAYERS:
    correct_features = torch.load(
        f"results/features/correct_confident_layer{layer_idx}.pt"
    )
    ignorance_features = torch.load(
        f"results/features/wrong_confident_layer{layer_idx}.pt"
    )
    
    results = find_ignorance_features(
        correct_features, ignorance_features, layer_idx, top_k=50
    )
    all_layer_results[layer_idx] = results

# 결과 저장
import json
with open("results/features/ignorance_feature_candidates.json", "w") as f:
    json.dump(all_layer_results, f, indent=2)

print("\n완료. 결과 저장: results/features/ignorance_feature_candidates.json")
```

---

## Step 4: Steering 실험

```python
# scripts/05_steering_experiment.py

import torch
import json
from tqdm import tqdm


def steer_and_evaluate(model, tokenizer, case, sae, 
                        feature_indices, magnitude, layer_idx):
    """
    Feature steering 적용 후 답변 평가
    
    Args:
        feature_indices: amplify할 feature 인덱스 리스트
        magnitude: steering 강도 (1.0 = 2배 증폭, 2.0 = 3배 증폭)
    
    Returns:
        dict with original and steered results
    """
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        
        # SAE encode
        W_enc = sae["W_enc"].float().to(hidden.device)
        b_enc = sae["b_enc"].float().to(hidden.device)
        W_dec = sae["W_dec"].float().to(hidden.device)
        
        hidden_float = hidden.float()
        if "b_dec" in sae:
            hidden_float = hidden_float - sae["b_dec"].float().to(hidden.device)
        
        pre_act = hidden_float @ W_enc + b_enc
        
        # TopK
        k = 50
        topk_vals, topk_idx = torch.topk(pre_act, k, dim=-1)
        topk_vals = torch.relu(topk_vals)
        
        features = torch.zeros_like(pre_act)
        features.scatter_(-1, topk_idx, topk_vals)
        
        # Ignorance feature AMPLIFY
        # 이 feature들이 suppressed되어 있다는 가설 하에 증폭
        feature_tensor = torch.tensor(
            feature_indices, dtype=torch.long, device=hidden.device
        )
        features[:, :, feature_tensor] *= (1 + magnitude)
        
        # Decode back
        modified = features @ W_dec
        if "b_dec" in sae:
            modified = modified + sae["b_dec"].float().to(hidden.device)
        
        modified = modified.to(hidden.dtype)
        
        if rest is not None:
            return (modified,) + rest
        return modified
    
    formatted = format_question_for_qwen3(case["question"], case["options"])
    
    # Original (steering 없음)
    original_probs = get_answer_probabilities(model, tokenizer, formatted)
    original_answer = max(original_probs, key=original_probs.get)
    original_confidence = original_probs[original_answer]
    
    # Steered
    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    steered_probs = get_answer_probabilities(model, tokenizer, formatted)
    handle.remove()
    
    steered_answer = max(steered_probs, key=steered_probs.get)
    steered_confidence = steered_probs[steered_answer]
    
    # Entropy 계산 (불확실성 지표)
    # 높은 entropy = 불확실 = "모른다"에 가까운 상태
    def entropy(probs_dict):
        import math
        return -sum(p * math.log(p + 1e-10) 
                   for p in probs_dict.values())
    
    return {
        "question": case["question"][:100] + "...",
        "correct_answer": case["correct_answer"],
        
        "original_answer": original_answer,
        "original_confidence": original_confidence,
        "original_entropy": entropy(original_probs),
        "original_probs": original_probs,
        
        "steered_answer": steered_answer,
        "steered_confidence": steered_confidence,
        "steered_entropy": entropy(steered_probs),
        "steered_probs": steered_probs,
        
        "confidence_change": steered_confidence - original_confidence,
        "entropy_change": entropy(steered_probs) - entropy(original_probs),
        
        # 핵심 지표:
        # confidence가 낮아지고 entropy가 높아지면
        # → 모델이 더 불확실해진 것 = steering 성공
        "became_uncertain": steered_confidence < 0.5 and original_confidence >= 0.7
    }


# 실험 설계
# ignorance_cases (WRONG_CONFIDENT)에서 pilot 실험
with open("data/processed/wrong_confident.json") as f:
    ignorance_cases = json.load(f)

# 사용할 feature 후보 (Step 3에서 찾은 것)
with open("results/features/ignorance_feature_candidates.json") as f:
    candidates = json.load(f)

# Layer 20을 primary로 사용 (조정 가능)
PRIMARY_LAYER = 20
top_features = candidates[str(PRIMARY_LAYER)]["top_feature_indices"][:20]
sae = load_sae(PRIMARY_LAYER)

# Magnitude sweep
magnitudes = [0.5, 1.0, 2.0, 3.0, 5.0]
n_test_cases = 30  # pilot: 30개

print(f"Steering 실험 시작")
print(f"Layer: {PRIMARY_LAYER}, Features: {top_features[:5]}..., Cases: {n_test_cases}")

all_results = {}

for magnitude in magnitudes:
    print(f"\nMagnitude: {magnitude}")
    results = []
    
    for case in tqdm(ignorance_cases[:n_test_cases]):
        result = steer_and_evaluate(
            model, tokenizer, case, sae,
            feature_indices=top_features,
            magnitude=magnitude,
            layer_idx=PRIMARY_LAYER
        )
        results.append(result)
    
    # 집계
    became_uncertain = sum(r["became_uncertain"] for r in results)
    avg_conf_change = sum(r["confidence_change"] for r in results) / len(results)
    avg_entropy_change = sum(r["entropy_change"] for r in results) / len(results)
    
    print(f"  불확실해진 케이스: {became_uncertain}/{n_test_cases} ({became_uncertain/n_test_cases*100:.1f}%)")
    print(f"  평균 confidence 변화: {avg_conf_change:+.4f}")
    print(f"  평균 entropy 변화: {avg_entropy_change:+.4f}")
    
    all_results[magnitude] = {
        "became_uncertain_rate": became_uncertain / n_test_cases,
        "avg_confidence_change": avg_conf_change,
        "avg_entropy_change": avg_entropy_change,
        "individual_results": results
    }

# 저장
with open("results/steering/magnitude_sweep_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n실험 완료. 결과 저장: results/steering/magnitude_sweep_results.json")
```

---

## Step 5: 결과 분석 및 시각화

```python
# results_analysis.py

import json
import matplotlib.pyplot as plt
import numpy as np

def plot_steering_results(results_path="results/steering/magnitude_sweep_results.json"):
    with open(results_path) as f:
        results = json.load(f)
    
    magnitudes = [float(k) for k in results.keys()]
    uncertain_rates = [results[str(m)]["became_uncertain_rate"] for m in magnitudes]
    conf_changes = [results[str(m)]["avg_confidence_change"] for m in magnitudes]
    entropy_changes = [results[str(m)]["avg_entropy_change"] for m in magnitudes]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(magnitudes, uncertain_rates, "b-o")
    axes[0].set_xlabel("Steering Magnitude")
    axes[0].set_ylabel("Rate")
    axes[0].set_title("Ignorance Cases → Became Uncertain")
    axes[0].axhline(y=0.5, color='r', linestyle='--', label='50% threshold')
    axes[0].legend()
    
    axes[1].plot(magnitudes, conf_changes, "r-o")
    axes[1].set_xlabel("Steering Magnitude")
    axes[1].set_ylabel("Confidence Change")
    axes[1].set_title("Average Confidence Change\n(negative = less confident)")
    axes[1].axhline(y=0, color='gray', linestyle='--')
    
    axes[2].plot(magnitudes, entropy_changes, "g-o")
    axes[2].set_xlabel("Steering Magnitude")
    axes[2].set_ylabel("Entropy Change")
    axes[2].set_title("Average Entropy Change\n(positive = more uncertain)")
    axes[2].axhline(y=0, color='gray', linestyle='--')
    
    plt.tight_layout()
    plt.savefig("results/figures/steering_magnitude_sweep.png", dpi=150)
    plt.show()
    
    print("\n핵심 결과 요약:")
    for m in magnitudes:
        ur = results[str(m)]["became_uncertain_rate"]
        cc = results[str(m)]["avg_confidence_change"]
        print(f"  magnitude={m}: uncertain_rate={ur:.2%}, conf_change={cc:+.4f}")


plot_steering_results()
```

---

## 예상 결과 및 해석

### Pilot 실험 후 확인할 것

**긍정적 결과 (이 연구가 된다는 신호):**
```
magnitude=2.0에서:
  became_uncertain_rate > 30%
  avg_confidence_change < -0.1
  avg_entropy_change > 0.2
→ "Ignorance feature steering이 효과 있다"
→ 본 실험으로 진행
```

**부정적 결과 (수정 필요):**
```
모든 magnitude에서:
  became_uncertain_rate < 10%
  confidence 변화 없음
→ 가능한 원인:
  1. 레이어 선택 잘못됨 → TARGET_LAYERS 변경
  2. Feature 선택 기준 수정 필요 → T-test p-value 기준 강화
  3. Confidence threshold 조정 필요 → HIGH_CONF 값 변경
  4. Qwen3의 thinking mode가 개입 → 포맷 확인
```

---

## 주의사항 및 디버깅

### 자주 발생하는 오류

**1. VRAM OOM:**
```python
# 해결: float16 사용 + 배치 크기 1
model = AutoModelForCausalLM.from_pretrained(
    ...,
    torch_dtype=torch.float16,
    device_map="auto"
)
```

**2. SAE checkpoint 키 이름 다를 경우:**
```python
# checkpoint 구조 먼저 확인
sae = torch.load("layer20.sae.pt")
print(sae.keys())  # 실제 키 이름 확인 후 코드 수정
```

**3. Qwen3 토큰 ID 확인:**
```python
# A,B,C,D 토큰 ID 반드시 확인
for choice in ["A", "B", "C", "D"]:
    ids = tokenizer.encode(choice, add_special_tokens=False)
    print(f"{choice}: {ids}")
# 결과가 단일 토큰인지 확인 필수
```

**4. Hook이 제대로 작동하는지 확인:**
```python
# activation 저장 확인
test_activation = get_residual_stream_activation(
    model, tokenizer, "Test input", layer_idx=20
)
print(f"Activation shape: {test_activation.shape}")  # [4096]이어야 함
print(f"Non-zero values: {(test_activation != 0).sum()}")
```

---

## 체크리스트

```
□ Step 0: 환경 세팅
  □ Qwen3-8B 다운로드 완료
  □ Qwen-Scope SAE 다운로드 완료
  □ 모델 로드 확인 (inference 테스트)

□ Step 1: 케이스 수집
  □ MedQA 로드 확인
  □ A,B,C,D 토큰 ID 확인
  □ evaluate_case 단일 케이스 테스트
  □ 200개 케이스 분류 완료
  □ WRONG_CONFIDENT 케이스 최소 30개 확보

□ Step 2: Feature 추출
  □ SAE 로드 확인 (layer20.sae.pt)
  □ get_residual_stream_activation 테스트
  □ activation_to_sae_features 테스트
  □ Layer 15, 20, 25 feature 추출 완료

□ Step 3: Feature 후보 식별
  □ Mean difference 계산 완료
  □ T-test 완료
  □ Top-50 feature 저장 완료

□ Step 4: Steering 실험
  □ steer_and_evaluate 단일 케이스 테스트
  □ Magnitude sweep (0.5~5.0) 완료
  □ 결과 저장 완료

□ Step 5: 결과 분석
  □ 시각화 완료
  □ 결과 해석 및 문서화
```

---

## 실험 완료 후 교수님께 보고할 내용

```
1. 케이스 분포
   - 전체 N개 중 WRONG_CONFIDENT M개 (X%)
   - 이게 "ignorance suppression"의 base rate

2. Feature 분석
   - Layer Y에서 가장 큰 차이 발견
   - Top feature #XXXXX: correct에서 Z배 더 활성화

3. Steering 결과
   - magnitude=2.0에서 A%의 케이스가 uncertain으로 변화
   - confidence 평균 B만큼 감소

4. 결론
   - "된다" → 본 실험 진행 제안
   - "안 된다" → 어디서 막혔는지 + 수정 방향 제안
```
