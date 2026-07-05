import os
import torch

# --- 전역 환경 설정 (01_setup_full.py와 동일하게 유지) ---
BASE_DIR = os.environ.get("MEDICAL_MI_BASE_DIR", "/home/eagle0914/medical_mi")
DATA_ROOT = os.environ.get("MEDICAL_MI_DATA_ROOT", "/data/heejae")
os.environ.setdefault("HF_HOME", f"{DATA_ROOT}/.cache/huggingface")
os.environ.setdefault("TMPDIR", f"{DATA_ROOT}/tmp")

def get_sae_path(model_name, layer):
    """
    사용자가 확인한 layer{n}.sae.pt 구조를 최우선으로 탐색
    """
    sae_bases = [
        f"{DATA_ROOT}/checkpoints/sae/{model_name}",
        f"{BASE_DIR}/checkpoints/sae/{model_name}",
    ]

    path_options = []
    for sae_base in sae_bases:
        path_options.extend(
            [
                f"{sae_base}/layer{layer}.sae.pt",
                f"{sae_base}/layer_{layer}/res_64k/sae_weights.pt",
                f"{sae_base}/layer_{layer}/res_64k/params.pt",
            ]
        )
    
    for p in path_options:
        if os.path.exists(p):
            return p
    return None

def format_pubmedqa(item, tokenizer=None, include_context=True):
    """
    PubMedQA를 yes/no/maybe 질문으로 포맷
    """
    context_data = item.get("context", "")
    if isinstance(context_data, dict):
        contexts = context_data.get("contexts", [])
        context_text = " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    else:
        context_text = str(context_data)

    if len(context_text) > 1500:
        context_text = context_text[:1500] + "..."
    
    question = item["question"]
    
    if include_context:
        prompt = f"Context: {context_text}\n\nQuestion: {question}\n\nBased ONLY on the context above, answer with one word: yes, no, or maybe."
    else:
        prompt = f"Question: {question}\n\nAnswer with one word: yes, no, or maybe."
    
    if tokenizer is None:
        return prompt

    messages = [{"role": "user", "content": prompt}]
    
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

def get_ynm_probs(model, tokenizer, prompt):
    """yes/no/maybe 토큰 확률 반환"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    
    result = {}
    for word in ["yes", "no", "maybe"]:
        ids = []
        for variant in [word, word.capitalize(), " " + word, " " + word.capitalize()]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if tok:
                ids.append(tok[0])
        
        if ids:
            result[word] = max(probs[i].item() for i in ids)
        else:
            result[word] = 0.0
    
    total = sum(result.values())
    if total > 0:
        result = {k: v/total for k, v in result.items()}
    else:
        result = {"yes": 0.33, "no": 0.33, "maybe": 0.33}
    return result

def format_medqa(item, tokenizer=None):
    """
    MedQA 포맷: Question + Options A,B,C,D
    """
    question = item["question"]
    options = item["options"]
    opt_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
    
    prompt = f"Question: {question}\n\nOptions:\n{opt_str}\n\nAnswer with one letter (A, B, C, or D):"
    
    if tokenizer is None: return prompt

    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class ActivationFetcher:
    """
    Hook-기반 액티베이션 추출기 (레이어 인덱싱 일관성 보장)
    """
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.activation = None
        self.handle = None

    def hook_fn(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        self.activation = h[:, -1, :].detach()

    def __enter__(self):
        target_layer = self.model.model.layers[self.layer_idx]
        self.handle = target_layer.register_forward_hook(self.hook_fn)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle:
            self.handle.remove()

def get_activation_with_hook(model, tokenizer, prompt, layer_idx):
    with ActivationFetcher(model, layer_idx) as fetcher:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            model(**inputs)
        return fetcher.activation
