import json
import math
import os
import random
import urllib.request
from contextlib import contextmanager

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import ttest_ind
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_wrapper import SAEWrapper
from utils import (
    format_medqa,
    format_pubmedqa,
    get_activation_with_hook,
    get_sae_path,
    get_ynm_probs,
)


BASE_DIR = os.environ.get("MEDICAL_MI_BASE_DIR", "/workspace/medical_mi")
RESULTS_DIR = f"{BASE_DIR}/results"

MODELS = {
    "qwen3-8b": f"{BASE_DIR}/checkpoints/model/qwen3-8b",
    "qwen3.5-9b": f"{BASE_DIR}/checkpoints/model/qwen3.5-9b",
    "gemma3-12b-it": f"{BASE_DIR}/checkpoints/model/gemma3-12b-it",
}

PUBMEDQA_OFFICIAL_TEST_URLS = [
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json",
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/pqal_test_set.json",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_model_and_tokenizer(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Available: {sorted(MODELS)}")
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_name],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_pubmedqa_items():
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    return [dict(item) for item in dataset]


def pubmedqa_id(item, index=None):
    value = item.get("pubid", item.get("id", index))
    return str(value)


def item_map_by_id(items):
    return {pubmedqa_id(item, index): item for index, item in enumerate(items)}


def load_medqa_items(split="test", limit=None):
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options")[split]
    items = [dict(item) for item in dataset]
    return items[:limit] if limit else items


def _extract_pubmedqa_ids(payload):
    if isinstance(payload, dict):
        keys = set(payload.keys())
        if keys and all(str(key).isdigit() for key in keys):
            return {str(key) for key in keys}
        ids = set()
        for value in payload.values():
            ids.update(_extract_pubmedqa_ids(value))
        return ids
    if isinstance(payload, list):
        ids = set()
        for value in payload:
            if isinstance(value, (str, int)):
                ids.add(str(value))
            elif isinstance(value, dict):
                if "pubid" in value:
                    ids.add(str(value["pubid"]))
                elif "id" in value:
                    ids.add(str(value["id"]))
                else:
                    ids.update(_extract_pubmedqa_ids(value))
        return ids
    return set()


def load_official_pubmedqa_test_ids(path=None, allow_download=True):
    if path and os.path.exists(path):
        return _extract_pubmedqa_ids(read_json(path))
    if not allow_download:
        return set()
    for url in PUBMEDQA_OFFICIAL_TEST_URLS:
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            ids = _extract_pubmedqa_ids(payload)
            if ids:
                return ids
        except Exception:
            continue
    return set()


def make_pubmedqa_splits(items, folds=5, seed=13, official_test_ids=None):
    ids = [pubmedqa_id(item, index) for index, item in enumerate(items)]
    if official_test_ids:
        official = {str(item_id) for item_id in official_test_ids}
        test_ids = [item_id for item_id in ids if item_id in official]
        val_ids = [item_id for item_id in ids if item_id not in official]
        if test_ids and val_ids:
            return [{"fold": 0, "validation_ids": val_ids, "test_ids": test_ids}]

    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    fold_size = int(math.ceil(len(shuffled) / folds))
    split_defs = []
    for fold in range(folds):
        start = fold * fold_size
        end = min(start + fold_size, len(shuffled))
        test_ids = shuffled[start:end]
        test_set = set(test_ids)
        val_ids = [item_id for item_id in shuffled if item_id not in test_set]
        split_defs.append({"fold": fold, "validation_ids": val_ids, "test_ids": test_ids})
    return split_defs


def evaluate_pubmedqa(model, tokenizer, items, include_context=True, desc="PubMedQA"):
    results = []
    for index, item in enumerate(tqdm(items, desc=desc)):
        prompt = format_pubmedqa(item, tokenizer, include_context=include_context)
        probs = get_ynm_probs(model, tokenizer, prompt)
        prediction = max(probs, key=probs.get)
        target = item["final_decision"]
        results.append(
            {
                "item_id": pubmedqa_id(item, index),
                "ground_truth": target,
                "prediction": prediction,
                "probs": probs,
                "confidence": probs[prediction],
                "is_correct": prediction == target,
                "include_context": include_context,
            }
        )
    return results


def summarize_labels(labels):
    total = len(labels)
    correct = sum(1 for row in labels if row["is_correct"])
    return {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": correct / total if total else 0.0,
    }


def labels_by_id(labels):
    return {str(row["item_id"]): row for row in labels}


def select_cases(items_by_id, label_map, item_ids, want_correct):
    cases = []
    for item_id in item_ids:
        label = label_map.get(str(item_id))
        if label and label["is_correct"] == want_correct and str(item_id) in items_by_id:
            cases.append(items_by_id[str(item_id)])
    return cases


def load_sae(model_name, layer, device):
    sae_path = get_sae_path(model_name, layer)
    if not sae_path:
        raise FileNotFoundError(f"SAE checkpoint not found for {model_name} layer {layer}")
    sae_dict = torch.load(sae_path, map_location="cpu")
    suite = "qwen" if "qwen" in model_name else "gemma"
    return SAEWrapper(sae_dict, suite=suite).to(device)


def extract_feature_matrix(model, tokenizer, sae, items, layer, include_context=True, formatter=None):
    values = []
    for item in tqdm(items, desc=f"L{layer} activations", leave=False):
        if formatter:
            prompt = formatter(item, tokenizer)
        else:
            prompt = format_pubmedqa(item, tokenizer, include_context=include_context)
        act = get_activation_with_hook(model, tokenizer, prompt, layer)
        with torch.no_grad():
            values.append(sae.encode(act).detach().cpu().numpy())
    return np.concatenate(values, axis=0) if values else np.zeros((0, sae.d_sae), dtype=np.float32)


def discover_dominant_features(model, tokenizer, sae, correct_items, wrong_items, layer, max_cases=100, p_value=0.01):
    n_sample = min(len(correct_items), len(wrong_items), max_cases)
    if n_sample < 2:
        return {"correct_dominant": [], "wrong_dominant": [], "n_sample": n_sample}
    feat_correct = extract_feature_matrix(model, tokenizer, sae, correct_items[:n_sample], layer)
    feat_wrong = extract_feature_matrix(model, tokenizer, sae, wrong_items[:n_sample], layer)
    t_stats, p_vals = ttest_ind(feat_correct, feat_wrong, axis=0, equal_var=False, nan_policy="omit")
    t_stats = np.nan_to_num(t_stats)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    correct_dom = np.where((p_vals < p_value) & (t_stats > 0))[0]
    wrong_dom = np.where((p_vals < p_value) & (t_stats < 0))[0]
    return {
        "correct_dominant": correct_dom.tolist(),
        "wrong_dominant": wrong_dom.tolist(),
        "n_sample": n_sample,
    }


def mean_feature_activation(model, tokenizer, sae, items, layer, feature_idx, include_context=True, formatter=None):
    vals = []
    for item in items:
        if formatter:
            prompt = formatter(item, tokenizer)
        else:
            prompt = format_pubmedqa(item, tokenizer, include_context=include_context)
        act = get_activation_with_hook(model, tokenizer, prompt, layer)
        with torch.no_grad():
            vals.append(float(sae.encode(act)[0, feature_idx].item()))
    return float(np.mean(vals)) if vals else 0.0


def filter_context_specific_features(
    model,
    tokenizer,
    sae,
    pqa_items,
    medqa_items,
    layer,
    candidates,
    max_items=50,
    without_ratio=2.0,
    medqa_max=0.05,
):
    pqa_subset = pqa_items[:max_items]
    medqa_subset = medqa_items[:max_items]
    stats = {}
    selected = []
    for feature_idx in tqdm(candidates, desc=f"L{layer} context filter", leave=False):
        mean_with = mean_feature_activation(model, tokenizer, sae, pqa_subset, layer, feature_idx, True)
        mean_without = mean_feature_activation(model, tokenizer, sae, pqa_subset, layer, feature_idx, False)
        mean_medqa = mean_feature_activation(
            model,
            tokenizer,
            sae,
            medqa_subset,
            layer,
            feature_idx,
            formatter=format_medqa,
        )
        is_context_specific = mean_with > without_ratio * max(mean_without, 1e-8) and mean_medqa < medqa_max
        stats[str(feature_idx)] = {
            "mean_with_context": mean_with,
            "mean_without_context": mean_without,
            "mean_medqa": mean_medqa,
            "is_context_specific": bool(is_context_specific),
        }
        if is_context_specific:
            selected.append(int(feature_idx))
    return selected, stats


def build_steer_vec(sae, amplify=None, suppress=None, device=None):
    device = device or sae.W_dec.device
    steer_vec = torch.zeros(sae.W_dec.shape[1], device=device)
    for feature_idx in amplify or []:
        steer_vec += sae.W_dec[int(feature_idx), :].to(device)
    for feature_idx in suppress or []:
        steer_vec -= sae.W_dec[int(feature_idx), :].to(device)
    return steer_vec


@contextmanager
def steering_hook(model, layer_to_vec, alpha):
    handles = []

    def make_hook(steer_vec):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[0, -1, :] = hidden[0, -1, :] + alpha * steer_vec.to(device=hidden.device, dtype=hidden.dtype)
            return output

        return hook

    for layer, steer_vec in layer_to_vec.items():
        handles.append(model.model.layers[int(layer)].register_forward_hook(make_hook(steer_vec)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def predict_pubmedqa_with_steering(model, tokenizer, item, layer_to_vec, alpha):
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    with steering_hook(model, layer_to_vec, alpha):
        probs = get_ynm_probs(model, tokenizer, prompt)
    return probs, max(probs, key=probs.get)


def evaluate_steering_set(model, tokenizer, wrong_items, correct_items, layer_to_vec, alpha, max_cases=None):
    wrong_eval = wrong_items[:max_cases] if max_cases else wrong_items
    correct_eval = correct_items[:max_cases] if max_cases else correct_items
    recovered = []
    unrecovered = []
    corrupted = []
    kept_correct = []

    for item in tqdm(wrong_eval, desc=f"recovery alpha={alpha}", leave=False):
        probs, pred = predict_pubmedqa_with_steering(model, tokenizer, item, layer_to_vec, alpha)
        row = {"item_id": pubmedqa_id(item), "prediction": pred, "ground_truth": item["final_decision"], "probs": probs}
        if pred == item["final_decision"]:
            recovered.append(row)
        else:
            unrecovered.append(row)

    for item in tqdm(correct_eval, desc=f"corruption alpha={alpha}", leave=False):
        probs, pred = predict_pubmedqa_with_steering(model, tokenizer, item, layer_to_vec, alpha)
        row = {"item_id": pubmedqa_id(item), "prediction": pred, "ground_truth": item["final_decision"], "probs": probs}
        if pred != item["final_decision"]:
            corrupted.append(row)
        else:
            kept_correct.append(row)

    return {
        "n_wrong": len(wrong_eval),
        "n_correct": len(correct_eval),
        "recovered": recovered,
        "unrecovered": unrecovered,
        "corrupted": corrupted,
        "kept_correct": kept_correct,
        "recovery_rate": len(recovered) / len(wrong_eval) if wrong_eval else 0.0,
        "corruption_rate": len(corrupted) / len(correct_eval) if correct_eval else 0.0,
    }


def pick_best_alpha(alpha_results):
    if not alpha_results:
        return None
    return max(alpha_results, key=lambda row: (row["recovery_rate"] - row["corruption_rate"], row["recovery_rate"]))


def aggregate_rates(rows, key):
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, "values": values}


def feature_signal(model, tokenizer, sae, item, layer, features):
    if not features:
        return 0.0
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    act = get_activation_with_hook(model, tokenizer, prompt, layer)
    with torch.no_grad():
        encoded = sae.encode(act)[0]
    return float(encoded[[int(idx) for idx in features]].sum().item())


def reconstruction_error(model, tokenizer, sae, item, layer):
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    act = get_activation_with_hook(model, tokenizer, prompt, layer)
    with torch.no_grad():
        recon = sae.decode(sae.encode(act))
        error = torch.norm(act.float() - recon.float()) / max(torch.norm(act.float()).item(), 1e-8)
    return float(error.item())


def conflict_score(correct_signal, wrong_signal):
    correct_signal = max(float(correct_signal), 0.0)
    wrong_signal = max(float(wrong_signal), 0.0)
    denom = correct_signal + wrong_signal
    return wrong_signal / denom if denom > 0 else 0.0


def threshold_from_validation(scores, labels):
    if len(set(labels)) < 2:
        return {"threshold": 0.5, "auc": 0.0}
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    best_index = int(np.argmax(youden))
    return {"threshold": float(thresholds[best_index]), "auc": float(auc(fpr, tpr))}


def auc_score(scores, labels):
    if len(set(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))
