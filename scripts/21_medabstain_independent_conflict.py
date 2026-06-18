import argparse
import gc
import glob
import math
import os
import random

import numpy as np
import torch
from scipy.stats import ttest_ind
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

from split_experiment_utils import (
    BASE_DIR,
    RESULTS_DIR,
    conflict_score,
    get_activation_with_hook,
    load_model_and_tokenizer,
    load_sae,
    read_json,
    write_json,
)


def medabstain_dir():
    candidates = [
        f"{BASE_DIR}/data/raw/MedAbstain/data",
        f"{BASE_DIR}/data/raw/MedAbstain/MedAbstain/data",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_medabstain_items(path=None):
    if path is None:
        data_dir = medabstain_dir()
        if not data_dir:
            raise FileNotFoundError("MedAbstain data directory not found")
        matches = glob.glob(f"{data_dir}/perturbed_medqa*_test_noabst.json")
        if not matches:
            raise FileNotFoundError(f"No MedAbstain perturbed file found in {data_dir}")
        path = matches[0]
    return read_json(path), path


def format_medabstain(item, tokenizer):
    choices = item.get("choices", {})
    choice_str = "\n".join([f"{key}: {value}" for key, value in choices.items()])
    prompt = f"Question: {item['question']}\n\nChoices:\n{choice_str}\n\nAnswer with one letter."
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def letter_probs(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    result = {}
    for letter in ["A", "B", "C", "D", "E"]:
        ids = []
        for variant in [letter, " " + letter]:
            ids.extend(tokenizer.encode(variant, add_special_tokens=False))
        valid = [idx for idx in ids if idx < probs.shape[0]]
        result[letter] = max([probs[idx].item() for idx in valid], default=0.0)
    total = sum(result.values())
    return {key: value / total for key, value in result.items()} if total else result


def label_behaviors(model, tokenizer, items):
    rows = []
    for index, item in enumerate(tqdm(items, desc="MedAbstain labels")):
        probs = letter_probs(model, tokenizer, format_medabstain(item, tokenizer))
        pred = max(probs, key=probs.get)
        abstain_letter = item.get("abstain_idx", "E")
        rows.append(
            {
                "index": index,
                "prediction": pred,
                "abstain_letter": abstain_letter,
                "behavior": "correct_abstain" if pred == abstain_letter else "wrong_answered",
                "is_wrong_answered": int(pred != abstain_letter),
                "probs": probs,
            }
        )
    return rows


def make_folds(n_items, folds, seed):
    indices = list(range(n_items))
    random.Random(seed).shuffle(indices)
    fold_size = int(math.ceil(n_items / folds))
    splits = []
    for fold in range(folds):
        test = indices[fold * fold_size : min((fold + 1) * fold_size, n_items)]
        test_set = set(test)
        validation = [idx for idx in indices if idx not in test_set]
        splits.append({"fold": fold, "validation_indices": validation, "test_indices": test})
    return splits


def feature_matrix(model, tokenizer, sae, items, layer, indices):
    values = []
    for index in tqdm(indices, desc=f"L{layer} MedAbstain activations", leave=False):
        act = get_activation_with_hook(model, tokenizer, format_medabstain(items[index], tokenizer), layer)
        with torch.no_grad():
            values.append(sae.encode(act).detach().cpu().numpy())
    return np.concatenate(values, axis=0) if values else np.zeros((0, sae.d_sae), dtype=np.float32)


def discover_features(model, tokenizer, sae, items, labels, validation_indices, layer, max_cases, p_value):
    abstain = [idx for idx in validation_indices if labels[idx]["behavior"] == "correct_abstain"]
    answered = [idx for idx in validation_indices if labels[idx]["behavior"] == "wrong_answered"]
    n_sample = min(len(abstain), len(answered), max_cases)
    if n_sample < 2:
        return {"abstain_dominant": [], "answered_dominant": [], "n_sample": n_sample}
    feat_abstain = feature_matrix(model, tokenizer, sae, items, layer, abstain[:n_sample])
    feat_answered = feature_matrix(model, tokenizer, sae, items, layer, answered[:n_sample])
    t_stats, p_vals = ttest_ind(feat_abstain, feat_answered, axis=0, equal_var=False, nan_policy="omit")
    t_stats = np.nan_to_num(t_stats)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    return {
        "abstain_dominant": np.where((p_vals < p_value) & (t_stats > 0))[0].tolist(),
        "answered_dominant": np.where((p_vals < p_value) & (t_stats < 0))[0].tolist(),
        "n_sample": n_sample,
    }


def feature_signal(model, tokenizer, sae, item, layer, features):
    if not features:
        return 0.0
    act = get_activation_with_hook(model, tokenizer, format_medabstain(item, tokenizer), layer)
    with torch.no_grad():
        encoded = sae.encode(act)[0]
    return float(encoded[[int(idx) for idx in features]].sum().item())


def auc_score(scores, labels):
    if len(set(labels)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def main():
    parser = argparse.ArgumentParser(description="MedAbstain independent conflict-score discovery and AUC.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[18, 20, 22])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--max-feature-cases", type=int, default=100)
    parser.add_argument("--p-value", type=float, default=0.01)
    args = parser.parse_args()

    items, source_path = load_medabstain_items(args.data_path)
    items = items[: args.max_cases]
    model, tokenizer = load_model_and_tokenizer(args.model)
    labels = label_behaviors(model, tokenizer, items)
    splits = make_folds(len(items), args.folds, args.seed)

    layer_outputs = {}
    for layer in args.layers:
        sae = load_sae(args.model, layer, model.device)
        fold_rows = []
        for split in splits:
            selected = discover_features(
                model,
                tokenizer,
                sae,
                items,
                labels,
                split["validation_indices"],
                layer,
                args.max_feature_cases,
                args.p_value,
            )
            scores = []
            y_true = []
            case_rows = []
            for index in tqdm(split["test_indices"], desc=f"L{layer} fold {split['fold']} scores", leave=False):
                abstain_signal = feature_signal(model, tokenizer, sae, items[index], layer, selected["abstain_dominant"])
                answered_signal = feature_signal(model, tokenizer, sae, items[index], layer, selected["answered_dominant"])
                score = conflict_score(abstain_signal, answered_signal)
                scores.append(score)
                y_true.append(labels[index]["is_wrong_answered"])
                case_rows.append(
                    {
                        "index": index,
                        "score": score,
                        "label": labels[index]["is_wrong_answered"],
                        "behavior": labels[index]["behavior"],
                        "abstain_signal": abstain_signal,
                        "answered_signal": answered_signal,
                    }
                )
            fold_rows.append(
                {
                    "fold": split["fold"],
                    "n_validation": len(split["validation_indices"]),
                    "n_test": len(split["test_indices"]),
                    "selected_features": selected,
                    "test_auc": auc_score(scores, y_true),
                    "test_scores": case_rows,
                }
            )
        layer_outputs[str(layer)] = {
            "folds": fold_rows,
            "mean_test_auc": float(np.mean([row["test_auc"] for row in fold_rows])) if fold_rows else 0.0,
            "std_test_auc": float(np.std([row["test_auc"] for row in fold_rows], ddof=1)) if len(fold_rows) > 1 else 0.0,
        }
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    output = {
        "model": args.model,
        "source_path": source_path,
        "behavior_summary": {
            "total": len(labels),
            "correct_abstain": sum(1 for row in labels if row["behavior"] == "correct_abstain"),
            "wrong_answered": sum(1 for row in labels if row["behavior"] == "wrong_answered"),
        },
        "layers": layer_outputs,
        "labels": labels,
    }
    output_path = f"{RESULTS_DIR}/medabstain_independent/{args.model}_medabstain_independent_conflict.json"
    write_json(output_path, output)
    print(f"Saved MedAbstain independent conflict results to {output_path}")


if __name__ == "__main__":
    main()
