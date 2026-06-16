import argparse
import gc
import glob
import os

import numpy as np
import torch
from scipy.stats import ttest_ind
from tqdm import tqdm

from split_experiment_utils import (
    BASE_DIR,
    RESULTS_DIR,
    build_steer_vec,
    get_activation_with_hook,
    load_model_and_tokenizer,
    load_sae,
    read_json,
    steering_hook,
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


def predict_letter(model, tokenizer, item):
    prompt = format_medabstain(item, tokenizer)
    probs = letter_probs(model, tokenizer, prompt)
    return probs, max(probs, key=probs.get)


def predict_letter_steered(model, tokenizer, item, layer, steer_vec, alpha):
    prompt = format_medabstain(item, tokenizer)
    with steering_hook(model, {layer: steer_vec}, alpha):
        probs = letter_probs(model, tokenizer, prompt)
    return probs, max(probs, key=probs.get)


def abstain_letter_for(item):
    return item.get("abstain_idx", "E")


def label_medabstain_behavior(model, tokenizer, items):
    rows = []
    for index, item in enumerate(tqdm(items, desc="MedAbstain behavior labels")):
        probs, pred = predict_letter(model, tokenizer, item)
        abstain_letter = abstain_letter_for(item)
        rows.append(
            {
                "index": index,
                "prediction": pred,
                "abstain_letter": abstain_letter,
                "behavior": "correct_abstain" if pred == abstain_letter else "wrong_answered",
                "abstain_prob": probs.get(abstain_letter, 0.0),
                "max_prob": probs.get(pred, 0.0),
                "probs": probs,
            }
        )
    return rows


def feature_matrix(model, tokenizer, sae, items, layer):
    values = []
    for item in tqdm(items, desc=f"L{layer} MedAbstain features", leave=False):
        prompt = format_medabstain(item, tokenizer)
        act = get_activation_with_hook(model, tokenizer, prompt, layer)
        with torch.no_grad():
            values.append(sae.encode(act).detach().cpu().numpy())
    return np.concatenate(values, axis=0) if values else np.zeros((0, sae.d_sae), dtype=np.float32)


def discover_medabstain_features(model, tokenizer, items, labels, model_name, layer, max_cases, p_value):
    abstain_items = [items[row["index"]] for row in labels if row["behavior"] == "correct_abstain"]
    answered_items = [items[row["index"]] for row in labels if row["behavior"] == "wrong_answered"]
    n_sample = min(len(abstain_items), len(answered_items), max_cases)
    if n_sample < 2:
        return {
            "layer": layer,
            "n_sample": n_sample,
            "abstain_dominant": [],
            "answered_dominant": [],
        }

    sae = load_sae(model_name, layer, model.device)
    feat_abstain = feature_matrix(model, tokenizer, sae, abstain_items[:n_sample], layer)
    feat_answered = feature_matrix(model, tokenizer, sae, answered_items[:n_sample], layer)
    t_stats, p_vals = ttest_ind(feat_abstain, feat_answered, axis=0, equal_var=False, nan_policy="omit")
    t_stats = np.nan_to_num(t_stats)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    abstain_dom = np.where((p_vals < p_value) & (t_stats > 0))[0].tolist()
    answered_dom = np.where((p_vals < p_value) & (t_stats < 0))[0].tolist()
    del sae
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "layer": layer,
        "n_sample": n_sample,
        "abstain_dominant": abstain_dom,
        "answered_dominant": answered_dom,
        "n_abstain_dominant": len(abstain_dom),
        "n_answered_dominant": len(answered_dom),
    }


def diagnose_pubmedqa_feature_transfer(model, tokenizer, items, labels, cv_data, model_name, layer, fold, alpha):
    fold_data = next(row for row in cv_data["folds"] if row["fold"] == fold)
    cs_wrong = fold_data["layer_info"][str(layer)]["cs_wrong"]
    sae = load_sae(model_name, layer, model.device)
    steer_vec = build_steer_vec(sae, suppress=cs_wrong, device=model.device)

    rows = []
    for label in tqdm(labels, desc="MedAbstain format/steering diagnostics"):
        item = items[label["index"]]
        probs, pred = predict_letter(model, tokenizer, item)
        steered_probs, steered_pred = predict_letter_steered(model, tokenizer, item, layer, steer_vec, alpha)
        abstain_letter = label["abstain_letter"]
        rows.append(
            {
                "index": label["index"],
                "prediction": pred,
                "steered_prediction": steered_pred,
                "abstain_letter": abstain_letter,
                "abstain_prob": probs.get(abstain_letter, 0.0),
                "steered_abstain_prob": steered_probs.get(abstain_letter, 0.0),
                "delta_abstain_prob": steered_probs.get(abstain_letter, 0.0) - probs.get(abstain_letter, 0.0),
                "max_prob": probs.get(pred, 0.0),
                "steered_max_prob": steered_probs.get(steered_pred, 0.0),
                "behavior": label["behavior"],
                "steered_behavior": "correct_abstain" if steered_pred == abstain_letter else "wrong_answered",
            }
        )

    del sae
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "fold": fold,
        "layer": layer,
        "alpha": alpha,
        "cs_wrong": cs_wrong,
        "mean_delta_abstain_prob": float(np.mean([row["delta_abstain_prob"] for row in rows])) if rows else 0.0,
        "wrong_to_abstain": sum(
            1 for row in rows if row["behavior"] == "wrong_answered" and row["steered_behavior"] == "correct_abstain"
        ),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose why PubMedQA features fail to transfer to MedAbstain.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--cv-path", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[18, 20, 22])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--max-feature-cases", type=int, default=100)
    parser.add_argument("--p-value", type=float, default=0.01)
    args = parser.parse_args()

    items, source_path = load_medabstain_items(args.data_path)
    items = items[: args.max_cases]
    cv_path = args.cv_path or f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json"
    cv_data = read_json(cv_path)

    model, tokenizer = load_model_and_tokenizer(args.model)
    labels = label_medabstain_behavior(model, tokenizer, items)
    feature_results = [
        discover_medabstain_features(
            model,
            tokenizer,
            items,
            labels,
            args.model,
            layer,
            args.max_feature_cases,
            args.p_value,
        )
        for layer in args.layers
    ]

    fold_data = next(row for row in cv_data["folds"] if row["fold"] == args.fold)
    alpha = args.alpha
    if alpha is None:
        match = next((row for row in fold_data["steering_results"] if row["name"] == "context_specific_wrong"), None)
        alpha = match["selected_alpha"] if match else 10.0

    transfer_diagnostics = []
    for layer in args.layers:
        if str(layer) in fold_data["layer_info"]:
            transfer_diagnostics.append(
                diagnose_pubmedqa_feature_transfer(
                    model,
                    tokenizer,
                    items,
                    labels,
                    cv_data,
                    args.model,
                    layer,
                    args.fold,
                    alpha,
                )
            )

    pubmedqa_feature_sets = {
        str(layer): set(fold_data["layer_info"].get(str(layer), {}).get("cs_wrong", []))
        for layer in args.layers
    }
    overlaps = {}
    for result in feature_results:
        layer = str(result["layer"])
        pubmedqa_wrong = pubmedqa_feature_sets.get(layer, set())
        overlaps[layer] = {
            "pubmedqa_cs_wrong_count": len(pubmedqa_wrong),
            "medabstain_answered_count": len(result["answered_dominant"]),
            "answered_overlap": sorted(pubmedqa_wrong.intersection(result["answered_dominant"])),
            "abstain_overlap": sorted(pubmedqa_wrong.intersection(result["abstain_dominant"])),
        }

    output = {
        "model": args.model,
        "source_path": source_path,
        "cv_path": cv_path,
        "fold": args.fold,
        "layers": args.layers,
        "behavior_summary": {
            "total": len(labels),
            "correct_abstain": sum(1 for row in labels if row["behavior"] == "correct_abstain"),
            "wrong_answered": sum(1 for row in labels if row["behavior"] == "wrong_answered"),
        },
        "feature_results": feature_results,
        "pubmedqa_overlap": overlaps,
        "transfer_diagnostics": transfer_diagnostics,
        "labels": labels,
    }
    output_path = f"{RESULTS_DIR}/medabstain_diagnostics/{args.model}_medabstain_feature_diagnostics_fold{args.fold}.json"
    write_json(output_path, output)
    print(f"Saved MedAbstain diagnostics to {output_path}")


if __name__ == "__main__":
    main()
