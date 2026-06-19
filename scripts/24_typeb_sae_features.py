import argparse
import gc
import json
import math
import os
import random

import numpy as np
import torch
from scipy.stats import ttest_ind
from tqdm import tqdm

from split_experiment_utils import (
    BASE_DIR,
    RESULTS_DIR,
    get_activation_with_hook,
    item_map_by_id,
    load_medqa_items,
    load_model_and_tokenizer,
    load_pubmedqa_items,
    load_sae,
    write_json,
)
from utils import format_medqa, format_pubmedqa


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def max_prob(probs):
    return max(float(value) for value in probs.values()) if probs else 0.0


def row_id(row):
    return str(row.get("item_id", row.get("pubid", "")))


def load_conflict_set(path, model):
    candidates = [
        path,
        f"{RESULTS_DIR}/eval/{model}_conflict_set.json",
        f"{BASE_DIR}/{model}_conflict_set.json",
        f"{model}_conflict_set.json",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return read_json(candidate), candidate
    raise FileNotFoundError(f"Could not find conflict_set for {model}")


def classify_typeb(row, confident_threshold=0.7):
    gt = row["ground_truth"]
    prior = row["prior_answer"]
    ctx = row["context_answer"]
    ctx_conf = max_prob(row["context_probs"])
    if gt != "no":
        return "other"
    if ctx == "no":
        return "correct"
    if prior == ctx:
        return "typeB_confident_ignore" if ctx_conf >= confident_threshold else "typeB_weak"
    return "typeA_processing"


def add_behavioral_scores(row):
    prior_conf = max_prob(row["prior_probs"])
    ctx_conf = max_prob(row["context_probs"])
    same = float(row["prior_answer"] == row["context_answer"])
    return {
        "prior_confidence": prior_conf,
        "context_confidence": ctx_conf,
        "context_output_ignore": same * ctx_conf,
        "prior_context_lock": same * prior_conf,
        "positive_conf_shift": prior_conf - ctx_conf,
        "is_wrong": int(row["context_answer"] != row["ground_truth"]),
    }


def make_folds(rows, folds, seed):
    ids = [row["item_id"] for row in rows]
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    fold_size = int(math.ceil(len(shuffled) / folds))
    split_defs = []
    for fold in range(folds):
        test_ids = shuffled[fold * fold_size : min((fold + 1) * fold_size, len(shuffled))]
        test_set = set(test_ids)
        validation_ids = [item_id for item_id in shuffled if item_id not in test_set]
        split_defs.append({"fold": fold, "validation_ids": validation_ids, "test_ids": test_ids})
    return split_defs


def feature_matrix(model, tokenizer, sae, items, layer, formatter):
    values = []
    for item in tqdm(items, desc=f"L{layer} activations", leave=False):
        prompt = formatter(item, tokenizer)
        act = get_activation_with_hook(model, tokenizer, prompt, layer)
        with torch.no_grad():
            values.append(sae.encode(act).detach().cpu().numpy())
    return np.concatenate(values, axis=0) if values else np.zeros((0, sae.d_sae), dtype=np.float32)


def pubmedqa_formatter(item, tokenizer):
    return format_pubmedqa(item, tokenizer, include_context=True)


def medqa_formatter(item, tokenizer):
    return format_medqa(item, tokenizer)


def discover_typeb_features(model, tokenizer, sae, correct_items, typeb_items, layer, max_cases, p_value):
    n_sample = min(len(correct_items), len(typeb_items), max_cases)
    if n_sample < 2:
        return {
            "n_sample": n_sample,
            "typeb_dominant": [],
            "correct_dominant": [],
        }
    feat_correct = feature_matrix(model, tokenizer, sae, correct_items[:n_sample], layer, pubmedqa_formatter)
    feat_typeb = feature_matrix(model, tokenizer, sae, typeb_items[:n_sample], layer, pubmedqa_formatter)
    t_stats, p_vals = ttest_ind(feat_correct, feat_typeb, axis=0, equal_var=False, nan_policy="omit")
    t_stats = np.nan_to_num(t_stats)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    return {
        "n_sample": n_sample,
        "typeb_dominant": np.where((p_vals < p_value) & (t_stats < 0))[0].tolist(),
        "correct_dominant": np.where((p_vals < p_value) & (t_stats > 0))[0].tolist(),
    }


def mean_selected_features(model, tokenizer, sae, item, layer, features, formatter):
    if not features:
        return 0.0
    prompt = formatter(item, tokenizer)
    act = get_activation_with_hook(model, tokenizer, prompt, layer)
    with torch.no_grad():
        encoded = sae.encode(act)[0]
    return float(encoded[[int(idx) for idx in features]].sum().item())


def medqa_means(model, tokenizer, sae, medqa_items, layer, features, max_items):
    if not features:
        return 0.0
    vals = [
        mean_selected_features(model, tokenizer, sae, item, layer, features, medqa_formatter)
        for item in medqa_items[:max_items]
    ]
    return float(np.mean(vals)) if vals else 0.0


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Type-B SAE feature discovery.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 18, 20, 22, 24])
    parser.add_argument("--confident-threshold", type=float, default=0.7)
    parser.add_argument("--max-feature-cases", type=int, default=100)
    parser.add_argument("--max-medqa-items", type=int, default=50)
    parser.add_argument("--p-value", type=float, default=0.01)
    args = parser.parse_args()

    conflict_rows, source_path = load_conflict_set(args.conflict_set, args.model)
    pubmedqa = load_pubmedqa_items()
    item_by_id = item_map_by_id(pubmedqa)
    rows = []
    for row in conflict_rows:
        item_id = row_id(row)
        if item_id not in item_by_id:
            continue
        enriched = dict(row)
        enriched["item_id"] = item_id
        enriched["fine_label"] = classify_typeb(row, args.confident_threshold)
        enriched.update(add_behavioral_scores(row))
        rows.append(enriched)
    row_map = {row["item_id"]: row for row in rows}
    splits = make_folds(rows, args.folds, args.seed)
    medqa_items = load_medqa_items(limit=args.max_medqa_items)

    model, tokenizer = load_model_and_tokenizer(args.model)
    fold_results = []
    for split in splits:
        fold_id = split["fold"]
        validation_rows = [row_map[item_id] for item_id in split["validation_ids"]]
        test_rows = [row_map[item_id] for item_id in split["test_ids"]]
        layer_results = {}
        for layer in args.layers:
            sae = load_sae(args.model, layer, model.device)
            correct_rows = [row for row in validation_rows if row["fine_label"] == "correct"]
            confident_rows = [row for row in validation_rows if row["fine_label"] == "typeB_confident_ignore"]
            all_typeb_rows = [row for row in validation_rows if row["fine_label"] in {"typeB_confident_ignore", "typeB_weak"}]
            correct_items = [item_by_id[row["item_id"]] for row in correct_rows]
            confident_items = [item_by_id[row["item_id"]] for row in confident_rows]
            all_typeb_items = [item_by_id[row["item_id"]] for row in all_typeb_rows]

            confident_features = discover_typeb_features(
                model, tokenizer, sae, correct_items, confident_items, layer, args.max_feature_cases, args.p_value
            )
            all_typeb_features = discover_typeb_features(
                model, tokenizer, sae, correct_items, all_typeb_items, layer, args.max_feature_cases, args.p_value
            )

            test_scores = []
            for row in tqdm(test_rows, desc=f"Fold {fold_id} L{layer} test scores", leave=False):
                item = item_by_id[row["item_id"]]
                test_scores.append(
                    {
                        "item_id": row["item_id"],
                        "fine_label": row["fine_label"],
                        "is_wrong": row["is_wrong"],
                        "typeb_confident_signal": mean_selected_features(
                            model, tokenizer, sae, item, layer, confident_features["typeb_dominant"], pubmedqa_formatter
                        ),
                        "typeb_all_signal": mean_selected_features(
                            model, tokenizer, sae, item, layer, all_typeb_features["typeb_dominant"], pubmedqa_formatter
                        ),
                        "prior_confidence": row["prior_confidence"],
                        "context_output_ignore": row["context_output_ignore"],
                        "positive_conf_shift": row["positive_conf_shift"],
                    }
                )

            existing_wrong = []
            try:
                cv_data = read_json(f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json")
                cv_fold = next((fold for fold in cv_data["folds"] if fold["fold"] == fold_id), None)
                if cv_fold and str(layer) in cv_fold["layer_info"]:
                    existing_wrong = cv_fold["layer_info"][str(layer)]["dominant"]["wrong_dominant"]
            except Exception:
                existing_wrong = []

            layer_results[str(layer)] = {
                "confident": confident_features,
                "all_typeb": all_typeb_features,
                "overlap_with_existing_wrong": {
                    "confident_typeb_overlap": sorted(set(confident_features["typeb_dominant"]).intersection(existing_wrong)),
                    "all_typeb_overlap": sorted(set(all_typeb_features["typeb_dominant"]).intersection(existing_wrong)),
                    "existing_wrong_count": len(existing_wrong),
                },
                "medqa_mean": {
                    "confident_typeb": medqa_means(
                        model, tokenizer, sae, medqa_items, layer, confident_features["typeb_dominant"], args.max_medqa_items
                    ),
                    "all_typeb": medqa_means(
                        model, tokenizer, sae, medqa_items, layer, all_typeb_features["typeb_dominant"], args.max_medqa_items
                    ),
                },
                "test_scores": test_scores,
            }
            del sae
            gc.collect()
            torch.cuda.empty_cache()

        fold_results.append(
            {
                "fold": fold_id,
                "n_validation": len(validation_rows),
                "n_test": len(test_rows),
                "layer_results": layer_results,
            }
        )

    output = {
        "model": args.model,
        "source_path": source_path,
        "folds": fold_results,
        "splits": splits,
        "layers": args.layers,
    }
    output_path = f"{RESULTS_DIR}/typeb_features/{args.model}_typeb_sae_features.json"
    write_json(output_path, output)
    print(f"Saved Type-B SAE feature results to {output_path}")


if __name__ == "__main__":
    main()
