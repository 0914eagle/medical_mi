import argparse
import json
import math
import os
import random

import numpy as np

from split_experiment_utils import BASE_DIR, RESULTS_DIR, read_json, write_json


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
    splits = []
    for fold in range(folds):
        test_ids = shuffled[fold * fold_size : min((fold + 1) * fold_size, len(shuffled))]
        test_set = set(test_ids)
        validation_ids = [item_id for item_id in shuffled if item_id not in test_set]
        splits.append({"fold": fold, "validation_ids": validation_ids, "test_ids": test_ids})
    return splits


def normalize(values, value):
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def auc_score(scores, labels):
    if len(set(labels)) < 2:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        avg_rank = (rank + rank + (end - index) - 1) / 2.0
        rank_sum += avg_rank * sum(label for _, label in pairs[index:end])
        rank += end - index
        index = end
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def risk_at_coverage(rows, score_key, coverage):
    n_answer = max(1, int(round(len(rows) * coverage)))
    answered = sorted(rows, key=lambda row: row[score_key])[:n_answer]
    wrong = sum(1 for row in answered if row["is_wrong"])
    return {
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "risk": wrong / len(answered) if answered else 1.0,
        "selective_accuracy": 1.0 - (wrong / len(answered)) if answered else 0.0,
        "answered": len(answered),
        "wrong_answered": wrong,
    }


def summarize(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def tune_combination(validation_rows, keys):
    grids = [
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (0.25, 0.25, 0.25, 0.25),
        (0.4, 0.2, 0.2, 0.2),
        (0.2, 0.4, 0.2, 0.2),
        (0.2, 0.2, 0.4, 0.2),
        (0.2, 0.2, 0.2, 0.4),
        (0.1, 0.3, 0.3, 0.3),
    ]
    values = {key: [row[key] for row in validation_rows] for key in keys}
    best = None
    for weights in grids:
        scores = []
        for row in validation_rows:
            scores.append(sum(weight * normalize(values[key], row[key]) for key, weight in zip(keys, weights)))
        score_auc = auc_score(scores, [row["is_wrong"] for row in validation_rows])
        candidate = {"weights": dict(zip(keys, weights)), "validation_auc": score_auc}
        if best is None or score_auc > best["validation_auc"]:
            best = candidate
    return best


def apply_combination(rows, reference_rows, weights, output_key):
    values = {key: [row[key] for row in reference_rows] for key in weights}
    for row in rows:
        row[output_key] = sum(weight * normalize(values[key], row[key]) for key, weight in weights.items())


def load_typeb_feature_scores(path, layer):
    data = read_json(path)
    scores = {}
    for fold in data["folds"]:
        layer_result = fold["layer_results"].get(str(layer))
        if not layer_result:
            continue
        for row in layer_result["test_scores"]:
            scores.setdefault(fold["fold"], {})[row["item_id"]] = {
                "typeb_confident_signal": row["typeb_confident_signal"],
                "typeb_all_signal": row["typeb_all_signal"],
            }
    return scores


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Combined Type-B Risk-Coverage evaluation.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--typeb-features-path", default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--coverage", type=float, default=0.7)
    args = parser.parse_args()

    conflict_rows, source_path = load_conflict_set(args.conflict_set, args.model)
    rows = []
    for row in conflict_rows:
        enriched = dict(row)
        enriched["item_id"] = row_id(row)
        enriched.update(add_behavioral_scores(row))
        rows.append(enriched)

    feature_path = args.typeb_features_path or f"{RESULTS_DIR}/typeb_features/{args.model}_typeb_sae_features.json"
    feature_scores = load_typeb_feature_scores(feature_path, args.layer)
    splits = make_folds(rows, args.folds, args.seed)
    row_map = {row["item_id"]: row for row in rows}
    base_keys = [
        "prior_confidence",
        "context_output_ignore",
        "prior_context_lock",
        "positive_conf_shift",
        "typeb_confident_signal",
        "typeb_all_signal",
    ]
    combo_keys = ["context_output_ignore", "prior_confidence", "positive_conf_shift", "typeb_all_signal"]

    fold_results = []
    for split in splits:
        fold_id = split["fold"]
        validation_rows = [dict(row_map[item_id]) for item_id in split["validation_ids"]]
        test_rows = [dict(row_map[item_id]) for item_id in split["test_ids"]]
        for row in validation_rows + test_rows:
            row.update(feature_scores.get(fold_id, {}).get(row["item_id"], {"typeb_confident_signal": 0.0, "typeb_all_signal": 0.0}))
        combo = tune_combination(validation_rows, combo_keys)
        apply_combination(test_rows, validation_rows, combo["weights"], "combined_typeb")
        metrics = {}
        for key in base_keys + ["combined_typeb"]:
            metrics[key] = {
                "auc": auc_score([row[key] for row in test_rows], [row["is_wrong"] for row in test_rows]),
                "risk_at_coverage": risk_at_coverage(test_rows, key, args.coverage),
            }
        fold_results.append(
            {
                "fold": fold_id,
                "combination": combo,
                "metrics": metrics,
                "test_ids": split["test_ids"],
            }
        )

    summary = {}
    for key in base_keys + ["combined_typeb"]:
        summary[key] = {
            "auc": summarize([fold["metrics"][key]["auc"] for fold in fold_results]),
            "risk_at_coverage": summarize([fold["metrics"][key]["risk_at_coverage"]["risk"] for fold in fold_results]),
            "selective_accuracy_at_coverage": summarize(
                [fold["metrics"][key]["risk_at_coverage"]["selective_accuracy"] for fold in fold_results]
            ),
        }

    output = {
        "model": args.model,
        "source_path": source_path,
        "typeb_features_path": feature_path,
        "layer": args.layer,
        "coverage": args.coverage,
        "summary": summary,
        "folds": fold_results,
    }
    output_path = f"{RESULTS_DIR}/typeb_combined/{args.model}_typeb_combined_risk_coverage_L{args.layer}.json"
    write_json(output_path, output)
    print(f"Saved Type-B combined Risk-Coverage to {output_path}")


if __name__ == "__main__":
    main()
