import argparse
import json
import math
import os
import random
from collections import Counter

import numpy as np

BASE_DIR = os.environ.get("MEDICAL_MI_BASE_DIR", "/workspace/medical_mi")
RESULTS_DIR = f"{BASE_DIR}/results"


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


LABELS = ["yes", "no", "maybe"]


def max_prob(probs):
    return max(float(value) for value in probs.values()) if probs else 0.0


def row_id(row):
    return str(row.get("item_id", row.get("pubid", "")))


def classify_typeb(row, target_gt="no", confident_threshold=0.7):
    gt = row["ground_truth"]
    prior = row["prior_answer"]
    ctx = row["context_answer"]
    ctx_conf = max_prob(row["context_probs"])
    if target_gt != "all" and gt != target_gt:
        return "other"
    if ctx == gt:
        return "correct"
    if prior == ctx:
        return "typeB_confident_ignore" if ctx_conf >= confident_threshold else "typeB_weak"
    return "typeA_processing"


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


def make_folds(rows, folds, seed):
    ids = [row_id(row) for row in rows]
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


def add_behavioral_scores(row):
    prior_conf = max_prob(row["prior_probs"])
    ctx_conf = max_prob(row["context_probs"])
    same = float(row["prior_answer"] == row["context_answer"])
    disagreement = 1.0 - same
    context_wrong = float(row["context_answer"] != row["ground_truth"])
    scores = {
        "prior_confidence": prior_conf,
        "context_confidence": ctx_conf,
        "same_prior_context": same,
        "context_output_ignore": same * ctx_conf,
        "prior_context_lock": same * prior_conf,
        "negative_conf_shift": ctx_conf - prior_conf,
        "positive_conf_shift": prior_conf - ctx_conf,
        "disagreement": disagreement,
        "context_wrong_oracle": context_wrong,
    }
    return scores


def normalize(values, value):
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def tune_combined_score(validation_rows, signal_keys):
    grids = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (1.0 / 3, 1.0 / 3, 1.0 / 3),
        (0.2, 0.6, 0.2),
        (0.2, 0.2, 0.6),
        (0.6, 0.2, 0.2),
    ]
    score_values = {key: [row[key] for row in validation_rows] for key in signal_keys}
    best = None
    for weights in grids:
        scores = []
        for row in validation_rows:
            score = 0.0
            for key, weight in zip(signal_keys, weights):
                score += weight * normalize(score_values[key], row[key])
            scores.append(score)
        metric = auc_score(scores, [row["is_wrong"] for row in validation_rows])
        candidate = {"weights": dict(zip(signal_keys, weights)), "validation_auc": metric}
        if best is None or metric > best["validation_auc"]:
            best = candidate
    return best


def apply_combined_score(rows, weights, signal_keys, reference_rows):
    score_values = {key: [row[key] for row in reference_rows] for key in signal_keys}
    for row in rows:
        row["combined_behavioral"] = sum(
            weight * normalize(score_values[key], row[key])
            for key, weight in weights.items()
        )


def summarize(values):
    if not values:
        return {"mean": 0.0, "std": 0.0, "values": []}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 1/5: Type-B scale and context-output behavioral signals.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--coverage", type=float, default=0.7)
    parser.add_argument("--confident-threshold", type=float, default=0.7)
    parser.add_argument("--target-gt", choices=["no", "all"], default="no")
    args = parser.parse_args()

    data, source_path = load_conflict_set(args.conflict_set, args.model)
    rows = []
    for row in data:
        new_row = dict(row)
        new_row["item_id"] = row_id(row)
        new_row["fine_label"] = classify_typeb(row, args.target_gt, args.confident_threshold)
        new_row["is_wrong"] = int(row["context_answer"] != row["ground_truth"])
        new_row.update(add_behavioral_scores(row))
        rows.append(new_row)

    counts = Counter(row["fine_label"] for row in rows)
    wrong_counts = Counter(row["fine_label"] for row in rows if row["is_wrong"])
    splits = make_folds(rows, args.folds, args.seed)
    row_map = {row["item_id"]: row for row in rows}
    signal_keys = [
        "prior_confidence",
        "context_confidence",
        "same_prior_context",
        "context_output_ignore",
        "prior_context_lock",
        "positive_conf_shift",
        "negative_conf_shift",
        "disagreement",
    ]
    combined_keys = ["context_output_ignore", "prior_confidence", "positive_conf_shift"]

    fold_results = []
    for split in splits:
        validation_rows = [row_map[item_id] for item_id in split["validation_ids"]]
        test_rows = [dict(row_map[item_id]) for item_id in split["test_ids"]]
        combined = tune_combined_score(validation_rows, combined_keys)
        apply_combined_score(test_rows, combined["weights"], combined_keys, validation_rows)
        fold_metrics = {}
        for key in signal_keys + ["combined_behavioral"]:
            scores = [row[key] for row in test_rows]
            labels = [row["is_wrong"] for row in test_rows]
            fold_metrics[key] = {
                "auc": auc_score(scores, labels),
                "risk_at_coverage": risk_at_coverage(test_rows, key, args.coverage),
            }
        fold_results.append(
            {
                "fold": split["fold"],
                "combined_selection": combined,
                "metrics": fold_metrics,
                "test_ids": split["test_ids"],
            }
        )

    summary = {}
    for key in signal_keys + ["combined_behavioral"]:
        summary[key] = {
            "auc": summarize([fold["metrics"][key]["auc"] for fold in fold_results]),
            "risk_at_coverage": summarize(
                [fold["metrics"][key]["risk_at_coverage"]["risk"] for fold in fold_results]
            ),
            "selective_accuracy_at_coverage": summarize(
                [fold["metrics"][key]["risk_at_coverage"]["selective_accuracy"] for fold in fold_results]
            ),
        }

    output = {
        "model": args.model,
        "source_path": source_path,
        "target_gt": args.target_gt,
        "confident_threshold": args.confident_threshold,
        "coverage": args.coverage,
        "counts": dict(counts),
        "wrong_counts": dict(wrong_counts),
        "typeB_cases": [
            {
                "item_id": row["item_id"],
                "fine_label": row["fine_label"],
                "ground_truth": row["ground_truth"],
                "prior_answer": row["prior_answer"],
                "context_answer": row["context_answer"],
                "prior_confidence": row["prior_confidence"],
                "context_confidence": row["context_confidence"],
            }
            for row in rows
            if row["fine_label"].startswith("typeB")
        ],
        "summary": summary,
        "folds": fold_results,
    }
    output_path = f"{RESULTS_DIR}/typeb_behavioral/{args.model}_typeb_behavioral_signals.json"
    write_json(output_path, output)
    print(f"Saved Type-B behavioral signal results to {output_path}")
    print(f"Counts: {dict(counts)}")
    print(f"Wrong counts: {dict(wrong_counts)}")


if __name__ == "__main__":
    main()
