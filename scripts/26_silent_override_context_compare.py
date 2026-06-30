import argparse
import gc
import json
import os
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm import tqdm

from split_experiment_utils import (
    BASE_DIR,
    RESULTS_DIR,
    conflict_score,
    feature_signal,
    item_map_by_id,
    load_model_and_tokenizer,
    load_pubmedqa_items,
    load_sae,
    read_json,
    write_json,
)


def path_candidates(path, model, subdir, filename):
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend(
        [
            f"{RESULTS_DIR}/{subdir}/{filename}",
            f"{BASE_DIR}/results/{subdir}/{filename}",
            f"{BASE_DIR}/{filename}",
            filename,
        ]
    )
    seen = set()
    deduped = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def load_first_existing(candidates, label, required=True):
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return read_json(candidate), candidate
    if required:
        raise FileNotFoundError(f"Could not find {label}. Tried: {candidates}")
    return None, None


def max_prob(probs):
    return max(float(value) for value in probs.values()) if probs else 0.0


def row_id(row):
    return str(row.get("item_id", row.get("pubid", "")))


def group_stats(values):
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "values": []}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "values": [float(value) for value in values],
    }


def compare_groups(a, b):
    result = {
        "ttest_ind_welch": None,
        "mannwhitneyu": None,
    }
    if len(a) >= 2 and len(b) >= 2:
        t_stat, t_p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
        u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
        result["ttest_ind_welch"] = {"statistic": float(t_stat), "p_value": float(t_p)}
        result["mannwhitneyu"] = {"statistic": float(u_stat), "p_value": float(u_p)}
    return result


def classify(row, computed_conflict_score, noisy_threshold, prior_threshold):
    if row["context_answer"] == row["ground_truth"]:
        return "correct"
    if computed_conflict_score >= noisy_threshold:
        return "noisy_wrong"
    if max_prob(row.get("prior_probs", {})) >= prior_threshold:
        return "silent_wrong"
    return "other_wrong"


def relax_silent_threshold_if_needed(rows, min_cases, initial_threshold):
    if min_cases <= 0:
        return initial_threshold
    for threshold in [initial_threshold, 0.65, 0.6, 0.55, 0.5]:
        n_silent = sum(
            1
            for row in rows
            if row["context_answer"] != row["ground_truth"]
            and row["computed_conflict_score"] < row["noisy_threshold"]
            and max_prob(row.get("prior_probs", {})) >= threshold
        )
        if n_silent >= min_cases:
            return threshold
    return 0.5


def feature_ids_from_typeb_fold(layer_result, source):
    if source == "all_typeb":
        return layer_result.get("all_typeb", {}).get("correct_dominant", [])
    if source == "confident":
        return layer_result.get("confident", {}).get("correct_dominant", [])
    raise ValueError("--correct-dominant-source must be one of: all_typeb, confident")


def build_fold_inputs(typeb_data, cv_data, layer, correct_dominant_source):
    fold_inputs = {}
    typeb_by_fold = {fold["fold"]: fold for fold in typeb_data.get("folds", [])}
    cv_by_fold = {fold["fold"]: fold for fold in cv_data.get("folds", [])} if cv_data else {}
    for fold_id, typeb_fold in typeb_by_fold.items():
        typeb_layer = typeb_fold.get("layer_results", {}).get(str(layer), {})
        cv_layer = cv_by_fold.get(fold_id, {}).get("layer_info", {}).get(str(layer), {})
        fold_inputs[fold_id] = {
            "test_ids": [str(item_id) for item_id in typeb_fold.get("test_ids", [])],
            "correct_dominant_features": feature_ids_from_typeb_fold(typeb_layer, correct_dominant_source),
            "context_specific_features": cv_layer.get("cs_correct", []),
            "context_specific_wrong_features": cv_layer.get("cs_wrong", []),
        }
    return fold_inputs


def activation_cache_key(item_id, layer, features):
    return f"{item_id}|L{layer}|{','.join(str(int(feature)) for feature in features)}"


def cached_feature_signal(cache, model, tokenizer, sae, item, item_id, layer, features):
    features = [int(feature) for feature in features]
    if not features:
        return 0.0
    key = activation_cache_key(item_id, layer, features)
    if key not in cache:
        cache[key] = feature_signal(model, tokenizer, sae, item, layer, features)
    return float(cache[key])


def main():
    parser = argparse.ArgumentParser(
        description="Compare context feature activation across correct, noisy wrong, and silent wrong cases."
    )
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--typeb-features-path", default=None)
    parser.add_argument("--context-cv-path", default=None)
    parser.add_argument("--correct-dominant-source", default="all_typeb", choices=["all_typeb", "confident"])
    parser.add_argument("--noisy-threshold", type=float, default=0.6)
    parser.add_argument("--prior-threshold", type=float, default=0.7)
    parser.add_argument("--min-silent-cases", type=int, default=20)
    parser.add_argument("--max-cases-per-fold", type=int, default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only validate input files and print available group counts; does not load model or SAE.",
    )
    args = parser.parse_args()

    conflict_rows, conflict_path = load_first_existing(
        path_candidates(args.conflict_set, args.model, "eval", f"{args.model}_conflict_set.json"),
        "conflict set",
    )
    typeb_data, typeb_path = load_first_existing(
        path_candidates(args.typeb_features_path, args.model, "typeb_features", f"{args.model}_typeb_sae_features.json"),
        "Type-B SAE feature results",
    )
    cv_data, cv_path = load_first_existing(
        path_candidates(args.context_cv_path, args.model, "steering_context_specific", f"{args.model}_context_specific_cv.json"),
        "context-specific CV results",
    )

    row_by_id = {row_id(row): row for row in conflict_rows}
    fold_inputs = build_fold_inputs(typeb_data, cv_data, args.layer, args.correct_dominant_source)
    missing_context = [
        fold_id for fold_id, fold in fold_inputs.items() if not fold["context_specific_features"] or not fold["context_specific_wrong_features"]
    ]
    if missing_context:
        raise ValueError(
            "Missing context-specific cs_correct/cs_wrong features for folds "
            f"{missing_context}. Pass --context-cv-path pointing to *_context_specific_cv.json."
        )

    if args.dry_run:
        covered_ids = {item_id for fold in fold_inputs.values() for item_id in fold["test_ids"]}
        rows = [row_by_id[item_id] for item_id in covered_ids if item_id in row_by_id]
        base_counts = defaultdict(int)
        for row in rows:
            base_counts["correct" if row["context_answer"] == row["ground_truth"] else "wrong"] += 1
        print(json.dumps({"conflict_path": conflict_path, "typeb_path": typeb_path, "context_cv_path": cv_path, "counts": base_counts}, indent=2))
        return

    items = load_pubmedqa_items()
    items_by_id = item_map_by_id(items)
    model, tokenizer = load_model_and_tokenizer(args.model)
    sae = load_sae(args.model, args.layer, model.device)
    cache = {}

    cases = []
    for fold_id, fold in fold_inputs.items():
        fold_rows = [row_by_id[item_id] for item_id in fold["test_ids"] if item_id in row_by_id and item_id in items_by_id]
        if args.max_cases_per_fold:
            fold_rows = fold_rows[: args.max_cases_per_fold]
        for row in tqdm(fold_rows, desc=f"Fold {fold_id} L{args.layer} signals", leave=False):
            item_id = row_id(row)
            item = items_by_id[item_id]
            correct_dom = cached_feature_signal(
                cache, model, tokenizer, sae, item, item_id, args.layer, fold["correct_dominant_features"]
            )
            context_spec = cached_feature_signal(
                cache, model, tokenizer, sae, item, item_id, args.layer, fold["context_specific_features"]
            )
            context_wrong = cached_feature_signal(
                cache, model, tokenizer, sae, item, item_id, args.layer, fold["context_specific_wrong_features"]
            )
            cscore = row.get("conflict_score")
            if cscore is None:
                cscore = conflict_score(context_spec, context_wrong)
            enriched = dict(row)
            enriched.update(
                {
                    "item_id": item_id,
                    "fold": fold_id,
                    "computed_conflict_score": float(cscore),
                    "noisy_threshold": args.noisy_threshold,
                    "prior_confidence": max_prob(row.get("prior_probs", {})),
                    "correct_dominant_activation": correct_dom,
                    "context_specific_activation": context_spec,
                    "context_specific_wrong_activation": context_wrong,
                    "n_correct_dominant_features": len(fold["correct_dominant_features"]),
                    "n_context_specific_features": len(fold["context_specific_features"]),
                    "n_context_specific_wrong_features": len(fold["context_specific_wrong_features"]),
                }
            )
            cases.append(enriched)

    final_prior_threshold = relax_silent_threshold_if_needed(cases, args.min_silent_cases, args.prior_threshold)
    for row in cases:
        row["prior_threshold"] = final_prior_threshold
        row["group"] = classify(row, row["computed_conflict_score"], args.noisy_threshold, final_prior_threshold)

    groups = defaultdict(list)
    for row in cases:
        groups[row["group"]].append(row)

    metrics = ["correct_dominant_activation", "context_specific_activation", "context_specific_wrong_activation", "computed_conflict_score", "prior_confidence"]
    summary = {}
    for metric in metrics:
        summary[metric] = {
            group: group_stats([row[metric] for row in rows])
            for group, rows in groups.items()
            if group in {"correct", "noisy_wrong", "silent_wrong", "other_wrong"}
        }
        summary[metric]["comparisons"] = {
            "correct_vs_silent_wrong": compare_groups(
                [row[metric] for row in groups["correct"]],
                [row[metric] for row in groups["silent_wrong"]],
            ),
            "correct_vs_noisy_wrong": compare_groups(
                [row[metric] for row in groups["correct"]],
                [row[metric] for row in groups["noisy_wrong"]],
            ),
            "silent_wrong_vs_noisy_wrong": compare_groups(
                [row[metric] for row in groups["silent_wrong"]],
                [row[metric] for row in groups["noisy_wrong"]],
            ),
        }

    table = []
    for group in ["correct", "noisy_wrong", "silent_wrong", "other_wrong"]:
        if group not in groups:
            continue
        table.append(
            {
                "group": group,
                "n": len(groups[group]),
                "correct_dom_mean": summary["correct_dominant_activation"].get(group, {}).get("mean", 0.0),
                "context_spec_mean": summary["context_specific_activation"].get(group, {}).get("mean", 0.0),
                "conflict_score_mean": summary["computed_conflict_score"].get(group, {}).get("mean", 0.0),
                "prior_confidence_mean": summary["prior_confidence"].get(group, {}).get("mean", 0.0),
            }
        )

    output = {
        "model": args.model,
        "layer": args.layer,
        "source_paths": {
            "conflict_set": conflict_path,
            "typeb_features": typeb_path,
            "context_specific_cv": cv_path,
        },
        "thresholds": {
            "noisy_threshold": args.noisy_threshold,
            "initial_prior_threshold": args.prior_threshold,
            "final_prior_threshold": final_prior_threshold,
            "min_silent_cases": args.min_silent_cases,
        },
        "feature_definition": {
            "correct_dominant": f"typeb_features.fold.layer_results[{args.layer}].{args.correct_dominant_source}.correct_dominant",
            "context_specific": f"context_cv.fold.layer_info[{args.layer}].cs_correct",
            "conflict_score": "existing row.conflict_score if present, else cs_wrong / (cs_correct + cs_wrong)",
        },
        "group_counts": {group: len(rows) for group, rows in sorted(groups.items())},
        "table": table,
        "summary": summary,
        "cases": cases,
    }

    output_path = args.output_path or f"{RESULTS_DIR}/silent_override/{args.model}_silent_override_context_compare_L{args.layer}.json"
    write_json(output_path, output)
    print(f"Saved silent-override context comparison to {output_path}")
    print(json.dumps({"group_counts": output["group_counts"], "final_prior_threshold": final_prior_threshold, "table": table}, indent=2))

    del sae
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
