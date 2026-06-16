import argparse
import gc

import numpy as np
import torch
from tqdm import tqdm

from split_experiment_utils import (
    RESULTS_DIR,
    aggregate_rates,
    build_steer_vec,
    conflict_score,
    feature_signal,
    item_map_by_id,
    labels_by_id,
    load_model_and_tokenizer,
    load_pubmedqa_items,
    load_sae,
    predict_pubmedqa_with_steering,
    read_json,
    write_json,
)


def get_original_row(labels, item_id):
    row = labels.get(str(item_id))
    if row is None:
        raise KeyError(f"Missing label for item_id={item_id}")
    return row


def score_cases(model, tokenizer, sae, items_by_id, labels, item_ids, layer, correct_features, wrong_features):
    rows = []
    for item_id in tqdm(item_ids, desc=f"L{layer} conflict scores", leave=False):
        item_id = str(item_id)
        item = items_by_id.get(item_id)
        if not item or item_id not in labels:
            continue
        correct_signal = feature_signal(model, tokenizer, sae, item, layer, correct_features)
        wrong_signal = feature_signal(model, tokenizer, sae, item, layer, wrong_features)
        label = labels[item_id]
        rows.append(
            {
                "item_id": item_id,
                "score": conflict_score(correct_signal, wrong_signal),
                "ground_truth": label["ground_truth"],
                "original_prediction": label["prediction"],
                "original_is_correct": label["is_correct"],
                "correct_signal": correct_signal,
                "wrong_signal": wrong_signal,
            }
        )
    return rows


def add_steered_predictions(model, tokenizer, items_by_id, rows, layer_to_vec, alpha):
    for row in tqdm(rows, desc=f"steered predictions alpha={alpha}", leave=False):
        item = items_by_id[row["item_id"]]
        probs, pred = predict_pubmedqa_with_steering(model, tokenizer, item, layer_to_vec, alpha)
        row["steered_prediction"] = pred
        row["steered_probs"] = probs
        row["steered_is_correct"] = pred == row["ground_truth"]
    return rows


def metric_no_intervention(rows):
    total = len(rows)
    correct = sum(1 for row in rows if row["original_is_correct"])
    wrong = total - correct
    return {
        "accuracy": correct / total if total else 0.0,
        "selective_accuracy": correct / total if total else 0.0,
        "coverage": 1.0 if total else 0.0,
        "safety_wrong_abstain_rate": 0.0,
        "answered": total,
        "abstained": 0,
        "correct_answered": correct,
        "wrong_answered": wrong,
        "wrong_abstained": 0,
    }


def metric_always_steer(rows):
    total = len(rows)
    correct = sum(1 for row in rows if row["steered_is_correct"])
    wrong = total - correct
    return {
        "accuracy": correct / total if total else 0.0,
        "selective_accuracy": correct / total if total else 0.0,
        "coverage": 1.0 if total else 0.0,
        "safety_wrong_abstain_rate": 0.0,
        "answered": total,
        "abstained": 0,
        "correct_answered": correct,
        "wrong_answered": wrong,
        "wrong_abstained": 0,
    }


def apply_cai_policy(rows, t_low, t_high):
    evaluated = []
    answered = 0
    correct_answered = 0
    wrong_answered = 0
    wrong_abstained = 0
    abstained = 0
    original_wrong = sum(1 for row in rows if not row["original_is_correct"])

    for row in rows:
        score = row["score"]
        if score < t_low:
            action = "pass"
            prediction = row["original_prediction"]
            is_correct = row["original_is_correct"]
        elif score <= t_high:
            action = "steer"
            prediction = row["steered_prediction"]
            is_correct = row["steered_is_correct"]
        else:
            action = "abstain"
            prediction = None
            is_correct = None

        case_row = dict(row)
        case_row.update({"action": action, "policy_prediction": prediction, "policy_is_correct": is_correct})
        evaluated.append(case_row)

        if action == "abstain":
            abstained += 1
            if not row["original_is_correct"]:
                wrong_abstained += 1
        else:
            answered += 1
            if is_correct:
                correct_answered += 1
            else:
                wrong_answered += 1

    total = len(rows)
    return {
        "selective_accuracy": correct_answered / answered if answered else 0.0,
        "coverage": answered / total if total else 0.0,
        "safety_wrong_abstain_rate": wrong_abstained / original_wrong if original_wrong else 0.0,
        "answered": answered,
        "abstained": abstained,
        "correct_answered": correct_answered,
        "wrong_answered": wrong_answered,
        "wrong_abstained": wrong_abstained,
        "cases": evaluated,
    }


def candidate_thresholds(rows):
    scores = np.array([row["score"] for row in rows], dtype=np.float32)
    if scores.size == 0:
        return [0.0], [1.0]
    quantiles = np.linspace(0.05, 0.95, 19)
    values = sorted(set(float(np.quantile(scores, q)) for q in quantiles))
    return values, values


def select_thresholds(validation_rows, min_coverage):
    lows, highs = candidate_thresholds(validation_rows)
    candidates = []
    for t_low in lows:
        for t_high in highs:
            if t_low > t_high:
                continue
            metric = apply_cai_policy(validation_rows, t_low, t_high)
            if metric["coverage"] < min_coverage:
                continue
            candidates.append(
                {
                    "t_low": t_low,
                    "t_high": t_high,
                    "selective_accuracy": metric["selective_accuracy"],
                    "coverage": metric["coverage"],
                    "safety_wrong_abstain_rate": metric["safety_wrong_abstain_rate"],
                }
            )
    if not candidates:
        metric = apply_cai_policy(validation_rows, 0.0, 1.0)
        return {
            "t_low": 0.0,
            "t_high": 1.0,
            "selective_accuracy": metric["selective_accuracy"],
            "coverage": metric["coverage"],
            "safety_wrong_abstain_rate": metric["safety_wrong_abstain_rate"],
        }
    return max(
        candidates,
        key=lambda row: (
            row["selective_accuracy"],
            row["coverage"],
            row["safety_wrong_abstain_rate"],
        ),
    )


def selected_alpha_for_fold(fold, steering_set):
    match = next((row for row in fold["steering_results"] if row["name"] == steering_set), None)
    if not match:
        return 10.0
    return match["selected_alpha"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAI pass/steer/abstain policy with validation thresholds.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--cv-path", default=None)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--steering-set", default="context_specific_wrong")
    parser.add_argument("--min-coverage", type=float, default=0.7)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    cv_path = args.cv_path or f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json"
    cv_data = read_json(cv_path)
    eval_data = read_json(f"{RESULTS_DIR}/eval_split/{args.model}_pubmedqa_split_eval.json")
    labels = labels_by_id(eval_data["labels_with_context"])
    items = load_pubmedqa_items()
    items_by_id = item_map_by_id(items)
    split_by_fold = {row["fold"]: row for row in cv_data["splits"]}

    model, tokenizer = load_model_and_tokenizer(args.model)
    fold_results = []
    for fold in cv_data["folds"]:
        fold_id = fold["fold"]
        if str(args.layer) not in fold["layer_info"]:
            continue
        split = split_by_fold[fold_id]
        validation_ids = split["validation_ids"][: args.max_cases] if args.max_cases else split["validation_ids"]
        test_ids = split["test_ids"][: args.max_cases] if args.max_cases else split["test_ids"]
        layer_info = fold["layer_info"][str(args.layer)]
        sae = load_sae(args.model, args.layer, model.device)
        steer_vec = build_steer_vec(sae, suppress=layer_info["cs_wrong"], device=model.device)
        layer_to_vec = {args.layer: steer_vec}
        alpha = selected_alpha_for_fold(fold, args.steering_set)

        validation_rows = score_cases(
            model,
            tokenizer,
            sae,
            items_by_id,
            labels,
            validation_ids,
            args.layer,
            layer_info["cs_correct"],
            layer_info["cs_wrong"],
        )
        test_rows = score_cases(
            model,
            tokenizer,
            sae,
            items_by_id,
            labels,
            test_ids,
            args.layer,
            layer_info["cs_correct"],
            layer_info["cs_wrong"],
        )
        validation_rows = add_steered_predictions(model, tokenizer, items_by_id, validation_rows, layer_to_vec, alpha)
        test_rows = add_steered_predictions(model, tokenizer, items_by_id, test_rows, layer_to_vec, alpha)
        thresholds = select_thresholds(validation_rows, args.min_coverage)
        validation_cai = apply_cai_policy(validation_rows, thresholds["t_low"], thresholds["t_high"])
        test_cai = apply_cai_policy(test_rows, thresholds["t_low"], thresholds["t_high"])
        fold_results.append(
            {
                "fold": fold_id,
                "layer": args.layer,
                "alpha": alpha,
                "thresholds": thresholds,
                "validation": {
                    "no_intervention": metric_no_intervention(validation_rows),
                    "always_steer": metric_always_steer(validation_rows),
                    "cai": {key: value for key, value in validation_cai.items() if key != "cases"},
                },
                "test": {
                    "no_intervention": metric_no_intervention(test_rows),
                    "always_steer": metric_always_steer(test_rows),
                    "cai": {key: value for key, value in test_cai.items() if key != "cases"},
                },
                "test_cases": test_cai["cases"],
            }
        )
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    summary = {}
    for baseline in ["no_intervention", "always_steer", "cai"]:
        rows = [fold["test"][baseline] for fold in fold_results]
        summary[baseline] = {
            "selective_accuracy": aggregate_rates(rows, "selective_accuracy"),
            "coverage": aggregate_rates(rows, "coverage"),
            "safety_wrong_abstain_rate": aggregate_rates(rows, "safety_wrong_abstain_rate"),
        }
        if baseline in ["no_intervention", "always_steer"]:
            summary[baseline]["accuracy"] = aggregate_rates(rows, "accuracy")

    output = {
        "model": args.model,
        "cv_path": cv_path,
        "layer": args.layer,
        "steering_set": args.steering_set,
        "min_coverage": args.min_coverage,
        "summary": summary,
        "folds": fold_results,
    }
    output_path = f"{RESULTS_DIR}/cai_policy/{args.model}_cai_policy_L{args.layer}.json"
    write_json(output_path, output)
    print(f"Saved CAI policy evaluation to {output_path}")


if __name__ == "__main__":
    main()
