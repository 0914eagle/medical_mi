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


def selected_alpha_for_fold(fold, steering_set):
    match = next((row for row in fold["steering_results"] if row["name"] == steering_set), None)
    return match["selected_alpha"] if match else 10.0


def score_cases(model, tokenizer, sae, items_by_id, labels, item_ids, layer, correct_features, wrong_features):
    rows = []
    for item_id in tqdm(item_ids, desc=f"L{layer} scores", leave=False):
        item_id = str(item_id)
        item = items_by_id.get(item_id)
        label = labels.get(item_id)
        if not item or not label:
            continue
        correct_signal = feature_signal(model, tokenizer, sae, item, layer, correct_features)
        wrong_signal = feature_signal(model, tokenizer, sae, item, layer, wrong_features)
        rows.append(
            {
                "item_id": item_id,
                "score": conflict_score(correct_signal, wrong_signal),
                "ground_truth": label["ground_truth"],
                "original_prediction": label["prediction"],
                "original_is_correct": label["is_correct"],
            }
        )
    return rows


def add_steered_predictions(model, tokenizer, items_by_id, rows, layer_to_vec, alpha):
    for row in tqdm(rows, desc=f"steered predictions alpha={alpha}", leave=False):
        probs, pred = predict_pubmedqa_with_steering(model, tokenizer, items_by_id[row["item_id"]], layer_to_vec, alpha)
        row["steered_prediction"] = pred
        row["steered_probs"] = probs
        row["steered_is_correct"] = pred == row["ground_truth"]
    return rows


def metrics_for_answered(rows, answered_rows, prediction_key, correct_key):
    total = len(rows)
    answered = len(answered_rows)
    correct = sum(1 for row in answered_rows if row[correct_key])
    original_wrong = sum(1 for row in rows if not row["original_is_correct"])
    abstained_ids = {row["item_id"] for row in rows} - {row["item_id"] for row in answered_rows}
    wrong_abstained = sum(1 for row in rows if row["item_id"] in abstained_ids and not row["original_is_correct"])
    return {
        "coverage": answered / total if total else 0.0,
        "risk": 1.0 - (correct / answered) if answered else 1.0,
        "selective_accuracy": correct / answered if answered else 0.0,
        "answered": answered,
        "abstained": total - answered,
        "correct_answered": correct,
        "wrong_answered": answered - correct,
        "wrong_abstained": wrong_abstained,
        "safety_wrong_abstain_rate": wrong_abstained / original_wrong if original_wrong else 0.0,
        "prediction_key": prediction_key,
    }


def baseline_at_coverage(rows, coverage, prediction_key, correct_key):
    n_answer = max(1, int(round(len(rows) * coverage)))
    answered = sorted(rows, key=lambda row: row["score"])[:n_answer]
    return metrics_for_answered(rows, answered, prediction_key, correct_key)


def apply_cai(rows, t_low, t_high):
    evaluated = []
    for row in rows:
        if row["score"] > t_high:
            action = "abstain"
            is_correct = None
            prediction = None
        elif row["score"] < t_low:
            action = "pass"
            is_correct = row["original_is_correct"]
            prediction = row["original_prediction"]
        else:
            action = "steer"
            is_correct = row["steered_is_correct"]
            prediction = row["steered_prediction"]
        case = dict(row)
        case.update({"action": action, "policy_prediction": prediction, "policy_is_correct": is_correct})
        evaluated.append(case)
    answered = [row for row in evaluated if row["action"] != "abstain"]
    total = len(evaluated)
    correct = sum(1 for row in answered if row["policy_is_correct"])
    original_wrong = sum(1 for row in evaluated if not row["original_is_correct"])
    wrong_abstained = sum(1 for row in evaluated if row["action"] == "abstain" and not row["original_is_correct"])
    return {
        "coverage": len(answered) / total if total else 0.0,
        "risk": 1.0 - (correct / len(answered)) if answered else 1.0,
        "selective_accuracy": correct / len(answered) if answered else 0.0,
        "answered": len(answered),
        "abstained": total - len(answered),
        "correct_answered": correct,
        "wrong_answered": len(answered) - correct,
        "wrong_abstained": wrong_abstained,
        "safety_wrong_abstain_rate": wrong_abstained / original_wrong if original_wrong else 0.0,
        "cases": evaluated,
    }


def choose_cai_thresholds(validation_rows, target_coverage):
    scores = np.array([row["score"] for row in validation_rows], dtype=np.float32)
    if scores.size == 0:
        return {"t_low": 0.0, "t_high": 1.0, "validation": {"coverage": 0.0, "selective_accuracy": 0.0}}
    t_high = float(np.quantile(scores, target_coverage))
    low_candidates = sorted(set(float(np.quantile(scores, q)) for q in np.linspace(0.0, target_coverage, 21)))
    best = None
    for t_low in low_candidates:
        if t_low > t_high:
            continue
        metric = apply_cai(validation_rows, t_low, t_high)
        candidate = {
            "t_low": t_low,
            "t_high": t_high,
            "validation": {key: value for key, value in metric.items() if key != "cases"},
        }
        if best is None or (
            candidate["validation"]["selective_accuracy"],
            candidate["validation"]["coverage"],
            candidate["validation"]["safety_wrong_abstain_rate"],
        ) > (
            best["validation"]["selective_accuracy"],
            best["validation"]["coverage"],
            best["validation"]["safety_wrong_abstain_rate"],
        ):
            best = candidate
    return best


def curve_auc(curve, y_key):
    points = sorted((row["coverage"], row[y_key]) for row in curve)
    if len(points) < 2:
        return 0.0
    x = np.array([p[0] for p in points], dtype=np.float32)
    y = np.array([p[1] for p in points], dtype=np.float32)
    return float(np.trapz(y, x))


def main():
    parser = argparse.ArgumentParser(description="Fair CAI Risk-Coverage comparison at matched coverage.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--cv-path", default=None)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--steering-set", default="context_specific_wrong")
    parser.add_argument("--coverages", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
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

        curve_rows = []
        for coverage in args.coverages:
            thresholds = choose_cai_thresholds(validation_rows, coverage)
            cai_metric = apply_cai(test_rows, thresholds["t_low"], thresholds["t_high"])
            no_intervention = baseline_at_coverage(test_rows, coverage, "original_prediction", "original_is_correct")
            always_steer = baseline_at_coverage(test_rows, coverage, "steered_prediction", "steered_is_correct")
            curve_rows.append(
                {
                    "target_coverage": coverage,
                    "thresholds": thresholds,
                    "test": {
                        "no_intervention": no_intervention,
                        "always_steer": always_steer,
                        "cai": {key: value for key, value in cai_metric.items() if key != "cases"},
                    },
                    "cai_cases": cai_metric["cases"],
                }
            )
        fold_results.append(
            {
                "fold": fold_id,
                "layer": args.layer,
                "alpha": alpha,
                "curve": curve_rows,
                "auc": {
                    "no_intervention_risk": curve_auc([row["test"]["no_intervention"] for row in curve_rows], "risk"),
                    "always_steer_risk": curve_auc([row["test"]["always_steer"] for row in curve_rows], "risk"),
                    "cai_risk": curve_auc([row["test"]["cai"] for row in curve_rows], "risk"),
                    "no_intervention_selective_accuracy": curve_auc([row["test"]["no_intervention"] for row in curve_rows], "selective_accuracy"),
                    "always_steer_selective_accuracy": curve_auc([row["test"]["always_steer"] for row in curve_rows], "selective_accuracy"),
                    "cai_selective_accuracy": curve_auc([row["test"]["cai"] for row in curve_rows], "selective_accuracy"),
                },
            }
        )
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    summary = {"by_coverage": {}, "auc": {}}
    for coverage in args.coverages:
        key = str(coverage)
        summary["by_coverage"][key] = {}
        for method in ["no_intervention", "always_steer", "cai"]:
            rows = [
                curve_row["test"][method]
                for fold in fold_results
                for curve_row in fold["curve"]
                if curve_row["target_coverage"] == coverage
            ]
            summary["by_coverage"][key][method] = {
                "risk": aggregate_rates(rows, "risk"),
                "selective_accuracy": aggregate_rates(rows, "selective_accuracy"),
                "coverage": aggregate_rates(rows, "coverage"),
                "safety_wrong_abstain_rate": aggregate_rates(rows, "safety_wrong_abstain_rate"),
            }
    for key in [
        "no_intervention_risk",
        "always_steer_risk",
        "cai_risk",
        "no_intervention_selective_accuracy",
        "always_steer_selective_accuracy",
        "cai_selective_accuracy",
    ]:
        summary["auc"][key] = aggregate_rates([fold["auc"] for fold in fold_results], key)

    output = {
        "model": args.model,
        "cv_path": cv_path,
        "layer": args.layer,
        "steering_set": args.steering_set,
        "coverages": args.coverages,
        "summary": summary,
        "folds": fold_results,
    }
    output_path = f"{RESULTS_DIR}/risk_coverage/{args.model}_cai_risk_coverage_L{args.layer}.json"
    write_json(output_path, output)
    print(f"Saved CAI risk-coverage evaluation to {output_path}")


if __name__ == "__main__":
    main()
