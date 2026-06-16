import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

from split_experiment_utils import (
    RESULTS_DIR,
    conflict_score,
    feature_signal,
    item_map_by_id,
    labels_by_id,
    load_model_and_tokenizer,
    load_pubmedqa_items,
    load_sae,
    reconstruction_error,
    read_json,
    write_json,
)


def group_stats(values):
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    return {"n": len(values), "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def compare(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
    return {"mannwhitney_u": float(stat), "p_value": float(p_value)}


def main():
    parser = argparse.ArgumentParser(description="Analyze recovered vs unrecovered PubMedQA steering cases.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--cv-path", default=None)
    parser.add_argument("--steering-set", default="context_specific_wrong")
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args()

    cv_path = args.cv_path or f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json"
    cv_data = read_json(cv_path)
    eval_data = read_json(f"{RESULTS_DIR}/eval_split/{args.model}_pubmedqa_split_eval.json")
    labels_without = labels_by_id(eval_data["labels_without_context"])
    items = load_pubmedqa_items()
    items_by_id = item_map_by_id(items)

    model, tokenizer = load_model_and_tokenizer(args.model)
    sae = load_sae(args.model, args.layer, model.device)

    rows = []
    for fold in cv_data["folds"]:
        layer_info = fold["layer_info"].get(str(args.layer))
        if not layer_info:
            continue
        cs_wrong = layer_info["cs_wrong"]
        cs_correct = layer_info["cs_correct"]
        for result in fold["steering_results"]:
            if result["name"] != args.steering_set:
                continue
            for status_name in ["recovered", "unrecovered"]:
                for case in result["test"][status_name]:
                    item_id = str(case["item_id"])
                    item = items_by_id.get(item_id)
                    if not item:
                        continue
                    wrong_signal = feature_signal(model, tokenizer, sae, item, args.layer, cs_wrong)
                    correct_signal = feature_signal(model, tokenizer, sae, item, args.layer, cs_correct)
                    prior = labels_without.get(item_id, {})
                    rows.append(
                        {
                            "fold": fold["fold"],
                            "item_id": item_id,
                            "status": status_name,
                            "conflict_score": conflict_score(correct_signal, wrong_signal),
                            "prior_confidence": prior.get("confidence", 0.0),
                            "context_feature_activation": correct_signal,
                            "wrong_feature_activation": wrong_signal,
                            "reconstruction_error": reconstruction_error(model, tokenizer, sae, item, args.layer),
                        }
                    )

    metrics = ["conflict_score", "prior_confidence", "context_feature_activation", "wrong_feature_activation", "reconstruction_error"]
    by_status = defaultdict(list)
    for row in rows:
        by_status[row["status"]].append(row)

    summary = {}
    for metric in metrics:
        recovered_values = [row[metric] for row in by_status["recovered"]]
        unrecovered_values = [row[metric] for row in by_status["unrecovered"]]
        summary[metric] = {
            "recovered": group_stats(recovered_values),
            "unrecovered": group_stats(unrecovered_values),
            "test": compare(recovered_values, unrecovered_values),
        }

    output = {
        "model": args.model,
        "cv_path": cv_path,
        "steering_set": args.steering_set,
        "layer": args.layer,
        "summary": summary,
        "cases": rows,
    }
    output_path = f"{RESULTS_DIR}/ceiling_analysis/{args.model}_{args.steering_set}_L{args.layer}.json"
    write_json(output_path, output)
    print(f"Saved ceiling analysis to {output_path}")


if __name__ == "__main__":
    main()
