import argparse

from split_experiment_utils import (
    RESULTS_DIR,
    auc_score,
    conflict_score,
    feature_signal,
    item_map_by_id,
    labels_by_id,
    load_model_and_tokenizer,
    load_pubmedqa_items,
    load_sae,
    read_json,
    threshold_from_validation,
    write_json,
)


def score_ids(model, tokenizer, sae, items_by_id, item_ids, labels, layer, correct_features, wrong_features):
    rows = []
    for item_id in item_ids:
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
                "is_wrong": int(not label["is_correct"]),
                "correct_signal": correct_signal,
                "wrong_signal": wrong_signal,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Validation-thresholded conflict score CV.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--cv-path", default=None)
    parser.add_argument("--layer", type=int, default=20)
    args = parser.parse_args()

    cv_path = args.cv_path or f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json"
    cv_data = read_json(cv_path)
    eval_data = read_json(f"{RESULTS_DIR}/eval_split/{args.model}_pubmedqa_split_eval.json")
    labels = labels_by_id(eval_data["labels_with_context"])
    items = load_pubmedqa_items()
    items_by_id = item_map_by_id(items)

    model, tokenizer = load_model_and_tokenizer(args.model)
    sae = load_sae(args.model, args.layer, model.device)

    fold_rows = []
    for fold in cv_data["folds"]:
        layer_info = fold["layer_info"].get(str(args.layer))
        if not layer_info:
            continue
        split = cv_data.get("splits", [])
        validation_ids = None
        test_ids = None
        if split:
            match = next((row for row in split if row["fold"] == fold["fold"]), None)
            if match:
                validation_ids = match["validation_ids"]
                test_ids = match["test_ids"]
        if validation_ids is None:
            validation_ids = [case["item_id"] for result in fold["steering_results"] for case in result["test"].get("kept_correct", [])]
            test_ids = [case["item_id"] for result in fold["steering_results"] for case in result["test"].get("recovered", []) + result["test"].get("unrecovered", [])]

        val_rows = score_ids(
            model,
            tokenizer,
            sae,
            items_by_id,
            validation_ids,
            labels,
            args.layer,
            layer_info["cs_correct"],
            layer_info["cs_wrong"],
        )
        test_rows = score_ids(
            model,
            tokenizer,
            sae,
            items_by_id,
            test_ids,
            labels,
            args.layer,
            layer_info["cs_correct"],
            layer_info["cs_wrong"],
        )
        threshold_info = threshold_from_validation([row["score"] for row in val_rows], [row["is_wrong"] for row in val_rows])
        threshold = threshold_info["threshold"]
        tp = sum(1 for row in test_rows if row["score"] >= threshold and row["is_wrong"])
        fp = sum(1 for row in test_rows if row["score"] >= threshold and not row["is_wrong"])
        tn = sum(1 for row in test_rows if row["score"] < threshold and not row["is_wrong"])
        fn = sum(1 for row in test_rows if row["score"] < threshold and row["is_wrong"])
        fold_rows.append(
            {
                "fold": fold["fold"],
                "validation_auc": threshold_info["auc"],
                "threshold": threshold,
                "test_auc": auc_score([row["score"] for row in test_rows], [row["is_wrong"] for row in test_rows]),
                "test_confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                "validation_scores": val_rows,
                "test_scores": test_rows,
            }
        )

    output = {"model": args.model, "layer": args.layer, "cv_path": cv_path, "folds": fold_rows}
    output_path = f"{RESULTS_DIR}/conflict_score/{args.model}_conflict_score_cv_L{args.layer}.json"
    write_json(output_path, output)
    print(f"Saved conflict score CV to {output_path}")


if __name__ == "__main__":
    main()
