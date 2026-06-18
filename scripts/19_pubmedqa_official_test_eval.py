import argparse
import os

from sklearn.metrics import f1_score

from split_experiment_utils import (
    RESULTS_DIR,
    evaluate_pubmedqa,
    item_map_by_id,
    labels_by_id,
    load_model_and_tokenizer,
    load_official_pubmedqa_test_ids,
    load_pubmedqa_items,
    summarize_labels,
    write_json,
)


LABELS = ["yes", "no", "maybe"]


def macro_f1(rows):
    if not rows:
        return 0.0
    y_true = [row["ground_truth"] for row in rows]
    y_pred = [row["prediction"] for row in rows]
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def get_rows_for_ids(existing_rows, item_ids):
    row_map = labels_by_id(existing_rows)
    return [row_map[str(item_id)] for item_id in item_ids if str(item_id) in row_map]


def main():
    parser = argparse.ArgumentParser(description="Evaluate PubMedQA official 500 test split with accuracy and macro-F1.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--official-test-ids", default=None)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    official_ids = load_official_pubmedqa_test_ids(
        path=args.official_test_ids,
        allow_download=args.allow_download,
    )
    if not official_ids:
        raise RuntimeError(
            "Official PubMedQA test IDs not found. Pass --official-test-ids /path/to/test_ground_truth.json "
            "or rerun with --allow-download if the server has network access."
        )

    items = load_pubmedqa_items()
    items_by_id = item_map_by_id(items)
    test_ids = [item_id for item_id in official_ids if str(item_id) in items_by_id]
    test_items = [items_by_id[str(item_id)] for item_id in test_ids]

    eval_path = f"{RESULTS_DIR}/eval_split/{args.model}_pubmedqa_split_eval.json"
    with_context = []
    without_context = []
    if os.path.exists(eval_path) and not args.force_rerun:
        import json

        with open(eval_path, "r") as f:
            eval_data = json.load(f)
        with_context = get_rows_for_ids(eval_data.get("labels_with_context", []), test_ids)
        without_context = get_rows_for_ids(eval_data.get("labels_without_context", []), test_ids)

    if len(with_context) != len(test_items) or len(without_context) != len(test_items):
        model, tokenizer = load_model_and_tokenizer(args.model)
        with_context = evaluate_pubmedqa(model, tokenizer, test_items, include_context=True, desc="Official PubMedQA with context")
        without_context = evaluate_pubmedqa(model, tokenizer, test_items, include_context=False, desc="Official PubMedQA no context")

    output = {
        "model": args.model,
        "official_test_id_count": len(official_ids),
        "matched_test_count": len(test_items),
        "with_context": {
            **summarize_labels(with_context),
            "macro_f1": macro_f1(with_context),
        },
        "without_context": {
            **summarize_labels(without_context),
            "macro_f1": macro_f1(without_context),
        },
        "labels_with_context": with_context,
        "labels_without_context": without_context,
    }
    output_path = f"{RESULTS_DIR}/official_pubmedqa/{args.model}_official_test_eval.json"
    write_json(output_path, output)
    print(f"Saved official PubMedQA test evaluation to {output_path}")
    print(f"With context: accuracy={output['with_context']['accuracy']:.2%}, macro_f1={output['with_context']['macro_f1']:.3f}")
    print(f"No context: accuracy={output['without_context']['accuracy']:.2%}, macro_f1={output['without_context']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
