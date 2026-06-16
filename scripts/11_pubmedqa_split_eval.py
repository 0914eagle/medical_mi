import argparse
import os

from split_experiment_utils import (
    RESULTS_DIR,
    evaluate_pubmedqa,
    load_model_and_tokenizer,
    load_official_pubmedqa_test_ids,
    load_pubmedqa_items,
    make_pubmedqa_splits,
    summarize_labels,
    write_json,
)


def subset_summary(labels, ids):
    id_set = {str(item_id) for item_id in ids}
    return summarize_labels([row for row in labels if str(row["item_id"]) in id_set])


def main():
    parser = argparse.ArgumentParser(description="PubMedQA split-aware context/no-context evaluation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--official-test-ids", default=None)
    parser.add_argument("--use-official-test", action="store_true")
    args = parser.parse_args()

    items = load_pubmedqa_items()
    official_ids = set()
    if args.official_test_ids or args.use_official_test:
        official_ids = load_official_pubmedqa_test_ids(
            path=args.official_test_ids,
            allow_download=args.use_official_test,
        )
    splits = make_pubmedqa_splits(items, folds=args.folds, seed=args.seed, official_test_ids=official_ids)

    model, tokenizer = load_model_and_tokenizer(args.model)
    with_context = evaluate_pubmedqa(model, tokenizer, items, include_context=True, desc="PubMedQA with context")
    without_context = evaluate_pubmedqa(model, tokenizer, items, include_context=False, desc="PubMedQA no context")

    fold_summaries = []
    for split in splits:
        fold_summaries.append(
            {
                "fold": split["fold"],
                "n_validation": len(split["validation_ids"]),
                "n_test": len(split["test_ids"]),
                "with_context_validation": subset_summary(with_context, split["validation_ids"]),
                "with_context_test": subset_summary(with_context, split["test_ids"]),
                "without_context_validation": subset_summary(without_context, split["validation_ids"]),
                "without_context_test": subset_summary(without_context, split["test_ids"]),
            }
        )

    result = {
        "model": args.model,
        "split_strategy": "official_test_ids" if official_ids else f"{args.folds}_fold_cv_seed_{args.seed}",
        "official_test_id_count": len(official_ids),
        "overall_with_context": summarize_labels(with_context),
        "overall_without_context": summarize_labels(without_context),
        "fold_summaries": fold_summaries,
        "splits": splits,
        "labels_with_context": with_context,
        "labels_without_context": without_context,
    }

    output_path = f"{RESULTS_DIR}/eval_split/{args.model}_pubmedqa_split_eval.json"
    write_json(output_path, result)
    print(f"Saved split evaluation to {output_path}")
    print(f"With context accuracy: {result['overall_with_context']['accuracy']:.2%}")
    print(f"No context accuracy: {result['overall_without_context']['accuracy']:.2%}")


if __name__ == "__main__":
    main()
