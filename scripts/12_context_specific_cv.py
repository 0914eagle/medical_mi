import argparse
import gc
import os

import torch

from split_experiment_utils import (
    RESULTS_DIR,
    aggregate_rates,
    build_steer_vec,
    discover_dominant_features,
    ensure_dir,
    evaluate_pubmedqa,
    evaluate_steering_set,
    filter_context_specific_features,
    item_map_by_id,
    labels_by_id,
    load_medqa_items,
    load_model_and_tokenizer,
    load_official_pubmedqa_test_ids,
    load_pubmedqa_items,
    load_sae,
    make_pubmedqa_splits,
    pick_best_alpha,
    read_json,
    select_cases,
    write_json,
)


def load_or_create_labels(model_name, model, tokenizer, items):
    path = f"{RESULTS_DIR}/eval_split/{model_name}_pubmedqa_split_eval.json"
    if os.path.exists(path):
        return read_json(path)["labels_with_context"]
    return evaluate_pubmedqa(model, tokenizer, items, include_context=True, desc="PubMedQA labels")


def layer_to_vec_from_features(model_name, layer, model_device, suppress=None, amplify=None):
    sae = load_sae(model_name, layer, model_device)
    steer_vec = build_steer_vec(sae, amplify=amplify, suppress=suppress, device=model_device)
    del sae
    gc.collect()
    torch.cuda.empty_cache()
    return steer_vec


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe context-specific steering with validation/test splits.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 18, 22])
    parser.add_argument("--multi-layers", type=int, nargs="*", default=[18, 20, 22])
    parser.add_argument("--alphas", type=float, nargs="+", default=[5.0, 7.0, 10.0, 15.0, 20.0])
    parser.add_argument("--max-feature-cases", type=int, default=100)
    parser.add_argument("--max-filter-items", type=int, default=50)
    parser.add_argument("--max-steering-cases", type=int, default=None)
    parser.add_argument("--official-test-ids", default=None)
    parser.add_argument("--use-official-test", action="store_true")
    parser.add_argument("--single-feature", type=int, default=28696)
    args = parser.parse_args()

    items = load_pubmedqa_items()
    items_by_id = item_map_by_id(items)
    official_ids = set()
    if args.official_test_ids or args.use_official_test:
        official_ids = load_official_pubmedqa_test_ids(
            path=args.official_test_ids,
            allow_download=args.use_official_test,
        )
    splits = make_pubmedqa_splits(items, folds=args.folds, seed=args.seed, official_test_ids=official_ids)
    medqa_items = load_medqa_items(limit=args.max_filter_items)

    model, tokenizer = load_model_and_tokenizer(args.model)
    labels = load_or_create_labels(args.model, model, tokenizer, items)
    label_map = labels_by_id(labels)

    fold_results = []
    for split in splits:
        fold = split["fold"]
        print(f"\n=== Fold {fold} ===")
        val_correct = select_cases(items_by_id, label_map, split["validation_ids"], want_correct=True)
        val_wrong = select_cases(items_by_id, label_map, split["validation_ids"], want_correct=False)
        test_correct = select_cases(items_by_id, label_map, split["test_ids"], want_correct=True)
        test_wrong = select_cases(items_by_id, label_map, split["test_ids"], want_correct=False)

        layer_info = {}
        for layer in args.layers:
            print(f"Fold {fold}: selecting features on validation, layer {layer}")
            sae = load_sae(args.model, layer, model.device)
            dominant = discover_dominant_features(
                model,
                tokenizer,
                sae,
                val_correct,
                val_wrong,
                layer,
                max_cases=args.max_feature_cases,
            )
            candidates = sorted(set(dominant["correct_dominant"] + dominant["wrong_dominant"]))
            cs_all, cs_stats = filter_context_specific_features(
                model,
                tokenizer,
                sae,
                [items_by_id[item_id] for item_id in split["validation_ids"] if item_id in items_by_id],
                medqa_items,
                layer,
                candidates,
                max_items=args.max_filter_items,
            )
            layer_info[str(layer)] = {
                "dominant": dominant,
                "context_specific_stats": cs_stats,
                "cs_wrong": [idx for idx in dominant["wrong_dominant"] if idx in cs_all],
                "cs_correct": [idx for idx in dominant["correct_dominant"] if idx in cs_all],
            }
            del sae
            gc.collect()
            torch.cuda.empty_cache()

        steering_sets = []
        primary_layer = args.layers[0]
        primary = layer_info[str(primary_layer)]
        steering_sets.append(
            {
                "name": "all_wrong_dominant",
                "layers": {str(primary_layer): {"suppress": primary["dominant"]["wrong_dominant"], "amplify": []}},
            }
        )
        steering_sets.append(
            {
                "name": "context_specific_wrong",
                "layers": {str(primary_layer): {"suppress": primary["cs_wrong"], "amplify": []}},
            }
        )
        steering_sets.append(
            {
                "name": f"single_{args.single_feature}_suppress",
                "layers": {str(primary_layer): {"suppress": [args.single_feature], "amplify": []}},
            }
        )

        multi_layers = [layer for layer in args.multi_layers if str(layer) in layer_info]
        if len(multi_layers) > 1:
            steering_sets.append(
                {
                    "name": "multi_layer_context_specific_wrong",
                    "layers": {
                        str(layer): {"suppress": layer_info[str(layer)]["cs_wrong"], "amplify": []}
                        for layer in multi_layers
                    },
                }
            )

        set_results = []
        for steering_set in steering_sets:
            if not any(cfg["suppress"] or cfg["amplify"] for cfg in steering_set["layers"].values()):
                continue
            layer_to_vec = {}
            for layer_str, cfg in steering_set["layers"].items():
                layer_to_vec[int(layer_str)] = layer_to_vec_from_features(
                    args.model,
                    int(layer_str),
                    model.device,
                    suppress=cfg["suppress"],
                    amplify=cfg["amplify"],
                )

            alpha_rows = []
            for alpha in args.alphas:
                val_result = evaluate_steering_set(
                    model,
                    tokenizer,
                    val_wrong,
                    val_correct,
                    layer_to_vec,
                    alpha,
                    max_cases=args.max_steering_cases,
                )
                alpha_rows.append(
                    {
                        "alpha": alpha,
                        "recovery_rate": val_result["recovery_rate"],
                        "corruption_rate": val_result["corruption_rate"],
                    }
                )
            best = pick_best_alpha(alpha_rows)
            test_result = evaluate_steering_set(
                model,
                tokenizer,
                test_wrong,
                test_correct,
                layer_to_vec,
                best["alpha"],
                max_cases=args.max_steering_cases,
            )
            set_results.append(
                {
                    "name": steering_set["name"],
                    "selected_layers": steering_set["layers"],
                    "validation_alpha_sweep": alpha_rows,
                    "selected_alpha": best["alpha"],
                    "test": test_result,
                }
            )

        fold_results.append(
            {
                "fold": fold,
                "n_validation_correct": len(val_correct),
                "n_validation_wrong": len(val_wrong),
                "n_test_correct": len(test_correct),
                "n_test_wrong": len(test_wrong),
                "layer_info": layer_info,
                "steering_results": set_results,
            }
        )

    summary = {}
    set_names = sorted({row["name"] for fold in fold_results for row in fold["steering_results"]})
    for set_name in set_names:
        rows = [row["test"] for fold in fold_results for row in fold["steering_results"] if row["name"] == set_name]
        summary[set_name] = {
            "recovery_rate": aggregate_rates(rows, "recovery_rate"),
            "corruption_rate": aggregate_rates(rows, "corruption_rate"),
        }

    output = {
        "model": args.model,
        "split_strategy": "official_test_ids" if official_ids else f"{args.folds}_fold_cv_seed_{args.seed}",
        "layers": args.layers,
        "alphas": args.alphas,
        "splits": splits,
        "folds": fold_results,
        "summary": summary,
    }
    output_path = f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json"
    write_json(output_path, output)
    print(f"Saved context-specific CV results to {output_path}")


if __name__ == "__main__":
    main()
