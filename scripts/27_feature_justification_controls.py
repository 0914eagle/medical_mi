import argparse
import gc
import json
import os
import random
import re
from collections import Counter

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

from split_experiment_utils import (
    BASE_DIR,
    RESULTS_DIR,
    get_activation_with_hook,
    item_map_by_id,
    load_model_and_tokenizer,
    load_pubmedqa_items,
    load_sae,
    read_json,
    write_json,
)
from utils import format_pubmedqa, get_ynm_probs


def path_candidates(path, subdir, filename):
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


def row_id(row):
    return str(row.get("item_id", row.get("pubid", "")))


def pubmedqa_context_text(item):
    context = item.get("context", "")
    if isinstance(context, dict):
        contexts = context.get("contexts", [])
        return " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    return str(context)


def with_context(item, context_text):
    cloned = dict(item)
    cloned["context"] = context_text
    return cloned


def sae_encode_prompt(model, tokenizer, sae, prompt, layer):
    act = get_activation_with_hook(model, tokenizer, prompt, layer)
    with torch.no_grad():
        return sae.encode(act)[0].detach().float().cpu().numpy()


def sae_encode_item(model, tokenizer, sae, item, layer):
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    return sae_encode_prompt(model, tokenizer, sae, prompt, layer)


def selected_signal(encoded, features):
    if not features:
        return 0.0
    return float(encoded[[int(feature) for feature in features]].sum())


def group_stats(values):
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
    }


def paired_test(before, after):
    if len(before) < 2 or len(after) < 2:
        return None
    diffs = np.asarray(before) - np.asarray(after)
    if np.allclose(diffs, 0):
        return {"wilcoxon_statistic": 0.0, "p_value": 1.0}
    stat, p_value = wilcoxon(before, after, zero_method="wilcox", alternative="two-sided")
    return {"wilcoxon_statistic": float(stat), "p_value": float(p_value)}


def behavior_label(row):
    return 1 if row["prior_answer"] != row["context_answer"] else 0


def is_behavior_conflict(row):
    return row.get("prior_answer") != row.get("ground_truth")


def binarize_feature(values, mode):
    arr = np.asarray(values)
    if mode == "positive":
        return (arr > 0).astype(int)
    if mode == "median":
        return (arr > np.median(arr)).astype(int)
    if mode == "q75":
        return (arr > np.quantile(arr, 0.75)).astype(int)
    raise ValueError("--mi-binarize must be one of: positive, median, q75")


def top_mi_features(acts, labels, top_k, mode):
    scores = []
    for feature_idx in tqdm(range(acts.shape[1]), desc="MI per feature", leave=False):
        feat_binary = binarize_feature(acts[:, feature_idx], mode)
        if len(set(feat_binary.tolist())) < 2:
            mi = 0.0
        else:
            mi = mutual_info_score(feat_binary, labels)
        scores.append((feature_idx, float(mi)))
    return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]


def build_fold_features(cv_data, layer, feature_source):
    fold_features = {}
    for fold in cv_data.get("folds", []):
        layer_info = fold.get("layer_info", {}).get(str(layer), {})
        cs_correct = [int(feature) for feature in layer_info.get("cs_correct", [])]
        cs_wrong = [int(feature) for feature in layer_info.get("cs_wrong", [])]
        if feature_source == "cs_correct":
            features = cs_correct
        elif feature_source == "cs_wrong":
            features = cs_wrong
        elif feature_source == "cs_both":
            features = sorted(set(cs_correct + cs_wrong))
        else:
            raise ValueError("--feature-source must be one of: cs_correct, cs_wrong, cs_both")
        fold_features[int(fold["fold"])] = {
            "features": features,
            "cs_correct": cs_correct,
            "cs_wrong": cs_wrong,
        }
    return fold_features


def build_item_to_fold(cv_data):
    item_to_fold = {}
    for split in cv_data.get("splits", []):
        fold_id = int(split["fold"])
        for item_id in split.get("test_ids", []):
            item_to_fold[str(item_id)] = fold_id
    return item_to_fold


def most_common_features(fold_features, max_features):
    counter = Counter()
    for fold in fold_features.values():
        counter.update(fold["features"])
    features = [feature for feature, _ in counter.most_common(max_features)]
    return features


FLIP_RULES = [
    (r"\bsignificant improvement\b", "no significant improvement"),
    (r"\bsignificantly improved\b", "did not significantly improve"),
    (r"\bwas effective\b", "was not effective"),
    (r"\bwere effective\b", "were not effective"),
    (r"\bis effective\b", "is not effective"),
    (r"\bare effective\b", "are not effective"),
    (r"\bwas associated with\b", "was not associated with"),
    (r"\bwere associated with\b", "were not associated with"),
    (r"\bis associated with\b", "is not associated with"),
    (r"\bare associated with\b", "are not associated with"),
    (r"\bincreased\b", "decreased"),
    (r"\bdecreased\b", "increased"),
    (r"\bhigher\b", "lower"),
    (r"\blower\b", "higher"),
    (r"\bimproved\b", "worsened"),
    (r"\bworsened\b", "improved"),
    (r"\bbeneficial\b", "not beneficial"),
    (r"\bno significant difference\b", "a significant difference"),
    (r"\bnot effective\b", "effective"),
    (r"\bdid not significantly improve\b", "significantly improved"),
    (r"\bno evidence\b", "evidence"),
]


def flip_conclusion(context_text):
    sentences = re.split(r"(?<=[.!?])\s+", context_text.strip())
    for sentence_index in range(len(sentences) - 1, -1, -1):
        sentence = sentences[sentence_index]
        for pattern, replacement in FLIP_RULES:
            flipped, n_subs = re.subn(pattern, replacement, sentence, count=1, flags=re.IGNORECASE)
            if n_subs:
                changed = sentences[:]
                changed[sentence_index] = flipped
                return " ".join(changed), True, {"sentence_index": sentence_index, "pattern": pattern, "replacement": replacement}
    return context_text, False, None


def shuffle_context(context_text, rng):
    words = context_text.split()
    rng.shuffle(words)
    return " ".join(words)


def truncate_to_token_length(tokenizer, text, target_len):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= target_len:
        return text
    return tokenizer.decode(tokens[:target_len], skip_special_tokens=True)


def make_length_control(item_id, real_context, tokenizer, filler_pool, rng):
    target_len = len(tokenizer.encode(real_context, add_special_tokens=False))
    candidates = [row for row in filler_pool if row["item_id"] != item_id]
    rng.shuffle(candidates)
    pieces = []
    source_ids = []
    for candidate in candidates:
        pieces.append(candidate["context"])
        source_ids.append(candidate["item_id"])
        combined = " ".join(pieces)
        if len(tokenizer.encode(combined, add_special_tokens=False)) >= target_len:
            return truncate_to_token_length(tokenizer, combined, target_len), source_ids
    return truncate_to_token_length(tokenizer, " ".join(pieces), target_len), source_ids


def text_excerpt(text, max_chars):
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def run_mi_experiment(model, tokenizer, sae, rows, items_by_id, layer, max_cases, top_k, binarize):
    selected_rows = [row for row in rows if is_behavior_conflict(row) and row_id(row) in items_by_id]
    selected_rows = selected_rows[:max_cases] if max_cases else selected_rows
    acts = []
    labels = []
    case_ids = []
    for row in tqdm(selected_rows, desc=f"L{layer} MI activations", leave=False):
        item_id = row_id(row)
        encoded = sae_encode_item(model, tokenizer, sae, items_by_id[item_id], layer)
        acts.append(encoded)
        labels.append(behavior_label(row))
        case_ids.append(item_id)

    if not acts:
        return {"n_cases": 0, "label_counts": {}, "top_features": [], "case_ids": []}

    acts = np.stack(acts, axis=0)
    labels = np.asarray(labels, dtype=np.int64)
    top_features = top_mi_features(acts, labels, top_k, binarize)
    return {
        "n_cases": len(case_ids),
        "label_counts": {"context_follow": int(labels.sum()), "memory_follow": int((labels == 0).sum())},
        "binarize": binarize,
        "top_features": [{"feature": int(feature), "mi": float(mi)} for feature, mi in top_features],
        "case_ids": case_ids,
    }


def summarize_control(rows, real_key, control_key):
    real = [row[real_key] for row in rows]
    control = [row[control_key] for row in rows]
    diffs = [row[real_key] - row[control_key] for row in rows]
    return {
        "real": group_stats(real),
        "control": group_stats(control),
        "diff_real_minus_control": group_stats(diffs),
        "paired_test": paired_test(real, control),
    }


def run_control_experiments(
    model,
    tokenizer,
    sae,
    rows,
    items_by_id,
    item_to_fold,
    fold_features,
    layer,
    max_samples,
    seed,
    store_text,
    text_max_chars,
):
    rng = random.Random(seed)
    eligible = [
        row
        for row in rows
        if row_id(row) in items_by_id
        and row_id(row) in item_to_fold
        and fold_features.get(item_to_fold[row_id(row)], {}).get("features")
    ]
    rng.shuffle(eligible)
    selected = eligible[:max_samples] if max_samples else eligible
    filler_pool = [{"item_id": item_id, "context": pubmedqa_context_text(item)} for item_id, item in items_by_id.items()]

    control_rows = []
    for row in tqdm(selected, desc=f"L{layer} content/length/shuffle controls", leave=False):
        item_id = row_id(row)
        item = items_by_id[item_id]
        fold_id = item_to_fold[item_id]
        features = fold_features[fold_id]["features"]
        real_context = pubmedqa_context_text(item)
        flipped_context, flipped_changed, flip_info = flip_conclusion(real_context)
        filler_context, filler_source_ids = make_length_control(item_id, real_context, tokenizer, filler_pool, rng)
        shuffled_context = shuffle_context(real_context, rng)

        variants = {
            "real": with_context(item, real_context),
            "flipped": with_context(item, flipped_context),
            "length_control": with_context(item, filler_context),
            "shuffled": with_context(item, shuffled_context),
        }
        signals = {}
        for name, variant in variants.items():
            encoded = sae_encode_item(model, tokenizer, sae, variant, layer)
            signals[name] = selected_signal(encoded, features)

        case_row = {
            "item_id": item_id,
            "fold": fold_id,
            "ground_truth": row.get("ground_truth"),
            "prior_answer": row.get("prior_answer"),
            "context_answer": row.get("context_answer"),
            "behavior_label": "C" if behavior_label(row) else "M",
            "n_features": len(features),
            "real_signal": signals["real"],
            "flipped_signal": signals["flipped"],
            "length_control_signal": signals["length_control"],
            "shuffled_signal": signals["shuffled"],
            "abs_real_minus_flipped": abs(signals["real"] - signals["flipped"]),
            "flip_changed": flipped_changed,
            "flip_info": flip_info,
            "length_control_source": "other_pubmedqa_abstracts_same_token_length",
            "length_control_source_item_ids": filler_source_ids,
            "real_context_tokens": len(tokenizer.encode(real_context, add_special_tokens=False)),
            "flipped_context_tokens": len(tokenizer.encode(flipped_context, add_special_tokens=False)),
            "length_control_tokens": len(tokenizer.encode(filler_context, add_special_tokens=False)),
            "shuffled_context_tokens": len(tokenizer.encode(shuffled_context, add_special_tokens=False)),
        }
        if store_text:
            case_row.update(
                {
                    "question": item.get("question"),
                    "real_context_excerpt": text_excerpt(real_context, text_max_chars),
                    "flipped_context_excerpt": text_excerpt(flipped_context, text_max_chars),
                    "length_control_excerpt": text_excerpt(filler_context, text_max_chars),
                    "shuffled_context_excerpt": text_excerpt(shuffled_context, text_max_chars),
                }
            )
        control_rows.append(case_row)

    flipped_rows = [row for row in control_rows if row["flip_changed"]]
    return {
        "n_samples": len(control_rows),
        "n_flipped_changed": len(flipped_rows),
        "feature_source_note": "Each item uses the context-specific feature set from its held-out CV test fold.",
        "input_generation_note": {
            "content_flip": "Rule-based regex flip over the last sentence matching a known conclusion phrase; inspect flip_info and stored excerpts before interpreting.",
            "length_control": "Other PubMedQA abstracts truncated to the same token length. This controls length, not medical-domain/context-ness.",
            "shuffled_control": "Same words as the original context, randomly permuted.",
            "text_stored": store_text,
            "text_max_chars": text_max_chars if store_text else 0,
        },
        "summary": {
            "content_flip_changed_only": summarize_control(flipped_rows, "real_signal", "flipped_signal"),
            "length_control": summarize_control(control_rows, "real_signal", "length_control_signal"),
            "shuffled_control": summarize_control(control_rows, "real_signal", "shuffled_signal"),
            "abs_real_minus_flipped_changed_only": group_stats([row["abs_real_minus_flipped"] for row in flipped_rows]),
        },
        "cases": control_rows,
    }


def predict_with_feature_steering(model, tokenizer, sae, item, layer, features, alpha):
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    steer_vec = torch.zeros(sae.W_dec.shape[1], device=model.device)
    for feature in features:
        steer_vec += sae.W_dec[int(feature), :].to(model.device)

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[0, -1, :] = hidden[0, -1, :] + alpha * steer_vec.to(device=hidden.device, dtype=hidden.dtype)
        return output

    handle = model.model.layers[int(layer)].register_forward_hook(hook)
    try:
        probs = get_ynm_probs(model, tokenizer, prompt)
    finally:
        handle.remove()
    return probs, max(probs, key=probs.get)


def summarize_steering_rows(rows):
    if not rows:
        return {
            "n": 0,
            "baseline_context_follow_rate": 0.0,
            "amplify_context_follow_rate": 0.0,
            "suppress_context_follow_rate": 0.0,
            "baseline_memory_follow_rate": 0.0,
            "amplify_memory_follow_rate": 0.0,
            "suppress_memory_follow_rate": 0.0,
        }
    return {
        "n": len(rows),
        "baseline_context_follow_rate": float(np.mean([row["baseline_context_follow"] for row in rows])),
        "amplify_context_follow_rate": float(np.mean([row["amplify_context_follow"] for row in rows])),
        "suppress_context_follow_rate": float(np.mean([row["suppress_context_follow"] for row in rows])),
        "baseline_memory_follow_rate": float(np.mean([row["baseline_memory_follow"] for row in rows])),
        "amplify_memory_follow_rate": float(np.mean([row["amplify_memory_follow"] for row in rows])),
        "suppress_memory_follow_rate": float(np.mean([row["suppress_memory_follow"] for row in rows])),
    }


def run_steering_experiment(model, tokenizer, sae, rows, items_by_id, item_to_fold, fold_features, layer, alpha, max_samples, seed):
    rng = random.Random(seed)
    eligible = [
        row
        for row in rows
        if is_behavior_conflict(row)
        and row_id(row) in items_by_id
        and row_id(row) in item_to_fold
        and fold_features.get(item_to_fold[row_id(row)], {}).get("features")
    ]
    rng.shuffle(eligible)
    selected = eligible[:max_samples] if max_samples else eligible
    steering_rows = []
    for row in tqdm(selected, desc=f"L{layer} steering controls", leave=False):
        item_id = row_id(row)
        item = items_by_id[item_id]
        fold_id = item_to_fold[item_id]
        features = fold_features[fold_id]["features"]
        ground_truth = row["ground_truth"]
        prior_answer = row["prior_answer"]
        amp_probs, amp_pred = predict_with_feature_steering(model, tokenizer, sae, item, layer, features, alpha)
        sup_probs, sup_pred = predict_with_feature_steering(model, tokenizer, sae, item, layer, features, -alpha)
        base_pred = row["context_answer"]
        steering_rows.append(
            {
                "item_id": item_id,
                "fold": fold_id,
                "n_features": len(features),
                "ground_truth": ground_truth,
                "prior_answer": prior_answer,
                "baseline_prediction": base_pred,
                "amplify_prediction": amp_pred,
                "suppress_prediction": sup_pred,
                "amplify_probs": amp_probs,
                "suppress_probs": sup_probs,
                "baseline_context_follow": int(base_pred == ground_truth),
                "amplify_context_follow": int(amp_pred == ground_truth),
                "suppress_context_follow": int(sup_pred == ground_truth),
                "baseline_memory_follow": int(base_pred == prior_answer),
                "amplify_memory_follow": int(amp_pred == prior_answer),
                "suppress_memory_follow": int(sup_pred == prior_answer),
            }
        )
    return {
        "alpha": alpha,
        "n_samples": len(steering_rows),
        "summary": summarize_steering_rows(steering_rows),
        "cases": steering_rows,
    }


def feature_overlap(mi_features, fold_features):
    mi_set = {int(row["feature"]) for row in mi_features}
    cv_set = set()
    for fold in fold_features.values():
        cv_set.update(fold["features"])
    overlap = sorted(mi_set.intersection(cv_set))
    return {
        "mi_top_k": len(mi_set),
        "context_feature_union": len(cv_set),
        "overlap_count": len(overlap),
        "overlap_features": overlap,
    }


def main():
    parser = argparse.ArgumentParser(description="Feature justification experiments for context-specific SAE features.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--context-cv-path", default=None)
    parser.add_argument("--feature-source", default="cs_correct", choices=["cs_correct", "cs_wrong", "cs_both"])
    parser.add_argument("--max-mi-cases", type=int, default=None)
    parser.add_argument("--max-control-samples", type=int, default=100)
    parser.add_argument("--mi-top-k", type=int, default=50)
    parser.add_argument("--mi-binarize", default="positive", choices=["positive", "median", "q75"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-mi", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--run-steering", action="store_true")
    parser.add_argument("--steering-alpha", type=float, default=10.0)
    parser.add_argument("--max-steering-samples", type=int, default=100)
    parser.add_argument("--store-control-text", action="store_true")
    parser.add_argument("--control-text-max-chars", type=int, default=1200)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conflict_rows, conflict_path = load_first_existing(
        path_candidates(args.conflict_set, "eval", f"{args.model}_conflict_set.json"),
        "conflict set",
    )
    cv_data, cv_path = load_first_existing(
        path_candidates(args.context_cv_path, "steering_context_specific", f"{args.model}_context_specific_cv.json"),
        "context-specific CV results",
    )
    fold_features = build_fold_features(cv_data, args.layer, args.feature_source)
    item_to_fold = build_item_to_fold(cv_data)
    n_feature_folds = sum(1 for fold in fold_features.values() if fold["features"])
    if n_feature_folds == 0:
        raise ValueError(f"No context-specific features found for layer {args.layer} and source {args.feature_source}")

    if args.dry_run:
        matched_rows = [row for row in conflict_rows if row_id(row) in item_to_fold]
        conflict_rows_only = [row for row in matched_rows if is_behavior_conflict(row)]
        print(
            json.dumps(
                {
                    "conflict_path": conflict_path,
                    "context_cv_path": cv_path,
                    "feature_source": args.feature_source,
                    "n_folds_with_features": n_feature_folds,
                    "n_rows": len(conflict_rows),
                    "n_rows_in_cv_test_splits": len(matched_rows),
                    "n_behavior_conflict_rows": len(conflict_rows_only),
                    "common_context_features": most_common_features(fold_features, 20),
                },
                indent=2,
            )
        )
        return

    items = load_pubmedqa_items()
    items_by_id = item_map_by_id(items)
    model, tokenizer = load_model_and_tokenizer(args.model)
    sae = load_sae(args.model, args.layer, model.device)

    output = {
        "model": args.model,
        "layer": args.layer,
        "source_paths": {"conflict_set": conflict_path, "context_specific_cv": cv_path},
        "feature_source": args.feature_source,
        "feature_definition": {
            "cs_correct": "context_cv.fold.layer_info[layer].cs_correct",
            "cs_wrong": "context_cv.fold.layer_info[layer].cs_wrong",
            "cs_both": "union of cs_correct and cs_wrong",
        },
        "common_context_features": most_common_features(fold_features, 50),
    }

    if not args.skip_mi:
        mi_result = run_mi_experiment(
            model,
            tokenizer,
            sae,
            conflict_rows,
            items_by_id,
            args.layer,
            args.max_mi_cases,
            args.mi_top_k,
            args.mi_binarize,
        )
        mi_result["overlap_with_context_features"] = feature_overlap(mi_result["top_features"], fold_features)
        output["experiment_1_behavior_mi"] = mi_result

    if not args.skip_controls:
        output["experiments_2_3_4_controls"] = run_control_experiments(
            model,
            tokenizer,
            sae,
            conflict_rows,
            items_by_id,
            item_to_fold,
            fold_features,
            args.layer,
            args.max_control_samples,
            args.seed,
            args.store_control_text,
            args.control_text_max_chars,
        )

    if args.run_steering:
        output["experiment_5_steering"] = run_steering_experiment(
            model,
            tokenizer,
            sae,
            conflict_rows,
            items_by_id,
            item_to_fold,
            fold_features,
            args.layer,
            args.steering_alpha,
            args.max_steering_samples,
            args.seed,
        )

    output_path = args.output_path or f"{RESULTS_DIR}/feature_justification/{args.model}_feature_justification_L{args.layer}_{args.feature_source}.json"
    write_json(output_path, output)
    print(f"Saved feature justification results to {output_path}")
    compact = {
        "output_path": output_path,
        "mi": output.get("experiment_1_behavior_mi", {}).get("label_counts"),
        "controls": output.get("experiments_2_3_4_controls", {}).get("summary"),
    }
    print(json.dumps(compact, indent=2))

    del sae
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
