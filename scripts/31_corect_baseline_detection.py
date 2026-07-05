import argparse
import gc
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from split_experiment_utils import (
    BASE_DIR,
    RESULTS_DIR,
    item_map_by_id,
    load_model_and_tokenizer,
    load_pubmedqa_items,
    read_json,
    write_json,
)
from utils import format_pubmedqa


ANSWER_KEYS = ["yes", "no", "maybe"]


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


def row_id(row):
    return str(row.get("item_id", row.get("pubid", "")))


def proxy_group(row):
    if row["context_answer"] == row["ground_truth"]:
        return "correct"
    if row["prior_answer"] == row["context_answer"]:
        return "silent_wrong"
    return "noisy_wrong"


def load_context_groups(path, model, layer):
    data, source = load_first_existing(
        path_candidates(path, model, "silent_override", f"{model}_silent_override_context_compare_L{layer}.json"),
        "silent-override context comparison",
        required=False,
    )
    if not data:
        return {}, None
    groups = {}
    for row in data.get("cases", []):
        groups[row_id(row)] = row.get("group")
    return groups, source


def answer_token_ids(tokenizer, answer):
    ids = []
    for variant in [answer, answer.capitalize(), " " + answer, " " + answer.capitalize()]:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded:
            ids.append(int(encoded[0]))
    return sorted(set(ids))


def layer_lens_probs(model, tokenizer, prompt, layer_indices, target_token_ids):
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**tokenized, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    max_layer_index = len(hidden_states) - 2
    norm = getattr(model.model, "norm", None)
    rows = {}
    for layer in layer_indices:
        if layer < 0 or layer > max_layer_index:
            continue
        hidden = hidden_states[layer + 1][:, -1, :]
        if norm is not None:
            hidden = norm(hidden)
        logits = model.lm_head(hidden)[0]
        probs = torch.softmax(logits.float(), dim=-1)
        rows[str(layer)] = {
            name: float(torch.max(probs[torch.tensor(token_ids, device=probs.device)]).item()) if token_ids else 0.0
            for name, token_ids in target_token_ids.items()
        }
    return rows


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


def group_stats(values):
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
    }


def summarize_cases(cases):
    by_group = defaultdict(list)
    for row in cases:
        by_group[row["group"]].append(row)
    metrics = [
        "corect_suppression_drop",
        "ctx_final_gold_minus_prior",
        "noctx_final_prior_minus_gold",
        "ctx_max_gold_minus_prior",
    ]
    summary = {}
    for metric in metrics:
        summary[metric] = {
            group: group_stats([row[metric] for row in rows])
            for group, rows in sorted(by_group.items())
        }
    comparisons = {}
    for positive_group in ["silent_wrong", "noisy_wrong", "other_wrong"]:
        pos = by_group.get(positive_group, [])
        neg = by_group.get("correct", [])
        if pos and neg:
            rows = pos + neg
            comparisons[f"{positive_group}_vs_correct_auc"] = {
                metric: auc_score([row[metric] for row in rows], [int(row["group"] == positive_group) for row in rows])
                for metric in metrics
            }
    return {
        "group_counts": {group: len(rows) for group, rows in sorted(by_group.items())},
        "metrics_by_group": summary,
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal CoRect-style logit-lens detection baseline.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20, help="Layer used only for default context-compare path.")
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 18, 20, 22, 24, 28, 32])
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--context-compare-path", default=None)
    parser.add_argument("--max-cases-per-group", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    conflict_rows, conflict_path = load_first_existing(
        path_candidates(args.conflict_set, args.model, "eval", f"{args.model}_conflict_set.json"),
        "conflict set",
    )
    context_groups, context_group_path = load_context_groups(args.context_compare_path, args.model, args.layer)
    items_by_id = item_map_by_id(load_pubmedqa_items())
    rng = random.Random(args.seed)
    grouped = defaultdict(list)
    for row in conflict_rows:
        item_id = row_id(row)
        if item_id not in items_by_id:
            continue
        group = context_groups.get(item_id) or proxy_group(row)
        enriched = dict(row)
        enriched["item_id"] = item_id
        enriched["group"] = group
        grouped[group].append(enriched)
    selected_rows = []
    for group, rows in grouped.items():
        rng.shuffle(rows)
        selected_rows.extend(rows[: args.max_cases_per_group] if args.max_cases_per_group else rows)

    model, tokenizer = load_model_and_tokenizer(args.model)
    cases = []
    for row in tqdm(selected_rows, desc="CoRect logit lens cases"):
        item = items_by_id[row["item_id"]]
        gold = row["ground_truth"]
        prior = row["prior_answer"]
        target_token_ids = {
            "gold": answer_token_ids(tokenizer, gold),
            "prior": answer_token_ids(tokenizer, prior),
        }
        ctx_prompt = format_pubmedqa(item, tokenizer, include_context=True)
        noctx_prompt = format_pubmedqa(item, tokenizer, include_context=False)
        ctx = layer_lens_probs(model, tokenizer, ctx_prompt, args.layers, target_token_ids)
        noctx = layer_lens_probs(model, tokenizer, noctx_prompt, args.layers, target_token_ids)
        valid_layers = [layer for layer in args.layers if str(layer) in ctx]
        if not valid_layers:
            continue
        layer_margins = {str(layer): ctx[str(layer)]["gold"] - ctx[str(layer)]["prior"] for layer in valid_layers}
        final_layer = str(valid_layers[-1])
        ctx_final_margin = layer_margins[final_layer]
        ctx_max_margin = max(layer_margins.values())
        noctx_final_prior_minus_gold = noctx[final_layer]["prior"] - noctx[final_layer]["gold"]
        cases.append(
            {
                "item_id": row["item_id"],
                "group": row["group"],
                "ground_truth": gold,
                "prior_answer": prior,
                "context_answer": row["context_answer"],
                "layers": valid_layers,
                "ctx_layer_probs": ctx,
                "noctx_layer_probs": noctx,
                "ctx_layer_gold_minus_prior": layer_margins,
                "ctx_max_gold_minus_prior": float(ctx_max_margin),
                "ctx_final_gold_minus_prior": float(ctx_final_margin),
                "noctx_final_prior_minus_gold": float(noctx_final_prior_minus_gold),
                "corect_suppression_drop": float(max(0.0, ctx_max_margin - ctx_final_margin)),
            }
        )

    output = {
        "model": args.model,
        "source_paths": {
            "conflict_set": conflict_path,
            "context_compare": context_group_path,
        },
        "layers": args.layers,
        "max_cases_per_group": args.max_cases_per_group,
        "method_note": "Minimal CoRect-style detection: logit lens over contextualized/non-contextualized prompts; no rectification.",
        "summary": summarize_cases(cases),
        "cases": cases,
    }
    output_path = args.output_path or f"{RESULTS_DIR}/baselines/{args.model}_corect_detection_L{args.layer}.json"
    write_json(output_path, output)
    print(f"Saved CoRect baseline detection to {output_path}")
    print(json.dumps(output["summary"], indent=2))
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
