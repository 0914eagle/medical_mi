import argparse
import json
import os

from split_experiment_utils import BASE_DIR, RESULTS_DIR, item_map_by_id, load_pubmedqa_items, read_json


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


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


def pubmedqa_context_text(item):
    context = item.get("context", "")
    if isinstance(context, dict):
        contexts = context.get("contexts", [])
        return " ".join(contexts) if isinstance(contexts, list) else str(contexts)
    return str(context)


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
    return {row_id(row): row.get("group") for row in data.get("cases", [])}, source


def main():
    parser = argparse.ArgumentParser(description="Export PubMedQA conflict cases for realistic_knowledge_conflicts-style baselines.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--context-compare-path", default=None)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--output-manifest", default=None)
    args = parser.parse_args()

    conflict_rows, conflict_path = load_first_existing(
        path_candidates(args.conflict_set, args.model, "eval", f"{args.model}_conflict_set.json"),
        "conflict set",
    )
    groups, group_path = load_context_groups(args.context_compare_path, args.model, args.layer)
    items_by_id = item_map_by_id(load_pubmedqa_items())
    output_jsonl = args.output_jsonl or f"{RESULTS_DIR}/baselines/{args.model}_pubmedqa_realistic_kc_L{args.layer}.jsonl"
    output_manifest = args.output_manifest or f"{RESULTS_DIR}/baselines/{args.model}_pubmedqa_realistic_kc_L{args.layer}_manifest.json"
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    counts = {}
    n_written = 0
    with open(output_jsonl, "w") as f:
        for row in conflict_rows:
            item_id = row_id(row)
            item = items_by_id.get(item_id)
            if not item:
                continue
            group = groups.get(item_id) or proxy_group(row)
            counts[group] = counts.get(group, 0) + 1
            record = {
                "id": item_id,
                "question": item.get("question"),
                "context": pubmedqa_context_text(item),
                "answer": row.get("ground_truth"),
                "gold_answer": row.get("ground_truth"),
                "parametric_answer": row.get("prior_answer"),
                "contextual_answer": row.get("context_answer"),
                "group": group,
                "prior_probs": row.get("prior_probs"),
                "context_probs": row.get("context_probs"),
                "source": "PubMedQA pqa_labeled + qwen conflict_set",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    manifest = {
        "model": args.model,
        "layer": args.layer,
        "source_paths": {"conflict_set": conflict_path, "context_compare": group_path},
        "output_jsonl": output_jsonl,
        "n_written": n_written,
        "group_counts": counts,
        "schema": {
            "question": "PubMedQA question",
            "context": "PubMedQA abstract/context",
            "answer/gold_answer": "ground-truth yes/no/maybe",
            "parametric_answer": "model answer without context",
            "contextual_answer": "model answer with context",
            "group": "correct/noisy_wrong/silent_wrong/other_wrong if context_compare is available, otherwise behavior proxy",
        },
        "realistic_kc_note": "Use this JSONL as the medical question/context/answer source when adapting realistic_knowledge_conflicts.",
    }
    write_json(output_manifest, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
