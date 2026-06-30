import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

try:
    from scipy.stats import mannwhitneyu, ttest_ind
except ImportError:
    mannwhitneyu = None
    ttest_ind = None


ANSWER_KEYS = ["yes", "no", "maybe"]
BASE_DIR = os.environ.get("MEDICAL_MI_BASE_DIR", "/workspace/medical_mi")
RESULTS_DIR = f"{BASE_DIR}/results"


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


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


def prior_metrics(row):
    prior_probs = row["prior_probs"]
    context_probs = row["context_probs"]
    prior_answer = row["prior_answer"]
    prior_vector = np.array([float(prior_probs[key]) for key in ANSWER_KEYS], dtype=np.float64)
    prior_held = float(context_probs.get(prior_answer, 0.0))
    prior_orig = float(prior_probs.get(prior_answer, 0.0))
    sorted_prior = np.sort(prior_vector)[::-1]
    return {
        "resistance": float(prior_held / (prior_orig + 1e-10)),
        "prior_held": prior_held,
        "prior_orig": prior_orig,
        "max_prob": float(prior_vector.max()),
        "entropy": float(-np.sum(prior_vector * np.log(prior_vector + 1e-10))),
        "margin": float(sorted_prior[0] - sorted_prior[1]) if len(sorted_prior) > 1 else 0.0,
    }


def behavior_label(row):
    return "C" if row["prior_answer"] != row["context_answer"] else "M"


def proxy_group(row):
    if row["context_answer"] == row["ground_truth"]:
        return "correct"
    if row["prior_answer"] == row["context_answer"]:
        return "silent_wrong"
    return "noisy_wrong"


def group_stats(values):
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
    }


def compare_groups(a, b):
    if len(a) < 2 or len(b) < 2 or mannwhitneyu is None or ttest_ind is None:
        return {"ttest_ind_welch": None, "mannwhitneyu": None}
    t_stat, t_p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
    return {
        "ttest_ind_welch": {"statistic": float(t_stat), "p_value": float(t_p)},
        "mannwhitneyu": {"statistic": float(u_stat), "p_value": float(u_p)},
    }


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
        groups[row_id(row)] = {
            "group": row.get("group"),
            "computed_conflict_score": row.get("computed_conflict_score"),
            "context_specific_activation": row.get("context_specific_activation"),
            "correct_dominant_activation": row.get("correct_dominant_activation"),
        }
    return groups, source


def main():
    parser = argparse.ArgumentParser(description="Data-only resistance and behavior-label analysis for silent override.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--context-compare-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conflict_rows, conflict_path = load_first_existing(
        path_candidates(args.conflict_set, args.model, "eval", f"{args.model}_conflict_set.json"),
        "conflict set",
    )
    context_groups, context_group_path = load_context_groups(args.context_compare_path, args.model, args.layer)
    group_source = "context_compare" if context_groups else "behavior_proxy"

    if args.dry_run:
        print(
            json.dumps(
                {
                    "conflict_path": conflict_path,
                    "context_compare_path": context_group_path,
                    "group_source_if_run": group_source,
                    "n_conflict_rows": len(conflict_rows),
                    "n_context_group_rows": len(context_groups),
                },
                indent=2,
            )
        )
        return

    rows = []
    for row in conflict_rows:
        item_id = row_id(row)
        metrics = prior_metrics(row)
        context_meta = context_groups.get(item_id, {})
        enriched = dict(row)
        enriched.update(metrics)
        enriched.update(
            {
                "item_id": item_id,
                "behavior_label": behavior_label(row),
                "group": context_meta.get("group") or proxy_group(row),
                "group_source": group_source,
                "computed_conflict_score": context_meta.get("computed_conflict_score"),
                "context_specific_activation": context_meta.get("context_specific_activation"),
                "correct_dominant_activation": context_meta.get("correct_dominant_activation"),
            }
        )
        rows.append(enriched)

    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)

    metrics = ["resistance", "prior_held", "prior_orig", "max_prob", "entropy", "margin"]
    summary = {}
    for metric in metrics:
        summary[metric] = {
            group: group_stats([row[metric] for row in group_rows])
            for group, group_rows in sorted(by_group.items())
        }
        summary[metric]["comparisons"] = {
            "silent_wrong_vs_noisy_wrong": compare_groups(
                [row[metric] for row in by_group["silent_wrong"]],
                [row[metric] for row in by_group["noisy_wrong"]],
            ),
            "silent_wrong_vs_correct": compare_groups(
                [row[metric] for row in by_group["silent_wrong"]],
                [row[metric] for row in by_group["correct"]],
            ),
            "noisy_wrong_vs_correct": compare_groups(
                [row[metric] for row in by_group["noisy_wrong"]],
                [row[metric] for row in by_group["correct"]],
            ),
        }

    cross_tab = Counter((row["group"], row["behavior_label"]) for row in rows)
    behavior_by_group = {}
    for group, group_rows in sorted(by_group.items()):
        n_group = len(group_rows)
        behavior_by_group[group] = {
            "n": n_group,
            "context_follow_C": sum(1 for row in group_rows if row["behavior_label"] == "C"),
            "memory_follow_M": sum(1 for row in group_rows if row["behavior_label"] == "M"),
            "memory_follow_rate": (
                sum(1 for row in group_rows if row["behavior_label"] == "M") / n_group if n_group else 0.0
            ),
        }

    table = []
    for group in ["correct", "noisy_wrong", "silent_wrong", "other_wrong"]:
        if group not in by_group:
            continue
        table.append(
            {
                "group": group,
                "n": len(by_group[group]),
                "behavior_M_rate": behavior_by_group[group]["memory_follow_rate"],
                "resistance_mean": summary["resistance"][group]["mean"],
                "prior_held_mean": summary["prior_held"][group]["mean"],
                "max_prob_mean": summary["max_prob"][group]["mean"],
                "entropy_mean": summary["entropy"][group]["mean"],
                "margin_mean": summary["margin"][group]["mean"],
            }
        )

    output = {
        "model": args.model,
        "layer": args.layer,
        "source_paths": {
            "conflict_set": conflict_path,
            "context_compare": context_group_path,
        },
        "group_source": group_source,
        "group_counts": {group: len(group_rows) for group, group_rows in sorted(by_group.items())},
        "behavior_cross_tab": {f"{group}|{label}": count for (group, label), count in sorted(cross_tab.items())},
        "behavior_by_group": behavior_by_group,
        "summary": summary,
        "table": table,
        "cases": rows,
    }

    output_path = args.output_path or f"{RESULTS_DIR}/silent_override/{args.model}_resistance_behavior_L{args.layer}.json"
    write_json(output_path, output)
    print(f"Saved resistance behavior analysis to {output_path}")
    print(json.dumps({"group_source": group_source, "group_counts": output["group_counts"], "table": table}, indent=2))


if __name__ == "__main__":
    main()
