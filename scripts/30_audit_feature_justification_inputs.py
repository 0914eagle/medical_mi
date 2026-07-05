import argparse
import json
import os


BASE_DIR = os.environ.get("MEDICAL_MI_BASE_DIR", "/home/eagle0914/medical_mi")
RESULTS_DIR = f"{BASE_DIR}/results"


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def compact(text, max_chars):
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def default_feature_path(model, layer, feature_source):
    return f"{RESULTS_DIR}/feature_justification/{model}_feature_justification_L{layer}_{feature_source}.json"


def markdown_case(case, index, max_chars):
    lines = [
        f"## Case {index}: item_id={case.get('item_id')} fold={case.get('fold')}",
        "",
        f"- ground_truth: `{case.get('ground_truth')}`",
        f"- prior_answer: `{case.get('prior_answer')}`",
        f"- context_answer: `{case.get('context_answer')}`",
        f"- behavior_label: `{case.get('behavior_label')}`",
        f"- flip_changed: `{case.get('flip_changed')}`",
        f"- flip_info: `{json.dumps(case.get('flip_info'), ensure_ascii=False)}`",
        f"- real_signal: `{case.get('real_signal')}`",
        f"- flipped_signal: `{case.get('flipped_signal')}`",
        f"- length_control_signal: `{case.get('length_control_signal')}`",
        f"- shuffled_signal: `{case.get('shuffled_signal')}`",
        f"- length_control_source: `{case.get('length_control_source')}`",
        f"- length_control_source_item_ids: `{case.get('length_control_source_item_ids')}`",
        "",
        "### Question",
        "",
        compact(case.get("question"), max_chars),
        "",
        "### Real Context",
        "",
        compact(case.get("real_context_excerpt"), max_chars),
        "",
        "### Flipped Context",
        "",
        compact(case.get("flipped_context_excerpt"), max_chars),
        "",
        "### Length Control",
        "",
        compact(case.get("length_control_excerpt"), max_chars),
        "",
        "### Shuffled Context",
        "",
        compact(case.get("shuffled_context_excerpt"), max_chars),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract human-readable examples for feature-justification input audit.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--feature-source", default="cs_correct")
    parser.add_argument("--feature-justification-path", default=None)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_path = args.feature_justification_path or default_feature_path(args.model, args.layer, args.feature_source)
    data = read_json(input_path)
    controls = data.get("experiments_2_3_4_controls", {})
    cases = controls.get("cases", [])
    if not cases:
        raise ValueError(f"No control cases found in {input_path}")

    text_fields = ["real_context_excerpt", "flipped_context_excerpt", "length_control_excerpt", "shuffled_context_excerpt"]
    has_text = any(any(case.get(field) for field in text_fields) for case in cases)
    if not has_text:
        raise ValueError(
            "This result file does not include stored control text. Re-run 27/29 with --store-control-text --force."
        )

    flipped_cases = [case for case in cases if case.get("flip_changed")]
    length_cases = cases[: args.n]
    selected = flipped_cases[: args.n]
    output_dir = args.output_dir or f"{RESULTS_DIR}/feature_justification/audit"
    ensure_dir(output_dir)

    jsonl_path = f"{output_dir}/{args.model}_L{args.layer}_{args.feature_source}_audit_examples.jsonl"
    with open(jsonl_path, "w") as f:
        for case in selected:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    md_path = f"{output_dir}/{args.model}_L{args.layer}_{args.feature_source}_audit_examples.md"
    with open(md_path, "w") as f:
        f.write(f"# Feature Justification Input Audit\n\n")
        f.write(f"- source: `{input_path}`\n")
        f.write(f"- n_control_cases: `{len(cases)}`\n")
        f.write(f"- n_flip_changed: `{len(flipped_cases)}`\n")
        f.write(f"- length_control_note: `{controls.get('input_generation_note', {}).get('length_control')}`\n\n")
        f.write("".join(markdown_case(case, idx + 1, args.max_chars) for idx, case in enumerate(selected)))

    summary = {
        "source": input_path,
        "n_control_cases": len(cases),
        "n_flip_changed": len(flipped_cases),
        "jsonl_path": jsonl_path,
        "markdown_path": md_path,
        "length_control_sources_first_n": [
            {
                "item_id": case.get("item_id"),
                "source": case.get("length_control_source"),
                "source_item_ids": case.get("length_control_source_item_ids"),
                "real_tokens": case.get("real_context_tokens"),
                "length_control_tokens": case.get("length_control_tokens"),
            }
            for case in length_cases
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
