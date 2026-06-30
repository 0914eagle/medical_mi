import argparse
import json
import os
import subprocess


BASE_DIR = os.environ.get("MEDICAL_MI_BASE_DIR", "/workspace/medical_mi")
RESULTS_DIR = f"{BASE_DIR}/results"
SCRIPTS_DIR = f"{BASE_DIR}/scripts"


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_cmd(cmd, skip_existing_path=None, force=False):
    if skip_existing_path and os.path.exists(skip_existing_path) and not force:
        print(f"\n>>> Skipping existing: {skip_existing_path}")
        return {"cmd": cmd, "status": "skipped_existing", "output_path": skip_existing_path}
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return {"cmd": cmd, "status": "completed", "output_path": skip_existing_path}


def existing_or_default(path, default_path):
    return path or default_path


def compact_json(path, keys):
    if not path or not os.path.exists(path):
        return None
    data = read_json(path)
    return {key: data.get(key) for key in keys}


def invalid_context_compare(path):
    if not os.path.exists(path):
        return False
    try:
        data = read_json(path)
    except Exception:
        return True
    counts = data.get("group_counts", {})
    return sum(int(value) for value in counts.values()) == 0


def invalid_resistance(path, context_compare_path):
    if not os.path.exists(path):
        return False
    if not os.path.exists(context_compare_path):
        return False
    try:
        data = read_json(path)
    except Exception:
        return True
    return data.get("group_source") != "context_compare"


def main():
    parser = argparse.ArgumentParser(
        description="Run the post-CV silent-override and feature-justification experiment suite."
    )
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--typeb-features-path", default=None)
    parser.add_argument("--context-cv-path", default=None)
    parser.add_argument("--feature-source", default="cs_correct", choices=["cs_correct", "cs_wrong", "cs_both"])
    parser.add_argument("--correct-dominant-source", default="all_typeb", choices=["all_typeb", "confident"])
    parser.add_argument("--noisy-threshold", type=float, default=0.6)
    parser.add_argument("--prior-threshold", type=float, default=0.7)
    parser.add_argument("--min-silent-cases", type=int, default=20)
    parser.add_argument("--max-cases-per-fold", type=int, default=None)
    parser.add_argument("--max-mi-cases", type=int, default=None)
    parser.add_argument("--max-control-samples", type=int, default=100)
    parser.add_argument("--mi-top-k", type=int, default=50)
    parser.add_argument("--mi-binarize", default="positive", choices=["positive", "median", "q75"])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--run-steering", action="store_true")
    parser.add_argument("--steering-alpha", type=float, default=10.0)
    parser.add_argument("--max-steering-samples", type=int, default=100)
    parser.add_argument("--store-control-text", action="store_true")
    parser.add_argument("--control-text-max-chars", type=int, default=1200)
    parser.add_argument("--skip-context-compare", action="store_true")
    parser.add_argument("--skip-resistance", action="store_true")
    parser.add_argument("--skip-feature-justification", action="store_true")
    parser.add_argument("--skip-mi", action="store_true")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute outputs even if default output files already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Validate commands and inputs without model/SAE-heavy execution.")
    parser.add_argument("--summary-path", default=None)
    args = parser.parse_args()

    conflict_set = existing_or_default(args.conflict_set, f"{BASE_DIR}/{args.model}_conflict_set.json")
    typeb_features_path = existing_or_default(
        args.typeb_features_path,
        f"{BASE_DIR}/{args.model}_typeb_sae_features.json",
    )
    context_cv_path = existing_or_default(
        args.context_cv_path,
        f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json",
    )
    context_compare_path = f"{RESULTS_DIR}/silent_override/{args.model}_silent_override_context_compare_L{args.layer}.json"
    resistance_path = f"{RESULTS_DIR}/silent_override/{args.model}_resistance_behavior_L{args.layer}.json"
    feature_justification_path = (
        f"{RESULTS_DIR}/feature_justification/{args.model}_feature_justification_L{args.layer}_{args.feature_source}.json"
    )
    summary_path = args.summary_path or f"{RESULTS_DIR}/silent_override/{args.model}_silent_override_suite_L{args.layer}.json"

    required_inputs = {"conflict_set": conflict_set}
    if not args.skip_context_compare:
        required_inputs["typeb_features"] = typeb_features_path
        required_inputs["context_specific_cv"] = context_cv_path
    if not args.skip_feature_justification:
        required_inputs["context_specific_cv"] = context_cv_path
    missing = {name: path for name, path in required_inputs.items() if not os.path.exists(path)}
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    steps = []
    if not args.skip_context_compare:
        cmd = [
            "python3",
            f"{SCRIPTS_DIR}/26_silent_override_context_compare.py",
            "--model",
            args.model,
            "--layer",
            str(args.layer),
            "--conflict-set",
            conflict_set,
            "--typeb-features-path",
            typeb_features_path,
            "--context-cv-path",
            context_cv_path,
            "--correct-dominant-source",
            args.correct_dominant_source,
            "--noisy-threshold",
            str(args.noisy_threshold),
            "--prior-threshold",
            str(args.prior_threshold),
            "--min-silent-cases",
            str(args.min_silent_cases),
            "--output-path",
            context_compare_path,
        ]
        if args.max_cases_per_fold:
            cmd += ["--max-cases-per-fold", str(args.max_cases_per_fold)]
        if args.dry_run:
            cmd += ["--dry-run"]
        recompute = args.force or invalid_context_compare(context_compare_path)
        steps.append(run_cmd(cmd, skip_existing_path=None if args.dry_run else context_compare_path, force=recompute))

    if not args.skip_resistance:
        cmd = [
            "python3",
            f"{SCRIPTS_DIR}/28_resistance_behavior_analysis.py",
            "--model",
            args.model,
            "--layer",
            str(args.layer),
            "--conflict-set",
            conflict_set,
            "--context-compare-path",
            context_compare_path,
            "--output-path",
            resistance_path,
        ]
        if args.dry_run:
            cmd += ["--dry-run"]
        recompute = args.force or invalid_resistance(resistance_path, context_compare_path)
        steps.append(run_cmd(cmd, skip_existing_path=None if args.dry_run else resistance_path, force=recompute))

    if not args.skip_feature_justification:
        cmd = [
            "python3",
            f"{SCRIPTS_DIR}/27_feature_justification_controls.py",
            "--model",
            args.model,
            "--layer",
            str(args.layer),
            "--conflict-set",
            conflict_set,
            "--context-cv-path",
            context_cv_path,
            "--feature-source",
            args.feature_source,
            "--max-control-samples",
            str(args.max_control_samples),
            "--mi-top-k",
            str(args.mi_top_k),
            "--mi-binarize",
            args.mi_binarize,
            "--seed",
            str(args.seed),
            "--output-path",
            feature_justification_path,
        ]
        if args.max_mi_cases:
            cmd += ["--max-mi-cases", str(args.max_mi_cases)]
        if args.skip_mi:
            cmd += ["--skip-mi"]
        if args.skip_controls:
            cmd += ["--skip-controls"]
        if args.run_steering:
            cmd += [
                "--run-steering",
                "--steering-alpha",
                str(args.steering_alpha),
                "--max-steering-samples",
                str(args.max_steering_samples),
            ]
        if args.store_control_text:
            cmd += ["--store-control-text", "--control-text-max-chars", str(args.control_text_max_chars)]
        if args.dry_run:
            cmd += ["--dry-run"]
        steps.append(run_cmd(cmd, skip_existing_path=None if args.dry_run else feature_justification_path, force=args.force))

    summary = {
        "model": args.model,
        "layer": args.layer,
        "inputs": required_inputs,
        "outputs": {
            "context_compare": context_compare_path,
            "resistance_behavior": resistance_path,
            "feature_justification": feature_justification_path,
        },
        "steps": steps,
        "context_compare_summary": compact_json(context_compare_path, ["group_counts", "thresholds", "table"]),
        "resistance_summary": compact_json(resistance_path, ["group_source", "group_counts", "behavior_by_group", "table"]),
        "feature_justification_summary": compact_json(
            feature_justification_path,
            ["feature_source", "common_context_features", "experiment_1_behavior_mi", "experiments_2_3_4_controls", "experiment_5_steering"],
        ),
    }
    if args.dry_run:
        print("\nDry-run suite summary:")
        print(json.dumps(summary, indent=2))
    else:
        write_json(summary_path, summary)
        print(f"\nSaved suite summary to {summary_path}")


if __name__ == "__main__":
    main()
