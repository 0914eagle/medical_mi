import argparse
import json
import os
import subprocess


BASE_DIR = os.environ.get("MEDICAL_MI_BASE_DIR", "/home/eagle0914/medical_mi")
DATA_ROOT = os.environ.get("MEDICAL_MI_DATA_ROOT", "/data/heejae")
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
    env = os.environ.copy()
    env.setdefault("MEDICAL_MI_BASE_DIR", BASE_DIR)
    env.setdefault("MEDICAL_MI_DATA_ROOT", DATA_ROOT)
    env.setdefault("HF_HOME", f"{DATA_ROOT}/.cache/huggingface")
    env.setdefault("TMPDIR", f"{DATA_ROOT}/tmp")
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)
    return {"cmd": cmd, "status": "completed", "output_path": skip_existing_path}


def compact(path, keys):
    if not path or not os.path.exists(path):
        return None
    data = read_json(path)
    return {key: data.get(key) for key in keys}


def main():
    parser = argparse.ArgumentParser(description="Run baseline methods for silent-override blind-spot testing.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--context-compare-path", default=None)
    parser.add_argument("--context-cv-path", default=None)
    parser.add_argument("--run-setup", action="store_true", help="Download models/SAEs/datasets into MEDICAL_MI_DATA_ROOT.")
    parser.add_argument("--skip-corect", action="store_true")
    parser.add_argument("--skip-spare", action="store_true")
    parser.add_argument("--skip-realistic-kc-export", action="store_true")
    parser.add_argument("--corect-layers", type=int, nargs="+", default=[4, 8, 12, 16, 18, 20, 22, 24, 28, 32])
    parser.add_argument("--max-cases-per-group", type=int, default=50)
    parser.add_argument("--feature-source", default="cs_correct", choices=["cs_correct", "cs_wrong", "cs_both"])
    parser.add_argument("--max-mi-cases", type=int, default=None)
    parser.add_argument("--max-control-samples", type=int, default=100)
    parser.add_argument("--run-spare-steering", action="store_true")
    parser.add_argument("--max-steering-samples", type=int, default=100)
    parser.add_argument("--steering-alpha", type=float, default=10.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-path", default=None)
    args = parser.parse_args()

    os.makedirs(f"{DATA_ROOT}/checkpoints/model", exist_ok=True)
    os.makedirs(f"{DATA_ROOT}/checkpoints/sae", exist_ok=True)
    os.makedirs(f"{DATA_ROOT}/.cache/huggingface", exist_ok=True)
    os.makedirs(f"{DATA_ROOT}/tmp", exist_ok=True)

    conflict_set = args.conflict_set or f"{BASE_DIR}/{args.model}_conflict_set.json"
    context_compare_path = args.context_compare_path or f"{RESULTS_DIR}/silent_override/{args.model}_silent_override_context_compare_L{args.layer}.json"
    context_cv_path = args.context_cv_path or f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json"
    corect_path = f"{RESULTS_DIR}/baselines/{args.model}_corect_detection_L{args.layer}.json"
    spare_path = f"{RESULTS_DIR}/feature_justification/{args.model}_feature_justification_L{args.layer}_{args.feature_source}.json"
    realistic_path = f"{RESULTS_DIR}/baselines/{args.model}_pubmedqa_realistic_kc_L{args.layer}.jsonl"
    realistic_manifest = f"{RESULTS_DIR}/baselines/{args.model}_pubmedqa_realistic_kc_L{args.layer}_manifest.json"
    summary_path = args.summary_path or f"{RESULTS_DIR}/baselines/{args.model}_baseline_suite_L{args.layer}.json"

    missing = []
    for path in [conflict_set]:
        if not os.path.exists(path):
            missing.append(path)
    if not args.skip_spare and not os.path.exists(context_cv_path):
        missing.append(context_cv_path)
    if missing:
        raise FileNotFoundError(f"Missing required input files: {missing}")

    steps = []
    if args.run_setup:
        steps.append(run_cmd(["python3", f"{SCRIPTS_DIR}/01_setup_full.py"], force=True))

    if not args.skip_corect:
        cmd = [
            "python3",
            f"{SCRIPTS_DIR}/31_corect_baseline_detection.py",
            "--model",
            args.model,
            "--layer",
            str(args.layer),
            "--layers",
            *[str(layer) for layer in args.corect_layers],
            "--conflict-set",
            conflict_set,
            "--context-compare-path",
            context_compare_path,
            "--max-cases-per-group",
            str(args.max_cases_per_group),
            "--output-path",
            corect_path,
        ]
        steps.append(run_cmd(cmd, skip_existing_path=corect_path, force=args.force))

    if not args.skip_spare:
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
            "--store-control-text",
            "--output-path",
            spare_path,
        ]
        if args.max_mi_cases:
            cmd += ["--max-mi-cases", str(args.max_mi_cases)]
        if args.run_spare_steering:
            cmd += [
                "--run-steering",
                "--max-steering-samples",
                str(args.max_steering_samples),
                "--steering-alpha",
                str(args.steering_alpha),
            ]
        steps.append(run_cmd(cmd, skip_existing_path=spare_path, force=args.force))

    if not args.skip_realistic_kc_export:
        cmd = [
            "python3",
            f"{SCRIPTS_DIR}/32_export_realistic_kc_pubmedqa.py",
            "--model",
            args.model,
            "--layer",
            str(args.layer),
            "--conflict-set",
            conflict_set,
            "--context-compare-path",
            context_compare_path,
            "--output-jsonl",
            realistic_path,
            "--output-manifest",
            realistic_manifest,
        ]
        steps.append(run_cmd(cmd, skip_existing_path=realistic_path, force=args.force))

    summary = {
        "model": args.model,
        "layer": args.layer,
        "base_dir": BASE_DIR,
        "data_root": DATA_ROOT,
        "inputs": {
            "conflict_set": conflict_set,
            "context_compare": context_compare_path,
            "context_cv": context_cv_path,
        },
        "outputs": {
            "corect": corect_path,
            "spare": spare_path,
            "realistic_kc_jsonl": realistic_path,
            "realistic_kc_manifest": realistic_manifest,
        },
        "steps": steps,
        "corect_summary": compact(corect_path, ["summary", "method_note"]),
        "spare_summary": compact(spare_path, ["feature_source", "experiment_1_behavior_mi", "experiments_2_3_4_controls", "experiment_5_steering"]),
        "realistic_kc_summary": compact(realistic_manifest, ["n_written", "group_counts", "schema", "realistic_kc_note"]),
    }
    write_json(summary_path, summary)
    print(f"\nSaved baseline suite summary to {summary_path}")


if __name__ == "__main__":
    main()
