import argparse
import os
import shutil
import subprocess
from datetime import datetime

from split_experiment_utils import BASE_DIR


SCRIPTS_DIR = f"{BASE_DIR}/scripts"
RESULTS_DIR = f"{BASE_DIR}/results"
BACKUP_TARGETS = [
    "eval_split",
    "steering_context_specific",
    "ceiling_analysis",
    "conflict_score",
    "medabstain_transfer",
    "medabstain_diagnostics",
    "cai_policy",
]


def run_cmd(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def backup_existing_results(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = f"{RESULTS_DIR}/backups/{timestamp}_{model}"
    copied = []
    os.makedirs(backup_root, exist_ok=True)

    for target in BACKUP_TARGETS:
        src_dir = f"{RESULTS_DIR}/{target}"
        if not os.path.isdir(src_dir):
            continue
        matching_files = [
            filename
            for filename in os.listdir(src_dir)
            if filename.startswith(model) or model in filename
        ]
        if not matching_files:
            continue
        dst_dir = f"{backup_root}/{target}"
        os.makedirs(dst_dir, exist_ok=True)
        for filename in matching_files:
            src = f"{src_dir}/{filename}"
            dst = f"{dst_dir}/{filename}"
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied.append(src)

    manifest_path = f"{backup_root}/manifest.txt"
    with open(manifest_path, "w") as f:
        for path in copied:
            f.write(f"{path}\n")

    if copied:
        print(f"\nBacked up {len(copied)} result files to {backup_root}")
    else:
        print(f"\nNo existing result files matched model={model}; created empty backup dir at {backup_root}")
    return backup_root


def main():
    parser = argparse.ArgumentParser(description="Run split-safe experiments from master_experiment_plan.md.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--layers", nargs="+", default=["20", "18", "22"])
    parser.add_argument("--backup-existing", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-steering", action="store_true")
    parser.add_argument("--run-ceiling", action="store_true")
    parser.add_argument("--run-conflict", action="store_true")
    parser.add_argument("--run-medabstain", action="store_true")
    parser.add_argument("--run-medabstain-diagnostics", action="store_true")
    parser.add_argument("--run-cai", action="store_true")
    parser.add_argument("--rxllm-jsonl", default=None)
    args = parser.parse_args()

    if args.backup_existing:
        backup_existing_results(args.model)

    if not args.skip_eval:
        run_cmd(["python3", f"{SCRIPTS_DIR}/11_pubmedqa_split_eval.py", "--model", args.model, "--folds", str(args.folds)])

    if not args.skip_steering:
        run_cmd(
            [
                "python3",
                f"{SCRIPTS_DIR}/12_context_specific_cv.py",
                "--model",
                args.model,
                "--folds",
                str(args.folds),
                "--layers",
                *args.layers,
            ]
        )

    if args.run_ceiling:
        run_cmd(["python3", f"{SCRIPTS_DIR}/13_ceiling_analysis.py", "--model", args.model])

    if args.run_conflict:
        run_cmd(["python3", f"{SCRIPTS_DIR}/14_conflict_score_cv.py", "--model", args.model])

    if args.run_medabstain:
        run_cmd(["python3", f"{SCRIPTS_DIR}/15_medabstain_behavior_transfer.py", "--model", args.model])

    if args.run_medabstain_diagnostics:
        run_cmd(["python3", f"{SCRIPTS_DIR}/17_medabstain_feature_diagnostics.py", "--model", args.model])

    if args.run_cai:
        run_cmd(["python3", f"{SCRIPTS_DIR}/18_cai_policy_eval.py", "--model", args.model])

    if args.rxllm_jsonl:
        run_cmd(
            [
                "python3",
                f"{SCRIPTS_DIR}/16_rxllm_clinical_validation.py",
                "--model",
                args.model,
                "--input-jsonl",
                args.rxllm_jsonl,
            ]
        )


if __name__ == "__main__":
    main()
