import argparse
import subprocess

from split_experiment_utils import BASE_DIR


SCRIPTS_DIR = f"{BASE_DIR}/scripts"


def run_cmd(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run split-safe experiments from master_experiment_plan.md.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--layers", nargs="+", default=["20", "18", "22"])
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-steering", action="store_true")
    parser.add_argument("--run-ceiling", action="store_true")
    parser.add_argument("--run-conflict", action="store_true")
    parser.add_argument("--run-medabstain", action="store_true")
    parser.add_argument("--rxllm-jsonl", default=None)
    args = parser.parse_args()

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
