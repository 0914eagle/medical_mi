import argparse
import json

import torch

from split_experiment_utils import (
    RESULTS_DIR,
    build_steer_vec,
    conflict_score,
    get_activation_with_hook,
    load_model_and_tokenizer,
    load_sae,
    read_json,
    steering_hook,
    write_json,
)
from utils import get_ynm_probs


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_rxllm(item, tokenizer):
    context = item.get("context", item.get("patient_context", ""))
    question = item.get("question", item.get("clinical_question", ""))
    prompt = (
        f"Patient context: {context}\n\n"
        f"Clinical question: {question}\n\n"
        "Based ONLY on the patient context and clinical guideline, answer with one word: yes, no, or maybe."
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def feature_sum(model, tokenizer, sae, item, layer, features):
    if not features:
        return 0.0
    prompt = format_rxllm(item, tokenizer)
    act = get_activation_with_hook(model, tokenizer, prompt, layer)
    with torch.no_grad():
        encoded = sae.encode(act)[0]
    return float(encoded[[int(idx) for idx in features]].sum().item())


def predict(model, tokenizer, item, layer=None, steer_vec=None, alpha=0.0):
    prompt = format_rxllm(item, tokenizer)
    if steer_vec is not None and layer is not None:
        with steering_hook(model, {layer: steer_vec}, alpha):
            probs = get_ynm_probs(model, tokenizer, prompt)
    else:
        probs = get_ynm_probs(model, tokenizer, prompt)
    return probs, max(probs, key=probs.get)


def main():
    parser = argparse.ArgumentParser(description="Rx-LLM/clean clinical validation using PubMedQA-selected features.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--cv-path", default=None)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    cv_path = args.cv_path or f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json"
    cv_data = read_json(cv_path)
    fold = next(row for row in cv_data["folds"] if row["fold"] == args.fold)
    layer_info = fold["layer_info"][str(args.layer)]
    cs_wrong = layer_info["cs_wrong"]
    cs_correct = layer_info["cs_correct"]
    alpha = args.alpha
    if alpha is None:
        match = next((row for row in fold["steering_results"] if row["name"] == "context_specific_wrong"), None)
        alpha = match["selected_alpha"] if match else 10.0

    rows = load_jsonl(args.input_jsonl)
    model, tokenizer = load_model_and_tokenizer(args.model)
    sae = load_sae(args.model, args.layer, model.device)
    steer_vec = build_steer_vec(sae, suppress=cs_wrong, device=model.device)

    case_results = []
    for index, item in enumerate(rows):
        target = item.get("answer", item.get("ground_truth", item.get("label")))
        probs, pred = predict(model, tokenizer, item)
        steered_probs, steered_pred = predict(model, tokenizer, item, args.layer, steer_vec, alpha)
        correct_signal = feature_sum(model, tokenizer, sae, item, args.layer, cs_correct)
        wrong_signal = feature_sum(model, tokenizer, sae, item, args.layer, cs_wrong)
        case_results.append(
            {
                "index": index,
                "target": target,
                "prediction": pred,
                "steered_prediction": steered_pred,
                "is_correct": pred == target if target else None,
                "is_correct_steered": steered_pred == target if target else None,
                "conflict_score": conflict_score(correct_signal, wrong_signal),
                "correct_signal": correct_signal,
                "wrong_signal": wrong_signal,
                "probs": probs,
                "steered_probs": steered_probs,
            }
        )

    labeled = [row for row in case_results if row["is_correct"] is not None]
    summary = {
        "total": len(case_results),
        "labeled": len(labeled),
        "accuracy": sum(row["is_correct"] for row in labeled) / len(labeled) if labeled else None,
        "steered_accuracy": sum(row["is_correct_steered"] for row in labeled) / len(labeled) if labeled else None,
        "recoveries": sum((not row["is_correct"]) and row["is_correct_steered"] for row in labeled),
        "corruptions": sum(row["is_correct"] and (not row["is_correct_steered"]) for row in labeled),
    }
    output = {
        "model": args.model,
        "input_jsonl": args.input_jsonl,
        "cv_path": cv_path,
        "fold": args.fold,
        "layer": args.layer,
        "alpha": alpha,
        "cs_wrong": cs_wrong,
        "cs_correct": cs_correct,
        "summary": summary,
        "cases": case_results,
    }
    output_path = f"{RESULTS_DIR}/rxllm_validation/{args.model}_rxllm_L{args.layer}_fold{args.fold}.json"
    write_json(output_path, output)
    print(f"Saved Rx-LLM validation to {output_path}")


if __name__ == "__main__":
    main()
