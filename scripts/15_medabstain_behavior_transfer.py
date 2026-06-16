import argparse
import glob
import os

import torch

from split_experiment_utils import (
    BASE_DIR,
    RESULTS_DIR,
    build_steer_vec,
    feature_signal,
    get_activation_with_hook,
    load_model_and_tokenizer,
    load_sae,
    read_json,
    steering_hook,
    write_json,
)


def medabstain_dir():
    candidates = [
        f"{BASE_DIR}/data/raw/MedAbstain/data",
        f"{BASE_DIR}/data/raw/MedAbstain/MedAbstain/data",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_medabstain_items(path=None):
    if path is None:
        data_dir = medabstain_dir()
        if not data_dir:
            raise FileNotFoundError("MedAbstain data directory not found")
        matches = glob.glob(f"{data_dir}/perturbed_medqa*_test_noabst.json")
        if not matches:
            raise FileNotFoundError(f"No MedAbstain perturbed file found in {data_dir}")
        path = matches[0]
    return read_json(path), path


def format_medabstain(item, tokenizer):
    choices = item.get("choices", {})
    choice_str = "\n".join([f"{key}: {value}" for key, value in choices.items()])
    prompt = f"Question: {item['question']}\n\nChoices:\n{choice_str}\n\nAnswer with one letter."
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def letter_probs(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    result = {}
    for letter in ["A", "B", "C", "D", "E"]:
        ids = []
        for variant in [letter, " " + letter]:
            ids.extend(tokenizer.encode(variant, add_special_tokens=False))
        valid = [idx for idx in ids if idx < probs.shape[0]]
        result[letter] = max([probs[idx].item() for idx in valid], default=0.0)
    total = sum(result.values())
    return {key: value / total for key, value in result.items()} if total else result


def letter_probs_with_steering(model, tokenizer, prompt, layer, steer_vec, alpha):
    with steering_hook(model, {layer: steer_vec}, alpha):
        return letter_probs(model, tokenizer, prompt)


def medabstain_feature_signal(model, tokenizer, sae, item, layer, features):
    if not features:
        return 0.0
    prompt = format_medabstain(item, tokenizer)
    act = get_activation_with_hook(model, tokenizer, prompt, layer)
    with torch.no_grad():
        encoded = sae.encode(act)[0]
    return float(encoded[[int(idx) for idx in features]].sum().item())


def main():
    parser = argparse.ArgumentParser(description="MedAbstain behavior-based transfer validation.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--cv-path", default=None)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    cv_path = args.cv_path or f"{RESULTS_DIR}/steering_context_specific/{args.model}_context_specific_cv.json"
    cv_data = read_json(cv_path)
    fold = next(row for row in cv_data["folds"] if row["fold"] == args.fold)
    layer_info = fold["layer_info"][str(args.layer)]
    cs_wrong = layer_info["cs_wrong"]

    items, source_path = load_medabstain_items(args.data_path)
    items = items[: args.max_cases]
    model, tokenizer = load_model_and_tokenizer(args.model)
    sae = load_sae(args.model, args.layer, model.device)
    steer_vec = build_steer_vec(sae, suppress=cs_wrong, device=model.device)

    alpha = args.alpha
    if alpha is None:
        match = next((row for row in fold["steering_results"] if row["name"] == "context_specific_wrong"), None)
        alpha = match["selected_alpha"] if match else 10.0

    rows = []
    for index, item in enumerate(items):
        prompt = format_medabstain(item, tokenizer)
        probs = letter_probs(model, tokenizer, prompt)
        pred = max(probs, key=probs.get)
        steered_probs = letter_probs_with_steering(model, tokenizer, prompt, args.layer, steer_vec, alpha)
        steered_pred = max(steered_probs, key=steered_probs.get)
        answer = item.get("answer_idx", item.get("answer", item.get("label")))
        abstain_letter = item.get("abstain_idx", "E")
        behavior = "correct_abstain" if pred == abstain_letter else "wrong_answered"
        steered_behavior = "correct_abstain" if steered_pred == abstain_letter else "wrong_answered"
        rows.append(
            {
                "index": index,
                "prediction": pred,
                "steered_prediction": steered_pred,
                "answer": answer,
                "abstain_letter": abstain_letter,
                "behavior": behavior,
                "steered_behavior": steered_behavior,
                "wrong_feature_signal": feature_signal(model, tokenizer, sae, item, args.layer, cs_wrong)
                if "final_decision" in item
                else medabstain_feature_signal(model, tokenizer, sae, item, args.layer, cs_wrong),
                "probs": probs,
                "steered_probs": steered_probs,
            }
        )

    output = {
        "model": args.model,
        "source_path": source_path,
        "cv_path": cv_path,
        "fold": args.fold,
        "layer": args.layer,
        "alpha": alpha,
        "cs_wrong": cs_wrong,
        "summary": {
            "total": len(rows),
            "correct_abstain": sum(1 for row in rows if row["behavior"] == "correct_abstain"),
            "wrong_answered": sum(1 for row in rows if row["behavior"] == "wrong_answered"),
            "steered_correct_abstain": sum(1 for row in rows if row["steered_behavior"] == "correct_abstain"),
            "wrong_to_abstain": sum(
                1
                for row in rows
                if row["behavior"] == "wrong_answered" and row["steered_behavior"] == "correct_abstain"
            ),
        },
        "cases": rows,
    }
    output_path = f"{RESULTS_DIR}/medabstain_transfer/{args.model}_medabstain_transfer_L{args.layer}_fold{args.fold}.json"
    write_json(output_path, output)
    print(f"Saved MedAbstain transfer results to {output_path}")


if __name__ == "__main__":
    main()
