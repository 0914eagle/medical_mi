import argparse
import gc
import os
import random
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from split_experiment_utils import (
    BASE_DIR,
    MODELS,
    RESULTS_DIR,
    item_map_by_id,
    load_pubmedqa_items,
    read_json,
    write_json,
)
from utils import format_pubmedqa


def max_prob(probs):
    return max(float(value) for value in probs.values()) if probs else 0.0


def row_id(row):
    return str(row.get("item_id", row.get("pubid", "")))


def classify_typeb(row, confident_threshold=0.7):
    gt = row["ground_truth"]
    prior = row["prior_answer"]
    ctx = row["context_answer"]
    ctx_conf = max_prob(row["context_probs"])
    if gt != "no":
        return "other"
    if ctx == "no":
        return "correct"
    if prior == ctx:
        return "typeB_confident_ignore" if ctx_conf >= confident_threshold else "typeB_weak"
    return "typeA_processing"


def load_conflict_set(path, model):
    candidates = [
        path,
        f"{RESULTS_DIR}/eval/{model}_conflict_set.json",
        f"{BASE_DIR}/{model}_conflict_set.json",
        f"{model}_conflict_set.json",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return read_json(candidate), candidate
    raise FileNotFoundError(f"Could not find conflict_set for {model}")


def load_attention_model(model_name, attn_implementation):
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name], trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[model_name],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[model_name],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = attn_implementation
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = attn_implementation
    model.config.output_attentions = True
    model.eval()
    return model, tokenizer


def context_token_indices(prompt, tokenizer):
    start = prompt.find("Context:")
    question = prompt.find("\n\nQuestion:", start)
    if start < 0 or question < 0:
        return []
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=2048,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    indices = []
    for index, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start:
            continue
        if tok_start >= question:
            continue
        if tok_end > tok_start:
            indices.append(index)
    return indices


def context_attention(model, tokenizer, item, layer):
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    ctx_indices = context_token_indices(prompt, tokenizer)
    if not ctx_indices:
        return None
    tokenized = tokenized.to(model.device)
    with torch.no_grad():
        outputs = model(**tokenized, output_attentions=True)
    attentions = outputs.attentions
    if attentions is None:
        return context_attention_with_hook(model, tokenizer, prompt, ctx_indices, layer)
    if layer >= len(attentions):
        hooked = context_attention_with_hook(model, tokenizer, prompt, ctx_indices, layer)
        if hooked is not None:
            hooked["fallback_reason"] = "model_output_layer_out_of_range"
            hooked["model_output_attention_layers"] = len(attentions)
            return hooked
        return {"error": "layer_out_of_range", "n_attention_layers": len(attentions), "requested_layer": layer}
    attn = attentions[layer][0, :, -1, ctx_indices].sum(dim=-1).detach().float().cpu().numpy()
    return {
        "mean": float(attn.mean()),
        "max_head": float(attn.max()),
        "min_head": float(attn.min()),
        "heads": attn.tolist(),
        "n_context_tokens": len(ctx_indices),
    }


def context_attention_with_hook(model, tokenizer, prompt, ctx_indices, layer):
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        return None
    if layer >= len(model.model.layers):
        return {"error": "transformer_layer_out_of_range", "n_transformer_layers": len(model.model.layers), "requested_layer": layer}

    captured = {}
    layer_module = model.model.layers[layer]
    attention_module, attention_module_name = find_attention_module(layer_module)
    if attention_module is None:
        return {
            "error": "attention_module_not_found",
            "layer_children": [name for name, _ in layer_module.named_children()],
            "requested_layer": layer,
        }

    def hook_fn(module, inputs, output):
        tensor = find_attention_tensor(output)
        if tensor is not None:
            captured["attn"] = tensor.detach()

    handle = attention_module.register_forward_hook(hook_fn)
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    try:
        with torch.no_grad():
            model(**tokenized, output_attentions=True, use_cache=False)
    finally:
        handle.remove()

    if "attn" not in captured:
        return None
    attn_tensor = captured["attn"]
    if attn_tensor.dim() != 4:
        return {"error": "unexpected_hook_attention_shape", "shape": list(attn_tensor.shape)}
    attn = attn_tensor[0, :, -1, ctx_indices].sum(dim=-1).detach().float().cpu().numpy()
    return {
        "mean": float(attn.mean()),
        "max_head": float(attn.max()),
        "min_head": float(attn.min()),
        "heads": attn.tolist(),
        "n_context_tokens": len(ctx_indices),
        "source": "self_attn_forward_hook",
        "attention_module_name": attention_module_name,
    }


def find_attention_module(layer_module):
    preferred_names = [
        "self_attn",
        "attention",
        "attn",
        "linear_attn",
        "linear_attention",
        "hybrid_attn",
        "hybrid_attention",
    ]
    for name in preferred_names:
        if hasattr(layer_module, name):
            return getattr(layer_module, name), name

    for name, module in layer_module.named_children():
        lowered = name.lower()
        class_name = module.__class__.__name__.lower()
        if "attn" in lowered or "attention" in lowered or "attn" in class_name or "attention" in class_name:
            return module, name

    for name, module in layer_module.named_modules():
        if not name:
            continue
        lowered = name.lower()
        class_name = module.__class__.__name__.lower()
        if "attn" in lowered or "attention" in lowered or "attn" in class_name or "attention" in class_name:
            return module, name
    return None, None


def find_attention_tensor(output):
    if torch.is_tensor(output):
        if output.dim() == 4:
            return output
        return None
    if isinstance(output, (tuple, list)):
        for value in output:
            found = find_attention_tensor(value)
            if found is not None:
                return found
    if isinstance(output, dict):
        for value in output.values():
            found = find_attention_tensor(value)
            if found is not None:
                return found
    return None


def attention_diagnostics(model, tokenizer, item):
    prompt = format_pubmedqa(item, tokenizer, include_context=True)
    tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**tokenized, output_attentions=True, use_cache=False)
    attentions = outputs.attentions
    shapes = [list(attn.shape) for attn in attentions] if attentions is not None else []
    layer_children = []
    layer_module_names = []
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        layer_children = [name for name, _ in model.model.layers[0].named_children()]
        layer_module_names = [name for name, _ in model.model.layers[0].named_modules() if name][:20]
    return {
        "config_num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        "config_num_attention_heads": getattr(model.config, "num_attention_heads", None),
        "config_num_key_value_heads": getattr(model.config, "num_key_value_heads", None),
        "config_attn_implementation": getattr(model.config, "_attn_implementation", None),
        "actual_transformer_layers": len(model.model.layers) if hasattr(model, "model") and hasattr(model.model, "layers") else None,
        "returned_attention_layers": len(attentions) if attentions is not None else 0,
        "attention_shapes_first_3": shapes[:3],
        "layer0_children": layer_children,
        "layer0_named_modules_first_20": layer_module_names,
    }


def summarize(values):
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: context attention for correct/typeA/typeB groups.")
    parser.add_argument("--model", default="qwen3.5-9b")
    parser.add_argument("--conflict-set", default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 18, 20, 22, 24])
    parser.add_argument("--max-cases-per-group", type=int, default=30)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--confident-threshold", type=float, default=0.7)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()

    data, source_path = load_conflict_set(args.conflict_set, args.model)
    pubmedqa = load_pubmedqa_items()
    items_by_id = item_map_by_id(pubmedqa)
    grouped = defaultdict(list)
    for row in data:
        label = classify_typeb(row, args.confident_threshold)
        item = items_by_id.get(row_id(row))
        if item and label in {"correct", "typeA_processing", "typeB_confident_ignore", "typeB_weak"}:
            grouped[label].append({"row": row, "item": item})

    rng = random.Random(args.seed)
    sampled = {}
    for label, rows in grouped.items():
        shuffled = rows[:]
        rng.shuffle(shuffled)
        sampled[label] = shuffled[: args.max_cases_per_group]

    model, tokenizer = load_attention_model(args.model, args.attn_implementation)
    diagnostic_item = next(iter(sampled.values()))[0]["item"]
    diagnostics = attention_diagnostics(model, tokenizer, diagnostic_item)
    print("Attention diagnostics:")
    print(diagnostics)
    if args.diagnose_only:
        output_path = f"{RESULTS_DIR}/typeb_attention/{args.model}_typeb_attention_diagnostics.json"
        write_json(
            output_path,
            {
                "model": args.model,
                "source_path": source_path,
                "diagnostics": diagnostics,
                "sample_counts": {label: len(rows) for label, rows in sampled.items()},
            },
        )
        print(f"Saved attention diagnostics to {output_path}")
        return

    layer_results = {}
    case_results = []
    for layer in args.layers:
        layer_group_values = defaultdict(list)
        layer_head_values = defaultdict(list)
        for label, rows in sampled.items():
            for entry in tqdm(rows, desc=f"L{layer} {label}", leave=False):
                attn = context_attention(model, tokenizer, entry["item"], layer)
                if attn is None:
                    continue
                if "error" in attn:
                    case_results.append(
                        {
                            "layer": layer,
                            "group": label,
                            "item_id": row_id(entry["row"]),
                            "attention": attn,
                        }
                    )
                    break
                layer_group_values[label].append(attn["mean"])
                layer_head_values[label].append(attn["heads"])
                case_results.append(
                    {
                        "layer": layer,
                        "group": label,
                        "item_id": row_id(entry["row"]),
                        "attention": attn,
                    }
                )
        group_summary = {label: summarize(values) for label, values in layer_group_values.items()}
        head_summary = {}
        for label, heads in layer_head_values.items():
            arr = np.array(heads, dtype=np.float32)
            if arr.size:
                head_summary[label] = {
                    "mean_by_head": arr.mean(axis=0).tolist(),
                    "std_by_head": arr.std(axis=0, ddof=1).tolist() if arr.shape[0] > 1 else [0.0] * arr.shape[1],
                }
        layer_results[str(layer)] = {
            "group_summary": group_summary,
            "head_summary": head_summary,
        }
        gc.collect()
        torch.cuda.empty_cache()

    output = {
        "model": args.model,
        "source_path": source_path,
        "layers": args.layers,
        "attention_diagnostics": diagnostics,
        "max_cases_per_group": args.max_cases_per_group,
        "sample_counts": {label: len(rows) for label, rows in sampled.items()},
        "layer_results": layer_results,
        "cases": case_results,
    }
    output_path = f"{RESULTS_DIR}/typeb_attention/{args.model}_typeb_context_attention.json"
    write_json(output_path, output)
    print(f"Saved Type-B attention results to {output_path}")


if __name__ == "__main__":
    main()
