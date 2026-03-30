import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

from galaxy_rft_structured_common import (
    CLASS_NAMES,
    SYSTEM_PROMPT,
    build_problem,
    canonical_label,
    extract_answer,
)


def load_model(model_name_or_path: str, dtype: torch.dtype, attn_implementation: str):
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if AutoModelForImageTextToText is not None:
        try:
            return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)
        except Exception:
            pass
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)


def build_messages(image_path: str, prompt_variant: int, include_few_shot: bool):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": build_problem(prompt_variant, include_definitions=True, include_few_shot=include_few_shot)},
            ],
        },
    ]


def majority_vote(predictions: list[str]) -> str:
    counts = Counter(predictions)
    best = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return best[0][0]


def compute_metrics(predictions):
    label_to_idx = {label: idx for idx, label in enumerate(CLASS_NAMES)}
    confusion = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})

    num_correct = 0
    for item in predictions:
        gold = canonical_label(item["label"])
        pred = canonical_label(item["prediction"])
        if gold in label_to_idx and pred in label_to_idx:
            confusion[label_to_idx[gold]][label_to_idx[pred]] += 1
        per_class[gold]["total"] += 1
        if pred == gold:
            num_correct += 1
            per_class[gold]["correct"] += 1

    metrics = {
        "num_samples": len(predictions),
        "accuracy": (num_correct / len(predictions)) if predictions else 0.0,
        "per_class_accuracy": {
            label: (stats["correct"] / stats["total"] if stats["total"] else 0.0)
            for label, stats in per_class.items()
        },
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": confusion,
        },
    }
    return metrics


def save_error_analysis(predictions, output_prefix: str):
    errors = [item for item in predictions if not item["correct"]]
    errors = sorted(errors, key=lambda x: (x["label"], x["prediction"], x["image_path"]))
    path = output_prefix + "_errors.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(errors[:200], handle, ensure_ascii=False, indent=2)
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate a structured-think Visual-RFT galaxy classification model.")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--processor_name_or_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--num_votes", type=int, default=3)
    parser.add_argument("--include_few_shot", action="store_true")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    processor_name_or_path = args.processor_name_or_path or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_name_or_path, trust_remote_code=True, use_fast=False)
    model = load_model(args.model_name_or_path, dtype=dtype, attn_implementation=args.attn_implementation)
    model.eval()
    model.cuda()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    predictions = []
    for row in tqdm(rows, desc="Evaluating"):
        image_path = row["filepath"].strip()
        label = canonical_label(row["caption"])
        image = Image.open(image_path).convert("RGB")
        vote_preds = []
        vote_outputs = []

        for vote_idx in range(args.num_votes):
            messages = build_messages(image_path, prompt_variant=vote_idx, include_few_shot=args.include_few_shot)
            prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            prompt_length = inputs["input_ids"].shape[1]
            trimmed_ids = generated_ids[:, prompt_length:]
            response = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            pred_label = extract_answer(response)
            vote_preds.append(pred_label)
            vote_outputs.append(response)

        final_prediction = majority_vote(vote_preds)
        is_correct = final_prediction == label
        predictions.append(
            {
                "image_path": image_path,
                "label": label,
                "prediction": final_prediction,
                "correct": is_correct,
                "vote_predictions": vote_preds,
                "vote_outputs": vote_outputs,
            }
        )

    with open(args.output_path, "w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    metrics = compute_metrics(predictions)
    metrics_path = args.output_path.replace(".jsonl", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    error_path = save_error_analysis(predictions, args.output_path.replace(".jsonl", ""))

    print(f"Evaluated {metrics['num_samples']} samples")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Predictions saved to {args.output_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Error analysis saved to {error_path}")


if __name__ == "__main__":
    main()
