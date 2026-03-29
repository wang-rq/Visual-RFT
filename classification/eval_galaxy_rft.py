import argparse
import csv
import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

from galaxy_rft_common import CLASS_NAMES, SYSTEM_PROMPT, build_problem, extract_answer, normalize_label


def load_model(model_name_or_path: str, dtype: torch.dtype, attn_implementation: str):
    model_kwargs = {
        "torch_dtype": dtype,
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


def build_messages(image_path: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": build_problem()},
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Visual-RFT-style galaxy classification model.")
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--processor_name_or_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    processor_name_or_path = args.processor_name_or_path or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_name_or_path, trust_remote_code=True)
    model = load_model(args.model_name_or_path, dtype=dtype, attn_implementation=args.attn_implementation)
    model.eval()
    model.cuda()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    with open(args.csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    num_correct = 0
    predictions = []

    for row in tqdm(rows, desc="Evaluating"):
        image_path = row["filepath"].strip()
        label = normalize_label(row["caption"])
        messages = build_messages(image_path)
        prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(image_path).convert("RGB")
        inputs = processor(
            text=[prompt_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

        prompt_length = inputs["input_ids"].shape[1]
        trimmed_ids = generated_ids[:, prompt_length:]
        response = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        pred_label = extract_answer(response)
        is_correct = pred_label == label
        num_correct += int(is_correct)

        predictions.append(
            {
                "image_path": image_path,
                "label": label,
                "prediction": pred_label,
                "correct": is_correct,
                "raw_response": response,
            }
        )

    with open(args.output_path, "w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    accuracy = num_correct / len(predictions) if predictions else 0.0
    print(f"Evaluated {len(predictions)} samples")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Predictions saved to {args.output_path}")
    print(f"Known labels: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
