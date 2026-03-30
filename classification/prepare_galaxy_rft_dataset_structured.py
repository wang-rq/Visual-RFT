import argparse
import csv
import json
import os
import random
from collections import Counter
from copy import deepcopy

from datasets import Dataset, DatasetDict, Image

from galaxy_rft_structured_common import (
    CLASS_NAMES,
    build_problem,
    build_sft_messages,
    build_solution,
    canonical_label,
    random_prompt_indices,
)


def load_rows(csv_path: str, max_samples: int | None, seed: int):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    random.Random(seed).shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def build_augmented_records(base_record: dict, split_name: str, augment_train: bool):
    records = [base_record]
    if split_name != "train" or not augment_train:
        return records

    for aug_name, suffix in [
        ("crop_center", "Focus on the central structure before deciding."),
        ("crop_zoom", "Inspect the inner morphology and arm/bar structure carefully."),
    ]:
        aug_record = deepcopy(base_record)
        aug_record["augmentation"] = aug_name
        aug_record["problem"] = aug_record["problem"] + "\n\n" + suffix
        aug_record["messages"] = build_sft_messages(aug_record["image_path"], aug_record["problem"], aug_record["solution"])
        records.append(aug_record)
    return records


def write_jsonl(path: str, records: list[dict]):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_split(csv_path: str, split_name: str, max_samples: int | None, seed: int, augment_train: bool):
    rows = load_rows(csv_path, max_samples=max_samples, seed=seed)
    records = []
    skipped = 0
    prompt_indices = random_prompt_indices(seed, len(rows) if rows else 0)
    for row in rows:
        image_path = row["filepath"].strip()
        label = canonical_label(row["caption"])
        if label not in CLASS_NAMES:
            skipped += 1
            continue
        if not os.path.exists(image_path):
            skipped += 1
            continue
        prompt_index = prompt_indices[len(records) % len(prompt_indices)] if prompt_indices else 0
        include_few_shot = split_name == "train" and (len(records) % 3 == 0)
        problem = build_problem(variant_index=prompt_index, include_definitions=True, include_few_shot=include_few_shot)
        solution = build_solution(label)
        base_record = {
            "image": image_path,
            "problem": problem,
            "solution": solution,
            "caption": label,
            "image_path": image_path,
            "split": split_name,
            "augmentation": "none",
            "messages": build_sft_messages(image_path, problem, solution),
        }
        records.extend(build_augmented_records(base_record, split_name=split_name, augment_train=augment_train))

    dataset = Dataset.from_list(records).cast_column("image", Image())
    label_counts = Counter(item["caption"] for item in records)
    print(f"[{split_name}] kept {len(records)} samples from {csv_path}; skipped {skipped} rows.")
    return dataset, label_counts, records


def main():
    parser = argparse.ArgumentParser(description="Build a structured-think Visual-RFT DatasetDict for galaxy morphology classification.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment_train", action="store_true")
    args = parser.parse_args()

    train_dataset, train_counts, train_records = build_split(
        args.train_csv, "train", args.max_train_samples, args.seed, augment_train=args.augment_train
    )
    test_dataset, test_counts, test_records = build_split(
        args.test_csv, "test", args.max_test_samples, args.seed + 1, augment_train=False
    )

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)

    write_jsonl(os.path.join(args.output_dir, "train_sft.jsonl"), train_records)
    write_jsonl(os.path.join(args.output_dir, "test_sft.jsonl"), test_records)

    metadata = {
        "class_names": CLASS_NAMES,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_label_counts": dict(train_counts),
        "test_label_counts": dict(test_counts),
        "augment_train": args.augment_train,
        "reasoning_format": "structured_key_value",
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Saved DatasetDict to {args.output_dir}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
