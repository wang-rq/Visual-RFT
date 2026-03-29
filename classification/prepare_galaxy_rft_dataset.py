import argparse
import csv
import json
import os
import random
from collections import Counter

from datasets import Dataset, DatasetDict, Image

from galaxy_rft_common import CLASS_NAMES, build_problem, build_solution, normalize_label


def load_rows(csv_path: str, max_samples: int | None, seed: int):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    random.Random(seed).shuffle(rows)
    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def build_split(csv_path: str, split_name: str, max_samples: int | None, seed: int):
    rows = load_rows(csv_path, max_samples=max_samples, seed=seed)
    records = []
    skipped = 0
    for row in rows:
        image_path = row["filepath"].strip()
        label = normalize_label(row["caption"])
        if label not in CLASS_NAMES:
            skipped += 1
            continue
        if not os.path.exists(image_path):
            skipped += 1
            continue
        records.append(
            {
                "image": image_path,
                "problem": build_problem(),
                "solution": build_solution(label),
                "caption": label,
                "image_path": image_path,
                "split": split_name,
            }
        )

    dataset = Dataset.from_list(records).cast_column("image", Image())
    label_counts = Counter(item["caption"] for item in records)
    print(
        f"[{split_name}] kept {len(records)} samples from {csv_path}; "
        f"skipped {skipped} rows."
    )
    return dataset, label_counts


def main():
    parser = argparse.ArgumentParser(description="Build a Visual-RFT DatasetDict for galaxy morphology classification.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dataset, train_counts = build_split(args.train_csv, "train", args.max_train_samples, args.seed)
    test_dataset, test_counts = build_split(args.test_csv, "test", args.max_test_samples, args.seed + 1)

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})
    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)

    metadata = {
        "class_names": CLASS_NAMES,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_label_counts": dict(train_counts),
        "test_label_counts": dict(test_counts),
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Saved DatasetDict to {args.output_dir}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
