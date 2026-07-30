import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from PIL import Image


def load_food101_subset(split: str = "test", n_samples: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    dataset = load_dataset("food101", split=split)
    subset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))
    examples = []

    for item in subset:
        image = item["image"]
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        label = dataset.features["label"].int2str(item["label"])
        examples.append({"image": image, "label": label})

    return examples


def save_food101_subset(target_path: str, split: str = "test", n_samples: int = 100, seed: int = 42) -> None:
    examples = load_food101_subset(split=split, n_samples=n_samples, seed=seed)
    output_path = Path(target_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir = output_path.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, example in enumerate(examples):
            image = example["image"]
            image_filename = f"food101_{idx:04d}.jpg"
            image_path = image_dir / image_filename
            image.save(image_path, format="JPEG")
            handle.write(
                json.dumps(
                    {
                        "index": idx,
                        "label": example["label"],
                        "image_path": str(image_path),
                    }
                )
                + "\n"
            )


def load_food101_subset_from_jsonl(source_path: str) -> list[dict[str, Any]]:
    examples = []
    source_file = Path(source_path)
    if not source_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_file}")

    with source_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            examples.append(json.loads(line.strip()))
    return examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a small Food101 subset for benchmark experiments.")
    parser.add_argument("--output", type=str, default="data/food101_subset.jsonl")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    save_food101_subset(target_path=args.output, split=args.split, n_samples=args.size, seed=args.seed)
