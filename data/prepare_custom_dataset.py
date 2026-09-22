import json
from pathlib import Path
from typing import Any
from PIL import Image


def load_and_convert_dataset(
    dataset_name: str = "food101",
    split: str = "test",
    n_samples: int = 1000,
    seed: int = 42,
    output_jsonl: str = "data/huge_dataset.jsonl",
) -> str:
    """
    Downloads any HuggingFace image dataset at scale and prepares it for VLM confidence benchmarking.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' package is required. Install via: pip install datasets")

    print(f"Loading HuggingFace dataset '{dataset_name}' (split='{split}')...")
    ds = load_dataset(dataset_name, split=split)

    if n_samples > 0 and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))

    out_file = Path(output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    img_dir = out_file.parent / f"images_{out_file.stem}"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"Saving {len(ds)} dataset samples to '{img_dir}' and indexing in '{out_file}'...")

    for idx, item in enumerate(ds):
        # Handle various HuggingFace dataset image column names
        img_val = item.get("image") or item.get("img")
        if img_val is None:
            continue

        if hasattr(img_val, "convert"):
            img = img_val.convert("RGB")
        elif isinstance(img_val, (str, Path)):
            img = Image.open(img_val).convert("RGB")
        else:
            continue

        # Handle label key
        label_val = item.get("label") or item.get("category") or item.get("class") or item.get("caption") or "unknown"
        if hasattr(ds.features.get("label", None), "int2str") and isinstance(label_val, int):
            label_name = ds.features["label"].int2str(label_val)
        else:
            label_name = str(label_val)

        img_filename = f"sample_{idx:06d}.jpg"
        img_path = img_dir / img_filename
        img.save(img_path, format="JPEG")

        rec = {
            "index": idx,
            "label": label_name,
            "image_path": str(img_path),
        }
        records.append(rec)

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully converted and saved {len(records)} samples to '{out_file}'.")
    return str(out_file)


def create_jsonl_from_local_folder(
    image_dir: str,
    output_jsonl: str = "data/local_huge_dataset.jsonl",
    default_label: str = "object",
) -> str:
    """
    Scans a local directory containing thousands/millions of images and builds a JSONL dataset index.
    """
    img_path = Path(image_dir)
    if not img_path.exists():
        raise FileNotFoundError(f"Local image directory '{image_dir}' does not exist.")

    out_file = Path(output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = [p for p in img_path.rglob("*") if p.suffix.lower() in extensions]

    print(f"Found {len(image_files)} local images in '{image_dir}'. Building index...")

    records = []
    for idx, p in enumerate(image_files):
        # Subfolder name can act as category/label if organized in subfolders
        label_name = p.parent.name if p.parent != img_path else default_label
        rec = {
            "index": idx,
            "label": label_name,
            "image_path": str(p),
        }
        records.append(rec)

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Successfully indexed {len(records)} local images into '{out_file}'.")
    return str(out_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare large/huge external datasets for VLM benchmark.")
    parser.add_argument("--hf_dataset", type=str, default=None, help="HuggingFace dataset name (e.g. food101, cifar100).")
    parser.add_argument("--local_dir", type=str, default=None, help="Local directory path containing images.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples (0 for all).")
    parser.add_argument("--output", type=str, default="data/huge_dataset.jsonl")

    args = parser.parse_args()

    if args.hf_dataset:
        load_and_convert_dataset(
            dataset_name=args.hf_dataset,
            split=args.split,
            n_samples=args.n_samples,
            output_jsonl=args.output,
        )
    elif args.local_dir:
        create_jsonl_from_local_folder(
            image_dir=args.local_dir,
            output_jsonl=args.output,
        )
    else:
        print("Please specify either --hf_dataset <name> or --local_dir <path>.")
