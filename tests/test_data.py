from pathlib import Path
import pytest

from data.prepare_food101 import (
    load_food101_subset,
    save_food101_subset,
    load_food101_subset_from_jsonl,
)


def test_synthetic_data_loading():
    examples = load_food101_subset(n_samples=5, synthetic=True)
    assert len(examples) == 5
    assert "image" in examples[0]
    assert "label" in examples[0]


def test_save_and_load_food101_jsonl(tmp_path):
    target_file = tmp_path / "subset.jsonl"
    save_food101_subset(target_path=str(target_file), n_samples=3, synthetic=True)

    assert target_file.exists()
    records = load_food101_subset_from_jsonl(str(target_file))
    assert len(records) == 3
    assert Path(records[0]["image_path"]).exists()
