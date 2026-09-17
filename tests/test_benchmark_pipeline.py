from pathlib import Path
import pytest
from experiments.run_benchmark import run_benchmark


def test_full_benchmark_mock_pipeline(tmp_path):
    output_dir = tmp_path / "results"
    data_path = tmp_path / "data" / "subset.jsonl"

    summary = run_benchmark(
        model_type="smolvlm",
        mock=True,
        data_path=str(data_path),
        n_samples=3,
        output_dir=str(output_dir),
        synthetic_data=True,
    )

    assert len(summary) > 0
    assert (output_dir / "summary_metrics.csv").exists()
    assert (output_dir / "summary_metrics.json").exists()
    assert (output_dir / "plots" / "verbalized_reliability_clean.png").exists()
    assert (output_dir / "plots" / "internal_reliability_clean.png").exists()
    assert (output_dir / "plots" / "accuracy_vs_severity.png").exists()
