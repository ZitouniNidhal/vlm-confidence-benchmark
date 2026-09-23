import argparse
import csv
import json
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from confidence.internal import compute_internal_confidence_from_probs  # noqa: E402
from confidence.verbalized import extract_verbalized_confidence, normalize_confidence  # noqa: E402
from data.prepare_food101 import load_food101_subset_from_jsonl, save_food101_subset  # noqa: E402
from degradations.pipeline import apply_degradations  # noqa: E402
from evaluation.accuracy import accuracy_score, match_label  # noqa: E402
from evaluation.auroc import error_detection_auroc  # noqa: E402
from evaluation.calibration import (  # noqa: E402
    adaptive_expected_calibration_error,
    brier_score,
    calibration_curve_data,
    expected_calibration_error,
)
from evaluation.plotting import (  # noqa: E402
    plot_confidence_histogram,
    plot_degradation_effects,
    plot_reliability_curve,
)
from models.qwen2vl import Qwen2VLM  # noqa: E402
from models.smolvlm import SmolVLM  # noqa: E402


DEGRADATION_TYPES = ["blur", "jpeg", "lowlight", "glare", "resample", "rotation", "noise", "fog"]
SEVERITY_LEVELS = ["low", "mid", "high"]


def get_model(model_type: str, mock: bool = False):
    if model_type.lower() == "qwen2vl":
        return Qwen2VLM(mock=mock)
    elif model_type.lower() == "smolvlm":
        return SmolVLM(mock=mock)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_benchmark(
    model_type: str = "smolvlm",
    mock: bool = False,
    data_path: str = "data/food101_subset.jsonl",
    n_samples: int = 100,
    output_dir: str = "results",
    prompt: str = 'Please answer with a label and a confidence percentage. Example: "I am 80% confident this is pizza."',
    synthetic_data: bool = False,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load or prepare dataset
    jsonl_path = Path(data_path)
    if not jsonl_path.exists():
        print(f"Dataset path {jsonl_path} not found. Preparing Food101 subset...")
        save_food101_subset(target_path=str(jsonl_path), n_samples=n_samples, synthetic=synthetic_data)

    data_records = load_food101_subset_from_jsonl(str(jsonl_path))
    if n_samples < len(data_records):
        data_records = data_records[:n_samples]

    print(f"Loaded {len(data_records)} samples for evaluation.")

    # Step 2: Initialize model
    print(f"Initializing {model_type} (mock={mock})...")
    model = get_model(model_type, mock=mock)

    # Build evaluation matrix: (Degradation, Severity)
    conditions = [("clean", "none")]
    for deg in DEGRADATION_TYPES:
        for sev in SEVERITY_LEVELS:
            conditions.append((deg, sev))

    summary_rows = []
    clean_verb_confs, clean_int_confs, clean_targets = [], [], []

    # Store metric progression for plotting
    severity_metrics = {
        "verbalized_ece": {deg: [] for deg in DEGRADATION_TYPES},
        "internal_ece": {deg: [] for deg in DEGRADATION_TYPES},
        "verbalized_auroc": {deg: [] for deg in DEGRADATION_TYPES},
        "internal_auroc": {deg: [] for deg in DEGRADATION_TYPES},
        "accuracy": {deg: [] for deg in DEGRADATION_TYPES},
    }

    print("Running benchmark evaluation across degradation conditions...")
    for deg, sev in tqdm(conditions, desc="Conditions"):
        preds = []
        targets = []
        verb_confs = []
        int_confs = []
        binary_correct = []
        binary_errors = []

        operations = [] if deg == "clean" else [(deg, sev)]

        for rec in data_records:
            image_path = rec["image_path"]
            ground_truth = rec["label"]
            img = Image.open(image_path).convert("RGB")

            # Apply degradation pipeline
            if operations:
                img_degraded = apply_degradations(img, operations=operations)
            else:
                img_degraded = img

            answer_text, token_probs = model.generate(img_degraded, prompt=prompt)

            is_correct = 1 if match_label(answer_text, ground_truth) else 0
            is_error = 1 - is_correct

            verb_raw = extract_verbalized_confidence(answer_text)
            verb_norm = normalize_confidence(verb_raw)
            int_norm = compute_internal_confidence_from_probs(token_probs)

            preds.append(answer_text)
            targets.append(ground_truth)
            verb_confs.append(verb_norm)
            int_confs.append(int_norm)
            binary_correct.append(is_correct)
            binary_errors.append(is_error)

        # Compute metrics
        acc = accuracy_score(preds, targets)
        verb_ece = expected_calibration_error(verb_confs, binary_correct)
        int_ece = expected_calibration_error(int_confs, binary_correct)
        verb_aece = adaptive_expected_calibration_error(verb_confs, binary_correct)
        int_aece = adaptive_expected_calibration_error(int_confs, binary_correct)
        verb_brier = brier_score(verb_confs, binary_correct)
        int_brier = brier_score(int_confs, binary_correct)
        verb_auroc = error_detection_auroc(verb_confs, binary_errors)
        int_auroc = error_detection_auroc(int_confs, binary_errors)

        row = {
            "model": model_type,
            "degradation": deg,
            "severity": sev,
            "accuracy": round(acc, 4),
            "verbalized_ece": round(verb_ece, 4),
            "internal_ece": round(int_ece, 4),
            "verbalized_adaptive_ece": round(verb_aece, 4),
            "internal_adaptive_ece": round(int_aece, 4),
            "verbalized_brier": round(verb_brier, 4),
            "internal_brier": round(int_brier, 4),
            "verbalized_auroc": round(verb_auroc, 4),
            "internal_auroc": round(int_auroc, 4),
        }
        summary_rows.append(row)

        if deg == "clean":
            clean_verb_confs = verb_confs
            clean_int_confs = int_confs
            clean_targets = binary_correct
        else:
            severity_metrics["verbalized_ece"][deg].append(verb_ece)
            severity_metrics["internal_ece"][deg].append(int_ece)
            severity_metrics["verbalized_auroc"][deg].append(verb_auroc)
            severity_metrics["internal_auroc"][deg].append(int_auroc)
            severity_metrics["accuracy"][deg].append(acc)

    # Save summary CSV
    csv_path = out_dir / "summary_metrics.csv"
    fieldnames = list(summary_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Save summary JSON
    json_path = out_dir / "summary_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    # Generate Plots
    print("Generating evaluation plots...")

    # 1. Clean Reliability Curves
    bin_confs_v, bin_accs_v = calibration_curve_data(clean_verb_confs, clean_targets)
    plot_reliability_curve(
        bin_confs_v,
        bin_accs_v,
        title=f"Verbalized Confidence Reliability ({model_type.upper()} Clean)",
        output_path=str(plots_dir / "verbalized_reliability_clean.png"),
    )

    bin_confs_i, bin_accs_i = calibration_curve_data(clean_int_confs, clean_targets)
    plot_reliability_curve(
        bin_confs_i,
        bin_accs_i,
        title=f"Internal Confidence Reliability ({model_type.upper()} Clean)",
        output_path=str(plots_dir / "internal_reliability_clean.png"),
    )

    # 2. Confidence Histograms
    plot_confidence_histogram(
        clean_verb_confs,
        title="Verbalized Confidence Distribution (Clean)",
        output_path=str(plots_dir / "verbalized_histogram_clean.png"),
    )
    plot_confidence_histogram(
        clean_int_confs,
        title="Internal Confidence Distribution (Clean)",
        output_path=str(plots_dir / "internal_histogram_clean.png"),
    )

    # 3. Degradation Impact Curves
    plot_degradation_effects(
        SEVERITY_LEVELS,
        severity_metrics["accuracy"],
        metric_name="Accuracy",
        output_path=str(plots_dir / "accuracy_vs_severity.png"),
    )
    plot_degradation_effects(
        SEVERITY_LEVELS,
        severity_metrics["verbalized_auroc"],
        metric_name="Verbalized Error Detection AUROC",
        output_path=str(plots_dir / "verbalized_auroc_vs_severity.png"),
    )
    plot_degradation_effects(
        SEVERITY_LEVELS,
        severity_metrics["internal_auroc"],
        metric_name="Internal Error Detection AUROC",
        output_path=str(plots_dir / "internal_auroc_vs_severity.png"),
    )

    print(f"Benchmark completed successfully! Results saved to '{out_dir}'.")
    return summary_rows


def main():
    parser = argparse.ArgumentParser(description="Run full VLM Confidence Benchmark.")
    parser.add_argument("--model", type=str, default="smolvlm", choices=["smolvlm", "qwen2vl"])
    parser.add_argument("--mock", action="store_true", help="Run with mock model for rapid execution/testing.")
    parser.add_argument("--data_path", type=str, default="data/food101_subset.jsonl")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic dataset fallback.")
    args = parser.parse_args()

    run_benchmark(
        model_type=args.model,
        mock=args.mock,
        data_path=args.data_path,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        synthetic_data=args.synthetic,
    )


if __name__ == "__main__":
    main()
