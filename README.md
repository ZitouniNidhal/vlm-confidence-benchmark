# VLM Confidence Benchmark 🎯👁️

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)](tests/)

A reproducible, modular benchmark for evaluating the **confidence calibration** and **error detection** capabilities of Vision-Language Models (VLMs) under realistic visual image degradations.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Methodology](#-methodology)
  - [1. Data & Subsets](#1-data--subsets)
  - [2. Image Degradation Pipeline](#2-image-degradation-pipeline)
  - [3. Supported Models](#3-supported-models)
  - [4. Confidence Signals](#4-confidence-signals)
  - [5. Evaluation Metrics](#5-evaluation-metrics)
- [Installation](#-installation)
- [Quick Start & Data Preparation](#-quick-start--data-preparation)
- [Running Benchmarks & Experiments](#-running-benchmarks--experiments)
  - [Full Benchmark Execution](#full-benchmark-execution)
  - [Single-Image Inference](#single-image-inference)
- [Python API Usage](#-python-api-usage)
- [Output Artifacts & Visualization](#-output-artifacts--visualization)
- [Running Unit Tests](#-running-unit-tests)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Overview

Vision-Language Models (VLMs) are deployed in critical real-world environments—such as autonomous robotics, medical diagnostics, and remote sensing—where input images are frequently corrupted by environmental conditions (e.g., blur, fog, noise, low light). 

This benchmark evaluates how vision-language models express and calibrate their uncertainty under degrading image conditions. Specifically, we evaluate and contrast two fundamental confidence mechanisms:
1. **Verbalized Confidence**: The confidence percentage explicitly stated by the model in natural language output (e.g., *"I am 80% confident this is a pizza"*).
2. **Internal Confidence**: Confidence estimated from output token probability distributions over generated answer tokens.

---

## 📁 Project Architecture

```
vlm-confidence-benchmark/
├── confidence/                 # Confidence signal extraction modules
│   ├── internal.py             # Log probability, distribution & predictive entropy calculations
│   └── verbalized.py           # Regex-based extraction of percentages, fractions & qualitative phrases
├── configs/                    # Model hyperparameter and prompt configurations
│   ├── qwen2vl.yaml            # Qwen2-VL config
│   └── smolvlm.yaml            # SmolVLM config
├── data/                       # Dataset loading & dataset generation tools
│   └── prepare_food101.py      # Food101 subset downloader and synthetic offline generator
├── degradations/               # Modular visual corruption operators
│   ├── blur.py                 # Gaussian blur operator
│   ├── fog.py                  # Synthetic fog corruption
│   ├── glare.py                # Overexposure / specular glare operator
│   ├── jpeg.py                 # JPEG compression artifacts
│   ├── lowlight.py             # Low-light attenuation
│   ├── noise.py                # Additive Gaussian noise operator
│   ├── pipeline.py             # Pipeline compositor & severity parameter maps
│   ├── resample.py             # Downscaling and upscaling resampling
│   └── rotation.py             # Image rotation operator
├── evaluation/                 # Metrics & plotting tools
│   ├── accuracy.py             # Match labeling & accuracy metrics
│   ├── auroc.py                # Error detection AUROC
│   ├── calibration.py          # Expected Calibration Error (ECE) & Brier score
│   └── plotting.py             # Reliability curves, histograms & degradation plots
├── experiments/                # CLI runners for benchmarking
│   ├── run_benchmark.py        # Complete benchmark matrix evaluator
│   ├── run_single_image.py     # Unified single-image evaluator & comparison plotter
│   ├── run_qwen2vl.py          # Legacy single-image runner for Qwen2-VL
│   └── run_smolvlm.py          # Legacy single-image runner for SmolVLM
├── models/                     # Model wrapper interfaces
│   ├── base.py                 # BaseVLMModel abstract base class
│   ├── qwen2vl.py              # Qwen2-VL-2B-Instruct interface & mock runner
│   └── smolvlm.py              # SmolVLM-Instruct interface & mock runner
├── results/                    # Generated benchmark outputs, CSV summaries & plots
├── tests/                      # Pytest test suite
├── pytest.ini                  # Pytest configuration
├── requirements.txt            # Dependency requirements
└── README.md                   # Project documentation
```

---

## 🧩 Methodology

### 1. Data & Subsets
- Evaluates classification on a standard subset of the **Food101** dataset.
- Supports **synthetic data generation** for offline testing, local debugging, and continuous integration environments without requiring external Hugging Face dataset downloads.

### 2. Image Degradation Pipeline
Evaluates robustness across **8 corruption families** with standardized `low`, `mid`, and `high` severity presets:
- **JPEG compression** (`jpeg`): Low quality factors.
- **Gaussian blur** (`blur`): Varying radius parameters.
- **Gaussian noise** (`noise`): Additive image variance.
- **Fog** (`fog`): Synthetic fog blend with opacity controls.
- **Low light** (`lowlight`): Brightness reduction scaling.
- **Glare** (`glare`): Overexposure saturation.
- **Rotation** (`rotation`): Geometric rotation angles.
- **Resampling** (`resample`): Downscaling followed by reconstruction upscaling.

### 3. Supported Models
- **[SmolVLM-Instruct](https://huggingface.co/HuggingFace/smolvlm-instruct)**: Lightweight multimodal model designed for efficient inference.
- **[Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)**: 2B parameter Vision-Language Model.
- **Mock Mode (`--mock`)**: Execution mode generating deterministic predictions for rapid pipeline validation without requiring GPU resources or heavy model weights.

### 4. Confidence Signals
- **Verbalized Confidence**: Parsed via regex patterns from model output (e.g., *"90%"*, *"0.85"*, *"low confidence"* mapped to 0.25).
- **Internal Confidence**: Computed from average or product token probabilities assigned to generated sequence tokens.

### 5. Evaluation Metrics
- **Accuracy**: Exact & normalized keyword prediction match against ground truth.
- **Expected Calibration Error (ECE)**: Measures discrepancy between predicted confidence and empirical accuracy across probability bins.
- **Brier Score**: Mean squared error of confidence scores relative to binary correctness.
- **AUROC for Error Detection**: Primary metric measuring how effectively confidence discriminates correct predictions from errors.

---

## ⚙️ Installation

1. **Clone repository**:
   ```bash
   git clone https://github.com/ZitouniNidhal/vlm-confidence-benchmark.git
   cd vlm-confidence-benchmark
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quick Start & Data Preparation

Prepare the evaluation dataset (Food101 subset or synthetic fallback):

```bash
# Prepare 100-sample Food101 dataset subset
python -m data.prepare_food101 --output data/food101_subset.jsonl --size 100

# Prepare synthetic dataset for testing without external downloads
python -m data.prepare_food101 --output data/synthetic_subset.jsonl --size 50
```

---

## 📊 Running Benchmarks & Experiments

### Full Benchmark Execution

Run the complete benchmark matrix across all degradation types and severities:

```bash
# Run SmolVLM in mock mode (fast, CPU-friendly)
python -m experiments.run_benchmark --model smolvlm --mock --n_samples 50

# Run Qwen2-VL with synthetic dataset
python -m experiments.run_benchmark --model qwen2vl --mock --synthetic

# Run SmolVLM with actual model weights on CUDA
python -m experiments.run_benchmark --model smolvlm --data_path data/food101_subset.jsonl
```

### Single-Image Inference

Run single-image evaluation with automatic degradation, verbalized vs internal confidence comparison, and optional side-by-side plot generation:

```bash
# Run single image with SmolVLM (mock) under blur degradation
python -m experiments.run_single_image --image data/images/food101_0000.jpg --model smolvlm --mock --degradation blur --severity high

# Run single image with Qwen2-VL (mock) and save side-by-side plot
python -m experiments.run_single_image --image data/images/food101_0000.jpg --model qwen2vl --mock --degradation fog --output_plot results/plots/single_image_eval.png
```

---

## 💻 Python API Usage

### Applying Degradations
```python
from PIL import Image
from degradations.pipeline import apply_degradations

img = Image.open("path/to/image.jpg")

# Apply preset severity levels
degraded_img = apply_degradations(
    img,
    operations=[
        ("blur", "mid"),
        ("jpeg", "high"),
    ]
)

# Apply customized degradation parameters
custom_img = apply_degradations(
    img,
    operations=[
        ("fog", {"severity": "mid", "opacity": 0.4}),
        ("rotation", {"angle": 15.0}),
    ]
)
```

### Extracting Confidence & Computing Metrics
```python
from confidence.verbalized import extract_verbalized_confidence, normalize_confidence
from confidence.internal import compute_internal_confidence_from_probs
from evaluation.calibration import expected_calibration_error, brier_score
from evaluation.auroc import error_detection_auroc

# Verbalized confidence parsing
raw_conf = extract_verbalized_confidence("I am 85% confident this is a pizza.")
norm_verb_conf = normalize_confidence(raw_conf)  # Returns 0.85

# Internal token probability confidence
token_probs = [0.95, 0.88, 0.92]
int_conf = compute_internal_confidence_from_probs(token_probs)

# Calibration & AUROC metrics
confidences = [0.90, 0.80, 0.60, 0.40]
binary_targets = [1, 1, 0, 0]          # 1 for correct, 0 for error
binary_errors = [0, 0, 1, 1]           # 1 for error, 0 for correct

ece = expected_calibration_error(confidences, binary_targets)
brier = brier_score(confidences, binary_targets)
auroc = error_detection_auroc(confidences, binary_errors)

print(f"ECE: {ece:.4f}, Brier: {brier:.4f}, Error Detection AUROC: {auroc:.4f}")
```

---

## 📈 Output Artifacts & Visualization

When `run_benchmark.py` finishes, artifacts are generated under the specified `--output_dir` (default `results/`):

- `summary_metrics.csv`: Tabular evaluation metrics across all (degradation, severity) conditions.
- `summary_metrics.json`: JSON output of all evaluation results.
- `plots/`:
  - `verbalized_reliability_clean.png`: Reliability curve for verbalized confidence under clean images.
  - `internal_reliability_clean.png`: Reliability curve for internal confidence under clean images.
  - `verbalized_histogram_clean.png`: Confidence distribution histogram (verbalized).
  - `internal_histogram_clean.png`: Confidence distribution histogram (internal).
  - `accuracy_vs_severity.png`: Model accuracy vs. degradation severity level.
  - `verbalized_auroc_vs_severity.png`: Verbalized AUROC performance vs. degradation severity.
  - `internal_auroc_vs_severity.png`: Internal AUROC performance vs. degradation severity.

---

## 🧪 Running Unit Tests

The test suite covers model loading, degradation pipeline transformations, confidence parsing, evaluation metrics, and end-to-end benchmark execution:

```bash
pytest
```

---

## ⚙️ Configuration

Model prompts and generation hyperparameters are specified in `configs/`:
- `configs/smolvlm.yaml`
- `configs/qwen2vl.yaml`

Example configuration structure:
```yaml
model_name: HuggingFace/smolvlm-instruct
prompt_template: 'Please answer with a label and a confidence percentage. Example: "I am 80% confident this is pizza."'
max_new_tokens: 64
temperature: 0.0
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit Pull Requests for new VLM models, degradation operators, or evaluation metrics. Please refer to [CONTRIBUTORS.md](CONTRIBUTORS.md) for details on how to add yourself to the project contributors.


## 📜 License

Distributed under the MIT License. See `LICENSE` for details.



