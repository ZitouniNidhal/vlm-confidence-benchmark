# VLM Confidence Benchmark

This repository provides a reproducible benchmark for evaluating **small Vision-Language Models (VLMs)** under realistic image degradations.  
We compare two confidence signals:

- **Verbalized confidence**: when the model explicitly states its confidence in natural language.  
- **Internal confidence**: derived from token probabilities in the model’s output distribution.  

---

## 📖 Background

Vision-Language Models are increasingly deployed in real-world scenarios where input images may be corrupted (e.g., blur, compression, low light).  
Understanding how models express and calibrate confidence is critical for safe deployment in sensitive domains such as healthcare or autonomous systems.

This benchmark is inspired by recent work on confidence evaluation in VLMs, focusing on lightweight models that can run on modest hardware.

---

## 🧩 Methodology

1. **Dataset**  
   - A 100-sample subset of Food101 is used for multiple-choice classification.  
   - Each sample includes an image and candidate labels.

2. **Image Degradations**
   - Eight corruption families:  
     - JPEG compression  
     - Gaussian blur  
     - Gaussian noise  
     - Fog  
     - Low light  
     - Glare (overexposure)  
     - Rotation  
     - Resampling (downscale + upscale)  
   - Each applied at low / mid / high severity to mirror VLM-RobustBench-style robustness evaluation.

3. **Models**
   - [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)  
   - [SmolVLM-Instruct](https://huggingface.co/HuggingFace/smolvlm-instruct)

4. **Confidence Signals**  
   - *Verbalized*: extracted via regex from generated text.  
   - *Internal*: computed from token probability distributions.

5. **Metrics**  
   - Accuracy  
   - Expected Calibration Error (ECE)  
   - Brier Score  
   - AUROC for error detection (primary metric)

---

## ⚙️ Installation

```bash
git clone https://github.com/ZitouniNidhal/vlm-confidence-benchmark.git
cd vlm-confidence-benchmark
pip install -r requirements.txt
```

## 🔧 Usage

You can now compose multiple image degradations in a single call, including severity presets:

```python
from PIL import Image
from degradations.pipeline import apply_degradations

image = Image.open("path/to/image.jpg")
augmented = apply_degradations(
    image,
    operations=[
        ("blur", "low"),
        ("jpeg", "mid"),
        ("noise", "high"),
    ],
)
```

You can also override preset severity values directly:

```python
augmented = apply_degradations(
    image,
    operations=[
        ("fog", {"severity": "mid", "opacity": 0.35}),
        ("rotation", {"angle": 20.0}),
    ],
)
```
