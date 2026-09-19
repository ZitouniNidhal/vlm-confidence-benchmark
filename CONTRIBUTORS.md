# Project Contributors 👥

Thank you to everyone who has contributed to the **VLM Confidence Benchmark**! Every contribution helps advance transparent, calibrated, and robust Vision-Language Model evaluation under real-world visual corruptions.

---

## 🌟 Project Lead & Core Maintainer

### **Nidhal Zitouni** ([@ZitouniNidhal](https://github.com/ZitouniNidhal))
> **Role**: *Project Founder, Core Architect & Lead Developer*
>
> **Contributions**:
> - **Benchmark Architecture**: Architected the end-to-end VLM uncertainty benchmark matrix across model families, degradation presets, and evaluation metrics.
> - **Confidence Signals Engine**: Designed dual confidence extraction algorithms contrasting **Verbalized Confidence** (natural language regex parsing) and **Internal Confidence** (sequence log-probability distributions).
> - **Visual Corruption Pipeline**: Implemented the modular degradation suite covering 8 environmental corruption families (Blur, Fog, Glare, JPEG compression, Low light, Noise, Resampling, Rotation).
> - **Evaluation & Calibration Suite**: Implemented Expected Calibration Error (ECE), Brier Score, and Error Detection AUROC evaluation tools with automated plotting routines.

---

## 👥 Core Contributors & Contribution Areas

| Contributor | GitHub Profile | Primary Module / Domain | Description of Contributions |
| :--- | :--- | :--- | :--- |
| **Nidhal Zitouni** | [@ZitouniNidhal](https://github.com/ZitouniNidhal) | **Core Engine & Architecture** | Created project framework, model interfaces (`SmolVLM`, `Qwen2-VL`), degradation pipeline, evaluation metrics, and Pytest validation suite. |
| *Community Contributors* | [Open a PR](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/pulls) | **VLM Model Interfaces** | Adding support for new open-weights Vision-Language Models (e.g., LLaVA, InternVL, Florence-2). |
| *Community Contributors* | [Open a PR](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/pulls) | **Visual Degradation Operators** | Expanding corruption filters (e.g., rain, snow, motion blur, sensor artifacts). |
| *Community Contributors* | [Open a PR](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/pulls) | **Calibration & Uncertainty Metrics** | Adding advanced calibration metrics (e.g., Adaptive ECE, Temperature Scaling, Conformal Prediction). |

---

## 💡 How to Become a Contributor

We welcome contributions from researchers, engineers, and open-source enthusiasts!

### Contribution Categories:
1. 🤖 **New VLM Models**: Implement interfaces in [`models/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/models) and YAML configs in [`configs/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/configs).
2. 🌫️ **Visual Degradations**: Add custom image corruption functions under [`degradations/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/degradations).
3. 📊 **Evaluation & Calibration Metrics**: Enhance metric implementations in [`evaluation/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/evaluation).
4. 🧪 **Tests & Docs**: Expand test coverage in [`tests/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/tests) or clarify project documentation.

### Adding Yourself to the Contributors Table:
1. Fork the repository and create a descriptive feature branch.
2. Add your name, GitHub link, module area, and description to the table above.
3. Run `pytest` to ensure all tests pass cleanly.
4. Submit a Pull Request.
