# Project Contributors & Commit History 👥

Thank you to everyone who has contributed to the **VLM Confidence Benchmark**! Every contribution helps advance transparent, calibrated, and robust Vision-Language Model evaluation under real-world visual corruptions.

---

## 🌟 Project Lead & Core Maintainer

### **Nidhal Zitouni** ([@ZitouniNidhal](https://github.com/ZitouniNidhal))
> **Role**: *Project Founder, Core Architect & Lead Developer*
>
> **Contributions**:
> - **Benchmark Architecture**: Architected the end-to-end VLM uncertainty benchmark matrix across model families, degradation presets, and evaluation metrics.
> - **Confidence Signals Engine**: Designed dual confidence extraction algorithms contrasting **Verbalized Confidence** (natural language regex parsing) and **Internal Confidence** (sequence log-probability distributions & predictive entropy).
> - **Visual Corruption Pipeline**: Implemented the modular degradation suite covering 8 environmental corruption families (Blur, Fog, Glare, JPEG compression, Low light, Noise, Resampling, Rotation).
> - **Evaluation & Calibration Suite**: Implemented Expected Calibration Error (ECE), Brier Score, and Error Detection AUROC evaluation tools with automated plotting routines.

### **Collaborators & AI Pair Assistants**
- **Adesh Kashyap** ([@adeshkashyap](https://github.com/adeshkashyap)) - *Co-author & Contributor*
- **Antigravity Assistant** ([@google](https://github.com/google)) - *Pair Programming & Refactoring Assistant* (Co-authored commits enhancing calibration documentation, `BaseVLMModel` abstraction, predictive sequence entropy, fraction parsing, and non-interactive plotting configurations).

---

## 📜 Commit-by-Commit Contribution Log

Below is the comprehensive list of repository commits documented one-by-one:

| Commit Hash | Date | Author / Contributor | Module / Component | Commit Message & Summary |
| :--- | :--- | :--- | :--- | :--- |
| [`f9919a1`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/f9919a1) | 2026-09-20 | Nidhal Zitouni | `evaluation/`, `pytest` | Configured non-interactive `matplotlib.use("Agg")` for clean automated executions and added `pytest.ini`. |
| [`7b4eae9`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/7b4eae9) | 2026-09-20 | Nidhal Zitouni | `experiments/` | Created `run_single_image.py` CLI script taking `--image`, `--model`, `--degradation`, `--severity`, `--output_plot`. |
| [`65ae7a1`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/65ae7a1) | 2026-09-20 | Nidhal Zitouni | `models/`, `README.md` | Defined `BaseVLMModel` using `abc.ABC` with abstract method `generate(image, prompt, max_new_tokens, temperature)`. |
| [`ea1c1fa`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/ea1c1fa) | 2026-09-20 | Nidhal Zitouni | `tests/` | Added unit tests for `BaseVLMModel` inheritance, predictive sequence entropy, fraction parsing, and single image runner. |
| [`d2f29cd`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/d2f29cd) | 2026-09-20 | Nidhal Zitouni | `confidence/`, `models/` | Refactored `SmolVLM` and `Qwen2VLM` under abstract `BaseVLMModel`, expanded verbalized confidence regex parsing (fractions & qualitative terms), and added sequence entropy. |
| [`260ab59`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/260ab59) | 2026-09-19 | Nidhal Zitouni | Git Branching | Merge branch `main` into `feature/pair-extraordinaire`. |
| [`155c974`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/155c974) | 2026-09-19 | Nidhal Zitouni | Documentation | `docs`: Enrich `CONTRIBUTORS.md` with detailed descriptions for all project domains. |
| [`dc780a7`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/dc780a7) | 2026-09-19 | Nidhal Zitouni | Pull Request #3 | Merge pull request #3: Benchmark components, tests, and configuration verification. |
| [`c3f1c95`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/c3f1c95) | 2026-09-19 | Nidhal Zitouni | `.gitignore`, `tests/` | Benchmark components, tests, and configuration changes verification. |
| [`5434011`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/5434011) | 2026-09-18 | Nidhal Zitouni | Pull Request #2 | Merge pull request #2: Calibration module documentation updates. |
| [`ebd1ff8`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/ebd1ff8) | 2026-09-18 | Nidhal Zitouni | `evaluation/calibration.py` | `docs`: Enhance calibration module docstring (co-authored with Antigravity Assistant). |
| [`0851358`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/0851358) | 2026-09-18 | Nidhal Zitouni | Pull Request #1 | Merge pull request #1: Add contributing section to `README.md`. |
| [`a2b1b19`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/a2b1b19) | 2026-09-18 | Nidhal Zitouni | `README.md` | `docs`: Add contributing section to README. |
| [`921801b`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/921801b) | 2026-09-18 | Nidhal Zitouni | `README.md` | Update README with detailed documentation covering architecture and capabilities. |
| [`b4d309b`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/b4d309b) | 2026-09-17 | Nidhal Zitouni | Core Project Files | Comprehensive update across models, confidence, evaluation, degradations, and unit tests. |
| [`c4ebaac`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/c4ebaac) | 2026-08-08 | ZITOUNI Nidhal | `degradations/` | Add fog and noise degradation functions with severity presets support. |
| [`2ea0998`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/2ea0998) | 2026-08-03 | ZITOUNI Nidhal | `tests/conftest.py` | Add `conftest.py` for test configuration and path setup. |
| [`d985dfe`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/d985dfe) | 2026-08-03 | ZITOUNI Nidhal | `degradations/` | Add `__init__.py` to expose `apply_degradations` function. |
| [`09811da`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/09811da) | 2026-08-03 | ZITOUNI Nidhal | `degradations/pipeline.py` | Add usage instructions and implement `apply_degradations` function. |
| [`be73166`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/be73166) | 2026-07-31 | ZITOUNI Nidhal | `.gitignore` | Add `.gitignore` to exclude dataset files, results, and cache. |
| [`fedeb5f`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/fedeb5f) | 2026-07-31 | ZITOUNI Nidhal | `data/prepare_food101.py` | Enhance Food101 subset saving: save images to disk and update JSON format. |
| [`67a0a02`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/67a0a02) | 2026-07-30 | ZITOUNI Nidhal | `confidence/internal.py` | Refactor internal confidence computation and update run scripts for Qwen2VLM and SmolVLM. |
| [`badebdd`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/badebdd) | 2026-07-30 | ZITOUNI Nidhal | `configs/` | Add Qwen2VLM and SmolVLM YAML configurations. |
| [`7e36165`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/7e36165) | 2026-07-30 | ZITOUNI Nidhal | `models/` | Add Qwen2VLM and SmolVLM model classes for image processing and text generation. |
| [`eedae89`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/eedae89) | 2026-07-30 | ZITOUNI Nidhal | `degradations/resample.py` | Add `apply_resample` function for image downscaling and upscaling resampling. |
| [`3dc0b2c`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/3dc0b2c) | 2026-07-30 | ZITOUNI Nidhal | `degradations/` | Add glare overexposure and lowlight attenuation operators. |
| [`0bce81e`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/0bce81e) | 2026-07-30 | ZITOUNI Nidhal | `degradations/` | Add Gaussian blur and JPEG compression corruption operators. |
| [`8f67940`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/8f67940) | 2026-07-30 | ZITOUNI Nidhal | `evaluation/calibration.py` | Add Brier score and expected calibration error (ECE) functions. |
| [`a5f00aa`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/a5f00aa) | 2026-07-30 | ZITOUNI Nidhal | `evaluation/accuracy.py` | Add `accuracy_score` and label normalization matching logic. |
| [`69c8310`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/69c8310) | 2026-07-30 | ZITOUNI Nidhal | `README.md` | Fix clone URL in installation instructions in README. |
| [`0ec13fa`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/0ec13fa) | 2026-07-30 | ZITOUNI Nidhal | `README.md` | Initial fix for repository clone instructions. |
| [`2867d4d`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/2867d4d) | 2026-07-29 | ZITOUNI Nidhal | `requirements.txt` | Update requirements.txt with dependencies for PIL, PyTorch, SciPy, Scikit-learn, and Matplotlib. |
| [`07b30b7`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/07b30b7) | 2026-07-29 | ZITOUNI Nidhal | `requirements.txt` | Initial requirements specification. |
| [`05dda30`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/05dda30) | 2026-07-29 | ZITOUNI Nidhal | `README.md` | Initial README documentation with project overview and setup. |
| [`df450a5`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/df450a5) | 2026-07-29 | ZITOUNI Nidhal | `confidence/verbalized.py` | Add regex extraction and normalization for verbalized confidence. |
| [`082384d`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/082384d) | 2026-07-29 | ZITOUNI Nidhal | `confidence/internal.py` | Implement initial token softmax logprobs internal confidence extraction. |
| [`255bf16`](https://github.com/ZitouniNidhal/vlm-confidence-benchmark/commit/255bf16) | 2026-07-29 | ZITOUNI Nidhal | Root Repository | Initial repository creation and commit. |

---

## 👥 Core Contributors Overview

| Contributor | GitHub Profile | Primary Module / Domain | Description of Contributions |
| :--- | :--- | :--- | :--- |
| **Nidhal Zitouni** | [@ZitouniNidhal](https://github.com/ZitouniNidhal) | **Core Engine & Architecture** | Created project framework, model interfaces (`SmolVLM`, `Qwen2-VL`), degradation pipeline, evaluation metrics, and Pytest validation suite across all listed commits. |
| **Adesh Kashyap** | [@adeshkashyap](https://github.com/adeshkashyap) | **Co-Author & Contributor** | Co-authored commits on merged benchmark calibration and documentation pull requests. |
| **Antigravity Assistant** | [@google](https://github.com/google) | **Pair Programming & Refactoring** | Co-authored commits enhancing calibration documentation, `BaseVLMModel` abstraction, predictive sequence entropy, fraction parsing, and non-interactive plotting configurations. |

---

## 💡 How to Become a Contributor

We welcome contributions from researchers, engineers, and open-source enthusiasts!

### Contribution Categories:
1. 🤖 **New VLM Models**: Implement interfaces in [`models/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/models) inheriting from `BaseVLMModel` and YAML configs in [`configs/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/configs).
2. 🌫️ **Visual Degradations**: Add custom image corruption functions under [`degradations/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/degradations).
3. 📊 **Evaluation & Calibration Metrics**: Enhance metric implementations in [`evaluation/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/evaluation).
4. 🧪 **Tests & Docs**: Expand test coverage in [`tests/`](file:///c:/Users/nidha/Desktop/github%20reposistory/vlm-confidence-benchmark/tests) or clarify project documentation.

### Adding Yourself to the Contributors Table:
1. Fork the repository and create a descriptive feature branch.
2. Add your commit summary, name, GitHub link, module area, and description to the tables above.
3. Run `pytest` to ensure all tests pass cleanly.
4. Submit a Pull Request.
