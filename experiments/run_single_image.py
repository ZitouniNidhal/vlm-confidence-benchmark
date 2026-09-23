import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from confidence.internal import compute_internal_confidence_from_probs, compute_sequence_entropy  # noqa: E402
from confidence.verbalized import extract_verbalized_confidence, normalize_confidence  # noqa: E402
from degradations.pipeline import apply_degradations  # noqa: E402
from evaluation.accuracy import match_label  # noqa: E402
from models.qwen2vl import Qwen2VLM  # noqa: E402
from models.smolvlm import SmolVLM  # noqa: E402


def get_model(model_type: str, mock: bool = False):
    if model_type.lower() == "qwen2vl":
        return Qwen2VLM(mock=mock)
    elif model_type.lower() == "smolvlm":
        return SmolVLM(mock=mock)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_single_image_eval(
    image_path: str,
    model_type: str = "smolvlm",
    mock: bool = False,
    degradation: str | None = "blur",
    severity: str = "mid",
    ground_truth: str | None = None,
    output_plot: str | None = None,
):
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image path does not exist: {path}")

    img_clean = Image.open(path).convert("RGB")

    # Apply degradation if specified
    if degradation and degradation.lower() != "none":
        img_eval = apply_degradations(img_clean, operations=[(degradation, severity)])
        deg_str = f"{degradation} ({severity})"
    else:
        img_eval = img_clean
        deg_str = "clean"

    print("\n--- VLM Single-Image Confidence Evaluation ---")
    print(f"Model: {model_type} (mock={mock})")
    print(f"Image: {image_path}")
    print(f"Degradation: {deg_str}")
    if ground_truth:
        print(f"Ground Truth Label: {ground_truth}")

    model = get_model(model_type, mock=mock)
    prompt = 'Please answer with a label and a confidence percentage. Example: "I am 80% confident this is pizza."'

    answer_text, token_probs = model.generate(img_eval, prompt=prompt)

    verb_raw = extract_verbalized_confidence(answer_text)
    verb_norm = normalize_confidence(verb_raw)
    int_norm = compute_internal_confidence_from_probs(token_probs)
    seq_entropy = compute_sequence_entropy(token_probs)

    print("\nResults:")
    print(f"  Raw Output: '{answer_text}'")
    print(f"  Verbalized Confidence: {verb_norm * 100:.1f}%")
    print(f"  Internal Confidence:   {int_norm * 100:.1f}%")
    print(f"  Sequence Entropy:      {seq_entropy:.4f} bits")

    if ground_truth:
        is_correct = match_label(answer_text, ground_truth)
        print(f"  Match Ground Truth:    {'[CORRECT]' if is_correct else '[INCORRECT]'}")

    if output_plot:
        out_p = Path(output_plot)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(img_clean)
        axes[0].set_title("Clean Input Image")
        axes[0].axis("off")

        axes[1].imshow(img_eval)
        axes[1].set_title(f"Degraded: {deg_str}")
        axes[1].axis("off")

        plt.suptitle(
            f"Model: {model_type} | Verb Conf: {verb_norm*100:.0f}% | Int Conf: {int_norm*100:.0f}%",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(out_p, dpi=300)
        plt.close(fig)
        print(f"\nVisualization saved to {out_p}")

    return {
        "answer_text": answer_text,
        "verbalized_confidence": verb_norm,
        "internal_confidence": int_norm,
        "sequence_entropy": seq_entropy,
    }


def main():
    parser = argparse.ArgumentParser(description="Single image VLM confidence evaluation.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--model", type=str, default="smolvlm", choices=["smolvlm", "qwen2vl"])
    parser.add_argument("--mock", action="store_true", help="Use mock model.")
    parser.add_argument("--degradation", type=str, default="blur", help="Degradation type.")
    parser.add_argument("--severity", type=str, default="mid", choices=["low", "mid", "high"])
    parser.add_argument("--ground_truth", type=str, default=None, help="Optional ground truth label.")
    parser.add_argument("--output_plot", type=str, default=None, help="Path to save side-by-side plot.")
    args = parser.parse_args()

    run_single_image_eval(
        image_path=args.image,
        model_type=args.model,
        mock=args.mock,
        degradation=args.degradation,
        severity=args.severity,
        ground_truth=args.ground_truth,
        output_plot=args.output_plot,
    )


if __name__ == "__main__":
    main()
