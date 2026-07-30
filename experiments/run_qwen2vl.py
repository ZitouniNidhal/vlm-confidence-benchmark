import argparse
from pathlib import Path

from models.qwen2vl import Qwen2VLM
from confidence.verbalized import extract_verbalized_confidence, normalize_confidence
from confidence.internal import compute_internal_confidence


def main():
    parser = argparse.ArgumentParser(description="Run the Qwen2-VL benchmark.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image file.")
    parser.add_argument("--prompt", type=str, default="Please answer with a label and confidence percentage.")
    args = parser.parse_args()

    model = Qwen2VLM()
    answer_text, token_probs = model.generate(args.image, args.prompt)
    verbalized = extract_verbalized_confidence(answer_text)
    internal = compute_internal_confidence(
        logits=None,
        labels=[],
    )

    print(f"Answer: {answer_text}")
    print(f"Verbalized confidence: {verbalized}")
    print(f"Internal confidence: {internal}")


if __name__ == "__main__":
    main()
