import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from confidence.internal import compute_internal_confidence_from_probs
from confidence.verbalized import extract_verbalized_confidence, normalize_confidence
from models.qwen2vl import Qwen2VLM


def main():
    parser = argparse.ArgumentParser(description="Run the Qwen2-VL benchmark.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image file.")
    parser.add_argument("--prompt", type=str, default="Please answer with a label and confidence percentage.")
    parser.add_argument("--mock", action="store_true", help="Use mock model for testing.")
    args = parser.parse_args()

    model = Qwen2VLM(mock=args.mock)

    answer_text, token_probs = model.generate(args.image, args.prompt)
    verbalized = extract_verbalized_confidence(answer_text)
    normalized_verbalized = normalize_confidence(verbalized)
    internal = compute_internal_confidence_from_probs(token_probs)

    print(f"Answer: {answer_text}")
    print(f"Verbalized confidence: {verbalized} -> {normalized_verbalized:.2f}")
    print(f"Internal confidence: {internal:.2f}")


if __name__ == "__main__":
    main()
