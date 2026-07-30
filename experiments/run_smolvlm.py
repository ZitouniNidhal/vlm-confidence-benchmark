import argparse

from models.smolvlm import SmolVLM
from confidence.verbalized import extract_verbalized_confidence, normalize_confidence
from confidence.internal import compute_internal_confidence_from_probs


def main():
    parser = argparse.ArgumentParser(description="Run the SmolVLM benchmark.")
    parser.add_argument("--image", type=str, required=True, help="Path to an input image file.")
    parser.add_argument("--prompt", type=str, default="Please answer with a label and confidence percentage.")
    args = parser.parse_args()

    model = SmolVLM()
    answer_text, token_probs = model.generate(args.image, args.prompt)
    verbalized = extract_verbalized_confidence(answer_text)
    normalized_verbalized = normalize_confidence(verbalized)
    internal = compute_internal_confidence_from_probs(token_probs)

    print(f"Answer: {answer_text}")
    print(f"Verbalized confidence: {verbalized} -> {normalized_verbalized:.2f}")
    print(f"Internal confidence: {internal:.2f}")


if __name__ == "__main__":
    main()
