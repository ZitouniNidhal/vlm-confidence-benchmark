from typing import Sequence


def match_label(predicted_text: str, target_label: str) -> bool:
    """
    Check whether the generated answer text matches the ground-truth target label.
    Normalizes underscores to spaces and compares case-insensitively.
    """
    if not predicted_text or not target_label:
        return False

    pred_norm = predicted_text.lower().replace("_", " ")
    target_norm = target_label.lower().replace("_", " ")

    # Direct substring or exact match
    return target_norm in pred_norm


def evaluate_predictions(answer_texts: Sequence[str], targets: Sequence[str]) -> list[int]:
    """Return a binary list [1 for correct, 0 for incorrect] for a batch of predictions."""
    if len(answer_texts) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    return [1 if match_label(p, t) else 0 for p, t in zip(answer_texts, targets)]


def accuracy_score(preds: Sequence[int | str], targets: Sequence[int | str]) -> float:
    """Compute classification accuracy for integer or string sequence predictions."""
    if len(preds) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if len(preds) == 0:
        return 0.0

    if len(preds) > 0 and isinstance(preds[0], str) and isinstance(targets[0], str):
        correct = sum(1 for p, t in zip(preds, targets) if match_label(str(p), str(t)))
    else:
        correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return float(correct / len(preds))

