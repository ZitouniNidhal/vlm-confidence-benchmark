from typing import Sequence


def accuracy_score(preds: Sequence[int], targets: Sequence[int]) -> float:
    """Compute simple classification accuracy."""
    if len(preds) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    if len(preds) == 0:
        return 0.0
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / len(preds)
