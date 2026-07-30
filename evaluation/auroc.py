from typing import Sequence

from sklearn.metrics import roc_auc_score


def error_detection_auroc(confidences: Sequence[float], labels: Sequence[int]) -> float:
    """Compute AUROC for using confidence values to detect incorrect predictions.

    Args:
        confidences: Higher means more confident.
        labels: Binary labels where 1 indicates an error (incorrect prediction) and 0 indicates correct.
    """
    if len(confidences) != len(labels):
        raise ValueError("Confidences and labels must have the same length.")
    if len(confidences) == 0:
        return 0.0

    # For error detection, invert confidence so higher scores mean more likely to be wrong.
    error_scores = [1.0 - float(c) for c in confidences]
    return float(roc_auc_score(labels, error_scores))
