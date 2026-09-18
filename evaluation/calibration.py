"""Calibration evaluation metrics including Brier score, Expected Calibration Error (ECE), and reliability curve data."""

from typing import Sequence



def brier_score(confidences: Sequence[float], targets: Sequence[int]) -> float:
    """Compute Brier score for binary outcomes.

    Args:
        confidences: Confidence values between 0 and 1.
        targets: Binary labels where 1 indicates the positive event.
    """
    if len(confidences) != len(targets):
        raise ValueError("Confidences and targets must have the same length.")
    if len(confidences) == 0:
        return 0.0

    total = 0.0
    for c, t in zip(confidences, targets):
        total += (float(c) - float(t)) ** 2
    return total / len(confidences)


def expected_calibration_error(confidences: Sequence[float], targets: Sequence[int], n_bins: int = 10) -> float:
    """Compute expected calibration error (ECE) for binary predictions."""
    if len(confidences) != len(targets):
        raise ValueError("Confidences and targets must have the same length.")
    if len(confidences) == 0:
        return 0.0

    bins = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins
    bin_acc_sum = [0.0] * n_bins

    for c, t in zip(confidences, targets):
        c_val = max(0.0, min(1.0, float(c)))
        index = min(int(c_val * n_bins), n_bins - 1)
        bins[index] += 1
        bin_conf_sum[index] += c_val
        bin_acc_sum[index] += float(t)

    ece = 0.0
    total = len(confidences)
    for count, conf_sum, acc_sum in zip(bins, bin_conf_sum, bin_acc_sum):
        if count == 0:
            continue
        avg_conf = conf_sum / count
        avg_acc = acc_sum / count
        ece += (count / total) * abs(avg_conf - avg_acc)
    return float(ece)


def calibration_curve_data(confidences: Sequence[float], targets: Sequence[int], n_bins: int = 10) -> tuple[list[float], list[float]]:
    """
    Compute average confidence and accuracy per bin for reliability curves.
    Returns (bin_confidences, bin_accuracies).
    """
    if len(confidences) != len(targets) or len(confidences) == 0:
        return [], []

    bins = [0] * n_bins
    bin_conf_sum = [0.0] * n_bins
    bin_acc_sum = [0.0] * n_bins

    for c, t in zip(confidences, targets):
        c_val = max(0.0, min(1.0, float(c)))
        index = min(int(c_val * n_bins), n_bins - 1)
        bins[index] += 1
        bin_conf_sum[index] += c_val
        bin_acc_sum[index] += float(t)

    bin_confs, bin_accs = [], []
    for count, conf_sum, acc_sum in zip(bins, bin_conf_sum, bin_acc_sum):
        if count > 0:
            bin_confs.append(conf_sum / count)
            bin_accs.append(acc_sum / count)

    return bin_confs, bin_accs

