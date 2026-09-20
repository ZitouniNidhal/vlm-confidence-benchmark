"""Calibration evaluation metrics including Brier score, Expected Calibration Error (ECE), Adaptive ECE, and Temperature Scaling."""

import math
from typing import Sequence
import numpy as np
from scipy.optimize import minimize


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


def adaptive_expected_calibration_error(confidences: Sequence[float], targets: Sequence[int], n_bins: int = 10) -> float:
    """
    Compute Adaptive Expected Calibration Error (Adaptive ECE) using equal-frequency (quantile) binning.

    Args:
        confidences: Confidence values between 0 and 1.
        targets: Binary outcomes (1 for correct, 0 for incorrect).
        n_bins: Number of equal-frequency quantile bins.

    Returns:
        float: Adaptive ECE score between 0.0 and 1.0.
    """
    if len(confidences) != len(targets):
        raise ValueError("Confidences and targets must have the same length.")
    if len(confidences) == 0:
        return 0.0

    confs = np.clip(np.array(confidences, dtype=float), 0.0, 1.0)
    targs = np.array(targets, dtype=float)
    total = len(confs)

    sorted_indices = np.argsort(confs)
    sorted_confs = confs[sorted_indices]
    sorted_targs = targs[sorted_indices]

    bin_chunks_confs = np.array_split(sorted_confs, min(n_bins, total))
    bin_chunks_targs = np.array_split(sorted_targs, min(n_bins, total))

    aece = 0.0
    for b_confs, b_targs in zip(bin_chunks_confs, bin_chunks_targs):
        if len(b_confs) == 0:
            continue
        avg_conf = float(np.mean(b_confs))
        avg_acc = float(np.mean(b_targs))
        weight = len(b_confs) / total
        aece += weight * abs(avg_conf - avg_acc)

    return float(aece)


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


def apply_temperature_scaling(confidences: Sequence[float], temperature: float = 1.0) -> list[float]:
    """
    Apply temperature scaling to confidence probabilities:
    logit = log(p / (1 - p)), logit_scaled = logit / temperature, p_scaled = sigmoid(logit_scaled).
    """
    if temperature <= 0:
        raise ValueError("Temperature parameter T must be positive.")

    scaled_probs = []
    for c in confidences:
        c_clamped = max(1e-7, min(1.0 - 1e-7, float(c)))
        logit = math.log(c_clamped / (1.0 - c_clamped))
        scaled_logit = logit / temperature
        scaled_p = 1.0 / (1.0 + math.exp(-scaled_logit))
        scaled_probs.append(float(scaled_p))
    return scaled_probs


def fit_temperature_scaling(confidences: Sequence[float], targets: Sequence[int]) -> float:
    """
    Find optimal temperature T > 0 that minimizes Binary Cross Entropy log loss on (confidences, targets).
    """
    if len(confidences) != len(targets) or len(confidences) == 0:
        return 1.0

    confs = np.clip(np.array(confidences, dtype=float), 1e-7, 1.0 - 1e-7)
    targs = np.array(targets, dtype=float)
    logits = np.log(confs / (1.0 - confs))

    def nll_loss(T_arr):
        T = T_arr[0]
        if T <= 1e-6:
            return 1e9
        scaled_logits = logits / T
        probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        loss = -np.mean(targs * np.log(probs + 1e-12) + (1.0 - targs) * np.log(1.0 - probs + 1e-12))
        return loss

    res = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 10.0)], method="L-BFGS-B")
    return float(res.x[0]) if res.success else 1.0
