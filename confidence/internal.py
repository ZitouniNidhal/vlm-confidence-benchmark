import math
from typing import List
import torch


def compute_internal_confidence(logits: torch.Tensor, labels: list[int]) -> float:
    """
    Compute the average internal confidence score for a sequence of tokens.

    Args:
        logits (torch.Tensor): Tensor of shape [N, vocab_size] containing raw logits.
        labels (list[int]): List of token IDs corresponding to the generated output.

    Returns:
        float: Confidence score between 0.0 and 1.0.
    """
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor [N, vocab_size].")

    if len(labels) == 0:
        return 0.0

    probs = torch.nn.functional.softmax(logits, dim=-1)
    token_probs = probs[range(len(labels)), labels]
    confidence = token_probs.mean().item()
    return confidence


def compute_internal_confidence_from_probs(token_probs: list[float], aggregation: str = "mean") -> float:
    """
    Compute internal confidence directly from token-level probabilities.

    Args:
        token_probs: List of probabilities for generated tokens.
        aggregation: Aggregation method ("mean", "min", "geometric_mean").
    """
    if len(token_probs) == 0:
        return 0.0

    if aggregation == "min":
        return float(min(token_probs))
    elif aggregation == "geometric_mean":
        log_sum = sum(math.log(max(p, 1e-12)) for p in token_probs)
        return float(math.exp(log_sum / len(token_probs)))
    else:  # default "mean"
        return float(sum(token_probs) / len(token_probs))


def confidence_distribution(logits: torch.Tensor, labels: list[int]) -> list[float]:
    """Return the list of output token probabilities for given target labels."""
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor [N, vocab_size].")

    if len(labels) == 0:
        return []

    probs = torch.nn.functional.softmax(logits, dim=-1)
    token_probs = probs[range(len(labels)), labels]
    return token_probs.tolist()


def max_token_confidence(logits: torch.Tensor, labels: list[int]) -> float:
    """Return the maximum token probability among target labels."""
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor [N, vocab_size].")

    if len(labels) == 0:
        return 0.0

    probs = torch.nn.functional.softmax(logits, dim=-1)
    token_probs = probs[range(len(labels)), labels]
    return token_probs.max().item()


def compute_token_entropy(logits: torch.Tensor) -> float:
    """
    Compute average predictive entropy (in nats) across a sequence of token logits [N, vocab_size].
    Higher entropy indicates higher model uncertainty.
    """
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor [N, vocab_size].")
    if logits.shape[0] == 0:
        return 0.0

    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    entropy_per_step = -(probs * log_probs).sum(dim=-1)
    return entropy_per_step.mean().item()


def compute_sequence_entropy(token_probs: list[float]) -> float:
    """
    Compute binary predictive entropy (in bits) averaged over a sequence of top-token probabilities:
    H(p) = - [p log2(p) + (1-p) log2(1-p)].
    """
    if len(token_probs) == 0:
        return 0.0

    total_entropy = 0.0
    for p in token_probs:
        p_clamped = max(1e-12, min(1.0 - 1e-12, float(p)))
        h = -(p_clamped * math.log2(p_clamped) + (1.0 - p_clamped) * math.log2(1.0 - p_clamped))
        total_entropy += h

    return total_entropy / len(token_probs)
