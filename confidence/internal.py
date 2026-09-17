
import torch
from typing import List


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


import math


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
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor [N, vocab_size].")

    if len(labels) == 0:
        return []

    probs = torch.nn.functional.softmax(logits, dim=-1)
    token_probs = probs[range(len(labels)), labels]
    return token_probs.tolist()


def max_token_confidence(logits: torch.Tensor, labels: list[int]) -> float:
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor [N, vocab_size].")

    if len(labels) == 0:
        return 0.0

    probs = torch.nn.functional.softmax(logits, dim=-1)
    token_probs = probs[range(len(labels)), labels]
    return token_probs.max().item()
