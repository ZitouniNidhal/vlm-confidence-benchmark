
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

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Extract probabilities for the generated tokens
    token_probs = probs[range(len(labels)), labels]

    # Average probability across tokens
    confidence = token_probs.mean().item()
    return confidence


def confidence_distribution(logits: torch.Tensor, labels: list[int]) -> list[float]:
   
    if logits.ndim != 2:
        raise ValueError("Logits must be a 2D tensor [N, vocab_size].")

    if len(labels) == 0:
        return []

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Extract probabilities for the generated tokens
    token_probs = probs[range(len(labels)), labels]

    return token_probs.tolist()

def max_token_confidence(logits: torch.Tensor, labels: list[int]) -> float:
   
    probs = torch.nn.functional.softmax(logits, dim=-1)
    token_probs = probs[range(len(labels)), labels]
    return token_probs.max().item()
