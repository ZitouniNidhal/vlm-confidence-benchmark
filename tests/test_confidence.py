import torch
from confidence.verbalized import (
    extract_verbalized_confidence,
    normalize_confidence,
    has_confidence_statement,
)
from confidence.internal import (
    compute_internal_confidence,
    compute_internal_confidence_from_probs,
    confidence_distribution,
    max_token_confidence,
    compute_token_entropy,
    compute_sequence_entropy,
)


def test_extract_verbalized_confidence():
    assert extract_verbalized_confidence("I am 85% confident this is pizza.") == 85
    assert extract_verbalized_confidence("Confidence: 90 percent.") == 90
    assert extract_verbalized_confidence("My confidence is 75.") == 75
    assert extract_verbalized_confidence("Confidence of 0.82") == 82
    assert extract_verbalized_confidence("No numbers here.") is None
    assert extract_verbalized_confidence("") is None


def test_extract_verbalized_confidence_fractions_and_qualitative():
    # Fractions
    assert extract_verbalized_confidence("Confidence is 9/10.") == 90
    assert extract_verbalized_confidence("I am 85 out of 100 sure.") == 85
    # Qualitative
    assert extract_verbalized_confidence("I have high confidence in this prediction.") == 90
    assert extract_verbalized_confidence("I am moderately confident.") == 60
    assert extract_verbalized_confidence("I am uncertain about this image.") == 20


def test_normalize_confidence():
    assert normalize_confidence(85) == 0.85
    assert normalize_confidence(100) == 1.0
    assert normalize_confidence(0) == 0.0
    assert normalize_confidence(None) == 0.0


def test_has_confidence_statement():
    assert has_confidence_statement("I am 90% sure") is True
    assert has_confidence_statement("Just pizza") is False


def test_compute_internal_confidence_tensor():
    # 3 tokens, vocab size 5
    logits = torch.tensor([
        [2.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 4.0],
    ])
    labels = [0, 1, 4]
    conf = compute_internal_confidence(logits, labels)
    assert 0.0 <= conf <= 1.0
    assert conf > 0.5

    dist = confidence_distribution(logits, labels)
    assert len(dist) == 3

    max_c = max_token_confidence(logits, labels)
    assert max_c >= conf


def test_compute_internal_confidence_from_probs():
    probs = [0.9, 0.8, 0.7]
    assert abs(compute_internal_confidence_from_probs(probs, "mean") - 0.8) < 1e-5
    assert abs(compute_internal_confidence_from_probs(probs, "min") - 0.7) < 1e-5
    assert compute_internal_confidence_from_probs([], "mean") == 0.0


def test_entropy_computations():
    logits = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0],  # uniform logits -> highest entropy
        [10.0, 0.0, 0.0, 0.0, 0.0], # peaked logit -> lowest entropy
    ])
    token_entropy = compute_token_entropy(logits)
    assert token_entropy > 0.0

    probs_certain = [0.99, 0.99]
    probs_uncertain = [0.5, 0.5]
    ent_low = compute_sequence_entropy(probs_certain)
    ent_high = compute_sequence_entropy(probs_uncertain)
    assert ent_high > ent_low
