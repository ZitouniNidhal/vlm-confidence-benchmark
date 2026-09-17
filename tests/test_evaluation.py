import pytest
from evaluation.accuracy import accuracy_score, match_label, evaluate_predictions
from evaluation.calibration import expected_calibration_error, brier_score, calibration_curve_data
from evaluation.auroc import error_detection_auroc


def test_match_label():
    assert match_label("I am 90% confident this is pizza.", "pizza") is True
    assert match_label("This looks like french fries.", "french_fries") is True
    assert match_label("This is a hamburger", "tacos") is False
    assert match_label("", "pizza") is False


def test_evaluate_predictions():
    answers = ["It is pizza", "It is sushi", "It is burger"]
    targets = ["pizza", "tacos", "burger"]
    binary = evaluate_predictions(answers, targets)
    assert binary == [1, 0, 1]


def test_accuracy_score():
    assert accuracy_score([1, 0, 1, 1], [1, 1, 1, 1]) == 0.75
    assert accuracy_score(["pizza", "sushi"], ["pizza", "tacos"]) == 0.5
    assert accuracy_score([], []) == 0.0


def test_brier_score():
    confidences = [0.9, 0.8, 0.2]
    targets = [1, 1, 0]
    score = brier_score(confidences, targets)
    assert 0.0 <= score <= 1.0


def test_expected_calibration_error():
    confidences = [0.9, 0.8, 0.7, 0.2]
    targets = [1, 1, 1, 0]
    ece = expected_calibration_error(confidences, targets, n_bins=5)
    assert 0.0 <= ece <= 1.0

    confs, accs = calibration_curve_data(confidences, targets, n_bins=5)
    assert len(confs) == len(accs)


def test_error_detection_auroc():
    # High confidence on correct (error=0), low confidence on error (error=1)
    confidences = [0.9, 0.85, 0.1, 0.2]
    errors = [0, 0, 1, 1]
    auroc = error_detection_auroc(confidences, errors)
    assert auroc == 1.0

    # Single class fallback
    assert error_detection_auroc([0.9, 0.8], [0, 0]) == 0.5
