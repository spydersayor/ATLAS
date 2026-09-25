import pytest

from src.evaluator import calculate_binary_metrics


def test_binary_metrics_basic_case():
    y_true = [1, 0, 1, 0]
    y_scores = [0.95, 0.20, 0.80, 0.70]

    result = calculate_binary_metrics(
        y_true,
        y_scores,
        threshold=0.5,
    )

    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["tn"] == 1

    assert result["precision"] == 2 / 3
    assert result["recall"] == 1.0

    expected_f05 = (
        1.25
        * (2 / 3)
        * 1.0
        / ((0.25 * (2 / 3)) + 1.0)
    )

    assert result["f0.5"] == expected_f05


def test_binary_metrics_threshold_is_inclusive():
    y_true = [1, 0]
    y_scores = [0.5, 0.5]

    result = calculate_binary_metrics(
        y_true,
        y_scores,
        threshold=0.5,
    )

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["tn"] == 0


def test_binary_metrics_all_correct():
    y_true = [1, 1, 0, 0]
    y_scores = [0.9, 0.8, 0.2, 0.1]

    result = calculate_binary_metrics(
        y_true,
        y_scores,
        threshold=0.5,
    )

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 2

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f0.5"] == 1.0


def test_binary_metrics_all_false_positives():
    y_true = [0, 0, 0]
    y_scores = [0.9, 0.8, 0.7]

    result = calculate_binary_metrics(
        y_true,
        y_scores,
        threshold=0.5,
    )

    assert result["tp"] == 0
    assert result["fp"] == 3
    assert result["fn"] == 0
    assert result["tn"] == 0

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f0.5"] == 0.0


def test_binary_metrics_all_false_negatives():
    y_true = [1, 1, 1]
    y_scores = [0.1, 0.2, 0.3]

    result = calculate_binary_metrics(
        y_true,
        y_scores,
        threshold=0.5,
    )

    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 3
    assert result["tn"] == 0

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f0.5"] == 0.0


def test_binary_metrics_all_true_negatives():
    y_true = [0, 0, 0]
    y_scores = [0.1, 0.2, 0.3]

    result = calculate_binary_metrics(
        y_true,
        y_scores,
        threshold=0.5,
    )

    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 3

    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f0.5"] == 0.0


def test_binary_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        calculate_binary_metrics(
            [1, 0, 1],
            [0.9, 0.2],
            threshold=0.5,
        )


def test_binary_metrics_rejects_invalid_labels():
    with pytest.raises(ValueError):
        calculate_binary_metrics(
            [1, 2, 0],
            [0.9, 0.8, 0.1],
            threshold=0.5,
        )


def test_binary_metrics_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        calculate_binary_metrics(
            [1, 0],
            [0.9, 0.1],
            threshold=1.5,
        )


def test_binary_metrics_accepts_numpy_arrays():
    import numpy as np

    y_true = np.array([1, 0, 1])
    y_scores = np.array([0.9, 0.2, 0.8])

    result = calculate_binary_metrics(
        y_true,
        y_scores,
        threshold=0.5,
    )

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["tn"] == 1
    