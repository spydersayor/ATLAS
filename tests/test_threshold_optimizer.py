"""
Tests for ATLAS threshold optimization.
"""

import numpy as np

from src.threshold_optimizer import ThresholdOptimizer


def test_f05_calculation():
    optimizer = ThresholdOptimizer()

    result = optimizer.evaluate_threshold(
        y_true=np.array([1, 1, 0]),
        y_scores=np.array([0.95, 0.90, 0.80]),
        threshold=0.85,
    )

    assert np.isclose(result.precision, 1.0)
    assert np.isclose(result.recall, 1.0)
    assert np.isclose(result.f05, 1.0)

    print("[PASS] F0.5 calculation")


def test_confusion_matrix_counts():
    optimizer = ThresholdOptimizer()

    result = optimizer.evaluate_threshold(
        y_true=np.array([1, 1, 0, 0]),
        y_scores=np.array([0.95, 0.80, 0.70, 0.10]),
        threshold=0.75,
    )

    assert result.true_positives == 2
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.true_negatives == 2

    print("[PASS] Confusion matrix counts")


def test_threshold_optimization():
    optimizer = ThresholdOptimizer(
        min_threshold=0.10,
        max_threshold=0.90,
        num_thresholds=9,
    )

    y_true = np.array([
        1, 1, 1, 1,
        0, 0, 0, 0,
    ])

    y_scores = np.array([
        0.95, 0.90, 0.85, 0.80,
        0.30, 0.25, 0.20, 0.10,
    ])

    result = optimizer.optimize(
        y_true=y_true,
        y_scores=y_scores,
    )

    assert 0.10 <= result.threshold <= 0.90
    assert result.f05 > 0.0
    assert result.precision == 1.0
    assert result.recall == 1.0

    assert optimizer.get_best_result() == result

    print("[PASS] Threshold optimization")


def test_zero_division_is_safe():
    optimizer = ThresholdOptimizer()

    result = optimizer.evaluate_threshold(
        y_true=np.array([1, 1, 0, 0]),
        y_scores=np.array([0.10, 0.20, 0.30, 0.40]),
        threshold=0.90,
    )

    assert result.true_positives == 0
    assert result.false_positives == 0
    assert result.false_negatives == 2
    assert result.true_negatives == 2

    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f05 == 0.0

    print("[PASS] Zero-division handling")


def test_invalid_inputs_are_rejected():
    optimizer = ThresholdOptimizer()

    try:
        optimizer.evaluate_threshold(
            y_true=np.array([1, 0]),
            y_scores=np.array([0.9]),
            threshold=0.5,
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    try:
        optimizer.evaluate_threshold(
            y_true=np.array([1, 2]),
            y_scores=np.array([0.9, 0.1]),
            threshold=0.5,
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    try:
        optimizer.evaluate_threshold(
            y_true=np.array([1, 0]),
            y_scores=np.array([1.2, 0.1]),
            threshold=0.5,
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass

    print("[PASS] Invalid input validation")


def main():
    test_f05_calculation()
    test_confusion_matrix_counts()
    test_threshold_optimization()
    test_zero_division_is_safe()
    test_invalid_inputs_are_rejected()

    print("\nALL THRESHOLD OPTIMIZER TESTS PASSED")


if __name__ == "__main__":
    main()