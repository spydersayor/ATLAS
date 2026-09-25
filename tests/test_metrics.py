from src.evaluator import (
    calculate_entity_metrics,
    calculate_macro_metrics,
)


def test_perfect_match():
    metrics = calculate_entity_metrics(
        {"S2-1", "S3-7"},
        {"S2-1", "S3-7"},
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f0.5"] == 1.0


def test_partial_match():
    metrics = calculate_entity_metrics(
        {"S2-1", "S3-7"},
        {"S2-1"},
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert round(metrics["f0.5"], 6) == round(0.8333333333333334, 6)


def test_false_positive():
    metrics = calculate_entity_metrics(
        {"S2-1"},
        {"S2-999"},
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f0.5"] == 0.0


def test_true_match_with_extra_prediction():
    metrics = calculate_entity_metrics(
        {"S2-1"},
        {"S2-1", "S2-999"},
    )

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert round(metrics["f0.5"], 6) == round(0.5555555555555556, 6)


def test_zero_true_zero_prediction():
    metrics = calculate_entity_metrics(
        set(),
        set(),
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f0.5"] == 1.0


def test_zero_true_nonzero_prediction():
    metrics = calculate_entity_metrics(
        set(),
        {"S2-999"},
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f0.5"] == 0.0


def test_nonzero_true_zero_prediction():
    metrics = calculate_entity_metrics(
        {"S2-1"},
        set(),
    )

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f0.5"] == 0.0


def test_macro_metrics():
    ground_truth = {
        "S1-A": {"S2-1"},
        "S1-B": {"S2-2", "S3-2"},
        "S1-C": set(),
    }

    predictions = {
        "S1-A": {"S2-1"},
        "S1-B": {"S2-2"},
        "S1-C": set(),
    }

    metrics = calculate_macro_metrics(
        ground_truth,
        predictions,
    )

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.8333333333333334
    assert round(metrics["f0.5"], 6) == round(
    (1.0 + 0.8333333333333334 + 1.0) / 3,
    6,
    )