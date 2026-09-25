from src.evaluator import classify_candidate_errors


def test_classifies_all_error_types():
    ground_truth = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),  # True positive
        ("S1-1", "S2-11"),  # False positive
        ("S1-1", "S3-20"),  # False negative with candidate
        ("S1-2", "S2-999"), # False positive
    ]

    predictions = {
        ("S1-1", "S2-10"): 1,
        ("S1-1", "S2-11"): 1,
        ("S1-1", "S3-20"): 0,
        ("S1-2", "S2-999"): 1,
    }

    result = classify_candidate_errors(
        candidate_pairs,
        ground_truth,
        predictions,
    )

    assert result["counts"] == {
        "true_positives": 1,
        "false_positives": 2,
        "false_negatives_with_candidate": 1,
        "blocking_failures": 1,
    }

    assert set(result["true_positives"]) == {
        ("S1-1", "S2-10")
    }

    assert set(result["false_positives"]) == {
        ("S1-1", "S2-11"),
        ("S1-2", "S2-999"),
    }

    assert set(result["false_negatives_with_candidate"]) == {
        ("S1-1", "S3-20")
    }

    assert set(result["blocking_failures"]) == {
        ("S1-2", "S2-30")
    }


def test_missing_true_match_is_blocking_failure_not_classification_failure():
    ground_truth = {
        "S1-1": {"S2-10"},
    }

    candidate_pairs = []

    predictions = {}

    result = classify_candidate_errors(
        candidate_pairs,
        ground_truth,
        predictions,
    )

    assert result["counts"]["blocking_failures"] == 1

    assert result["blocking_failures"] == [
        ("S1-1", "S2-10")
    ]

    assert result["counts"]["false_negatives_with_candidate"] == 0


def test_no_blocking_failure_when_all_true_matches_are_retrieved():
    ground_truth = {
        "S1-1": {"S2-10", "S3-20"},
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
        ("S1-1", "S3-20"),
        ("S1-1", "S2-999"),
    ]

    predictions = {
        ("S1-1", "S2-10"): 1,
        ("S1-1", "S3-20"): 1,
        ("S1-1", "S2-999"): 0,
    }

    result = classify_candidate_errors(
        candidate_pairs,
        ground_truth,
        predictions,
    )

    assert result["counts"]["blocking_failures"] == 0
    assert result["counts"]["true_positives"] == 2
    assert result["counts"]["false_positives"] == 0
    assert result["counts"]["false_negatives_with_candidate"] == 0


def test_missing_prediction_raises_error():
    ground_truth = {
        "S1-1": {"S2-10"},
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
    ]

    predictions = {}

    try:
        classify_candidate_errors(
            candidate_pairs,
            ground_truth,
            predictions,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Missing prediction" in str(exc)


def test_invalid_prediction_raises_error():
    ground_truth = {
        "S1-1": {"S2-10"},
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
    ]

    predictions = {
        ("S1-1", "S2-10"): 2,
    }

    try:
        classify_candidate_errors(
            candidate_pairs,
            ground_truth,
            predictions,
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "0 and 1" in str(exc)


def test_no_match_entity_positive_candidate_is_false_positive():
    ground_truth = {
        "S1-1": set(),
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
    ]

    predictions = {
        ("S1-1", "S2-10"): 1,
    }

    result = classify_candidate_errors(
        candidate_pairs,
        ground_truth,
        predictions,
    )

    assert result["counts"]["true_positives"] == 0
    assert result["counts"]["false_positives"] == 1
    assert result["counts"]["false_negatives_with_candidate"] == 0
    assert result["counts"]["blocking_failures"] == 0

    assert result["false_positives"] == [
        ("S1-1", "S2-10")
    ]