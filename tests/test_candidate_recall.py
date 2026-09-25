from src.evaluator import calculate_candidate_recall


def test_full_candidate_recall():
    ground_truth = {
        "S1-A": {"S2-1", "S3-7"},
        "S1-B": {"S2-2"},
    }

    candidates = {
        "S1-A": {"S2-1", "S3-7"},
        "S1-B": {"S2-2"},
    }

    result = calculate_candidate_recall(
        ground_truth,
        candidates,
    )

    assert result["total_true_matches"] == 3
    assert result["retrieved_true_matches"] == 3
    assert result["candidate_recall"] == 1.0


def test_partial_candidate_recall():
    ground_truth = {
        "S1-A": {"S2-1", "S3-7"},
    }

    candidates = {
        "S1-A": {"S2-1"},
    }

    result = calculate_candidate_recall(
        ground_truth,
        candidates,
    )

    assert result["total_true_matches"] == 2
    assert result["retrieved_true_matches"] == 1
    assert result["candidate_recall"] == 0.5


def test_zero_candidate_recall():
    ground_truth = {
        "S1-A": {"S2-1", "S3-7"},
    }

    candidates = {
        "S1-A": {"S2-999"},
    }

    result = calculate_candidate_recall(
        ground_truth,
        candidates,
    )

    assert result["total_true_matches"] == 2
    assert result["retrieved_true_matches"] == 0
    assert result["candidate_recall"] == 0.0


def test_zero_match_entities_do_not_reduce_recall():
    ground_truth = {
        "S1-A": {"S2-1"},
        "S1-B": set(),
    }

    candidates = {
        "S1-A": {"S2-1"},
        "S1-B": {"S2-999"},
    }

    result = calculate_candidate_recall(
        ground_truth,
        candidates,
    )

    assert result["total_true_matches"] == 1
    assert result["retrieved_true_matches"] == 1
    assert result["candidate_recall"] == 1.0


def test_missing_s1_candidate_set():
    ground_truth = {
        "S1-A": {"S2-1"},
        "S1-B": {"S2-2"},
    }

    candidates = {
        "S1-A": {"S2-1"},
    }

    result = calculate_candidate_recall(
        ground_truth,
        candidates,
    )

    assert result["total_true_matches"] == 2
    assert result["retrieved_true_matches"] == 1
    assert result["candidate_recall"] == 0.5