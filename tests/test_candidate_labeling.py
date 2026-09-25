from src.evaluator import label_candidate_pairs


def test_positive_candidate_is_labeled_one():
    ground_truth = {
        "S1-A": {"S2-1", "S3-1"},
    }

    candidates = [
        ("S1-A", "S2-1"),
    ]

    result = label_candidate_pairs(
        candidates,
        ground_truth,
    )

    assert result == [
        {
            "source1_entity_id": "S1-A",
            "candidate_entity_id": "S2-1",
            "label": 1,
        }
    ]


def test_non_matching_candidate_is_labeled_zero():
    ground_truth = {
        "S1-A": {"S2-1"},
    }

    candidates = [
        ("S1-A", "S2-999"),
    ]

    result = label_candidate_pairs(
        candidates,
        ground_truth,
    )

    assert result[0]["label"] == 0


def test_multiple_true_matches_are_labeled_correctly():
    ground_truth = {
        "S1-A": {"S2-1", "S3-1"},
    }

    candidates = [
        ("S1-A", "S2-1"),
        ("S1-A", "S3-1"),
        ("S1-A", "S2-999"),
    ]

    result = label_candidate_pairs(
        candidates,
        ground_truth,
    )

    labels = [row["label"] for row in result]

    assert labels == [1, 1, 0]


def test_no_match_entity_candidates_are_all_zero():
    ground_truth = {
        "S1-A": set(),
    }

    candidates = [
        ("S1-A", "S2-1"),
        ("S1-A", "S3-1"),
    ]

    result = label_candidate_pairs(
        candidates,
        ground_truth,
    )

    assert [row["label"] for row in result] == [0, 0]


def test_missing_true_match_is_not_created_as_negative():
    ground_truth = {
        "S1-A": {"S2-1", "S3-1"},
    }

    candidates = [
        ("S1-A", "S2-1"),
    ]

    result = label_candidate_pairs(
        candidates,
        ground_truth,
    )

    assert len(result) == 1
    assert result[0]["label"] == 1

    # S3-1 is absent from candidates.
    # It must NOT appear as an artificial label=0 row.
    candidate_ids = {
        row["candidate_entity_id"]
        for row in result
    }

    assert "S3-1" not in candidate_ids


def test_s2_and_s3_candidates_are_supported():
    ground_truth = {
        "S1-A": {"S2-1", "S3-1"},
    }

    candidates = [
        ("S1-A", "S2-1"),
        ("S1-A", "S3-1"),
    ]

    result = label_candidate_pairs(
        candidates,
        ground_truth,
    )

    assert [row["label"] for row in result] == [1, 1]