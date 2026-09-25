from src.evaluator import calculate_retrieval_report


def test_retrieval_report_counts_retrieved_and_missed_matches():
    ground_truth = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
        "S1-3": set(),
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
        ("S1-1", "S3-20"),
        ("S1-2", "S2-999"),
    ]

    report = calculate_retrieval_report(ground_truth, candidate_pairs)

    assert report["total_true_matches"] == 3
    assert report["true_matches_retrieved"] == 2
    assert report["true_matches_missed"] == 1
    assert report["candidate_recall"] == 2 / 3


def test_retrieval_report_identifies_completely_missed_entities():
    ground_truth = {
        "S1-1": {"S2-10"},
        "S1-2": {"S3-20"},
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
    ]

    report = calculate_retrieval_report(ground_truth, candidate_pairs)

    assert report["s1_entities_with_completely_missed_true_matches"] == [
        "S1-2"
    ]


def test_retrieval_report_calculates_candidate_distribution():
    ground_truth = {
        "S1-1": {"S2-10"},
        "S1-2": {"S2-20"},
        "S1-3": set(),
        "S1-4": {"S3-30"},
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
        ("S1-1", "S2-11"),
        ("S1-2", "S2-20"),
        ("S1-2", "S2-21"),
        ("S1-2", "S2-22"),
        ("S1-4", "S3-30"),
    ]

    report = calculate_retrieval_report(ground_truth, candidate_pairs)

    assert report["candidate_count_mean"] == 1.5
    assert report["candidate_count_median"] == 1.5
    assert report["candidate_count_max"] == 3
    assert report["total_candidate_pairs"] == 6


def test_retrieval_report_separates_s2_and_s3():
    ground_truth = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
        "S1-3": {"S3-40"},
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
        ("S1-1", "S3-20"),
        ("S1-2", "S2-999"),
        ("S1-3", "S3-999"),
    ]

    report = calculate_retrieval_report(ground_truth, candidate_pairs)

    assert report["s2"]["true_matches"] == 2
    assert report["s2"]["retrieved"] == 1
    assert report["s2"]["missed"] == 1

    assert report["s3"]["true_matches"] == 2
    assert report["s3"]["retrieved"] == 1
    assert report["s3"]["missed"] == 1


def test_retrieval_report_tracks_no_match_entities():
    ground_truth = {
        "S1-1": set(),
        "S1-2": {"S2-10"},
        "S1-3": set(),
    }

    candidate_pairs = [
        ("S1-1", "S2-999"),
        ("S1-2", "S2-10"),
    ]

    report = calculate_retrieval_report(ground_truth, candidate_pairs)

    assert report["no_match"]["s1_entities"] == 2
    assert report["no_match"]["s1_entities_with_candidates"] == 1


def test_retrieval_report_tracks_multi_match_entities():
    ground_truth = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
        ("S1-1", "S3-20"),
        ("S1-2", "S2-30"),
    ]

    report = calculate_retrieval_report(ground_truth, candidate_pairs)

    assert report["multi_match"]["s1_entities"] == 1
    assert report["multi_match"]["true_matches"] == 2
    assert report["multi_match"]["fully_retrieved_entities"] == 1


def test_retrieval_report_handles_no_true_matches():
    ground_truth = {
        "S1-1": set(),
        "S1-2": set(),
    }

    candidate_pairs = [
        ("S1-1", "S2-10"),
        ("S1-2", "S3-20"),
    ]

    report = calculate_retrieval_report(ground_truth, candidate_pairs)

    assert report["total_true_matches"] == 0
    assert report["true_matches_retrieved"] == 0
    assert report["true_matches_missed"] == 0
    assert report["candidate_recall"] == 1.0
