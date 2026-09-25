from src.evaluator import calculate_candidate_distribution


def test_candidate_distribution_basic():
    candidates = {
        "S1-A": {"S2-1", "S2-2"},
        "S1-B": {"S2-3", "S2-4", "S2-5"},
        "S1-C": {"S2-6"},
        "S1-D": {"S2-7", "S2-8", "S2-9", "S2-10"},
    }

    result = calculate_candidate_distribution(candidates)

    assert result["total_candidate_pairs"] == 10
    assert result["num_source1_entities"] == 4
    assert result["avg_candidates_per_s1"] == 2.5
    assert result["median_candidates_per_s1"] == 2.5
    assert result["max_candidates_per_s1"] == 4


def test_candidate_distribution_empty():
    result = calculate_candidate_distribution({})

    assert result["total_candidate_pairs"] == 0
    assert result["num_source1_entities"] == 0
    assert result["avg_candidates_per_s1"] == 0.0
    assert result["median_candidates_per_s1"] == 0.0
    assert result["p90_candidates_per_s1"] == 0.0
    assert result["p95_candidates_per_s1"] == 0.0
    assert result["p99_candidates_per_s1"] == 0.0
    assert result["max_candidates_per_s1"] == 0.0


def test_candidate_distribution_zero_candidate_entity():
    candidates = {
        "S1-A": set(),
        "S1-B": {"S2-1", "S2-2"},
        "S1-C": {"S2-3"},
    }

    result = calculate_candidate_distribution(candidates)

    assert result["total_candidate_pairs"] == 3
    assert result["num_source1_entities"] == 3
    assert result["avg_candidates_per_s1"] == 1.0
    assert result["max_candidates_per_s1"] == 2