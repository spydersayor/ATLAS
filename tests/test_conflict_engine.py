"""
Tests for ATLAS conflict engine.
"""

import numpy as np
import pandas as pd

from src.conflict_engine import ConflictEngine


def make_candidates():
    """
    Create a small synthetic candidate set covering:
    - multiple matches
    - single match
    - zero match
    - S2 and S3 candidates
    """
    return pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-A",
                "S1-A",
                "S1-A",
                "S1-B",
                "S1-B",
                "S1-C",
                "S1-D",
            ],
            "candidate_entity_id": [
                "S2-101",
                "S2-102",
                "S3-201",
                "S2-301",
                "S3-302",
                "S2-401",
                "S3-501",
            ],
            "candidate_source": [
                "S2",
                "S2",
                "S3",
                "S2",
                "S3",
                "S2",
                "S3",
            ],
            "score": [
                0.98,
                0.96,
                0.30,
                0.95,
                0.40,
                0.49,
                0.20,
            ],
        }
    )


def test_single_match():
    """
    S1-B has:
        S2-301 -> 0.95
        S3-302 -> 0.40

    With threshold 0.90, only S2-301 should survive.
    """

    engine = ConflictEngine(
        threshold=0.90,
        conflict_gap=0.05,
    )

    decisions = engine.decide(make_candidates())

    row = decisions[
        decisions["source1_entity_id"] == "S1-B"
    ].iloc[0]

    assert row["candidate_count"] == 2
    assert row["accepted_count"] == 1

    assert row["matched_entity_ids"] == (
        "S2-301",
    )

    assert np.isclose(
        row["best_score"],
        0.95,
    )

    assert np.isclose(
        row["second_best_score"],
        0.40,
    )

    # Pandas may return numpy.bool_ instead of Python bool.
    assert not bool(row["has_conflict"])

    print("[PASS] Single-match decision")


def test_multiple_matches_are_preserved():
    """
    S1-A has two candidates above threshold.

    Both must be preserved because ATLAS supports
    multiple matches for one S1 entity.
    """

    engine = ConflictEngine(
        threshold=0.90,
        conflict_gap=0.05,
    )

    decisions = engine.decide(make_candidates())

    row = decisions[
        decisions["source1_entity_id"] == "S1-A"
    ].iloc[0]

    assert row["candidate_count"] == 3
    assert row["accepted_count"] == 2

    assert row["matched_entity_ids"] == (
        "S2-101",
        "S2-102",
    )

    assert np.isclose(
        row["best_score"],
        0.98,
    )

    assert np.isclose(
        row["second_best_score"],
        0.96,
    )

    print("[PASS] Multiple matches preserved")


def test_zero_match_is_supported():
    """
    S1-C has a candidate, but its score is below threshold.

    Therefore the entity should produce zero matches.
    """

    engine = ConflictEngine(
        threshold=0.90,
        conflict_gap=0.05,
    )

    decisions = engine.decide(make_candidates())

    row = decisions[
        decisions["source1_entity_id"] == "S1-C"
    ].iloc[0]

    assert row["candidate_count"] == 1
    assert row["accepted_count"] == 0

    assert row["matched_entity_ids"] == ()

    assert not bool(row["has_conflict"])

    print("[PASS] Zero-match decision")


def test_conflict_detection():
    """
    Two accepted candidates with very similar scores
    should be marked as potentially conflicting.

    Important:
        Conflict detection does NOT remove either match.
    """

    candidates = pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-CONFLICT",
                "S1-CONFLICT",
            ],
            "candidate_entity_id": [
                "S2-111",
                "S3-222",
            ],
            "candidate_source": [
                "S2",
                "S3",
            ],
            "score": [
                0.96,
                0.94,
            ],
        }
    )

    engine = ConflictEngine(
        threshold=0.90,
        conflict_gap=0.05,
    )

    decisions = engine.decide(candidates)

    row = decisions.iloc[0]

    assert row["candidate_count"] == 2
    assert row["accepted_count"] == 2

    assert bool(row["has_conflict"])

    assert np.isclose(
        row["score_gap"],
        0.02,
    )

    assert row["matched_entity_ids"] == (
        "S2-111",
        "S3-222",
    )

    print("[PASS] Conflict detection")


def test_non_conflicting_high_scores():
    """
    Multiple candidates may pass the threshold without
    necessarily being considered ambiguous.

    Here the score gap is much larger than conflict_gap.
    """

    candidates = pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-CLEAR",
                "S1-CLEAR",
            ],
            "candidate_entity_id": [
                "S2-111",
                "S3-222",
            ],
            "candidate_source": [
                "S2",
                "S3",
            ],
            "score": [
                0.99,
                0.70,
            ],
        }
    )

    engine = ConflictEngine(
        threshold=0.60,
        conflict_gap=0.05,
    )

    decisions = engine.decide(candidates)

    row = decisions.iloc[0]

    assert row["accepted_count"] == 2

    assert not bool(
        row["has_conflict"]
    )

    assert np.isclose(
        row["score_gap"],
        0.29,
    )

    print("[PASS] Non-conflicting candidates")


def test_match_table_contains_only_accepted_pairs():
    """
    get_match_table() should return only candidates whose
    scores meet or exceed the threshold.
    """

    engine = ConflictEngine(
        threshold=0.90,
        conflict_gap=0.05,
    )

    matches = engine.get_match_table(
        make_candidates()
    )

    assert len(matches) == 3

    assert set(
        matches["candidate_entity_id"]
    ) == {
        "S2-101",
        "S2-102",
        "S2-301",
    }

    assert (
        matches["score"] >= 0.90
    ).all()

    print("[PASS] Accepted match table")


def test_deterministic_ordering():
    """
    If scores are identical, candidate IDs should provide
    deterministic ordering.
    """

    candidates = pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-X",
                "S1-X",
                "S1-X",
            ],
            "candidate_entity_id": [
                "S2-B",
                "S2-A",
                "S3-C",
            ],
            "candidate_source": [
                "S2",
                "S2",
                "S3",
            ],
            "score": [
                0.95,
                0.95,
                0.95,
            ],
        }
    )

    engine = ConflictEngine(
        threshold=0.90,
    )

    decisions = engine.decide(
        candidates
    )

    row = decisions.iloc[0]

    assert row[
        "matched_entity_ids"
    ] == (
        "S2-A",
        "S2-B",
        "S3-C",
    )

    print("[PASS] Deterministic ordering")


def test_invalid_scores_are_rejected():
    """
    Scores outside the probability range [0, 1]
    must be rejected.
    """

    engine = ConflictEngine(
        threshold=0.90
    )

    invalid = make_candidates()

    invalid.loc[
        0,
        "score",
    ] = 1.5

    try:
        engine.decide(invalid)

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    print("[PASS] Invalid score validation")


def test_missing_columns_are_rejected():
    """
    Required candidate columns must be present.
    """

    engine = ConflictEngine(
        threshold=0.90
    )

    invalid = pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-X"
            ],
            "score": [
                0.95
            ],
        }
    )

    try:
        engine.decide(invalid)

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    print("[PASS] Missing-column validation")


def test_empty_dataframe():
    """
    Empty candidate input should return an empty result
    with the expected decision schema.
    """

    empty = pd.DataFrame(
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "score",
        ]
    )

    engine = ConflictEngine(
        threshold=0.90
    )

    result = engine.decide(empty)

    assert result.empty

    expected_columns = {
        "source1_entity_id",
        "candidate_count",
        "accepted_count",
        "best_score",
        "second_best_score",
        "score_gap",
        "has_conflict",
        "matched_entity_ids",
    }

    assert set(
        result.columns
    ) == expected_columns

    print("[PASS] Empty candidate handling")


def main():
    test_single_match()
    test_multiple_matches_are_preserved()
    test_zero_match_is_supported()
    test_conflict_detection()
    test_non_conflicting_high_scores()
    test_match_table_contains_only_accepted_pairs()
    test_deterministic_ordering()
    test_invalid_scores_are_rejected()
    test_missing_columns_are_rejected()
    test_empty_dataframe()

    print(
        "\nALL CONFLICT ENGINE TESTS PASSED"
    )


if __name__ == "__main__":
    main()