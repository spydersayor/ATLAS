"""
Tests for ATLAS output generator.
"""

import tempfile
from pathlib import Path

import pandas as pd

from src.output_generator import OutputGenerator


def make_test_source1():
    return pd.DataFrame(
        {
            "entity_id": [
                "S1-A",
                "S1-B",
                "S1-C",
                "S1-D",
            ],
            "business_name": [
                "Alpha",
                "Beta",
                "Gamma",
                "Delta",
            ],
            "business_address": [
                "Address A",
                "Address B",
                "Address C",
                "Address D",
            ],
            "country": [
                "India",
                "India",
                "USA",
                "France",
            ],
        }
    )


def make_candidate_pairs():
    return pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-A",
                "S1-A",
                "S1-B",
                "S1-B",
                "S1-C",
            ],
            "candidate_entity_id": [
                "S2-101",
                "S3-201",
                "S2-301",
                "S3-302",
                "S2-401",
            ],
            "candidate_source": [
                "S2",
                "S3",
                "S2",
                "S3",
                "S2",
            ],
            "blocker_sources": [
                ["name_exact"],
                ["address_exact"],
                ["name_exact"],
                ["country_exact"],
                ["name_ratio"],
            ],
            "num_blockers": [
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )


def make_decisions():
    return pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-A",
                "S1-B",
                "S1-C",
                "S1-D",
            ],
            "matched_entity_ids": [
                (
                    "S2-101",
                    "S3-201",
                ),
                (
                    "S2-301",
                ),
                (),
                (),
            ],
        }
    )


def test_build_matching_results():
    generator = OutputGenerator()

    result = generator.build_matching_results(
        test_source1=make_test_source1(),
        candidate_pairs=make_candidate_pairs(),
        decisions=make_decisions(),
    )

    assert list(result.columns) == [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    assert len(result) == 4

    row_a = result[
        result["source1_entity_id"] == "S1-A"
    ].iloc[0]

    assert row_a["matched_entity_ids"] == (
        "S2-101,S3-201"
    )

    row_b = result[
        result["source1_entity_id"] == "S1-B"
    ].iloc[0]

    assert row_b["matched_entity_ids"] == "S2-301"

    row_c = result[
        result["source1_entity_id"] == "S1-C"
    ].iloc[0]

    assert row_c["matched_entity_ids"] == ""

    row_d = result[
        result["source1_entity_id"] == "S1-D"
    ].iloc[0]

    assert row_d["matched_entity_ids"] == ""

    print("[PASS] Matching-results generation")


def test_every_test_s1_appears_once():
    generator = OutputGenerator()

    result = generator.build_matching_results(
        test_source1=make_test_source1(),
        candidate_pairs=make_candidate_pairs(),
        decisions=make_decisions(),
    )

    ids = result[
        "source1_entity_id"
    ].tolist()

    assert len(ids) == 4
    assert len(ids) == len(set(ids))

    assert set(ids) == {
        "S1-A",
        "S1-B",
        "S1-C",
        "S1-D",
    }

    print("[PASS] Every test S1 appears exactly once")


def test_matches_outside_candidates_are_removed():
    """
    If a decision accidentally contains an entity that
    wasn't present in candidate_pairs, the output layer
    must not include that entity in the final result.
    """

    decisions = make_decisions().copy()

    row_index = decisions.index[
        decisions["source1_entity_id"] == "S1-A"
    ][0]

    decisions.at[
        row_index,
        "matched_entity_ids",
    ] = (
        "S2-101",
        "S2-NOT-A-CANDIDATE",
    )

    generator = OutputGenerator()

    result = generator.build_matching_results(
        test_source1=make_test_source1(),
        candidate_pairs=make_candidate_pairs(),
        decisions=decisions,
    )

    row = result[
        result["source1_entity_id"] == "S1-A"
    ].iloc[0]

    assert row["matched_entity_ids"] == "S2-101"

    print("[PASS] Candidate-subset enforcement")


def test_candidate_output_schema():
    generator = OutputGenerator()

    result = generator.build_candidate_pairs(
        make_candidate_pairs()
    )

    assert list(result.columns) == [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "blocker_sources",
        "num_blockers",
    ]

    assert len(result) == 5

    print("[PASS] Candidate output schema")


def test_file_generation():
    """
    Test actual TSV file creation.

    This test is intentionally standalone and does not use
    pytest fixtures because our project tests are executed
    directly with:

        python -m tests.test_output_generator
    """

    with tempfile.TemporaryDirectory() as temp_dir:

        output_dir = Path(temp_dir)

        generator = OutputGenerator(
            output_dir=output_dir
        )

        matching_path, candidate_path = (
            generator.generate(
                test_source1=make_test_source1(),
                candidate_pairs=make_candidate_pairs(),
                decisions=make_decisions(),
            )
        )

        assert matching_path.exists()
        assert candidate_path.exists()

        assert matching_path.name == (
            "matching_results.tsv"
        )

        assert candidate_path.name == (
            "candidate_pairs.tsv"
        )

        matching = pd.read_csv(
            matching_path,
            sep="\t",
            dtype=str,
        )

        candidates = pd.read_csv(
            candidate_path,
            sep="\t",
            dtype=str,
        )

        assert len(matching) == 4
        assert len(candidates) == 5

        assert list(matching.columns) == [
            "source1_entity_id",
            "matched_entity_ids",
        ]

        assert list(candidates.columns) == [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "blocker_sources",
            "num_blockers",
        ]

    print("[PASS] TSV file generation")


def test_zero_match_is_preserved():
    generator = OutputGenerator()

    result = generator.build_matching_results(
        test_source1=make_test_source1(),
        candidate_pairs=make_candidate_pairs(),
        decisions=make_decisions(),
    )

    zero_match_rows = result[
        result["matched_entity_ids"] == ""
    ]

    assert len(zero_match_rows) == 2

    assert set(
        zero_match_rows[
            "source1_entity_id"
        ]
    ) == {
        "S1-C",
        "S1-D",
    }

    print("[PASS] Zero-match preservation")


def test_multiple_matches_are_preserved():
    generator = OutputGenerator()

    result = generator.build_matching_results(
        test_source1=make_test_source1(),
        candidate_pairs=make_candidate_pairs(),
        decisions=make_decisions(),
    )

    row = result[
        result["source1_entity_id"] == "S1-A"
    ].iloc[0]

    matches = row[
        "matched_entity_ids"
    ].split(",")

    assert matches == [
        "S2-101",
        "S3-201",
    ]

    print("[PASS] Multiple-match preservation")


def test_invalid_candidate_source_is_rejected():
    candidates = make_candidate_pairs().copy()

    candidates.loc[
        0,
        "candidate_source",
    ] = "S4"

    generator = OutputGenerator()

    try:
        generator.build_candidate_pairs(
            candidates
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    print("[PASS] Invalid candidate-source validation")


def test_missing_candidate_column_is_rejected():
    candidates = make_candidate_pairs().drop(
        columns=["num_blockers"]
    )

    generator = OutputGenerator()

    try:
        generator.build_candidate_pairs(
            candidates
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    print("[PASS] Missing-column validation")


def test_string_match_ids_are_supported():
    """
    ConflictEngine normally produces tuples, but the
    output layer should also accept comma-separated
    strings.
    """

    decisions = make_decisions().copy()

    row_index = decisions.index[
        decisions["source1_entity_id"] == "S1-A"
    ][0]

    decisions.at[
        row_index,
        "matched_entity_ids",
    ] = "S2-101,S3-201"

    generator = OutputGenerator()

    result = generator.build_matching_results(
        test_source1=make_test_source1(),
        candidate_pairs=make_candidate_pairs(),
        decisions=decisions,
    )

    row = result[
        result["source1_entity_id"] == "S1-A"
    ].iloc[0]

    assert row["matched_entity_ids"] == (
        "S2-101,S3-201"
    )

    print("[PASS] String match-ID parsing")


def main():
    test_build_matching_results()
    test_every_test_s1_appears_once()
    test_matches_outside_candidates_are_removed()
    test_candidate_output_schema()
    test_file_generation()
    test_zero_match_is_preserved()
    test_multiple_matches_are_preserved()
    test_invalid_candidate_source_is_rejected()
    test_missing_candidate_column_is_rejected()
    test_string_match_ids_are_supported()

    print(
        "\nALL OUTPUT GENERATOR TESTS PASSED"
    )


if __name__ == "__main__":
    main()