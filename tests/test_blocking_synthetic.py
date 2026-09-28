import pandas as pd

from src.blocking import (
    generate_candidate_pairs,
    validate_candidate_ids,
    validate_candidate_pairs,
)


def make_source(
    ids,
    names,
    addresses,
    countries,
):
    return pd.DataFrame(
        {
            "entity_id": ids,
            "business_name": names,
            "business_address": addresses,
            "country": countries,
            "name_fingerprint": names,
            "address_fingerprint": addresses,
            "country_fingerprint": countries,
        }
    )


def main():
    # ------------------------------------------------------------
    # Source 1
    # ------------------------------------------------------------

    source1 = make_source(
        ids=["S1_1", "S1_2", "S1_3", "S1_4"],
        names=[
            "alpha",
            "beta",
            "gamma",
            "no_match",
        ],
        addresses=[
            "addr1",
            "addr2",
            "addr3",
            "addr4",
        ],
        countries=[
            "india",
            "india",
            "usa",
            "france",
        ],
    )

    # ------------------------------------------------------------
    # Source 2
    # ------------------------------------------------------------

    source2 = make_source(
        ids=["S2_1", "S2_2", "S2_3", "S2_4"],
        names=[
            "alpha",       # S1_1 -> name_exact
            "different",   # S1_2 -> address_exact
            "gamma",       # S1_3 -> multiple blockers
            "irrelevant",
        ],
        addresses=[
            "different1",
            "addr2",       # S1_2 -> address_exact
            "addr3",       # S1_3 -> multiple blockers
            "different4",
        ],
        countries=[
            "india",
            "india",
            "usa",
            "france",
        ],
    )

    # ------------------------------------------------------------
    # Source 3
    # ------------------------------------------------------------

    source3 = make_source(
        ids=["S3_1", "S3_2", "S3_3"],
        names=[
            "something",
            "gamma",
            "another",
        ],
        addresses=[
            "something_else",
            "addr3",
            "another_address",
        ],
        countries=[
            "india",
            "usa",
            "france",
        ],
    )

    # ------------------------------------------------------------
    # Generate candidates
    # ------------------------------------------------------------

    candidates = generate_candidate_pairs(
        source1,
        source2,
        source3,
    )

    print("\n=== CANDIDATE PAIRS ===")
    print(candidates.to_string(index=False))

    # ------------------------------------------------------------
    # Contract validation
    # ------------------------------------------------------------

    validate_candidate_pairs(candidates)

    validate_candidate_ids(
        candidates,
        source1,
        source2,
        source3,
    )

    # ------------------------------------------------------------
    # Expected checks
    # ------------------------------------------------------------

    # S1_1 -> S2_1 through name_exact
    assert (
        (
            candidates["source1_entity_id"].eq("S1_1")
            & candidates["candidate_entity_id"].eq("S2_1")
        ).sum()
        == 1
    ), "S1_1 -> S2_1 was not generated."

    # S1_2 -> S2_2 through address_exact
    assert (
        (
            candidates["source1_entity_id"].eq("S1_2")
            & candidates["candidate_entity_id"].eq("S2_2")
        ).sum()
        == 1
    ), "S1_2 -> S2_2 was not generated."

    # ------------------------------------------------------------
    # S1_3 -> S2_3 must be retrieved by all 4 blockers
    # ------------------------------------------------------------

    s1_3_s2_3 = candidates[
        candidates["source1_entity_id"].eq("S1_3")
        & candidates["candidate_entity_id"].eq("S2_3")
        & candidates["candidate_source"].eq("S2")
    ]

    assert len(s1_3_s2_3) == 1, (
        "S1_3 -> S2_3 should exist exactly once after fusion."
    )

    row = s1_3_s2_3.iloc[0]

    assert row["num_blockers"] == 4, (
        f"Expected 4 blockers for S1_3 -> S2_3, "
        f"got {row['num_blockers']}."
    )

    expected_blockers = {
        "name_exact",
        "address_exact",
        "name_country",
        "address_country",
    }

    actual_blockers = set(
        str(row["blocker_sources"]).split(",")
    )

    assert actual_blockers == expected_blockers, (
        "Incorrect blocker provenance for S1_3 -> S2_3. "
        f"Expected {expected_blockers}, "
        f"got {actual_blockers}."
    )

    # ------------------------------------------------------------
    # S1_3 -> S3_2
    # ------------------------------------------------------------

    s1_3_s3_2 = candidates[
        candidates["source1_entity_id"].eq("S1_3")
        & candidates["candidate_entity_id"].eq("S3_2")
        & candidates["candidate_source"].eq("S3")
    ]

    assert len(s1_3_s3_2) == 1, (
        "S1_3 -> S3_2 should exist exactly once."
    )

    # ------------------------------------------------------------
    # S1_4 should have ZERO candidates
    # ------------------------------------------------------------

    s1_4_candidates = candidates[
        candidates["source1_entity_id"].eq("S1_4")
    ]

    assert s1_4_candidates.empty, (
        "S1_4 should have zero candidates."
    )

    # ------------------------------------------------------------
    # No duplicate candidate pairs
    # ------------------------------------------------------------

    duplicate_count = candidates.duplicated(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]
    ).sum()

    assert duplicate_count == 0, (
        f"Found {duplicate_count} duplicate candidate pairs."
    )

    # ------------------------------------------------------------
    # Candidate source must only be S2/S3
    # ------------------------------------------------------------

    valid_sources = {"S2", "S3"}

    actual_sources = set(
        candidates["candidate_source"].unique()
    )

    assert actual_sources.issubset(valid_sources), (
        f"Invalid candidate sources found: "
        f"{actual_sources - valid_sources}"
    )

    # ------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------

    print("\n========================================")
    print("SYNTHETIC BLOCKING TEST: PASS")
    print("========================================")


if __name__ == "__main__":
    main()