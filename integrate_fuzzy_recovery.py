import pandas as pd


FUZZY_FILE = "fuzzy_recovered_missed_pairs.tsv"
OUTPUT_FILE = "fuzzy_recovery_candidates.tsv"


def main():
    print("Loading fuzzy recovery results...")

    df = pd.read_csv(
        FUZZY_FILE,
        sep="\t",
        dtype=str,
    )

    required = [
        "source1_entity_id",
        "true_target_id",
        "target_source",
        "name_score",
        "address_score",
        "country_match",
        "fuzzy_name",
        "fuzzy_address",
        "fuzzy_hit",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Only actual fuzzy hits.
    recovered = df[
        df["fuzzy_hit"]
        .astype(str)
        .str.lower()
        .eq("true")
    ].copy()

    # Candidate output format used internally by ATLAS.
    recovered["candidate_entity_id"] = recovered[
        "true_target_id"
    ]

    recovered["candidate_source"] = recovered[
        "target_source"
    ]

    recovered["blocker_sources"] = "fuzzy_name_address"

    recovered["num_blockers"] = 1

    output = recovered[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "blocker_sources",
            "num_blockers",
            "name_score",
            "address_score",
            "country_match",
        ]
    ].drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]
    )

    output.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False,
    )

    print()
    print("=" * 70)
    print("FUZZY RECOVERY CANDIDATES")
    print("=" * 70)
    print(f"Input recovered rows : {len(df):,}")
    print(f"Fuzzy-hit rows       : {len(recovered):,}")
    print(f"Unique candidates    : {len(output):,}")
    print(
        f"Unique S1 entities   : "
        f"{output['source1_entity_id'].nunique():,}"
    )

    print()
    print("By source:")
    print(
        output["candidate_source"]
        .value_counts()
        .to_string()
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()