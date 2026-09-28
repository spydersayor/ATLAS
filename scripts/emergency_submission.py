from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.normalization import add_normalized_columns
from src.fingerprinting import add_fingerprint_columns
from src.blocking import BlockingConfig, generate_candidate_pairs


DATA = ROOT / "dataset"
TEST = DATA / "test"
OUT = ROOT / "output"

S1 = TEST / "test_source1.tsv"
S2 = TEST / "test_source2.tsv"
S3 = TEST / "test_source3.tsv"


def prepare(path, label):
    print(f"\nLoading {label}: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(f"{label}: {len(df):,} rows")

    print(f"{label}: normalization...")
    df = add_normalized_columns(df)

    print(f"{label}: fingerprinting...")
    df = add_fingerprint_columns(df, copy=True)

    return df


def main():

    print("=" * 70)
    print("ATLAS EMERGENCY TEST-ONLY SUBMISSION")
    print("=" * 70)

    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # TEST DATA ONLY
    # ------------------------------------------------------------

    s1 = prepare(S1, "TEST S1")
    s2 = prepare(S2, "TEST S2")
    s3 = prepare(S3, "TEST S3")

    # ------------------------------------------------------------
    # TIGHT BLOCKING ONLY
    # ------------------------------------------------------------

    config = BlockingConfig(
    enable_name_exact=True,
    enable_address_exact=True,
    enable_name_country=True,
    enable_address_country=True,

    enable_name_token=False,
    enable_name_token_country=False,
    enable_name_prefix_country=False,

    enable_address_token_country=False,
    enable_address_prefix_country=False,

    enable_name_soundex_country=False,

    enable_name_token_pair_country=False,
    enable_address_token_pair_country=False,
)

    print("\n" + "=" * 70)
    print("GENERATING TEST CANDIDATES")
    print("=" * 70)

    candidates = generate_candidate_pairs(
        source1=s1,
        source2=s2,
        source3=s3,
        config=config,
    )

    print(f"\nRaw candidates: {len(candidates):,}")

    # ------------------------------------------------------------
    # DEFENSIVE DEDUP
    # ------------------------------------------------------------

    candidates = candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ],
        keep="first",
    ).reset_index(drop=True)

    print(f"Deduplicated candidates: {len(candidates):,}")

    # ------------------------------------------------------------
    # SAVE candidate_pairs.tsv
    # ------------------------------------------------------------

    candidate_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "blocker_sources",
        "num_blockers",
    ]

    candidates[candidate_columns].to_csv(
        OUT / "candidate_pairs.tsv",
        sep="\t",
        index=False,
    )

    print("\nSaved:")
    print(OUT / "candidate_pairs.tsv")

    # ------------------------------------------------------------
    # EMERGENCY MATCHING RESULTS
    #
    # Use strongest available candidate(s):
    #   1. highest num_blockers
    #   2. deterministic candidate ID
    #
    # This is NOT ML scoring. It is only a submission fallback.
    # ------------------------------------------------------------

    candidates["num_blockers"] = pd.to_numeric(
        candidates["num_blockers"],
        errors="coerce",
    ).fillna(0)

    candidates = candidates.sort_values(
        [
            "source1_entity_id",
            "num_blockers",
            "candidate_entity_id",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        kind="mergesort",
    )

    # Take the strongest candidate for each S1.
    best = (
        candidates
        .drop_duplicates(
            subset=["source1_entity_id"],
            keep="first",
        )
    )

    matches = dict(
        zip(
            best["source1_entity_id"].astype(str),
            best["candidate_entity_id"].astype(str),
        )
    )

    result = pd.DataFrame({
        "source1_entity_id":
            s1["entity_id"].astype(str),
    })

    result["matched_entity_ids"] = (
        result["source1_entity_id"]
        .map(matches)
        .fillna("")
    )

    result.to_csv(
        OUT / "matching_results.tsv",
        sep="\t",
        index=False,
    )

    print("\nSaved:")
    print(OUT / "matching_results.tsv")

    # ------------------------------------------------------------
    # FINAL CHECK
    # ------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL OUTPUT")
    print("=" * 70)

    print(
        f"Test S1 rows: {len(s1):,}"
    )

    print(
        f"Candidate rows: {len(candidates):,}"
    )

    print(
        f"Matching rows: {len(result):,}"
    )

    print(
        f"Matched S1 rows: "
        f"{(result['matched_entity_ids'] != '').sum():,}"
    )

    assert len(result) == len(s1)

    assert list(result.columns) == [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    assert list(
        candidates[candidate_columns].columns
    ) == candidate_columns

    print("\nSUBMISSION FILES GENERATED.")
    print("\noutput/candidate_pairs.tsv")
    print("output/matching_results.tsv")


if __name__ == "__main__":
    main()
