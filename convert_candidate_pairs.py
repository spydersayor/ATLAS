from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TEST_SOURCE1 = Path("dataset/test/test_source1.tsv")

INPUT_CANDIDATES = Path("candidate_pairs_long.tsv")
OUTPUT_CANDIDATES = Path("candidate_pairs.tsv")

# Read the long candidate file in chunks to avoid loading the
# entire candidate set into RAM.
CHUNK_SIZE = 500_000


# ============================================================
# LOAD SOURCE 1
# ============================================================

def load_source1_ids() -> pd.Series:
    """
    Load every Source 1 entity from the actual test schema.

    test_source1.tsv contains:
        entity_id
        business_name
        business_address
        country

    The final candidate_pairs.tsv must contain exactly one row
    for every Source 1 entity, including entities with zero
    candidates.
    """

    s1 = pd.read_csv(
        TEST_SOURCE1,
        sep="\t",
        usecols=["entity_id"],
        dtype={"entity_id": "string"},
    )

    s1 = (
        s1["entity_id"]
        .dropna()
        .astype("string")
        .str.strip()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return s1


# ============================================================
# CONVERT LONG -> WIDE
# ============================================================

def convert_candidates() -> None:

    print("=" * 70)
    print("ATLAS Candidate Pair Converter")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. LOAD ALL SOURCE 1 IDS
    # --------------------------------------------------------

    print("\n[1/4] Loading Source 1...")

    source1_ids = load_source1_ids()

    print(
        f"Source 1 entities: {len(source1_ids):,}"
    )

    # --------------------------------------------------------
    # 2. CHECK INPUT
    # --------------------------------------------------------

    print("\n[2/4] Loading long-format candidates...")

    if not INPUT_CANDIDATES.exists():
        raise FileNotFoundError(
            f"\nInput candidate file not found:\n"
            f"  {INPUT_CANDIDATES}\n\n"
            f"This converter expects the blocking pipeline to first "
            f"create candidate_pairs_long.tsv."
        )

    # --------------------------------------------------------
    # CHUNKED READ
    # --------------------------------------------------------
    #
    # DO NOT load the entire candidate file into RAM.
    #
    # candidate_pairs_long.tsv may contain millions of rows.
    #
    # Each chunk is:
    #
    #   read
    #   clean
    #   deduplicate
    #   aggregate
    #   discarded
    #
    # Only the final per-S1 candidate mapping remains in memory.
    # --------------------------------------------------------

    candidate_map: dict[str, set[str]] = {}

    total_input_rows = 0
    total_valid_rows = 0

    reader = pd.read_csv(
        INPUT_CANDIDATES,
        sep="\t",
        dtype={
            "source1_entity_id": "string",
            "candidate_entity_id": "string",
            "candidate_source": "string",
        },
        usecols=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ],
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, candidates in enumerate(reader, start=1):

        total_input_rows += len(candidates)

        # ----------------------------------------------------
        # Defensive cleanup
        # ----------------------------------------------------

        candidates = candidates.dropna(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
            ]
        )

        candidates["source1_entity_id"] = (
            candidates["source1_entity_id"]
            .astype("string")
            .str.strip()
        )

        candidates["candidate_entity_id"] = (
            candidates["candidate_entity_id"]
            .astype("string")
            .str.strip()
        )

        candidates["candidate_source"] = (
            candidates["candidate_source"]
            .astype("string")
            .str.strip()
        )

        candidates = candidates[
            candidates["source1_entity_id"].ne("")
            & candidates["candidate_entity_id"].ne("")
        ]

        total_valid_rows += len(candidates)

        # ----------------------------------------------------
        # Deduplicate inside this chunk
        # ----------------------------------------------------

        candidates = candidates.drop_duplicates(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )

        # ----------------------------------------------------
        # Add to Python mapping
        # ----------------------------------------------------
        #
        # Using sets guarantees that the same candidate pair
        # appearing in different chunks is still represented once.
        # ----------------------------------------------------

        for row in candidates.itertuples(index=False):

            s1_id = str(row.source1_entity_id)
            candidate_id = str(row.candidate_entity_id)

            if s1_id not in candidate_map:
                candidate_map[s1_id] = set()

            candidate_map[s1_id].add(candidate_id)

        print(
            f"  Processed chunk {chunk_number:,} | "
            f"rows: {total_input_rows:,} | "
            f"S1 entities accumulated: {len(candidate_map):,}"
        )

        del candidates

    print(
        f"\nLong-format rows read : {total_input_rows:,}"
    )

    print(
        f"Valid candidate rows   : {total_valid_rows:,}"
    )

    print(
        f"S1 entities with candidates: "
        f"{len(candidate_map):,}"
    )

    # --------------------------------------------------------
    # 3. BUILD FINAL WIDE OUTPUT
    # --------------------------------------------------------

    print("\n[3/4] Building wide candidate_pairs.tsv...")

    output_rows = []

    total_candidate_pairs = 0
    max_candidates = 0

    for source1_id in source1_ids:

        s1_id = str(source1_id)

        candidate_ids = candidate_map.get(
            s1_id,
            set(),
        )

        if candidate_ids:

            # Sorting gives deterministic output.
            candidate_list = sorted(candidate_ids)

            candidate_string = ",".join(
                candidate_list
            )

            count = len(candidate_list)

        else:

            candidate_string = ""
            count = 0

        output_rows.append(
            (
                s1_id,
                candidate_string,
            )
        )

        total_candidate_pairs += count

        if count > max_candidates:
            max_candidates = count

    result = pd.DataFrame(
        output_rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_ids",
        ],
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    assert len(result) == len(source1_ids), (
        "ERROR: Final output does not contain exactly "
        "one row per Source 1 entity."
    )

    assert result["source1_entity_id"].is_unique, (
        "ERROR: Duplicate Source 1 entities found."
    )

    assert (
        result["source1_entity_id"]
        .astype(str)
        .equals(
            source1_ids.astype(str)
        )
    ), (
        "ERROR: Source 1 ordering/content changed unexpectedly."
    )

    # --------------------------------------------------------
    # 4. WRITE FINAL FILE
    # --------------------------------------------------------

    print("\n[4/4] Writing final candidate_pairs.tsv...")

    result.to_csv(
        OUTPUT_CANDIDATES,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    candidate_counts = (
        result["candidate_entity_ids"]
        .str.split(",")
        .str.len()
    )

    candidate_counts = candidate_counts.where(
        result["candidate_entity_ids"].ne(""),
        0,
    )

    with_candidates = int(
        (candidate_counts > 0).sum()
    )

    without_candidates = (
        len(result) - with_candidates
    )

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"Total S1 entities       : {len(result):,}"
    )

    print(
        f"S1 with candidates      : {with_candidates:,}"
    )

    print(
        f"S1 with zero candidates : {without_candidates:,}"
    )

    print(
        f"Total candidate pairs   : "
        f"{int(candidate_counts.sum()):,}"
    )

    print(
        f"Average candidates/S1   : "
        f"{candidate_counts.mean():.2f}"
    )

    print(
        f"Median candidates/S1    : "
        f"{candidate_counts.median():.0f}"
    )

    print(
        f"Maximum candidates/S1   : "
        f"{candidate_counts.max():,}"
    )

    print(
        f"\nOutput: {OUTPUT_CANDIDATES}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    convert_candidates()