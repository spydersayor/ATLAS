"""
ATLAS - Blocking + Fuzzy Recovery Recall Measurement

Measures:

1. Base blocking candidate recall
2. Fuzzy recovery recall contribution
3. Union recall = base blocking U fuzzy recovery
4. Remaining missed true pairs
5. Per-source recovery statistics

Important:
- S1 is kept at 2,000 for fast diagnostics.
- S2/S3 are FULL datasets so the recall ceiling is meaningful.
- Fuzzy recovery file contains only recovered candidates for the
  previously missed true pairs.
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_source
from src.normalization import add_normalized_columns
from src.fingerprinting import add_fingerprint_columns
from src.blocking import generate_candidate_pairs, BlockingConfig


# ================================================================
# CONFIG
# ================================================================

S1_LIMIT = 2000
S2_LIMIT = None
S3_LIMIT = None

GROUND_TRUTH_PATH = Path(
    "dataset/train/train_ground_truth.tsv"
)

FUZZY_RECOVERY_PATH = Path(
    "fuzzy_recovery_candidates.tsv"
)

ALL_BLOCKERS = [
    "name_exact",
    "address_exact",
    "name_country",
    "address_country",
    "name_token_country",
    "name_prefix_country",
    "address_token_country",
    "address_prefix_country",
    "name_token_pair_country",
    "address_token_pair_country",
    "name_token",
]


# ================================================================
# HELPERS
# ================================================================

def build_true_pairs(gt: pd.DataFrame, s1_ids: set[str]) -> pd.DataFrame:
    """
    Convert train_ground_truth.tsv from:

        S1 -> comma-separated target IDs

    into:

        source1_entity_id, matched_id
    """

    gt_slice = gt[
        gt["source1_entity_id"]
        .astype(str)
        .isin(s1_ids)
    ].copy()

    true_pairs = []

    for row in gt_slice.itertuples(index=False):

        source1_id = str(row.source1_entity_id)
        matched = row.matched_entity_ids

        if pd.isna(matched):
            continue

        matched = str(matched).strip()

        if not matched:
            continue

        for matched_id in matched.split(","):

            matched_id = matched_id.strip()

            if matched_id:
                true_pairs.append(
                    (
                        source1_id,
                        matched_id,
                    )
                )

    return pd.DataFrame(
        true_pairs,
        columns=[
            "source1_entity_id",
            "matched_id",
        ],
    )


def make_pair_set(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> set[tuple[str, str]]:

    return set(
        zip(
            df[source_col].astype(str),
            df[target_col].astype(str),
        )
    )


# ================================================================
# MAIN
# ================================================================

def main() -> None:

    print("=" * 70)
    print("ATLAS BLOCKING + FUZZY RECOVERY RECALL")
    print("=" * 70)

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    print("\n[1/7] Loading datasets...")

    s1 = load_source(
        "dataset/train/train_source1.tsv"
    ).head(S1_LIMIT)

    s2 = load_source(
        "dataset/train/train_source2.tsv"
    )

    s3 = load_source(
        "dataset/train/train_source3.tsv"
    )

    if S2_LIMIT is not None:
        s2 = s2.head(S2_LIMIT)

    if S3_LIMIT is not None:
        s3 = s3.head(S3_LIMIT)

    print(
        f"S1: {len(s1):,}"
    )

    print(
        f"S2: {len(s2):,}"
    )

    print(
        f"S3: {len(s3):,}"
    )

    # ============================================================
    # 2. PREPARE BLOCKING DATA
    # ============================================================

    print("\n[2/7] Preparing fingerprints...")

    s1f = add_fingerprint_columns(
        add_normalized_columns(s1)
    )

    s2f = add_fingerprint_columns(
        add_normalized_columns(s2)
    )

    s3f = add_fingerprint_columns(
        add_normalized_columns(s3)
    )

    # ============================================================
    # 3. GENERATE BASE BLOCKING CANDIDATES
    # ============================================================

    print("\n[3/7] Generating base blocking candidates...")

    config = BlockingConfig()

    base_candidates = generate_candidate_pairs(
        s1f,
        s2f,
        s3f,
        config,
    )

    print(
        f"Base candidate pairs: "
        f"{len(base_candidates):,}"
    )

    # ============================================================
    # 4. LOAD GROUND TRUTH
    # ============================================================

    print("\n[4/7] Loading ground truth...")

    gt = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str,
    )

    s1_ids = set(
        s1["entity_id"].astype(str)
    )

    true_pairs_df = build_true_pairs(
        gt,
        s1_ids,
    )

    print(
        f"Total true match pairs: "
        f"{len(true_pairs_df):,}"
    )

    print(
        f"S1 entities with >=1 match: "
        f"{true_pairs_df['source1_entity_id'].nunique():,}"
    )

    print(
        f"True singleton S1 entities: "
        f"{len(s1_ids) - true_pairs_df['source1_entity_id'].nunique():,}"
    )

    # ============================================================
    # 5. DETERMINE COVERABLE TRUE PAIRS
    # ============================================================

    s2_ids = set(
        s2["entity_id"].astype(str)
    )

    s3_ids = set(
        s3["entity_id"].astype(str)
    )

    def resolve_target(target_id: str) -> str:

        in_s2 = target_id in s2_ids
        in_s3 = target_id in s3_ids

        if in_s2 and not in_s3:
            return "S2"

        if in_s3 and not in_s2:
            return "S3"

        if in_s2 and in_s3:
            return "AMBIGUOUS"

        return "NOT_IN_SLICE"

    true_pairs_df["target_source"] = (
        true_pairs_df["matched_id"]
        .map(resolve_target)
    )

    print("\nTarget resolution:")

    print(
        true_pairs_df[
            "target_source"
        ].value_counts()
        .to_dict()
    )

    coverable = true_pairs_df[
        true_pairs_df["target_source"].isin(
            ["S2", "S3"]
        )
    ].copy()

    print(
        f"\nCoverable true pairs: "
        f"{len(coverable):,}"
    )

    # ============================================================
    # 6. BASE + FUZZY UNION
    # ============================================================

    print("\n[5/7] Building base candidate set...")

    base_pair_set = make_pair_set(
        base_candidates,
        "source1_entity_id",
        "candidate_entity_id",
    )

    print(
        f"Unique base pairs: "
        f"{len(base_pair_set):,}"
    )

    # ------------------------------------------------------------
    # Load fuzzy recovery
    # ------------------------------------------------------------

    print(
        "\n[6/7] Loading fuzzy recovery candidates..."
    )

    if not FUZZY_RECOVERY_PATH.exists():

        raise FileNotFoundError(
            f"Missing fuzzy recovery file:\n"
            f"{FUZZY_RECOVERY_PATH}\n\n"
            f"Run the fuzzy recovery pipeline first."
        )

    fuzzy = pd.read_csv(
        FUZZY_RECOVERY_PATH,
        sep="\t",
        dtype=str,
    )

    required_columns = {
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    }

    missing = (
        required_columns
        - set(fuzzy.columns)
    )

    if missing:

        raise ValueError(
            "Fuzzy recovery file is missing "
            f"columns: {sorted(missing)}"
        )

    fuzzy = fuzzy.dropna(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )

    fuzzy["source1_entity_id"] = (
        fuzzy["source1_entity_id"]
        .astype(str)
        .str.strip()
    )

    fuzzy["candidate_entity_id"] = (
        fuzzy["candidate_entity_id"]
        .astype(str)
        .str.strip()
    )

    fuzzy = fuzzy[
        fuzzy["candidate_entity_id"].ne("")
    ]

    fuzzy_pair_set = make_pair_set(
        fuzzy,
        "source1_entity_id",
        "candidate_entity_id",
    )

    print(
        f"Fuzzy recovery rows: "
        f"{len(fuzzy):,}"
    )

    print(
        f"Unique fuzzy pairs: "
        f"{len(fuzzy_pair_set):,}"
    )

    # ------------------------------------------------------------
    # UNION
    # ------------------------------------------------------------

    union_pair_set = (
        base_pair_set
        | fuzzy_pair_set
    )

    print(
        f"Union candidate pairs: "
        f"{len(union_pair_set):,}"
    )

    # ============================================================
    # RECALL CALCULATION
    # ============================================================

    print("\n[7/7] Evaluating recall...")

    coverable_keys = set(
        zip(
            coverable[
                "source1_entity_id"
            ].astype(str),

            coverable[
                "matched_id"
            ].astype(str),
        )
    )

    base_found = (
        coverable_keys
        & base_pair_set
    )

    fuzzy_found = (
        coverable_keys
        & fuzzy_pair_set
    )

    union_found = (
        coverable_keys
        & union_pair_set
    )

    base_recall = (
        len(base_found)
        / len(coverable_keys)
        if coverable_keys
        else 0.0
    )

    fuzzy_recovery_rate = (
        len(fuzzy_found)
        / len(coverable_keys)
        if coverable_keys
        else 0.0
    )

    union_recall = (
        len(union_found)
        / len(coverable_keys)
        if coverable_keys
        else 0.0
    )

    # ============================================================
    # RESULTS
    # ============================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Coverable true pairs       : "
        f"{len(coverable_keys):,}"
    )

    print(
        f"Base blocking found        : "
        f"{len(base_found):,}"
    )

    print(
        f"Fuzzy recovery found       : "
        f"{len(fuzzy_found):,}"
    )

    print(
        f"Union found                : "
        f"{len(union_found):,}"
    )

    print()

    print(
        f"Base blocking recall       : "
        f"{base_recall:.6f} "
        f"({base_recall * 100:.2f}%)"
    )

    print(
        f"Fuzzy recovery contribution: "
        f"{fuzzy_recovery_rate:.6f} "
        f"({fuzzy_recovery_rate * 100:.2f}%)"
    )

    print(
        f"Union recall               : "
        f"{union_recall:.6f} "
        f"({union_recall * 100:.2f}%)"
    )

    # ============================================================
    # NEWLY RECOVERED PAIRS
    # ============================================================

    newly_recovered = (
        fuzzy_found
        - base_pair_set
    )

    remaining_missed = (
        coverable_keys
        - union_pair_set
    )

    print()
    print("=" * 70)
    print("FUZZY RECOVERY IMPACT")
    print("=" * 70)

    print(
        f"Base misses                 : "
        f"{len(coverable_keys - base_pair_set):,}"
    )

    print(
        f"Newly recovered by fuzzy    : "
        f"{len(newly_recovered):,}"
    )

    print(
        f"Remaining missed            : "
        f"{len(remaining_missed):,}"
    )

    if len(coverable_keys - base_pair_set):

        recovery_of_base_misses = (
            len(newly_recovered)
            / len(coverable_keys - base_pair_set)
        )

        print(
            f"Recovery of base misses    : "
            f"{recovery_of_base_misses:.4f} "
            f"({recovery_of_base_misses * 100:.2f}%)"
        )

    # ============================================================
    # SOURCE BREAKDOWN
    # ============================================================

    print()
    print("=" * 70)
    print("FUZZY RECOVERY BY SOURCE")
    print("=" * 70)

    for source in ["S2", "S3"]:

        source_keys = set(
            zip(
                coverable.loc[
                    coverable["target_source"] == source,
                    "source1_entity_id",
                ].astype(str),

                coverable.loc[
                    coverable["target_source"] == source,
                    "matched_id",
                ].astype(str),
            )
        )

        base_source_found = (
            source_keys & base_pair_set
        )

        fuzzy_source_found = (
            source_keys & fuzzy_pair_set
        )

        union_source_found = (
            source_keys & union_pair_set
        )

        source_total = len(source_keys)

        print(
            f"\n{source}:"
        )

        print(
            f"  True pairs       : "
            f"{source_total:,}"
        )

        print(
            f"  Base found       : "
            f"{len(base_source_found):,}"
        )

        print(
            f"  Fuzzy found      : "
            f"{len(fuzzy_source_found):,}"
        )

        print(
            f"  Union found      : "
            f"{len(union_source_found):,}"
        )

        if source_total:

            print(
                f"  Union recall     : "
                f"{len(union_source_found) / source_total:.6f}"
            )

    # ============================================================
    # REMAINING MISSES
    # ============================================================

    remaining_df = pd.DataFrame(
        list(remaining_missed),
        columns=[
            "source1_entity_id",
            "true_target_id",
        ],
    )

    remaining_df.to_csv(
        "remaining_missed_after_fuzzy.tsv",
        sep="\t",
        index=False,
    )

    print()
    print(
        "Saved remaining misses to: "
        "remaining_missed_after_fuzzy.tsv"
    )

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()