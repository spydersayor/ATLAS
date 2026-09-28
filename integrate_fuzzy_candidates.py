from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_CANDIDATES = "candidate_pairs_long.tsv"
FUZZY_CANDIDATES = "fuzzy_recovery_candidates.tsv"

OUTPUT_LONG = "candidate_pairs_with_fuzzy.tsv"
OUTPUT_WIDE = "candidate_pairs.tsv"

S1_PATH = "dataset/train/train_source1.tsv"
GROUND_TRUTH_PATH = "dataset/train/train_ground_truth.tsv"


# ============================================================
# HELPERS
# ============================================================

KEY_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
]


def normalise_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise candidate columns and remove exact duplicate pairs."""

    required = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required candidate columns: {missing}"
        )

    for col in required:
        df[col] = df[col].astype(str).str.strip()

    df = df[
        (df["source1_entity_id"] != "")
        & (df["candidate_entity_id"] != "")
        & (df["candidate_source"].isin(["S2", "S3"]))
    ].copy()

    return df


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("FUZZY CANDIDATE INTEGRATION")
print("=" * 70)

print("\nLoading base candidates...")
base = pd.read_csv(
    BASE_CANDIDATES,
    sep="\t",
    dtype=str,
)

print(f"Base rows: {len(base):,}")

print("\nLoading fuzzy candidates...")
fuzzy = pd.read_csv(
    FUZZY_CANDIDATES,
    sep="\t",
    dtype=str,
)

print(f"Fuzzy rows: {len(fuzzy):,}")


# ============================================================
# NORMALISE
# ============================================================

base = normalise_candidate_frame(base)
fuzzy = normalise_candidate_frame(fuzzy)


# ============================================================
# ADD / NORMALISE PROVENANCE
# ============================================================

if "blocker_sources" not in base.columns:
    base["blocker_sources"] = ""

if "num_blockers" not in base.columns:
    base["num_blockers"] = "0"

base["blocker_sources"] = (
    base["blocker_sources"]
    .fillna("")
    .astype(str)
)

fuzzy["blocker_sources"] = "fuzzy_name_address"

if "num_blockers" not in fuzzy.columns:
    fuzzy["num_blockers"] = "1"

# Mark fuzzy rows explicitly.
fuzzy["is_fuzzy"] = "1"
base["is_fuzzy"] = "0"


# ============================================================
# KEEP ONLY USEFUL COLUMNS
# ============================================================

base_cols = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "blocker_sources",
    "num_blockers",
    "is_fuzzy",
]

fuzzy_cols = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "blocker_sources",
    "num_blockers",
    "is_fuzzy",
]

base = base[base_cols].copy()
fuzzy = fuzzy[fuzzy_cols].copy()


# ============================================================
# UNION
# ============================================================

print("\nCombining candidates...")

combined = pd.concat(
    [base, fuzzy],
    ignore_index=True,
)

print(f"Rows before dedup: {len(combined):,}")


# ============================================================
# MERGE DUPLICATE PAIRS
# ============================================================

def merge_pair_rows(group: pd.DataFrame) -> pd.Series:

    blocker_set = set()

    for value in group["blocker_sources"].fillna(""):
        for blocker in str(value).split(","):
            blocker = blocker.strip()

            if blocker:
                blocker_set.add(blocker)

    # Preserve deterministic blockers first.
    ordered_blockers = []

    deterministic_order = [
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

    for blocker in deterministic_order:
        if blocker in blocker_set:
            ordered_blockers.append(blocker)

    # Fuzzy provenance separately.
    if "fuzzy_name_address" in blocker_set:
        ordered_blockers.append("fuzzy_name_address")

    return pd.Series(
        {
            "source1_entity_id": group["source1_entity_id"].iloc[0],
            "candidate_entity_id": group["candidate_entity_id"].iloc[0],
            "candidate_source": group["candidate_source"].iloc[0],
            "blocker_sources": ",".join(ordered_blockers),
            "num_blockers": len(ordered_blockers),
            "is_fuzzy": (
                "1"
                if (group["is_fuzzy"] == "1").any()
                else "0"
            ),
        }
    )


print("Fusing duplicate candidate pairs...")

combined = (
    combined
    .groupby(
        KEY_COLUMNS,
        sort=False,
        dropna=False,
        group_keys=False,
    )
    .apply(
        merge_pair_rows,
        include_groups=False,
    )
    .reset_index(drop=True)
)

print(f"Rows after dedup: {len(combined):,}")


# ============================================================
# FUZZY CONTRIBUTION
# ============================================================

fuzzy_rows = combined["is_fuzzy"].eq("1")

print("\n" + "=" * 70)
print("FUZZY CONTRIBUTION")
print("=" * 70)

print(
    f"Final candidates containing fuzzy evidence: "
    f"{int(fuzzy_rows.sum()):,}"
)

print(
    f"Final candidates without fuzzy evidence: "
    f"{int((~fuzzy_rows).sum()):,}"
)

print(
    f"Fuzzy candidate percentage: "
    f"{100 * fuzzy_rows.mean():.3f}%"
)


# ============================================================
# PER-S1 CANDIDATE DISTRIBUTION
# ============================================================

candidate_counts = (
    combined
    .groupby("source1_entity_id")
    .size()
)

print("\n" + "=" * 70)
print("CANDIDATE COUNT DISTRIBUTION")
print("=" * 70)

print(f"S1 entities with candidates: {len(candidate_counts):,}")
print(f"Mean candidates/S1: {candidate_counts.mean():.2f}")
print(f"Median candidates/S1: {candidate_counts.median():.2f}")
print(f"P90 candidates/S1: {candidate_counts.quantile(0.90):.2f}")
print(f"P95 candidates/S1: {candidate_counts.quantile(0.95):.2f}")
print(f"P99 candidates/S1: {candidate_counts.quantile(0.99):.2f}")
print(f"Max candidates/S1: {candidate_counts.max():,}")


# ============================================================
# SINGLETON SAFETY CHECK
# ============================================================

print("\n" + "=" * 70)
print("SINGLETON SAFETY CHECK")
print("=" * 70)

s1 = pd.read_csv(
    S1_PATH,
    sep="\t",
    dtype=str,
    usecols=["entity_id"],
)

gt = pd.read_csv(
    GROUND_TRUTH_PATH,
    sep="\t",
    dtype=str,
)

s1_ids = set(
    s1["entity_id"].astype(str)
)

matched_s1 = set()

for row in gt.itertuples(index=False):

    s1_id = str(row.source1_entity_id)

    if s1_id not in s1_ids:
        continue

    matched = getattr(row, "matched_entity_ids", "")

    if pd.isna(matched):
        continue

    matched = str(matched).strip()

    if matched:
        matched_s1.add(s1_id)


singleton_ids = s1_ids - matched_s1

singleton_candidates = combined[
    combined["source1_entity_id"].isin(singleton_ids)
]

singleton_counts = (
    singleton_candidates
    .groupby("source1_entity_id")
    .size()
)

print(f"Total S1 entities: {len(s1_ids):,}")
print(f"True singleton S1 entities: {len(singleton_ids):,}")
print(
    f"Singletons receiving candidates: "
    f"{len(singleton_counts):,}"
)

if not singleton_counts.empty:

    print(
        f"Average candidates on affected singletons: "
        f"{singleton_counts.mean():.2f}"
    )

    print(
        f"Maximum candidates on affected singleton: "
        f"{singleton_counts.max():,}"
    )

    affected_singletons = (
        singleton_counts
        .sort_values(ascending=False)
        .head(20)
    )

    print("\nTop singleton candidate counts:")
    print(affected_singletons.to_string())

else:
    print("GOOD: no singleton received any candidate.")


# ============================================================
# SAVE LONG FORMAT
# ============================================================

print("\n" + "=" * 70)
print("SAVING")
print("=" * 70)

combined = combined.sort_values(
    [
        "source1_entity_id",
        "candidate_source",
        "candidate_entity_id",
    ],
    kind="stable",
)

combined.to_csv(
    OUTPUT_LONG,
    sep="\t",
    index=False,
)

print(f"Saved long format: {OUTPUT_LONG}")


# ============================================================
# CREATE REQUIRED WIDE FORMAT
# ============================================================

print("\nCreating submission-style wide candidate file...")

# Every S1 must exist, including zero-candidate entities.

candidate_lists = (
    combined
    .groupby("source1_entity_id", sort=False)["candidate_entity_id"]
    .agg(",".join)
)

wide = pd.DataFrame(
    {
        "source1_entity_id": s1["entity_id"].astype(str)
    }
)

wide["candidate_entity_ids"] = (
    wide["source1_entity_id"]
    .map(candidate_lists)
    .fillna("")
)

wide.to_csv(
    OUTPUT_WIDE,
    sep="\t",
    index=False,
)

print(f"Saved wide format: {OUTPUT_WIDE}")


# ============================================================
# FINAL SANITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL SANITY CHECKS")
print("=" * 70)

print(
    f"Wide rows: {len(wide):,}"
)

print(
    f"Expected S1 rows: {len(s1):,}"
)

print(
    f"Zero-candidate S1s: "
    f"{int(wide['candidate_entity_ids'].eq('').sum()):,}"
)

print(
    f"Non-zero-candidate S1s: "
    f"{int(wide['candidate_entity_ids'].ne('').sum()):,}"
)

duplicate_pairs = combined.duplicated(
    subset=KEY_COLUMNS
).sum()

print(
    f"Duplicate candidate pairs remaining: "
    f"{duplicate_pairs}"
)

if duplicate_pairs != 0:
    raise RuntimeError(
        "Duplicate candidate pairs remain!"
    )

if len(wide) != len(s1):
    raise RuntimeError(
        "Wide candidate file does not contain every S1 entity!"
    )

print("\nDONE.")
print()
print(f"LONG : {OUTPUT_LONG}")
print(f"WIDE : {OUTPUT_WIDE}")