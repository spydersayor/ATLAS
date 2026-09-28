from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_CANDIDATES = Path("candidate_pairs.tsv")
REMAINING_RECOVERED = Path("remaining_misses_recovered.tsv")

OUTPUT_CANDIDATES = Path("candidate_pairs_with_recovery.tsv")

# Conservative threshold.
# Start here; we can tune after measuring recall.
MIN_COMBINED_SCORE = 0.70


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("ATLAS REMAINING-MISS RECOVERY INTEGRATION")
print("=" * 70)

print("\n[1/5] Loading base candidates...")

base = pd.read_csv(
    BASE_CANDIDATES,
    sep="\t",
    dtype=str,
    usecols=[
        "source1_entity_id",
        "candidate_entity_ids",
    ],
)

print(f"Base S1 rows: {len(base):,}")


print("\n[2/5] Loading recovered pairs...")

recovered = pd.read_csv(
    REMAINING_RECOVERED,
    sep="\t",
    dtype=str,
)

print(f"Recovered rows: {len(recovered):,}")


# ============================================================
# FILTER RECOVERED PAIRS
# ============================================================

print("\n[3/5] Filtering recovered pairs...")

recovered["combined_score"] = pd.to_numeric(
    recovered["combined_score"],
    errors="coerce",
)

recovered = recovered.dropna(
    subset=[
        "source1_entity_id",
        "true_target_id",
        "combined_score",
    ]
)

recovered = recovered[
    recovered["combined_score"] >= MIN_COMBINED_SCORE
].copy()

recovered = recovered.rename(
    columns={
        "true_target_id": "candidate_entity_id"
    }
)

recovered = recovered[
    [
        "source1_entity_id",
        "candidate_entity_id",
        "target_source",
        "combined_score",
        "recovery_class",
    ]
]

recovered = recovered.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
    ]
)

print(
    f"Recovered pairs after threshold "
    f"({MIN_COMBINED_SCORE:.2f}): {len(recovered):,}"
)

print(
    f"Unique S1 entities: "
    f"{recovered['source1_entity_id'].nunique():,}"
)

print("\nRecovery classes:")
print(
    recovered["recovery_class"]
    .value_counts()
    .to_string()
)

print("\nBy source:")
print(
    recovered["target_source"]
    .value_counts()
    .to_string()
)


# ============================================================
# EXPAND BASE CANDIDATES TO LONG FORMAT
# ============================================================

print("\n[4/5] Expanding base candidate list...")

base_long = base[
    base["candidate_entity_ids"].fillna("").ne("")
].copy()

base_long["candidate_entity_id"] = (
    base_long["candidate_entity_ids"]
    .str.split(",")
)

base_long = base_long.explode(
    "candidate_entity_id"
)

base_long["candidate_entity_id"] = (
    base_long["candidate_entity_id"]
    .astype(str)
    .str.strip()
)

base_long = base_long[
    base_long["candidate_entity_id"].ne("")
]

base_long = base_long[
    [
        "source1_entity_id",
        "candidate_entity_id",
    ]
].drop_duplicates()

print(
    f"Base unique candidate pairs: "
    f"{len(base_long):,}"
)


# ============================================================
# UNION
# ============================================================

recovery_long = recovered[
    [
        "source1_entity_id",
        "candidate_entity_id",
    ]
].drop_duplicates()

before = len(base_long)

union = pd.concat(
    [
        base_long,
        recovery_long,
    ],
    ignore_index=True,
)

union = union.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
    ]
)

after = len(union)

print(f"Base pairs       : {before:,}")
print(f"Recovery pairs   : {len(recovery_long):,}")
print(f"Union pairs      : {after:,}")
print(f"New unique pairs : {after - before:,}")


# ============================================================
# CONVERT BACK TO WIDE FORMAT
# ============================================================

print("\n[5/5] Rebuilding candidate_pairs format...")

grouped = (
    union
    .groupby(
        "source1_entity_id",
        sort=False,
    )["candidate_entity_id"]
    .agg(",".join)
    .rename("candidate_entity_ids")
    .reset_index()
)

result = base[
    ["source1_entity_id"]
].copy()

result = result[
    ["source1_entity_id"]
].merge(
    grouped,
    on="source1_entity_id",
    how="left",
    sort=False,
)

result["candidate_entity_ids"] = (
    result["candidate_entity_ids"]
    .fillna("")
)


# ============================================================
# VALIDATION
# ============================================================

assert len(result) == len(base)

assert result["source1_entity_id"].is_unique

candidate_counts = (
    result["candidate_entity_ids"]
    .str.split(",")
    .str.len()
)

candidate_counts = candidate_counts.where(
    result["candidate_entity_ids"].ne(""),
    0,
)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_CANDIDATES,
    sep="\t",
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("INTEGRATION COMPLETE")
print("=" * 70)

print(
    f"Total S1 entities      : {len(result):,}"
)

print(
    f"Total candidate pairs  : "
    f"{candidate_counts.sum():,}"
)

print(
    f"S1 with candidates     : "
    f"{(candidate_counts > 0).sum():,}"
)

print(
    f"S1 with zero candidates: "
    f"{(candidate_counts == 0).sum():,}"
)

print(
    f"Average candidates/S1  : "
    f"{candidate_counts.mean():.2f}"
)

print(
    f"Median candidates/S1   : "
    f"{candidate_counts.median():.0f}"
)

print(
    f"Maximum candidates/S1  : "
    f"{candidate_counts.max():,}"
)

print(
    f"\nOutput: {OUTPUT_CANDIDATES}"
)

print("=" * 70)