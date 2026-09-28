import pandas as pd

BASE = "candidate_pairs_long.tsv"
FUZZY = "fuzzy_recovery_candidates.tsv"

print("=" * 70)
print("ATLAS BASE + FUZZY CANDIDATE UNION")
print("=" * 70)

# ------------------------------------------------------------
# Load base candidates
# ------------------------------------------------------------

print("\nLoading base candidates...")

base = pd.read_csv(
    BASE,
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
)

base = base.dropna(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]
)

base = base.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
    ]
)

print(f"Base rows       : {len(base):,}")
print(f"Base S1 entities : {base['source1_entity_id'].nunique():,}")


# ------------------------------------------------------------
# Load fuzzy recovery
# ------------------------------------------------------------

print("\nLoading fuzzy recovery...")

fuzzy = pd.read_csv(
    FUZZY,
    sep="\t",
    dtype={
        "source1_entity_id": "string",
        "candidate_entity_id": "string",
        "candidate_source": "string",
    },
)

fuzzy = fuzzy.dropna(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]
)

fuzzy = fuzzy.drop_duplicates(
    subset=[
        "source1_entity_id",
        "candidate_entity_id",
    ]
)

print(f"Fuzzy rows       : {len(fuzzy):,}")
print(f"Fuzzy S1 entities: {fuzzy['source1_entity_id'].nunique():,}")


# ------------------------------------------------------------
# Find genuinely NEW fuzzy candidates
# ------------------------------------------------------------

base_keys = pd.MultiIndex.from_frame(
    base[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ]
)

fuzzy_keys = pd.MultiIndex.from_frame(
    fuzzy[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ]
)

is_new = ~fuzzy_keys.isin(base_keys)

new_fuzzy = fuzzy.loc[is_new].copy()

print("\n" + "=" * 70)
print("FUZZY INCREMENT")
print("=" * 70)

print(f"Fuzzy candidates        : {len(fuzzy):,}")
print(f"Already in base         : {len(fuzzy) - len(new_fuzzy):,}")
print(f"NEW candidates added    : {len(new_fuzzy):,}")

print(
    f"New S1 entities touched : "
    f"{new_fuzzy['source1_entity_id'].nunique():,}"
)

if not new_fuzzy.empty:
    print("\nNew candidates by source:")
    print(new_fuzzy["candidate_source"].value_counts())


# ------------------------------------------------------------
# Candidate counts per S1
# ------------------------------------------------------------

base_counts = (
    base.groupby("source1_entity_id")
    .size()
    .rename("base_count")
)

new_counts = (
    new_fuzzy.groupby("source1_entity_id")
    .size()
    .rename("new_fuzzy_count")
)

counts = pd.concat(
    [base_counts, new_counts],
    axis=1,
).fillna(0)

counts["base_count"] = counts["base_count"].astype(int)
counts["new_fuzzy_count"] = counts["new_fuzzy_count"].astype(int)

counts["combined_count"] = (
    counts["base_count"] +
    counts["new_fuzzy_count"]
)


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CANDIDATE COUNT IMPACT")
print("=" * 70)

print("\nBASE:")
print(counts["base_count"].describe())

print("\nNEW FUZZY:")
print(counts["new_fuzzy_count"].describe())

print("\nCOMBINED:")
print(counts["combined_count"].describe())


print("\nTop combined candidate counts:")
print(
    counts["combined_count"]
    .sort_values(ascending=False)
    .head(30)
)


# ------------------------------------------------------------
# Distribution of fuzzy additions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FUZZY ADDITION DISTRIBUTION")
print("=" * 70)

print(
    counts["new_fuzzy_count"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# Save analysis
# ------------------------------------------------------------

counts.reset_index().to_csv(
    "fuzzy_union_candidate_stats.tsv",
    sep="\t",
    index=False,
)

new_fuzzy.to_csv(
    "fuzzy_new_candidates_only.tsv",
    sep="\t",
    index=False,
)

print("\nSaved:")
print("  fuzzy_union_candidate_stats.tsv")
print("  fuzzy_new_candidates_only.tsv")

print("\nDONE.")
