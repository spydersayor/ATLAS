from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from rapidfuzz.fuzz import ratio


# ============================================================
# CONFIG
# ============================================================

S1_LIMIT = 2000
FUZZY_THRESHOLD = 0.55

S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"

MISSED_PATH = "missed_true_pairs_diagnostics.tsv"

OUTPUT_PATH = "targeted_fuzzy_candidates.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_set(value: str) -> set[str]:
    return set(value.split())


# ============================================================
# FUZZY SIMILARITY
# ============================================================

def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def token_overlap(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)

    if not ta or not tb:
        return 0.0

    intersection = len(ta & tb)
    union = len(ta | tb)

    return intersection / union if union else 0.0


# ============================================================
# LOAD
# ============================================================

def load_source(path: str, limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    if limit is not None:
        df = df.head(limit)

    return df


# ============================================================
# BUILD TARGETED FUZZY CANDIDATES
# ============================================================

def generate_targeted_candidates(
    s1: pd.DataFrame,
    s2: pd.DataFrame,
    s3: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:

    targets = pd.concat(
        [
            s2.assign(candidate_source="S2"),
            s3.assign(candidate_source="S3"),
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Normalize only columns required by fuzzy matching.
    # --------------------------------------------------------

    s1 = s1.copy()
    targets = targets.copy()

    for df in (s1, targets):

        df["name_norm"] = df["business_name"].map(normalize)
        df["address_norm"] = df["business_address"].map(normalize)
        df["country_norm"] = df["country"].map(normalize)

    # --------------------------------------------------------
    # Country index.
    #
    # This is the important optimization:
    # DO NOT compare every S1 against every S2/S3 record.
    #
    # Fuzzy matching is performed only within the same country.
    # --------------------------------------------------------

    country_groups: dict[str, pd.DataFrame] = {}

    for country, group in targets.groupby("country_norm", sort=False):
        if country:
            country_groups[country] = group

    output_rows: list[dict] = []

    total_s1 = len(s1)

    print()
    print("=" * 70)
    print("TARGETED FUZZY BLOCKER")
    print("=" * 70)
    print(f"S1 records: {total_s1:,}")
    print(f"S2 records: {len(s2):,}")
    print(f"S3 records: {len(s3):,}")
    print(f"Threshold: {threshold:.2f}")
    print("Blocking key: country")
    print()

    for position, row in enumerate(
        s1.itertuples(index=False),
        start=1,
    ):

        s1_id = str(row.entity_id)

        name = normalize(row.business_name)
        address = normalize(row.business_address)
        country = normalize(row.country)

        if not country:
            continue

        candidates = country_groups.get(country)

        if candidates is None or candidates.empty:
            continue

        # ----------------------------------------------------
        # Convert only the relevant columns to arrays.
        # ----------------------------------------------------

        target_ids = candidates["entity_id"].to_numpy()
        target_sources = candidates["candidate_source"].to_numpy()

        target_names = candidates["name_norm"].to_numpy()
        target_addresses = candidates["address_norm"].to_numpy()

        for (
            target_id,
            target_source,
            target_name,
            target_address,
        ) in zip(
            target_ids,
            target_sources,
            target_names,
            target_addresses,
        ):

            name_score = fuzzy_score(
                name,
                target_name,
            )

            address_score = fuzzy_score(
                address,
                target_address,
            )

            # ------------------------------------------------
            # Targeted recovery rule.
            #
            # Strong evidence from either field is enough,
            # but require BOTH records to have usable text.
            # ------------------------------------------------

            name_hit = (
                bool(name)
                and bool(target_name)
                and name_score >= threshold
            )

            address_hit = (
                bool(address)
                and bool(target_address)
                and address_score >= threshold
            )

            if not (name_hit or address_hit):
                continue

            output_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": str(target_id),
                    "candidate_source": str(target_source),
                    "name_score": round(name_score, 6),
                    "address_score": round(address_score, 6),
                    "country_match": True,
                    "fuzzy_name": name_hit,
                    "fuzzy_address": address_hit,
                }
            )

        if position % 100 == 0 or position == total_s1:
            print(
                f"\rProcessed S1: {position:,}/{total_s1:,} | "
                f"fuzzy candidates: {len(output_rows):,}",
                end="",
                flush=True,
            )

    print()
    print()

    if not output_rows:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "name_score",
                "address_score",
                "country_match",
                "fuzzy_name",
                "fuzzy_address",
            ]
        )

    result = pd.DataFrame(output_rows)

    # Defensive deduplication.
    result = result.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]
    )

    return result


# ============================================================
# COMPARE AGAINST MISSED TRUE PAIRS
# ============================================================

def evaluate_recovery(
    fuzzy_candidates: pd.DataFrame,
    missed: pd.DataFrame,
) -> None:

    true_pairs = set(
        zip(
            missed["source1_entity_id"].astype(str),
            missed["true_target_id"].astype(str),
        )
    )

    candidate_pairs = set(
        zip(
            fuzzy_candidates["source1_entity_id"].astype(str),
            fuzzy_candidates["candidate_entity_id"].astype(str),
        )
    )

    recovered = true_pairs & candidate_pairs

    print("=" * 70)
    print("FUZZY CANDIDATE ANALYSIS")
    print("=" * 70)

    print(f"Missed true pairs:       {len(true_pairs):,}")
    print(f"Fuzzy candidates:        {len(candidate_pairs):,}")
    print(f"Recovered missed pairs:  {len(recovered):,}")

    recovery = (
        len(recovered) / len(true_pairs)
        if true_pairs
        else 0.0
    )

    print(f"Recovery rate:           {recovery:.2%}")

    # --------------------------------------------------------
    # Candidate efficiency.
    # --------------------------------------------------------

    efficiency = (
        len(recovered) / len(candidate_pairs)
        if candidate_pairs
        else 0.0
    )

    print(
        f"Recovery / candidate:    {efficiency:.6%}"
    )

    # --------------------------------------------------------
    # New candidates per recovered pair.
    # --------------------------------------------------------

    if recovered:
        print(
            f"Candidates per recovery: "
            f"{len(candidate_pairs) / len(recovered):.2f}"
        )

    # --------------------------------------------------------
    # Source breakdown.
    # --------------------------------------------------------

    recovered_rows = []

    for s1_id, target_id in recovered:
        source_rows = missed[
            (missed["source1_entity_id"].astype(str) == s1_id)
            & (missed["true_target_id"].astype(str) == target_id)
        ]

        for _, r in source_rows.iterrows():
            recovered_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "true_target_id": target_id,
                    "target_source": r["target_source"],
                }
            )

    if recovered_rows:
        recovered_df = pd.DataFrame(recovered_rows)

        print()
        print("Recovered by source:")
        print(
            recovered_df["target_source"]
            .value_counts()
            .to_string()
        )

    # --------------------------------------------------------
    # Candidate count per S1.
    # --------------------------------------------------------

    counts = (
        fuzzy_candidates
        .groupby("source1_entity_id")
        .size()
    )

    print()
    print("Candidates per S1:")
    print(f"  Mean:   {counts.mean():,.2f}")
    print(f"  Median: {counts.median():,.2f}")
    print(f"  P90:    {counts.quantile(0.90):,.2f}")
    print(f"  P95:    {counts.quantile(0.95):,.2f}")
    print(f"  P99:    {counts.quantile(0.99):,.2f}")
    print(f"  Max:    {counts.max():,.0f}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("Loading data...")

    s1 = load_source(
        S1_PATH,
        S1_LIMIT,
    )

    s2 = load_source(S2_PATH)
    s3 = load_source(S3_PATH)

    print(f"S1: {len(s1):,}")
    print(f"S2: {len(s2):,}")
    print(f"S3: {len(s3):,}")

    print()
    print("Loading missed true pairs...")

    missed = pd.read_csv(
        MISSED_PATH,
        sep="\t",
        dtype=str,
    )

    print(f"Missed pairs: {len(missed):,}")

    # --------------------------------------------------------
    # Generate targeted fuzzy candidates.
    # --------------------------------------------------------

    fuzzy_candidates = generate_targeted_candidates(
        s1=s1,
        s2=s2,
        s3=s3,
        threshold=FUZZY_THRESHOLD,
    )

    print()
    print(
        f"Generated fuzzy candidates: "
        f"{len(fuzzy_candidates):,}"
    )

    # --------------------------------------------------------
    # Evaluate.
    # --------------------------------------------------------

    evaluate_recovery(
        fuzzy_candidates,
        missed,
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    fuzzy_candidates.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()