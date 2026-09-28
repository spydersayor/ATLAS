from pathlib import Path
import re
import unicodedata

import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio, WRatio


# ============================================================
# CONFIG
# ============================================================

REMAINING = Path("remaining_misses_diagnostics.tsv")

OUTPUT = Path("remaining_misses_recovery_analysis.tsv")
RECOVERED = Path("remaining_misses_recovered.tsv")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Unicode normalization
    value = unicodedata.normalize("NFKD", value)

    # Remove accents where possible
    value = "".join(
        ch for ch in value
        if not unicodedata.combining(ch)
    )

    # Replace punctuation with spaces
    value = re.sub(r"[^a-z0-9]+", " ", value)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def compact_text(value):
    return re.sub(
        r"[^a-z0-9]",
        "",
        normalize_text(value)
    )


def tokens(value):
    return set(normalize_text(value).split())


# ============================================================
# TOKEN FEATURES
# ============================================================

def token_jaccard(a, b):
    ta = tokens(a)
    tb = tokens(b)

    if not ta or not tb:
        return 0.0

    return len(ta & tb) / len(ta | tb)


def token_containment(a, b):
    ta = tokens(a)
    tb = tokens(b)

    if not ta or not tb:
        return 0.0

    return max(
        len(ta & tb) / len(ta),
        len(ta & tb) / len(tb),
    )


def exact_token_overlap(a, b):
    ta = tokens(a)
    tb = tokens(b)

    return len(ta & tb)


# ============================================================
# ADDRESS NUMBER SIGNAL
# ============================================================

def extract_numbers(value):
    return set(
        re.findall(
            r"\d+",
            normalize_text(value)
        )
    )


def number_overlap(a, b):
    na = extract_numbers(a)
    nb = extract_numbers(b)

    if not na or not nb:
        return 0.0

    return len(na & nb) / len(na | nb)


# ============================================================
# CHARACTER FEATURES
# ============================================================

def char_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return ratio(a, b) / 100.0


def token_set_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return token_set_ratio(a, b) / 100.0


def weighted_similarity(a, b):
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return WRatio(a, b) / 100.0


# ============================================================
# ADDRESS-SPECIFIC FEATURES
# ============================================================

def address_score(a, b):
    if not a or not b:
        return 0.0

    scores = [
        char_similarity(a, b),
        token_set_similarity(a, b),
        token_jaccard(a, b),
        token_containment(a, b),
    ]

    return max(scores)


# ============================================================
# NAME-SPECIFIC FEATURES
# ============================================================

def name_score(a, b):
    if not a or not b:
        return 0.0

    scores = [
        char_similarity(a, b),
        token_set_similarity(a, b),
        token_jaccard(a, b),
        token_containment(a, b),
    ]

    return max(scores)


# ============================================================
# RECOVERY DECISION
# ============================================================

def classify(row):
    name = row["name_score"]
    address = row["address_score"]
    name_tok = row["name_token_set"]
    addr_tok = row["address_token_set"]
    numbers = row["address_number_overlap"]
    country = row["country_match"]

    # --------------------------------------------------------
    # VERY STRONG
    # --------------------------------------------------------

    if country and (
        name >= 0.82
        or address >= 0.82
        or (
            name_tok >= 0.78
            and address >= 0.65
        )
    ):
        return "STRONG"

    # --------------------------------------------------------
    # STRONG ADDRESS
    # --------------------------------------------------------

    if country and address >= 0.72 and numbers >= 0.50:
        return "ADDRESS_STRONG"

    # --------------------------------------------------------
    # STRONG NAME
    # --------------------------------------------------------

    if country and name >= 0.72 and name_tok >= 0.65:
        return "NAME_STRONG"

    # --------------------------------------------------------
    # COMBINED EVIDENCE
    # --------------------------------------------------------

    if country:
        combined = (
            0.55 * name
            + 0.45 * address
        )

        if combined >= 0.70:
            return "COMBINED"

        if (
            name >= 0.60
            and address >= 0.60
            and numbers >= 0.25
        ):
            return "COMBINED"

    # --------------------------------------------------------
    # WEAK / MANUAL REVIEW
    # --------------------------------------------------------

    if country and (
        name >= 0.55
        or address >= 0.55
    ):
        return "REVIEW"

    return "WEAK"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ATLAS REMAINING-MISS RECOVERY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print("\n[1/4] Loading remaining misses...")

    df = pd.read_csv(
        REMAINING,
        sep="\t",
        dtype=str,
    )

    print(
        f"Remaining misses: {len(df):,}"
    )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    print("\n[2/4] Computing recovery features...")

    df["s1_name_norm"] = df["s1_name"].map(normalize_text)
    df["target_name_norm"] = df["target_name"].map(normalize_text)

    df["s1_address_norm"] = df["s1_address"].map(normalize_text)
    df["target_address_norm"] = df["target_address"].map(normalize_text)

    df["country_match"] = (
        df["s1_country"]
        .fillna("")
        .str.lower()
        .str.strip()
        ==
        df["target_country"]
        .fillna("")
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    df["name_char"] = (
        df.apply(
            lambda r: char_similarity(
                r["s1_name"],
                r["target_name"],
            ),
            axis=1,
        )
    )

    df["name_token_set"] = (
        df.apply(
            lambda r: token_set_similarity(
                r["s1_name"],
                r["target_name"],
            ),
            axis=1,
        )
    )

    df["name_jaccard"] = (
        df.apply(
            lambda r: token_jaccard(
                r["s1_name"],
                r["target_name"],
            ),
            axis=1,
        )
    )

    df["name_containment"] = (
        df.apply(
            lambda r: token_containment(
                r["s1_name"],
                r["target_name"],
            ),
            axis=1,
        )
    )

    df["name_score"] = (
        df.apply(
            lambda r: name_score(
                r["s1_name"],
                r["target_name"],
            ),
            axis=1,
        )
    )

    # --------------------------------------------------------
    # ADDRESS
    # --------------------------------------------------------

    df["address_char"] = (
        df.apply(
            lambda r: char_similarity(
                r["s1_address"],
                r["target_address"],
            ),
            axis=1,
        )
    )

    df["address_token_set"] = (
        df.apply(
            lambda r: token_set_similarity(
                r["s1_address"],
                r["target_address"],
            ),
            axis=1,
        )
    )

    df["address_jaccard"] = (
        df.apply(
            lambda r: token_jaccard(
                r["s1_address"],
                r["target_address"],
            ),
            axis=1,
        )
    )

    df["address_containment"] = (
        df.apply(
            lambda r: token_containment(
                r["s1_address"],
                r["target_address"],
            ),
            axis=1,
        )
    )

    df["address_number_overlap"] = (
        df.apply(
            lambda r: number_overlap(
                r["s1_address"],
                r["target_address"],
            ),
            axis=1,
        )
    )

    df["address_score"] = (
        df.apply(
            lambda r: address_score(
                r["s1_address"],
                r["target_address"],
            ),
            axis=1,
        )
    )

    # --------------------------------------------------------
    # DECISION
    # --------------------------------------------------------

    df["recovery_class"] = df.apply(
        classify,
        axis=1,
    )

    # --------------------------------------------------------
    # OVERALL SCORE
    # --------------------------------------------------------

    df["combined_score"] = (
        0.55 * df["name_score"]
        + 0.45 * df["address_score"]
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    df = df.sort_values(
        [
            "recovery_class",
            "combined_score",
            "address_number_overlap",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    # --------------------------------------------------------
    # SAVE FULL ANALYSIS
    # --------------------------------------------------------

    print("\n[3/4] Saving analysis...")

    df.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # RECOVERED CANDIDATES
    # --------------------------------------------------------

    recovered_classes = {
        "STRONG",
        "ADDRESS_STRONG",
        "NAME_STRONG",
        "COMBINED",
    }

    recovered = df[
        df["recovery_class"].isin(
            recovered_classes
        )
    ].copy()

    recovered_out = recovered[
        [
            "source1_entity_id",
            "true_target_id",
            "target_source",
            "name_score",
            "address_score",
            "name_token_set",
            "address_token_set",
            "address_number_overlap",
            "country_match",
            "combined_score",
            "recovery_class",
        ]
    ]

    recovered_out.to_csv(
        RECOVERED,
        sep="\t",
        index=False,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n[4/4] RESULTS")
    print("=" * 70)

    print(
        f"Total remaining misses : {len(df):,}"
    )

    print(
        f"Potentially recoverable : {len(recovered):,}"
    )

    print(
        f"Potential recovery rate : "
        f"{len(recovered) / len(df):.4f}"
        if len(df)
        else "Potential recovery rate : N/A"
    )

    print("\nBy recovery class:")
    print(
        df["recovery_class"]
        .value_counts()
    )

    print("\nBy target source:")
    print(
        recovered_out["target_source"]
        .value_counts()
    )

    print("\nTop recovery candidates:")

    print(
        recovered_out[
            [
                "source1_entity_id",
                "true_target_id",
                "target_source",
                "name_score",
                "address_score",
                "address_number_overlap",
                "combined_score",
                "recovery_class",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Full analysis : {OUTPUT}"
    )

    print(
        f"Recovered     : {RECOVERED}"
    )


if __name__ == "__main__":
    main()