import pandas as pd
import re
import unicodedata
from collections import defaultdict


S1_PATH = "dataset/train/train_source1.tsv"
S2_PATH = "dataset/train/train_source2.tsv"
S3_PATH = "dataset/train/train_source3.tsv"
MISSED_PATH = "missed_true_pairs_diagnostics.tsv"

S1_LIMIT = 2000


def norm(value):
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()

    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def char_ngrams(text, n=3):
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()

    return {
        text[i:i+n]
        for i in range(len(text) - n + 1)
    }


def jaccard(a, b):
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


print("Loading data...")

s1 = pd.read_csv(S1_PATH, sep="\t", dtype=str).head(S1_LIMIT)
s2 = pd.read_csv(S2_PATH, sep="\t", dtype=str)
s3 = pd.read_csv(S3_PATH, sep="\t", dtype=str)

missed = pd.read_csv(MISSED_PATH, sep="\t", dtype=str)

print(f"S1: {len(s1):,}")
print(f"S2: {len(s2):,}")
print(f"S3: {len(s3):,}")
print(f"Missed pairs: {len(missed):,}")


# ------------------------------------------------------------
# Normalize only the missed-pair targets + corresponding S1.
# ------------------------------------------------------------

s1_lookup = (
    s1.set_index("entity_id")
    .to_dict("index")
)

s2_lookup = (
    s2.set_index("entity_id")
    .to_dict("index")
)

s3_lookup = (
    s3.set_index("entity_id")
    .to_dict("index")
)


# ------------------------------------------------------------
# Test whether a missed pair would be recovered by
# character n-gram similarity.
# ------------------------------------------------------------

results = []

for row in missed.itertuples(index=False):

    s1_id = str(row.source1_entity_id)
    target_id = str(row.true_target_id)
    source = str(row.target_source)

    s1_row = s1_lookup.get(s1_id)

    if source == "S2":
        target_row = s2_lookup.get(target_id)
    else:
        target_row = s3_lookup.get(target_id)

    if s1_row is None or target_row is None:
        continue

    s1_name = norm(s1_row.get("business_name", ""))
    target_name = norm(target_row.get("business_name", ""))

    s1_address = norm(s1_row.get("business_address", ""))
    target_address = norm(target_row.get("business_address", ""))

    s1_country = norm(s1_row.get("country", ""))
    target_country = norm(target_row.get("country", ""))

    name_score = jaccard(
        char_ngrams(s1_name, 3),
        char_ngrams(target_name, 3),
    )

    address_score = jaccard(
        char_ngrams(s1_address, 3),
        char_ngrams(target_address, 3),
    )

    country_match = (
        bool(s1_country)
        and bool(target_country)
        and s1_country == target_country
    )

    results.append(
        {
            "source1_entity_id": s1_id,
            "true_target_id": target_id,
            "target_source": source,
            "name_score": name_score,
            "address_score": address_score,
            "country_match": country_match,
        }
    )


df = pd.DataFrame(results)

print()
print("=" * 70)
print("FUZZY BLOCKER RECOVERY ANALYSIS")
print("=" * 70)

print(f"Analyzed missed pairs: {len(df):,}")


# ------------------------------------------------------------
# Test several thresholds.
# ------------------------------------------------------------

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
]


print()
print(
    f"{'threshold':>10} "
    f"{'name':>10} "
    f"{'address':>10} "
    f"{'name OR address':>18}"
)

print("-" * 55)

for threshold in thresholds:

    name_hits = (
        df["country_match"]
        & (df["name_score"] >= threshold)
    )

    address_hits = (
        df["country_match"]
        & (df["address_score"] >= threshold)
    )

    either_hits = name_hits | address_hits

    print(
        f"{threshold:>10.2f} "
        f"{name_hits.sum():>10,} "
        f"{address_hits.sum():>10,} "
        f"{either_hits.sum():>18,}"
    )


# ------------------------------------------------------------
# Recommended initial threshold.
# ------------------------------------------------------------

threshold = 0.55

df["fuzzy_name"] = (
    df["country_match"]
    & (df["name_score"] >= threshold)
)

df["fuzzy_address"] = (
    df["country_match"]
    & (df["address_score"] >= threshold)
)

df["fuzzy_hit"] = (
    df["fuzzy_name"]
    | df["fuzzy_address"]
)


print()
print("=" * 70)
print(f"INITIAL FUZZY BLOCKER @ {threshold:.2f}")
print("=" * 70)

print(
    f"Recovered missed pairs: "
    f"{df['fuzzy_hit'].sum():,} / {len(df):,}"
)

print(
    f"Recovery rate: "
    f"{100 * df['fuzzy_hit'].mean():.2f}%"
)

print()
print("By source:")

print(
    df[df["fuzzy_hit"]]
    .groupby("target_source")
    .size()
    .to_string()
)


# ------------------------------------------------------------
# Save recovered examples.
# ------------------------------------------------------------

recovered = df[df["fuzzy_hit"]].copy()

recovered = recovered.sort_values(
    ["name_score", "address_score"],
    ascending=False,
)

recovered.to_csv(
    "fuzzy_recovered_missed_pairs.tsv",
    sep="\t",
    index=False,
)

print()
print(
    "Saved: fuzzy_recovered_missed_pairs.tsv"
)


# ------------------------------------------------------------
# Print strongest examples.
# ------------------------------------------------------------

print()
print("=" * 70)
print("TOP RECOVERED MISSED PAIRS")
print("=" * 70)

print(
    recovered.head(30).to_string(index=False)
)
