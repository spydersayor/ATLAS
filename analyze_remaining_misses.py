import pandas as pd
from rapidfuzz.fuzz import ratio, token_set_ratio, token_sort_ratio


MISS = "remaining_misses_diagnostics.tsv"

df = pd.read_csv(MISS, sep="\t", dtype=str).fillna("")


def sim(a, b):
    return ratio(str(a), str(b)) / 100.0


def token_set(a, b):
    return token_set_ratio(str(a), str(b)) / 100.0


def token_sort(a, b):
    return token_sort_ratio(str(a), str(b)) / 100.0


df["name_ratio"] = df.apply(
    lambda r: sim(r["s1_name"], r["target_name"]),
    axis=1,
)

df["name_token_set"] = df.apply(
    lambda r: token_set(r["s1_name"], r["target_name"]),
    axis=1,
)

df["name_token_sort"] = df.apply(
    lambda r: token_sort(r["s1_name"], r["target_name"]),
    axis=1,
)

df["address_ratio"] = df.apply(
    lambda r: sim(r["s1_address"], r["target_address"]),
    axis=1,
)

df["address_token_set"] = df.apply(
    lambda r: token_set(r["s1_address"], r["target_address"]),
    axis=1,
)

df["address_token_sort"] = df.apply(
    lambda r: token_sort(r["s1_address"], r["target_address"]),
    axis=1,
)

df["country_match"] = (
    df["s1_country"].str.lower().str.strip()
    ==
    df["target_country"].str.lower().str.strip()
)


print("=" * 70)
print("REMAINING MISS DIAGNOSTICS")
print("=" * 70)

print(f"Total remaining misses: {len(df):,}")

print("\nBy source:")
print(df["target_source"].value_counts())


print("\n" + "=" * 70)
print("NAME SIMILARITY")
print("=" * 70)

print(
    df[
        ["name_ratio", "name_token_set", "name_token_sort"]
    ].describe()
)


print("\n" + "=" * 70)
print("ADDRESS SIMILARITY")
print("=" * 70)

print(
    df[
        ["address_ratio", "address_token_set", "address_token_sort"]
    ].describe()
)


print("\n" + "=" * 70)
print("THRESHOLD COVERAGE")
print("=" * 70)

for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:

    name = (df["name_token_set"] >= t)
    address = (df["address_token_set"] >= t)
    either = name | address

    print(
        f"{t:.2f}  "
        f"name={name.sum():4d}  "
        f"address={address.sum():4d}  "
        f"OR={either.sum():4d}"
    )


print("\n" + "=" * 70)
print("HIGH ADDRESS / LOW NAME CASES")
print("=" * 70)

interesting = df[
    (df["address_token_set"] >= 0.75)
    &
    (df["name_token_set"] < 0.50)
].copy()

print(f"Count: {len(interesting):,}")

print(
    interesting[
        [
            "source1_entity_id",
            "true_target_id",
            "target_source",
            "s1_name",
            "target_name",
            "s1_address",
            "target_address",
            "name_token_set",
            "address_token_set",
        ]
    ].head(30).to_string(index=False)
)


print("\n" + "=" * 70)
print("HIGH NAME / LOW ADDRESS CASES")
print("=" * 70)

interesting = df[
    (df["name_token_set"] >= 0.75)
    &
    (df["address_token_set"] < 0.50)
].copy()

print(f"Count: {len(interesting):,}")

print(
    interesting[
        [
            "source1_entity_id",
            "true_target_id",
            "target_source",
            "s1_name",
            "target_name",
            "s1_address",
            "target_address",
            "name_token_set",
            "address_token_set",
        ]
    ].head(30).to_string(index=False)
)


print("\n" + "=" * 70)
print("VERY LOW NAME + VERY LOW ADDRESS")
print("=" * 70)

hard = df[
    (df["name_token_set"] < 0.50)
    &
    (df["address_token_set"] < 0.50)
].copy()

print(f"Count: {len(hard):,}")

print(
    hard[
        [
            "source1_entity_id",
            "true_target_id",
            "target_source",
            "s1_name",
            "target_name",
            "s1_address",
            "target_address",
            "name_token_set",
            "address_token_set",
        ]
    ].head(50).to_string(index=False)
)


OUT = "remaining_misses_similarity.tsv"
df.to_csv(OUT, sep="\t", index=False)

print(f"\nSaved: {OUT}")
