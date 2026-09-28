from pathlib import Path
from collections import defaultdict
import csv
import re
import unicodedata
import gc
import time

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "test"
OUT = ROOT / "output"

S1 = DATA / "test_source1.tsv"
S2 = DATA / "test_source2.tsv"
S3 = DATA / "test_source3.tsv"

OUTPUT = OUT / "candidate_pairs.tsv"

CHUNK_SIZE = 25_000

# ------------------------------------------------------------
# Frequency caps.
#
# WHY: without a cap, a common name/address key can match tens of
# thousands of target records. Every S1 row that also hits that key
# then has to iterate the entire bucket -- this is a major hidden
# cost in the original script (no cap existed at all), and it also
# inflates candidate_pairs.tsv, which is penalized separately per the
# problem statement ("smaller candidate set... ranked higher").
# Buckets larger than the cap are dropped entirely after indexing
# (mirrors the same fix already validated in the pandas blocking.py
# pipeline: max_exact_key_frequency / max_signature_frequency = 100).
# ------------------------------------------------------------
MAX_BUCKET_SIZE = 100


# ============================================================
# FAST NORMALIZATION
#
# CHANGED: norm() now skips the unicodedata NFKD + combining-mark
# pass entirely when the string is already pure ASCII (str.isascii()
# is a fast C-level check). Most US records are pure ASCII, so this
# avoids the expensive Unicode decomposition on the majority of rows.
#
# CHANGED: token_signature_from_norm() takes an ALREADY-normalized
# string instead of re-normalizing from raw input. Every call site
# below normalizes a field exactly once and derives both the plain
# normalized value and its token signature from that single result --
# the original script normalized every field twice (once directly,
# once again inside token_signature()).
# ============================================================

_non_alnum = re.compile(r"[^a-z0-9]+")


def norm(value):
    if not value:
        return ""

    if not value.isascii():
        value = unicodedata.normalize("NFKD", value)
        value = "".join(
            c for c in value
            if not unicodedata.combining(c)
        )

    value = value.lower()
    value = _non_alnum.sub(" ", value)

    # The regex above already collapses any run of non-alnum chars
    # (including whitespace) into a single space, so the only thing
    # left to clean up is a possible leading/trailing space -- a plain
    # strip() is cheaper than the previous " ".join(value.split()),
    # which re-tokenized and rebuilt the whole string just to do the
    # same thing.
    return value.strip()


def token_signature_from_norm(normed):
    """Token signature derived from an ALREADY-normalized string.
    Do not pass raw input here -- call norm() first and reuse it."""

    if not normed:
        return ""

    return " ".join(sorted(set(normed.split())))


def prefix_from_norm(normed, length=8):
    """Prefix derived from an ALREADY-normalized string."""

    if not normed:
        return ""

    return normed[:length]


# ============================================================
# BLOCKER BITMASK
#
# WHY: candidates_for_row previously tracked provenance as a
# per-candidate Python set of strings ({"name_exact", "address_exact",
# ...}), unioned with .add() as more blockers matched, then converted
# to "name_exact|address_exact" via sorted()+join() at write time.
# That's a set object + set mutation for every single candidate hit,
# times ~1.7M S1 rows x 2 sources x up to 8 blockers.
#
# Using an int bitmask instead means each blocker hit is a single
# integer OR (found[cid] |= bit), and the string label is looked up
# once from a precomputed table (only 2**8 = 256 possible masks) at
# write time instead of being rebuilt from a set every time.
# ============================================================

BLOCKER_BITS = {
    "name_exact": 1,
    "address_exact": 2,
    "name_country": 4,
    "address_country": 8,
    "name_token_country": 16,
    "address_token_country": 32,
    "name_prefix_country": 64,
    "address_prefix_country": 128,
}

_BLOCKER_ORDER = [
    "address_country",
    "address_exact",
    "address_prefix_country",
    "address_token_country",
    "name_country",
    "name_exact",
    "name_prefix_country",
    "name_token_country",
]

# Precompute every possible mask -> "blocker1|blocker2|..." label once,
# up front (only 255 non-zero combinations for 8 bits) instead of
# rebuilding it per candidate at write time.
MASK_TO_LABEL = {}

for _mask in range(1, 256):
    _labels = [
        _name for _name in _BLOCKER_ORDER
        if _mask & BLOCKER_BITS[_name]
    ]
    MASK_TO_LABEL[_mask] = "|".join(_labels)


def mark(found, dictionary, key, bit):
    """Module-level (not a per-call closure) so it isn't redefined on
    every candidates_for_row() call. Mutates `found` (candidate_id ->
    bitmask) in place."""

    if not key:
        return

    ids = dictionary.get(key)

    if not ids:
        return

    for candidate_id in ids:
        found[candidate_id] = found.get(candidate_id, 0) | bit


# ============================================================
# INDEX
#
# IMPORTANT:
# We store ONLY IDs.
# No pandas.
# No groupby.
# No DataFrame copies.
#
# CHANGED: defaultdict(list) instead of manual get/set -- one dict
# operation per insert instead of a get-then-branch-then-set.
# ============================================================

def build_index(path, source):
    print(f"\nBuilding compact indexes for {source}...")

    name_exact = defaultdict(list)
    address_exact = defaultdict(list)

    name_country = defaultdict(list)
    address_country = defaultdict(list)

    name_token_country = defaultdict(list)
    address_token_country = defaultdict(list)

    name_prefix_country = defaultdict(list)
    address_prefix_country = defaultdict(list)

    count = 0

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
        newline="",
    ) as f:

        # CHANGED: csv.reader + explicit column indices instead of
        # DictReader. DictReader builds a new dict per row; plain
        # tuple/list indexing from csv.reader is measurably faster at
        # multi-million-row scale.
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}

        i_eid = col["entity_id"]
        i_name = col["business_name"]
        i_addr = col["business_address"]
        i_country = col["country"]

        for row in reader:

            eid = row[i_eid]

            # Normalize each field EXACTLY ONCE, then derive
            # token-signature / prefix from that single result.
            name = norm(row[i_name]) if len(row) > i_name else ""
            address = norm(row[i_addr]) if len(row) > i_addr else ""
            country = norm(row[i_country]) if len(row) > i_country else ""

            # ------------------------------------------------
            # Exact
            # ------------------------------------------------

            if name:
                name_exact[name].append(eid)

            if address:
                address_exact[address].append(eid)

            # ------------------------------------------------
            # Country composites
            # ------------------------------------------------

            if country:

                if name:
                    name_country[name + "\x1f" + country].append(eid)

                if address:
                    address_country[address + "\x1f" + country].append(eid)

                # ------------------------------------------------
                # Token signatures (derived from already-normalized
                # name/address -- NOT re-normalized from raw input)
                # ------------------------------------------------

                nt = token_signature_from_norm(name)
                at = token_signature_from_norm(address)

                if nt:
                    name_token_country[nt + "\x1f" + country].append(eid)

                if at:
                    address_token_country[at + "\x1f" + country].append(eid)

                # ------------------------------------------------
                # Prefix (also derived from already-normalized value)
                # ------------------------------------------------

                np_ = prefix_from_norm(name)
                ap_ = prefix_from_norm(address)

                if np_:
                    name_prefix_country[np_ + "\x1f" + country].append(eid)

                if ap_:
                    address_prefix_country[ap_ + "\x1f" + country].append(eid)

            count += 1

            if count % 500_000 == 0:
                print(
                    f"{source}: {count:,} rows indexed"
                )

    print(
        f"{source}: {count:,} rows complete"
    )

    indexes = {
        "name_exact": name_exact,
        "address_exact": address_exact,
        "name_country": name_country,
        "address_country": address_country,
        "name_token_country": name_token_country,
        "address_token_country": address_token_country,
        "name_prefix_country": name_prefix_country,
        "address_prefix_country": address_prefix_country,
    }

    # --------------------------------------------------------
    # Frequency cap: drop any bucket bigger than MAX_BUCKET_SIZE.
    #
    # This bounds the worst-case per-S1-row lookup cost (a huge
    # bucket would otherwise be iterated once for every S1 row that
    # also hits that key) and keeps candidate_pairs.tsv small, which
    # matters both for wall-clock time and for the separate
    # candidate-set-size scoring criterion.
    # --------------------------------------------------------
    dropped_keys = 0
    dropped_ids = 0

    for name_, bucket_index in indexes.items():
        oversized = [
            key for key, ids in bucket_index.items()
            if len(ids) > MAX_BUCKET_SIZE
        ]

        for key in oversized:
            dropped_ids += len(bucket_index[key])
            del bucket_index[key]

        dropped_keys += len(oversized)

    print(
        f"{source}: frequency cap removed {dropped_keys:,} oversized "
        f"keys ({dropped_ids:,} postings) above {MAX_BUCKET_SIZE}"
    )

    return indexes


# ============================================================
# CANDIDATES FOR ONE S1 ROW
#
# CHANGED: same single-normalize-then-derive pattern as build_index.
# Also collect() now uses dict.get with a plain "in" check removed --
# unchanged logically, just kept tight since this runs ~1.7M x 8 times.
# ============================================================

def candidates_for_row(eid, name_raw, address_raw, country_raw, source, indexes):

    name = norm(name_raw)
    address = norm(address_raw)
    country = norm(country_raw)

    nt = token_signature_from_norm(name)
    at = token_signature_from_norm(address)

    np_ = prefix_from_norm(name)
    ap_ = prefix_from_norm(address)

    found = {}  # candidate_id -> blocker bitmask

    # --------------------------------------------------------
    # Strong blockers first
    # --------------------------------------------------------

    mark(found, indexes["name_exact"], name, BLOCKER_BITS["name_exact"])
    mark(found, indexes["address_exact"], address, BLOCKER_BITS["address_exact"])

    if country:

        mark(
            found,
            indexes["name_country"],
            name + "\x1f" + country,
            BLOCKER_BITS["name_country"],
        )

        mark(
            found,
            indexes["address_country"],
            address + "\x1f" + country,
            BLOCKER_BITS["address_country"],
        )

        mark(
            found,
            indexes["name_token_country"],
            nt + "\x1f" + country,
            BLOCKER_BITS["name_token_country"],
        )

        mark(
            found,
            indexes["address_token_country"],
            at + "\x1f" + country,
            BLOCKER_BITS["address_token_country"],
        )

        mark(
            found,
            indexes["name_prefix_country"],
            np_ + "\x1f" + country,
            BLOCKER_BITS["name_prefix_country"],
        )

        mark(
            found,
            indexes["address_prefix_country"],
            ap_ + "\x1f" + country,
            BLOCKER_BITS["address_prefix_country"],
        )

    for candidate_id, mask in found.items():

        yield (
            eid,
            candidate_id,
            source,
            MASK_TO_LABEL[mask],
            mask.bit_count(),
        )


# ============================================================
# MAIN
# ============================================================

def main():

    start = time.time()

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if OUTPUT.exists():
        OUTPUT.unlink()

    print("=" * 70)
    print("ATLAS FAST CANDIDATE GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Build S2 index
    # --------------------------------------------------------

    s2_index = build_index(
        S2,
        "S2",
    )

    # --------------------------------------------------------
    # Build S3 index
    # --------------------------------------------------------

    s3_index = build_index(
        S3,
        "S3",
    )

    # --------------------------------------------------------
    # Open output
    #
    # CHANGED: explicit large buffer (1 MiB) on the output file, and
    # rows are batched into a list and flushed with writerows() every
    # CHUNK_SIZE S1 records instead of one writer.writerow() call per
    # candidate -- fewer Python-level function calls over the same
    # total row count.
    # --------------------------------------------------------

    columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "blocker_sources",
        "num_blockers",
    ]

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
        newline="",
        buffering=1 << 20,
    ) as out:

        writer = csv.writer(
            out,
            delimiter="\t",
        )

        writer.writerow(columns)

        total_s1 = 0
        total_candidates = 0
        row_buffer = []

        # ----------------------------------------------------
        # Stream S1
        # ----------------------------------------------------

        print("\nStreaming TEST S1...")

        with open(
            S1,
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as f:

            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            col = {name: i for i, name in enumerate(header)}

            i_eid = col["entity_id"]
            i_name = col["business_name"]
            i_addr = col["business_address"]
            i_country = col["country"]

            for row in reader:

                total_s1 += 1

                eid = row[i_eid]
                name_raw = row[i_name] if len(row) > i_name else ""
                address_raw = row[i_addr] if len(row) > i_addr else ""
                country_raw = row[i_country] if len(row) > i_country else ""

                # S2
                for candidate in candidates_for_row(
                    eid, name_raw, address_raw, country_raw,
                    "S2",
                    s2_index,
                ):
                    row_buffer.append(candidate)
                    total_candidates += 1

                # S3
                for candidate in candidates_for_row(
                    eid, name_raw, address_raw, country_raw,
                    "S3",
                    s3_index,
                ):
                    row_buffer.append(candidate)
                    total_candidates += 1

                if total_s1 % CHUNK_SIZE == 0:

                    writer.writerows(row_buffer)
                    row_buffer.clear()
                    out.flush()

                    elapsed = time.time() - start

                    print(
                        f"S1 processed: "
                        f"{total_s1:,} | "
                        f"candidates: "
                        f"{total_candidates:,} | "
                        f"time: "
                        f"{elapsed / 60:.1f} min"
                    )

            # Flush whatever's left in the buffer.
            if row_buffer:
                writer.writerows(row_buffer)
                row_buffer.clear()

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"S1 processed : {total_s1:,}"
    )

    print(
        f"Candidates    : {total_candidates:,}"
    )

    print(
        f"Time          : {elapsed / 60:.2f} minutes"
    )

    print(
        f"\nOUTPUT:"
    )

    print(
        OUTPUT
    )

    print("\nDONE.")

    del s2_index
    del s3_index

    gc.collect()


if __name__ == "__main__":
    main()