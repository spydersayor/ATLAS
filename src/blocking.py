"""
ATLAS - Blocking + Candidate Generation

Owner:
    Satyam - Blocking + Candidate Generation/Fusion

Purpose:
    Generate candidate pairs between Source 1 and Source 2/3
    using deterministic blocking strategies.

Pipeline position:

    RAW DATA
        ↓
    NORMALIZATION
        ↓
    FINGERPRINTING
        ↓
    BLOCKING  ← this module
        ↓
    CANDIDATE FUSION
        ↓
    FEATURE ENGINEERING
        ↓
    MATCHING

Important:
    - This module performs retrieval only.
    - It does NOT perform final entity matching.
    - It does NOT use ground truth.
    - It does NOT perform exhaustive O(N²) comparison.
    - Final matches must be a subset of generated candidate pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re
import pickle
import tempfile

import pandas as pd


# ============================================================================
# CONTRACT
# ============================================================================

REQUIRED_SOURCE_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}

REQUIRED_FINGERPRINT_COLUMNS = {
    "name_fingerprint",
    "address_fingerprint",
    "country_fingerprint",
}

REQUIRED_CANDIDATE_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "blocker_sources",
    "num_blockers",
}

VALID_CANDIDATE_SOURCES = {"S2", "S3"}


# ============================================================================
# BLOCKING STRATEGIES
# ============================================================================

BLOCKER_NAME_EXACT = "name_exact"
BLOCKER_ADDRESS_EXACT = "address_exact"
BLOCKER_NAME_COUNTRY = "name_country"
BLOCKER_ADDRESS_COUNTRY = "address_country"

BLOCKER_NAME_TOKEN = "name_token_country"
BLOCKER_NAME_PREFIX = "name_prefix_country"
BLOCKER_ADDRESS_TOKEN = "address_token_country"
BLOCKER_ADDRESS_PREFIX = "address_prefix_country"

BLOCKER_NAME_TOKEN_PAIR = "name_token_pair_country"
BLOCKER_ADDRESS_TOKEN_PAIR = "address_token_pair_country"

BLOCKER_NAME_NGRAM = "name_ngram_country"
BLOCKER_ADDRESS_NGRAM = "address_ngram_country"

# Simple, uncapped-key (no country component) high-recall token blocker
# over normalized business names. Named *_SIMPLE to avoid colliding with
# the existing BLOCKER_NAME_TOKEN ("name_token_country") constant above.
BLOCKER_NAME_TOKEN_SIMPLE = "name_token"

BLOCKER_NAMES = (
    BLOCKER_NAME_EXACT,
    BLOCKER_ADDRESS_EXACT,
    BLOCKER_NAME_COUNTRY,
    BLOCKER_ADDRESS_COUNTRY,
    BLOCKER_NAME_TOKEN,
    BLOCKER_NAME_PREFIX,
    BLOCKER_ADDRESS_TOKEN,
    BLOCKER_ADDRESS_PREFIX,
    BLOCKER_NAME_TOKEN_PAIR,
    BLOCKER_ADDRESS_TOKEN_PAIR,
    BLOCKER_NAME_TOKEN_SIMPLE,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class BlockingConfig:
    """
    Configuration for deterministic multi-strategy blocking.

    Token/prefix blockers are frequency-capped so that common
    tokens do not create an uncontrolled candidate explosion.
    """

    # Existing exact blockers
    enable_name_exact: bool = True
    enable_address_exact: bool = True
    enable_name_country: bool = True
    enable_address_country: bool = True

    # New bounded blockers
    enable_name_token_country: bool = True
    enable_name_prefix_country: bool = True
    enable_address_token_country: bool = True
    enable_address_prefix_country: bool = True

    # Token-pair blockers
    enable_name_token_pair_country: bool = True
    enable_address_token_pair_country: bool = True

    # Safety controls
    min_token_length: int = 3
    max_token_signatures_per_record: int = 3

    # Minimum token length considered when building the order-independent
    # two-token pair signature used by the *_token_pair_country blockers.
    pair_min_token_length: int = 2

    # Ignore signatures that occur too frequently in target data
    max_signature_frequency: int = 100

    # Prefix length used by prefix blocking
    prefix_length: int = 4

    # Number of source1 rows processed per chunk in _run_signature_block,
    # so per-blocker candidate output is never held in memory for the
    # entire source1 dataset at once.
    source1_chunk_size: int = 100_000

    # Simple, no-country, high-recall token blocker over
    # business_name_normalized. Complements name_token_country: no
    # country key, but tightened to reuse max_signature_frequency for its
    # postings cap (not a separate 1%-of-target formula), filter common
    # business stopwords, and require a longer minimum token length.
    enable_name_token: bool = True

    # Minimum token length for the name_token blocker specifically.
    # Raised above min_token_length (3) because this blocker has no
    # country key, so short tokens are far more likely to collide
    # across unrelated businesses.
    name_token_min_length: int = 4


# ============================================================================
# VALIDATION
# ============================================================================

def validate_source_dataframe(
    df: pd.DataFrame,
    source_name: str,
) -> None:
    """
    Validate the structural contract of a source DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{source_name} must be a pandas DataFrame."
        )

    missing = REQUIRED_SOURCE_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    if df["entity_id"].isna().any():
        raise ValueError(
            f"{source_name} contains null entity_id values."
        )

    if (
        df["entity_id"]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):
        raise ValueError(
            f"{source_name} contains empty entity_id values."
        )

    if df["entity_id"].duplicated().any():
        duplicate_count = int(
            df["entity_id"].duplicated().sum()
        )

        raise ValueError(
            f"{source_name} contains "
            f"{duplicate_count} duplicate entity_id values."
        )


def validate_fingerprints(
    df: pd.DataFrame,
    source_name: str,
) -> None:
    """
    Validate that required fingerprint columns exist.
    """

    missing = (
        REQUIRED_FINGERPRINT_COLUMNS
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{source_name} is missing required fingerprint "
            f"columns: {sorted(missing)}"
        )


def validate_candidate_pairs(
    candidate_pairs: pd.DataFrame,
) -> None:
    """
    Validate the ATLAS candidate-pair contract.
    """

    if not isinstance(candidate_pairs, pd.DataFrame):
        raise TypeError(
            "candidate_pairs must be a pandas DataFrame."
        )

    missing = (
        REQUIRED_CANDIDATE_COLUMNS
        - set(candidate_pairs.columns)
    )

    if missing:
        raise ValueError(
            "Candidate-pair contract violation. "
            f"Missing columns: {sorted(missing)}"
        )

    invalid_sources = set(
        candidate_pairs["candidate_source"].dropna().unique()
    ) - VALID_CANDIDATE_SOURCES

    if invalid_sources:
        raise ValueError(
            "Invalid candidate_source values: "
            f"{sorted(invalid_sources)}"
        )

    if candidate_pairs[
        ["source1_entity_id", "candidate_entity_id"]
    ].isna().any().any():
        raise ValueError(
            "Candidate IDs cannot contain null values."
        )

    if candidate_pairs[
        ["source1_entity_id", "candidate_entity_id"]
    ].astype(str).apply(
        lambda column: column.str.strip().eq("")
    ).any().any():
        raise ValueError(
            "Candidate IDs cannot contain empty values."
        )

    if candidate_pairs["num_blockers"].isna().any():
        raise ValueError(
            "num_blockers cannot contain null values."
        )

    if (
        candidate_pairs["num_blockers"]
        .astype(int)
        .lt(1)
        .any()
    ):
        raise ValueError(
            "num_blockers must be >= 1."
        )

    duplicate_keys = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]

    if candidate_pairs.duplicated(
        subset=duplicate_keys
    ).any():
        raise ValueError(
            "Duplicate candidate pairs detected."
        )


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _clean_key_series(
    series: pd.Series,
) -> pd.Series:
    """
    Convert blocking keys to safe strings.

    Empty strings remain empty and are excluded from indexing.
    """

    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


_TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _tokenize(value: str) -> list[str]:
    if not value:
        return []
    return _TOKEN_PATTERN.findall(str(value))


def _valid_tokens(value: str, min_length: int) -> list[str]:
    return [
        token
        for token in _tokenize(value)
        if len(token) >= min_length
    ]


def _token_signature(value: str, min_length: int) -> str:
    tokens = _valid_tokens(value, min_length)

    if not tokens:
        return ""

    return sorted(
        tokens,
        key=lambda token: (-len(token), token)
    )[0]


def _token_signatures(
    value: str,
    min_length: int,
    max_signatures: int,
) -> list[str]:
    """
    Return multiple deterministic token signatures.

    Longer tokens are preferred because they are generally more
    discriminative than short/common tokens.
    """
    tokens = sorted(
        set(_valid_tokens(value, min_length)),
        key=lambda token: (-len(token), token),
    )

    return tokens[:max_signatures]


# Common legal/business suffix tokens that are too generic to be
# discriminative on their own. Shared by the token-pair blockers and the
# no-country-key name_token blocker.
_COMMON_BUSINESS_STOPWORDS = {
    "ltd",
    "limited",
    "llc",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "pvt",
    "private",
    "plc",
    "llp",
    "lp",
    "services",
    "service",
}


def _token_pair_signature(
    value: str,
    min_token_length: int,
) -> str:
    """
    Return one deterministic, order-independent pair of informative tokens.

    The two longest distinct valid tokens are selected. Common legal/business
    suffix tokens are excluded so the pair remains reasonably discriminative.
    """

    tokens = {
        token
        for token in _valid_tokens(
            value,
            min_token_length,
        )
        if token.lower() not in _COMMON_BUSINESS_STOPWORDS
    }

    if len(tokens) < 2:
        return ""

    selected = sorted(
        tokens,
        key=lambda token: (-len(token), token),
    )[:2]

    # Order-independent pair.
    selected = sorted(selected)

    return "\x1e".join(selected)


def _prefix_signature(
    value: str,
    min_length: int,
    prefix_length: int,
) -> str:
    token = _token_signature(value, min_length)

    if not token:
        return ""

    if len(token) <= prefix_length:
        return token

    return token[:prefix_length]


def _make_composite_key(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    """
    Create a deterministic composite blocking key.

    A separator that is not expected in fingerprints is used.
    """

    left = _clean_key_series(left)
    right = _clean_key_series(right)

    return left + "\x1f" + right


def _prepare_target_index(
    target: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    """
    Prepare target-side lookup rows.

    Only non-empty blocking keys are retained.

    Duplicate target keys are intentionally NOT removed because
    multiple source records may legitimately share the same
    fingerprint and must all become candidates.
    """

    columns = [
        "entity_id",
        *key_columns,
    ]

    index = target[columns].copy()

    for column in key_columns:
        index[column] = _clean_key_series(
            index[column]
        )

    valid = pd.Series(
        True,
        index=index.index,
    )

    for column in key_columns:
        valid &= index[column].ne("")

    return index.loc[valid].copy()


# ============================================================================
# INDIVIDUAL BLOCKERS
# ============================================================================

def _run_single_key_block(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    target_source: str,
    key_column: str,
    blocker_name: str,
) -> pd.DataFrame:
    """
    Execute one exact-key blocker.

    Example:
        source1.name_fingerprint
            ↔
        source2.name_fingerprint

    Returns only:
        source1_entity_id
        candidate_entity_id
        candidate_source
        blocker_sources
        num_blockers
    """

    s1 = source1[
        ["entity_id", key_column]
    ].copy()

    target_index = _prepare_target_index(
        target,
        [key_column],
    )

    s1[key_column] = _clean_key_series(
        s1[key_column]
    )

    # Never index empty blocking keys.
    s1 = s1.loc[
        s1[key_column].ne("")
    ].copy()

    if s1.empty or target_index.empty:
        return _empty_candidate_frame()

    merged = s1.merge(
        target_index,
        on=key_column,
        how="inner",
        sort=False,
        copy=False,
        suffixes=("_s1", "_candidate"),
    )

    if merged.empty:
        return _empty_candidate_frame()

    # Defensive dedup: entity_id is validated unique per source, so an
    # exact-key merge should not be able to produce duplicate
    # (source1_entity_id, candidate_entity_id) pairs. Guard against it
    # anyway rather than trusting that assumption silently.
    merged = merged.drop_duplicates(
        subset=["entity_id_s1", "entity_id_candidate"]
    )

    assert not merged.duplicated(
        subset=["entity_id_s1", "entity_id_candidate"]
    ).any()

    result = pd.DataFrame(
        {
            "source1_entity_id": merged["entity_id_s1"],
            "candidate_entity_id": merged["entity_id_candidate"],
            "candidate_source": target_source,
            "blocker_sources": blocker_name,
            "num_blockers": 1,
        }
    )

    return result


def _run_composite_block(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    target_source: str,
    left_column: str,
    right_column: str,
    blocker_name: str,
) -> pd.DataFrame:
    """
    Execute an exact composite-key blocker.

    Example:
        name_fingerprint + country_fingerprint

    Only records with BOTH non-empty components participate.
    """

    s1 = source1[
        [
            "entity_id",
            left_column,
            right_column,
        ]
    ].copy()

    target_index = target[
        [
            "entity_id",
            left_column,
            right_column,
        ]
    ].copy()

    s1[left_column] = _clean_key_series(
        s1[left_column]
    )
    s1[right_column] = _clean_key_series(
        s1[right_column]
    )

    target_index[left_column] = _clean_key_series(
        target_index[left_column]
    )
    target_index[right_column] = _clean_key_series(
        target_index[right_column]
    )

    # Both components must be present.
    s1 = s1.loc[
        s1[left_column].ne("")
        & s1[right_column].ne("")
    ].copy()

    target_index = target_index.loc[
        target_index[left_column].ne("")
        & target_index[right_column].ne("")
    ].copy()

    if s1.empty or target_index.empty:
        return _empty_candidate_frame()

    s1["_blocking_key"] = _make_composite_key(
        s1[left_column],
        s1[right_column],
    )

    target_index["_blocking_key"] = _make_composite_key(
        target_index[left_column],
        target_index[right_column],
    )

    merged = s1[
        ["entity_id", "_blocking_key"]
    ].merge(
        target_index[
            ["entity_id", "_blocking_key"]
        ],
        on="_blocking_key",
        how="inner",
        sort=False,
        copy=False,
        suffixes=("_s1", "_candidate"),
    )

    if merged.empty:
        return _empty_candidate_frame()

    # Defensive dedup: same rationale as _run_single_key_block -- entity_id
    # uniqueness per source means this shouldn't produce duplicate pairs,
    # but assert it rather than assume it.
    merged = merged.drop_duplicates(
        subset=["entity_id_s1", "entity_id_candidate"]
    )

    assert not merged.duplicated(
        subset=["entity_id_s1", "entity_id_candidate"]
    ).any()

    result = pd.DataFrame(
        {
            "source1_entity_id": merged["entity_id_s1"],
            "candidate_entity_id": merged["entity_id_candidate"],
            "candidate_source": target_source,
            "blocker_sources": blocker_name,
            "num_blockers": 1,
        }
    )

    return result


def _empty_candidate_frame() -> pd.DataFrame:
    """
    Return an empty DataFrame with the exact candidate contract.
    """

    return pd.DataFrame(
        {
            "source1_entity_id": pd.Series(
                dtype="string"
            ),
            "candidate_entity_id": pd.Series(
                dtype="string"
            ),
            "candidate_source": pd.Series(
                dtype="string"
            ),
            "blocker_sources": pd.Series(
                dtype="string"
            ),
            "num_blockers": pd.Series(
                dtype="int64"
            ),
        }
    )


# ============================================================================
# CANDIDATE FUSION
# ============================================================================

def fuse_candidate_pairs(
    candidate_frames: Iterable[pd.DataFrame],
) -> pd.DataFrame:
    """
    Fuse candidate-pair frames while keeping blocker provenance.

    Uses a compact uint16 bitmask instead of expanding blocker_sources
    into Python lists. This is substantially more memory-efficient for
    large candidate sets.

    Each blocker is represented by one bit:
        name_exact                  -> 1
        address_exact               -> 2
        name_country                -> 4
        address_country             -> 8
        name_token_country          -> 16
        name_prefix_country         -> 32
        address_token_country       -> 64
        address_prefix_country      -> 128
        name_token_pair_country     -> 256
        address_token_pair_country  -> 512
        name_token                  -> 1024

    The public output contract remains unchanged:
        source1_entity_id
        candidate_entity_id
        candidate_source
        blocker_sources
        num_blockers
    """

    frames = [
        frame
        for frame in candidate_frames
        if frame is not None and not frame.empty
    ]

    if not frames:
        return _empty_candidate_frame()

    key_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]

    blocker_bits = {
        "name_exact": 1,
        "address_exact": 2,
        "name_country": 4,
        "address_country": 8,
        "name_token_country": 16,
        "name_prefix_country": 32,
        "address_token_country": 64,
        "address_prefix_country": 128,
        "name_token_pair_country": 256,
        "address_token_pair_country": 512,
        "name_token": 1024,
    }

    compact_frames: list[pd.DataFrame] = []

    for frame in frames:
        missing_columns = [
            column
            for column in key_columns + ["blocker_sources"]
            if column not in frame.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Candidate frame is missing required columns: {missing_columns}"
            )

        # Defensive dedup: each individual blocker call is expected to
        # already emit at most one row per (source1_entity_id,
        # candidate_entity_id, candidate_source) -- but fuse here on the
        # raw per-frame input too, since a groupby-sum over a duplicated
        # mask would silently double-count that blocker's bit.
        frame = frame.drop_duplicates(subset=key_columns)

        blocker_values = (
            frame["blocker_sources"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        masks = pd.Series(
            0,
            index=frame.index,
            dtype="uint16",
        )

        for blocker_name, bit in blocker_bits.items():
            matches = blocker_values.str.contains(
                rf"(?:^|,){blocker_name}(?:,|$)",
                regex=True,
                na=False,
            )

            masks = (
                masks
                | matches.astype("uint16") * bit
            )

        if masks.eq(0).any():
            raise ValueError(
                "Candidate frame contains an empty or invalid blocker_sources value."
            )

        compact = frame[
            key_columns
        ].copy()

        compact["blocker_mask"] = masks.to_numpy(
            dtype="uint16",
            copy=False,
        )

        compact_frames.append(compact)

    combined = pd.concat(
        compact_frames,
        ignore_index=True,
        copy=False,
    )

    if combined.empty:
        return _empty_candidate_frame()

    # Each generated blocker frame contributes a blocker at most once
    # for a given candidate pair (enforced by the per-frame drop_duplicates
    # above). Therefore, because blocker bits are powers of two, summing
    # the masks is equivalent to bitwise OR.
    fused = (
        combined
        .groupby(
            key_columns,
            sort=False,
            dropna=False,
            as_index=False,
        )["blocker_mask"]
        .sum()
    )

    if fused["blocker_mask"].gt(2047).any():
        raise ValueError(
            "Invalid blocker mask detected during candidate fusion."
        )

    blocker_order = [
        "address_country",
        "address_exact",
        "address_prefix_country",
        "address_token_country",
        "address_token_pair_country",
        "name_country",
        "name_exact",
        "name_prefix_country",
        "name_token",
        "name_token_country",
        "name_token_pair_country",
    ]

    mask_to_sources = {}

    for mask in range(1, 2048):
        sources = [
            blocker
            for blocker in blocker_order
            if mask & blocker_bits[blocker]
        ]

        mask_to_sources[mask] = ",".join(sources)

    fused["blocker_mask"] = fused["blocker_mask"].astype("uint16")

    fused["blocker_sources"] = fused["blocker_mask"].map(
        mask_to_sources
    )

    fused["num_blockers"] = (
        fused["blocker_mask"]
        .map(lambda mask: int(mask).bit_count())
        .astype("int64")
    )

    fused = fused.drop(
        columns=["blocker_mask"]
    )

    fused = fused[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "blocker_sources",
            "num_blockers",
        ]
    ]

    # Final defensive check: after the groupby-sum fusion above, there
    # must be exactly one row per (source1_entity_id, candidate_entity_id,
    # candidate_source) key -- that's the entire point of the groupby.
    # Assert it rather than silently trusting it.
    assert not fused.duplicated(subset=key_columns).any()

    fused = fused.sort_values(
        by=[
            "source1_entity_id",
            "candidate_source",
            "candidate_entity_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    validate_candidate_pairs(fused)

    return fused


# ============================================================================
# SIGNATURE-BASED (TOKEN / PREFIX) BLOCKER
# ============================================================================

def _run_signature_block(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    target_source: str,
    value_column: str,
    blocker_name: str,
    config: BlockingConfig,
) -> pd.DataFrame:
    """Generate bounded token/prefix blocking candidates.

    Uses a target-side inverted index and processes source1 signatures
    without constructing a large exploded merge DataFrame.
    """

    if target_source not in {"S2", "S3"}:
        raise ValueError(f"Unsupported target_source: {target_source}")

    if value_column not in source1.columns or value_column not in target.columns:
        raise KeyError(f"Missing blocking column: {value_column}")

    # Determine whether this blocker uses token signatures or prefixes.
    is_token_block = blocker_name in {
        BLOCKER_NAME_TOKEN,
        BLOCKER_ADDRESS_TOKEN,
    }

    is_prefix_block = blocker_name in {
        BLOCKER_NAME_PREFIX,
        BLOCKER_ADDRESS_PREFIX,
    }

    is_token_pair_block = blocker_name in {
        BLOCKER_NAME_TOKEN_PAIR,
        BLOCKER_ADDRESS_TOKEN_PAIR,
    }

    if not (
        is_token_block
        or is_prefix_block
        or is_token_pair_block
    ):
        raise ValueError(
            f"Unsupported signature blocker: {blocker_name}"
        )

    min_token_length = int(config.min_token_length)
    max_signature_frequency = int(config.max_signature_frequency)
    prefix_length = int(config.prefix_length)
    max_token_signatures = int(config.max_token_signatures_per_record)

    # ------------------------------------------------------------------
    # Build target-side bounded inverted index.
    # ------------------------------------------------------------------
    target_work = target[
        ["entity_id", value_column, "country_fingerprint"]
    ].copy()

    target_work[value_column] = target_work[value_column].fillna("").astype(str)
    target_work["country_fingerprint"] = (
        target_work["country_fingerprint"].fillna("").astype(str)
    )

    records: list[tuple[str, str, str]] = []

    for entity_id, value, country in target_work.itertuples(index=False, name=None):
        if is_token_block:
            signatures = _token_signatures(
                value,
                min_token_length,
                max_token_signatures,
            )

        elif is_token_pair_block:
            pair_signature = _token_pair_signature(
                value,
                int(config.pair_min_token_length),
            )

            signatures = (
                [pair_signature]
                if pair_signature
                else []
            )

        else:
            prefix = _prefix_signature(
                value,
                min_token_length,
                prefix_length,
            )

            signatures = [prefix] if prefix else []

        for signature in signatures:
            if not signature:
                continue
            records.append((signature, country, entity_id))

    if not records:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "blocker_sources",
                "num_blockers",
            ]
        )

    target_index = pd.DataFrame(
        records,
        columns=["signature", "country", "candidate_entity_id"],
    )

    # Frequency cap is applied to the actual blocking key.
    target_index["blocking_key"] = (
        target_index["signature"] + "\x1f" + target_index["country"]
    )

    frequencies = target_index["blocking_key"].value_counts()
    allowed_keys = frequencies[
        frequencies <= max_signature_frequency
    ].index

    target_index = target_index[
        target_index["blocking_key"].isin(allowed_keys)
    ][
        ["signature", "country", "candidate_entity_id"]
    ]

    if target_index.empty:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
                "blocker_sources",
                "num_blockers",
            ]
        )

    # Build compact Python dictionaries instead of doing a large pandas
    # many-to-many merge.
    index: dict[tuple[str, str], list[str]] = {}

    for signature, country, candidate_id in target_index.itertuples(
        index=False,
        name=None,
    ):
        key = (signature, country)
        index.setdefault(key, []).append(candidate_id)

    # ------------------------------------------------------------------
    # Generate candidates source1-record by source1-record, in bounded
    # chunks.
    #
    # IMPORTANT:
    # Do NOT keep every chunk DataFrame in RAM. Each completed chunk is
    # serialized to a temporary file and released before the next chunk.
    # ------------------------------------------------------------------
    source1_work = source1[
        ["entity_id", value_column, "country_fingerprint"]
    ].copy()

    source1_work[value_column] = source1_work[value_column].fillna("").astype(str)
    source1_work["country_fingerprint"] = (
        source1_work["country_fingerprint"].fillna("").astype(str)
    )

    chunk_size = int(config.source1_chunk_size)
    if chunk_size <= 0:
        raise ValueError("source1_chunk_size must be a positive integer.")

    output_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "blocker_sources",
        "num_blockers",
    ]

    temp_paths: list[str] = []

    try:
        with tempfile.TemporaryDirectory(prefix="atlas_blocking_") as temp_dir:

            for start in range(0, len(source1_work), chunk_size):
                chunk = source1_work.iloc[start : start + chunk_size]

                chunk_rows: list[tuple[str, str, str, str, int]] = []

                for source1_id, value, country in chunk.itertuples(
                    index=False,
                    name=None,
                ):
                    if is_token_block:
                        signatures = _token_signatures(
                            value,
                            min_token_length,
                            max_token_signatures,
                        )

                    elif is_token_pair_block:
                        pair_signature = _token_pair_signature(
                            value,
                            int(config.pair_min_token_length),
                        )

                        signatures = (
                            [pair_signature]
                            if pair_signature
                            else []
                        )

                    else:
                        prefix = _prefix_signature(
                            value,
                            min_token_length,
                            prefix_length,
                        )

                        signatures = [prefix] if prefix else []

                    if not signatures:
                        continue

                    # Deduplicate candidates within this blocker
                    # for the current Source1 record.
                    candidate_ids: set[str] = set()

                    for signature in signatures:
                        if not signature:
                            continue

                        matches = index.get((signature, country))

                        if matches:
                            candidate_ids.update(matches)

                    for candidate_id in candidate_ids:
                        chunk_rows.append(
                            (
                                source1_id,
                                candidate_id,
                                target_source,
                                blocker_name,
                                1,
                            )
                        )

                if not chunk_rows:
                    continue

                chunk_df = pd.DataFrame(
                    chunk_rows,
                    columns=output_columns,
                )

                # Write the completed chunk to disk instead of retaining
                # it in RAM.
                temp_path = f"{temp_dir}/chunk_{start}.pkl"

                with open(temp_path, "wb") as handle:
                    pickle.dump(
                        chunk_df,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                temp_paths.append(temp_path)

                # Explicitly release the current chunk objects.
                del chunk_df
                del chunk_rows
                del chunk

            if not temp_paths:
                return pd.DataFrame(columns=output_columns)

            # Reconstruct the final blocker DataFrame from the temporary
            # chunk files. The large Python list of tuples from all chunks
            # never exists in memory at the same time.
            result_frames: list[pd.DataFrame] = []

            for temp_path in temp_paths:
                with open(temp_path, "rb") as handle:
                    result_frames.append(pickle.load(handle))

            result = pd.concat(
                result_frames,
                ignore_index=True,
                copy=False,
            )

            del result_frames

            # Defensive dedup: chunks are built from disjoint source1
            # index ranges and candidate_ids is already deduped per
            # source1 record, so this shouldn't fire -- but assert it
            # rather than trust it silently across chunk boundaries.
            result = result.drop_duplicates(
                subset=["source1_entity_id", "candidate_entity_id"]
            )

            assert not result.duplicated(
                subset=["source1_entity_id", "candidate_entity_id"]
            ).any()

            return result

    finally:
        # TemporaryDirectory normally handles cleanup automatically.
        # Keep this block intentionally empty so cleanup remains managed
        # by the context manager.
        pass


# ============================================================================
# SIMPLE NAME-TOKEN BLOCKER (NO COUNTRY KEY, HIGH RECALL)
# ============================================================================

def _run_name_token_block(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    target_source: str,
    config: BlockingConfig,
) -> pd.DataFrame:
    """
    Generate candidates using shared business-name tokens.

    Uses an inverted index over normalized business names.

    Unlike name_token_country, this blocker has no country component
    in its key and is intentionally looser/higher-recall than the other
    signature blockers -- but it is tightened along three axes so it
    does not dominate the candidate set:

      1. Postings cap reuses config.max_signature_frequency (the same
         cap every other signature blocker uses), instead of a looser
         1%-of-target-size formula.
      2. Common business-suffix stopwords (ltd, inc, corp, company,
         services, ...) are excluded, matching the token-pair blockers.
      3. Minimum token length is config.name_token_min_length (default
         4, above the other blockers' default of 3), since a token with
         no country key needs to be more discriminative on its own.

    Requires business_name_normalized (produced upstream by
    normalization, before fingerprinting) on both source1 and target.
    """

    blocker_name = BLOCKER_NAME_TOKEN_SIMPLE

    required = {
        "business_name_normalized",
    }

    missing = required - set(source1.columns)
    if missing:
        raise ValueError(
            f"source1 is missing columns required for "
            f"{blocker_name}: {sorted(missing)}"
        )

    missing = required - set(target.columns)
    if missing:
        raise ValueError(
            f"{target_source} is missing columns required for "
            f"{blocker_name}: {sorted(missing)}"
        )

    min_length = int(config.name_token_min_length)

    def _significant_tokens(name: str) -> set[str]:
        return {
            token
            for token in str(name).split()
            if len(token) >= min_length
            and token.lower() not in _COMMON_BUSINESS_STOPWORDS
        }

    # ------------------------------------------------------------
    # Build target inverted index.
    # token -> target entity IDs
    # ------------------------------------------------------------

    target_index: dict[str, list[str]] = {}

    for row in target[
        ["entity_id", "business_name_normalized"]
    ].itertuples(index=False):

        entity_id, name = row

        for token in _significant_tokens(name):
            target_index.setdefault(token, []).append(
                str(entity_id)
            )

    if not target_index:
        return _empty_candidate_frame()

    # Ignore tokens more frequent than the shared blocking-frequency cap,
    # same bound used by every other signature blocker.
    max_postings = max(
        1,
        int(config.max_signature_frequency),
    )

    target_index = {
        token: ids
        for token, ids in target_index.items()
        if len(ids) <= max_postings
    }

    rows = []

    for row in source1[
        ["entity_id", "business_name_normalized"]
    ].itertuples(index=False):

        source1_id, name = row

        candidates = set()

        for token in _significant_tokens(name):
            candidates.update(
                target_index.get(token, [])
            )

        for candidate_id in candidates:
            rows.append(
                {
                    "source1_entity_id": str(source1_id),
                    "candidate_entity_id": str(candidate_id),
                    "candidate_source": target_source,
                    "blocker_sources": blocker_name,
                }
            )

    if not rows:
        return _empty_candidate_frame()

    result = pd.DataFrame(rows)

    # Defensive dedup: `candidates` is already a per-source1-record set,
    # so duplicate (source1_id, candidate_id) pairs shouldn't occur --
    # assert it rather than trust it.
    result = result.drop_duplicates(
        subset=["source1_entity_id", "candidate_entity_id"]
    )

    assert not result.duplicated(
        subset=["source1_entity_id", "candidate_entity_id"]
    ).any()

    result["num_blockers"] = 1

    return result[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "blocker_sources",
            "num_blockers",
        ]
    ]


# ============================================================================
# TARGET-SOURCE GENERATION
# ============================================================================

def generate_candidates_for_target(
    source1: pd.DataFrame,
    target: pd.DataFrame,
    target_source: str,
    config: BlockingConfig | None = None,
) -> pd.DataFrame:
    """
    Generate and fuse candidates between Source1 and one target source.

    Args:
        source1:
            Source 1 DataFrame containing normalized fingerprints.

        target:
            Source 2 or Source 3 DataFrame containing normalized
            fingerprints.

        target_source:
            Must be "S2" or "S3".

        config:
            Blocking configuration.

    Returns:
        Fused candidate-pair DataFrame.
    """

    if config is None:
        config = BlockingConfig()

    if target_source not in VALID_CANDIDATE_SOURCES:
        raise ValueError(
            "target_source must be 'S2' or 'S3'. "
            f"Received: {target_source}"
        )

    validate_source_dataframe(
        source1,
        "source1",
    )

    validate_source_dataframe(
        target,
        target_source,
    )

    validate_fingerprints(
        source1,
        "source1",
    )

    validate_fingerprints(
        target,
        target_source,
    )

    blocker_frames: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # name_exact
    # ------------------------------------------------------------------

    if config.enable_name_exact:
        blocker_frames.append(
            _run_single_key_block(
                source1=source1,
                target=target,
                target_source=target_source,
                key_column="name_fingerprint",
                blocker_name=BLOCKER_NAME_EXACT,
            )
        )

    # ------------------------------------------------------------------
    # address_exact
    # ------------------------------------------------------------------

    if config.enable_address_exact:
        blocker_frames.append(
            _run_single_key_block(
                source1=source1,
                target=target,
                target_source=target_source,
                key_column="address_fingerprint",
                blocker_name=BLOCKER_ADDRESS_EXACT,
            )
        )

    # ------------------------------------------------------------------
    # name_country
    # ------------------------------------------------------------------

    if config.enable_name_country:
        blocker_frames.append(
            _run_composite_block(
                source1=source1,
                target=target,
                target_source=target_source,
                left_column="name_fingerprint",
                right_column="country_fingerprint",
                blocker_name=BLOCKER_NAME_COUNTRY,
            )
        )

    # ------------------------------------------------------------------
    # address_country
    # ------------------------------------------------------------------

    if config.enable_address_country:
        blocker_frames.append(
            _run_composite_block(
                source1=source1,
                target=target,
                target_source=target_source,
                left_column="address_fingerprint",
                right_column="country_fingerprint",
                blocker_name=BLOCKER_ADDRESS_COUNTRY,
            )
        )

    # ------------------------------------------------------------------
    # name_token (simple, no country key, high recall)
    # ------------------------------------------------------------------

    if config.enable_name_token:
        blocker_frames.append(
            _run_name_token_block(
                source1=source1,
                target=target,
                target_source=target_source,
                config=config,
            )
        )

    # ------------------------------------------------------------------
    # name_token_country / name_prefix_country /
    # address_token_country / address_prefix_country
    # ------------------------------------------------------------------

    if config.enable_name_token_country:
        blocker_frames.append(
            _run_signature_block(
                source1=source1,
                target=target,
                target_source=target_source,
                value_column="business_name",
                blocker_name=BLOCKER_NAME_TOKEN,
                config=config,
            )
        )

    if config.enable_name_prefix_country:
        blocker_frames.append(
            _run_signature_block(
                source1=source1,
                target=target,
                target_source=target_source,
                value_column="business_name",
                blocker_name=BLOCKER_NAME_PREFIX,
                config=config,
            )
        )

    if config.enable_address_token_country:
        blocker_frames.append(
            _run_signature_block(
                source1=source1,
                target=target,
                target_source=target_source,
                value_column="business_address",
                blocker_name=BLOCKER_ADDRESS_TOKEN,
                config=config,
            )
        )

    if config.enable_address_prefix_country:
        blocker_frames.append(
            _run_signature_block(
                source1=source1,
                target=target,
                target_source=target_source,
                value_column="business_address",
                blocker_name=BLOCKER_ADDRESS_PREFIX,
                config=config,
            )
        )

    # ------------------------------------------------------------------
    # name_token_pair_country / address_token_pair_country
    # ------------------------------------------------------------------

    if config.enable_name_token_pair_country:
        blocker_frames.append(
            _run_signature_block(
                source1=source1,
                target=target,
                target_source=target_source,
                value_column="business_name",
                blocker_name=BLOCKER_NAME_TOKEN_PAIR,
                config=config,
            )
        )

    if config.enable_address_token_pair_country:
        blocker_frames.append(
            _run_signature_block(
                source1=source1,
                target=target,
                target_source=target_source,
                value_column="business_address",
                blocker_name=BLOCKER_ADDRESS_TOKEN_PAIR,
                config=config,
            )
        )

    return fuse_candidate_pairs(
        blocker_frames
    )


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def generate_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: BlockingConfig | None = None,
) -> pd.DataFrame:
    """
    Generate the complete fused S1 -> S2/S3 candidate set.

    This is the main entry point for the ATLAS pipeline.

    Args:
        source1:
            Source 1 DataFrame.

        source2:
            Source 2 DataFrame.

        source3:
            Source 3 DataFrame.

        config:
            Optional BlockingConfig.

    Returns:
        DataFrame following the exact ATLAS candidate-pair contract.

    Guarantees:
        - candidate_source is only S2 or S3
        - candidate IDs belong to their corresponding source
        - duplicate pairs are fused
        - blocker provenance is preserved
        - num_blockers matches provenance count
        - no ground truth is required
        - no exhaustive S1 x S2/S3 comparison is performed
    """

    if config is None:
        config = BlockingConfig()

    validate_source_dataframe(
        source1,
        "source1",
    )

    validate_source_dataframe(
        source2,
        "source2",
    )

    validate_source_dataframe(
        source3,
        "source3",
    )

    validate_fingerprints(
        source1,
        "source1",
    )

    validate_fingerprints(
        source2,
        "source2",
    )

    validate_fingerprints(
        source3,
        "source3",
    )

    # Generate S2 and S3 separately.
    s2_candidates = generate_candidates_for_target(
        source1=source1,
        target=source2,
        target_source="S2",
        config=config,
    )

    s3_candidates = generate_candidates_for_target(
        source1=source1,
        target=source3,
        target_source="S3",
        config=config,
    )

    # Final fusion across both target sources.
    final_candidates = fuse_candidate_pairs(
        [
            s2_candidates,
            s3_candidates,
        ]
    )

    validate_candidate_pairs(
        final_candidates
    )

    return final_candidates


# ============================================================================
# CONTRACT / ID VALIDATION
# ============================================================================

def validate_candidate_ids(
    candidate_pairs: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> None:
    """
    Verify that every generated candidate ID exists in the correct source.

    This is a structural validation step and does not use ground truth.
    """

    validate_candidate_pairs(candidate_pairs)

    s1_ids = set(
        source1["entity_id"].astype(str)
    )

    s2_ids = set(
        source2["entity_id"].astype(str)
    )

    s3_ids = set(
        source3["entity_id"].astype(str)
    )

    invalid_s1 = set(
        candidate_pairs["source1_entity_id"].astype(str)
    ) - s1_ids

    if invalid_s1:
        raise ValueError(
            "Candidate output contains Source1 IDs that do not "
            f"exist in source1. Examples: {list(invalid_s1)[:5]}"
        )

    s2_candidates = candidate_pairs.loc[
        candidate_pairs["candidate_source"].eq("S2"),
        "candidate_entity_id",
    ].astype(str)

    invalid_s2 = set(s2_candidates) - s2_ids

    if invalid_s2:
        raise ValueError(
            "Candidate output contains S2 IDs that do not "
            f"exist in source2. Examples: {list(invalid_s2)[:5]}"
        )

    s3_candidates = candidate_pairs.loc[
        candidate_pairs["candidate_source"].eq("S3"),
        "candidate_entity_id",
    ].astype(str)

    invalid_s3 = set(s3_candidates) - s3_ids

    if invalid_s3:
        raise ValueError(
            "Candidate output contains S3 IDs that do not "
            f"exist in source3. Examples: {list(invalid_s3)[:5]}"
        )


# ============================================================================
# SIMPLE SUMMARY
# ============================================================================

def summarize_candidates(
    candidate_pairs: pd.DataFrame,
) -> dict:
    """
    Return basic candidate-generation statistics.

    This is intentionally lightweight. Detailed benchmark and
    candidate-recall reporting will be implemented separately.
    """

    validate_candidate_pairs(candidate_pairs)

    if candidate_pairs.empty:
        return {
            "total_candidate_pairs": 0,
            "unique_source1_entities": 0,
            "s2_candidate_pairs": 0,
            "s3_candidate_pairs": 0,
            "average_candidates_per_source1": 0.0,
            "median_candidates_per_source1": 0.0,
            "max_candidates_per_source1": 0,
        }

    counts = (
        candidate_pairs
        .groupby("source1_entity_id")
        .size()
    )

    return {
        "total_candidate_pairs": int(
            len(candidate_pairs)
        ),
        "unique_source1_entities": int(
            candidate_pairs["source1_entity_id"].nunique()
        ),
        "s2_candidate_pairs": int(
            candidate_pairs["candidate_source"]
            .eq("S2")
            .sum()
        ),
        "s3_candidate_pairs": int(
            candidate_pairs["candidate_source"]
            .eq("S3")
            .sum()
        ),
        "average_candidates_per_source1": float(
            counts.mean()
        ),
        "median_candidates_per_source1": float(
            counts.median()
        ),
        "max_candidates_per_source1": int(
            counts.max()
        ),
    }