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

BLOCKER_NAMES = (
    BLOCKER_NAME_EXACT,
    BLOCKER_ADDRESS_EXACT,
    BLOCKER_NAME_COUNTRY,
    BLOCKER_ADDRESS_COUNTRY,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class BlockingConfig:
    """
    Configuration for candidate generation.

    The current baseline intentionally uses exact deterministic
    retrieval keys only.

    Token/approximate blockers will be added separately after
    candidate-volume benchmarking.
    """

    enable_name_exact: bool = True
    enable_address_exact: bool = True
    enable_name_country: bool = True
    enable_address_country: bool = True


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
    Fuse candidates generated by multiple blockers.

    If the same pair is retrieved by multiple blockers, only one
    candidate row survives.

    Example:

        pair A
            name_exact
            address_exact
            name_country

    becomes:

        blocker_sources =
            "address_exact,name_country,name_exact"

        num_blockers = 3

    Provenance is stored as a deterministic comma-separated string
    because Sayor's downstream feature module accepts comma-separated
    blocker representations.
    """

    frames = [
        frame
        for frame in candidate_frames
        if frame is not None and not frame.empty
    ]

    if not frames:
        return _empty_candidate_frame()

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    if combined.empty:
        return _empty_candidate_frame()

    key_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]

    # Normalize blocker representation.
    combined["blocker_sources"] = (
        combined["blocker_sources"]
        .fillna("")
        .astype(str)
        .map(
            lambda value: [
                blocker.strip()
                for blocker in value.split(",")
                if blocker.strip()
            ]
        )
    )

    # Aggregate all blocker provenance for each unique pair.
    fused = (
        combined
        .groupby(
            key_columns,
            sort=False,
            dropna=False,
        )
        .agg(
            blocker_sources=(
                "blocker_sources",
                lambda groups: sorted(
                    {
                        blocker
                        for group in groups
                        for blocker in group
                    }
                ),
            )
        )
        .reset_index()
    )

    fused["num_blockers"] = (
        fused["blocker_sources"]
        .map(len)
        .astype("int64")
    )

    # Downstream Sayor feature code accepts comma-separated
    # blocker representations.
    fused["blocker_sources"] = (
        fused["blocker_sources"]
        .map(",".join)
        .astype("string")
    )

    # Deterministic ordering.
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