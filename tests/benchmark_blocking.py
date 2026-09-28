"""
ATLAS - Controlled Blocking Benchmark

Owner:
    Satyam - Candidate Retrieval / Blocking

Purpose:
    Benchmark the implemented blocking + candidate fusion pipeline
    on a controlled sample of the real ATLAS test data.

Important:
    - Ground truth is NOT used.
    - ML matching is NOT used.
    - This benchmark evaluates candidate retrieval only.
    - Downstream Sayor modules are NOT modified.

Controlled sample:
    Source1: 20,000
    Source2: 50,000
    Source3: 50,000

Implemented blockers:
    - name_exact
    - address_exact
    - name_country
    - address_country
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.blocking import (
    BlockingConfig,
    generate_candidate_pairs,
    summarize_candidates,
    validate_candidate_ids,
)
from src.data_loader import load_test_sources
from src.fingerprinting import add_fingerprint_columns
from src.normalization import add_normalized_columns


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"C:\Users\Satyam Kumar\Downloads\6ab10eb3b23ba_student_resource"
    r"\student_resource\dataset"
)

RANDOM_STATE = 42

SOURCE1_SAMPLE_SIZE = 20_000
SOURCE2_SAMPLE_SIZE = 50_000
SOURCE3_SAMPLE_SIZE = 50_000


# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def sample_source(
    df: pd.DataFrame,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Randomly sample rows without replacement.
    """
    if len(df) <= sample_size:
        return df.copy().reset_index(drop=True)

    return df.sample(
        n=sample_size,
        random_state=random_state,
        replace=False,
    ).reset_index(drop=True)


def prepare_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same preprocessing expected by the blocking pipeline:

        raw data
            -> normalization
            -> fingerprinting
    """
    result = add_normalized_columns(
        df,
        copy=True,
    )

    result = add_fingerprint_columns(
        result,
        copy=False,
    )

    return result


def percentile(
    values: pd.Series,
    percentile_value: float,
) -> float:
    """
    Calculate a percentile safely.
    """
    if values.empty:
        return 0.0

    return float(
        np.percentile(
            values.to_numpy(),
            percentile_value,
        )
    )


def print_source_sizes(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> None:
    print_header("CONTROLLED SAMPLE SIZES")

    print(
        f"Source1 sample rows : {len(source1):,}"
    )
    print(
        f"Source2 sample rows : {len(source2):,}"
    )
    print(
        f"Source3 sample rows : {len(source3):,}"
    )


# ============================================================
# CANDIDATE DISTRIBUTION
# ============================================================

def print_candidate_distribution(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
) -> None:
    """
    Report candidate volume and candidates per Source1 entity.
    """

    print_header("CANDIDATE DISTRIBUTION")

    total_pairs = len(candidates)

    print(
        f"Total candidate pairs : {total_pairs:,}"
    )

    if candidates.empty:
        print("Unique Source1 IDs    : 0")
        print("S2 candidate pairs    : 0")
        print("S3 candidate pairs    : 0")
        print("Average candidates/S1 : 0.000")
        print("Median candidates/S1  : 0.000")
        print("P90 candidates/S1     : 0.000")
        print("P95 candidates/S1     : 0.000")
        print("P99 candidates/S1     : 0.000")
        print("Max candidates/S1     : 0")
        print(
            f"S1 rows with zero candidates: "
            f"{len(source1):,}"
        )
        return

    candidates_per_s1 = (
        candidates
        .groupby("source1_entity_id")
        .size()
        .astype("int64")
    )

    unique_s1 = int(
        candidates["source1_entity_id"].nunique()
    )

    s2_pairs = int(
        candidates["candidate_source"]
        .eq("S2")
        .sum()
    )

    s3_pairs = int(
        candidates["candidate_source"]
        .eq("S3")
        .sum()
    )

    zero_candidate_s1 = (
        len(source1) - unique_s1
    )

    print(
        f"Unique Source1 IDs    : {unique_s1:,}"
    )

    print(
        f"S2 candidate pairs    : {s2_pairs:,}"
    )

    print(
        f"S3 candidate pairs    : {s3_pairs:,}"
    )

    print()

    print(
        f"Average candidates/S1 : "
        f"{candidates_per_s1.mean():,.3f}"
    )

    print(
        f"Median candidates/S1  : "
        f"{candidates_per_s1.median():,.3f}"
    )

    print(
        f"P90 candidates/S1     : "
        f"{percentile(candidates_per_s1, 90):,.3f}"
    )

    print(
        f"P95 candidates/S1     : "
        f"{percentile(candidates_per_s1, 95):,.3f}"
    )

    print(
        f"P99 candidates/S1     : "
        f"{percentile(candidates_per_s1, 99):,.3f}"
    )

    print(
        f"Max candidates/S1     : "
        f"{int(candidates_per_s1.max()):,}"
    )

    print(
        f"S1 rows with zero candidates: "
        f"{zero_candidate_s1:,}"
    )


# ============================================================
# BLOCKER PROVENANCE
# ============================================================

def print_blocker_usage(
    candidates: pd.DataFrame,
) -> None:
    """
    Report how frequently each blocker appears in fused
    candidate provenance.

    This is provenance frequency over the fused candidate set.
    It is NOT an independent per-blocker candidate-volume
    benchmark.
    """

    print_header("BLOCKER PROVENANCE")

    if candidates.empty:
        print("No candidate pairs generated.")
        return

    blocker_counts: dict[str, int] = {}

    for value in candidates[
        "blocker_sources"
    ].fillna(""):

        blockers = [
            blocker.strip()
            for blocker in str(value).split(",")
            if blocker.strip()
        ]

        for blocker in blockers:
            blocker_counts[blocker] = (
                blocker_counts.get(blocker, 0) + 1
            )

    if not blocker_counts:
        print("No blocker provenance recorded.")
        return

    for blocker, count in sorted(
        blocker_counts.items()
    ):
        percentage = (
            count / len(candidates)
        ) * 100.0

        print(
            f"{blocker:20s} : "
            f"{count:12,} pairs "
            f"({percentage:6.2f}% of fused pairs)"
        )


# ============================================================
# CANDIDATE SUMMARY
# ============================================================

def print_candidate_summary(
    candidates: pd.DataFrame,
) -> None:
    """
    Render the existing candidate summary helper when available.
    """

    print_header("CANDIDATE SUMMARY")

    try:
        summary = summarize_candidates(
            candidates
        )

        if isinstance(summary, dict):
            for key, value in summary.items():
                print(
                    f"{key}: {value}"
                )
        else:
            print(summary)

    except Exception as exc:
        print(
            "Candidate summary helper could not "
            f"be rendered: {type(exc).__name__}: {exc}"
        )


# ============================================================
# CONTRACT VALIDATION
# ============================================================

def validate_candidate_contract(
    candidates: pd.DataFrame,
) -> None:
    """
    Validate the candidate-pair output contract independently
    of source-ID validation.
    """

    print_header("CONTRACT CHECKS")

    required_columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
        "blocker_sources",
        "num_blockers",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in candidates.columns
    ]

    if missing_columns:
        print(
            "Required columns : FAIL"
        )
        raise AssertionError(
            f"Missing candidate columns: {missing_columns}"
        )

    print(
        "Required columns : PASS"
    )

    duplicate_pairs = int(
        candidates.duplicated(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
            ]
        ).sum()
    )

    print(
        "Duplicate pairs  : "
        f"{'PASS' if duplicate_pairs == 0 else 'FAIL'} "
        f"({duplicate_pairs:,} duplicates)"
    )

    if duplicate_pairs:
        raise AssertionError(
            f"Found {duplicate_pairs:,} duplicate candidate pairs."
        )

    invalid_sources = int(
        (
            ~candidates[
                "candidate_source"
            ].isin(["S2", "S3"])
        ).sum()
    )

    print(
        "Source validation: "
        f"{'PASS' if invalid_sources == 0 else 'FAIL'} "
        f"({invalid_sources:,} invalid)"
    )

    if invalid_sources:
        raise AssertionError(
            f"Found {invalid_sources:,} invalid candidate sources."
        )

    invalid_blocker_counts = int(
        (
            pd.to_numeric(
                candidates["num_blockers"],
                errors="coerce",
            )
            .fillna(0)
            .lt(1)
        ).sum()
    )

    print(
        "Blocker counts   : "
        f"{'PASS' if invalid_blocker_counts == 0 else 'FAIL'} "
        f"({invalid_blocker_counts:,} invalid)"
    )

    if invalid_blocker_counts:
        raise AssertionError(
            "Found invalid num_blockers values."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "ATLAS CONTROLLED BLOCKING BENCHMARK"
    )

    print(
        f"Dataset root: {DATASET_ROOT}"
    )

    print(
        f"Random state : {RANDOM_STATE}"
    )

    # --------------------------------------------------------
    # Dataset existence check
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            "Dataset root does not exist:\n"
            f"{DATASET_ROOT}"
        )

    # --------------------------------------------------------
    # 1. LOAD FULL TEST SOURCES
    # --------------------------------------------------------

    print()
    print("Loading test sources...")

    load_start = time.perf_counter()

    source1, source2, source3 = (
        load_test_sources(
            DATASET_ROOT
        )
    )

    load_seconds = (
        time.perf_counter()
        - load_start
    )

    print(
        f"Loaded test data in "
        f"{load_seconds:.2f} seconds."
    )

    print()
    print("Full source sizes:")

    print(
        f"Source1 : {len(source1):,}"
    )

    print(
        f"Source2 : {len(source2):,}"
    )

    print(
        f"Source3 : {len(source3):,}"
    )

    # --------------------------------------------------------
    # 2. CONTROLLED SAMPLING
    # --------------------------------------------------------

    sample_start = time.perf_counter()

    source1 = sample_source(
        source1,
        SOURCE1_SAMPLE_SIZE,
        RANDOM_STATE,
    )

    source2 = sample_source(
        source2,
        SOURCE2_SAMPLE_SIZE,
        RANDOM_STATE,
    )

    source3 = sample_source(
        source3,
        SOURCE3_SAMPLE_SIZE,
        RANDOM_STATE,
    )

    sample_seconds = (
        time.perf_counter()
        - sample_start
    )

    print_source_sizes(
        source1,
        source2,
        source3,
    )

    # --------------------------------------------------------
    # 3. NORMALIZATION + FINGERPRINTING
    # --------------------------------------------------------

    print_header(
        "NORMALIZATION + FINGERPRINTING"
    )

    preprocess_start = time.perf_counter()

    source1 = prepare_source(
        source1
    )

    source2 = prepare_source(
        source2
    )

    source3 = prepare_source(
        source3
    )

    preprocess_seconds = (
        time.perf_counter()
        - preprocess_start
    )

    print(
        f"Preprocessing time: "
        f"{preprocess_seconds:.2f} seconds"
    )

    # --------------------------------------------------------
    # 4. BLOCKING CONFIGURATION
    # --------------------------------------------------------

    config = BlockingConfig(
        enable_name_exact=True,
        enable_address_exact=True,
        enable_name_country=True,
        enable_address_country=True,
    )

    print_header(
        "BLOCKING CONFIGURATION"
    )

    print(
        "name_exact      : ENABLED"
    )

    print(
        "address_exact   : ENABLED"
    )

    print(
        "name_country    : ENABLED"
    )

    print(
        "address_country : ENABLED"
    )

    print()
    print(
        "Ground truth    : NOT USED"
    )

    # --------------------------------------------------------
    # 5. CANDIDATE GENERATION + FUSION
    # --------------------------------------------------------

    print_header(
        "CANDIDATE GENERATION"
    )

    blocking_start = time.perf_counter()

    candidates = generate_candidate_pairs(
        source1=source1,
        source2=source2,
        source3=source3,
        config=config,
    )

    blocking_seconds = (
        time.perf_counter()
        - blocking_start
    )

    print(
        f"Blocking + fusion time: "
        f"{blocking_seconds:.2f} seconds"
    )

    # --------------------------------------------------------
    # 6. STRUCTURAL CANDIDATE-ID VALIDATION
    # --------------------------------------------------------

    print_header(
        "CANDIDATE VALIDATION"
    )

    validate_start = time.perf_counter()

    # IMPORTANT:
    # Actual src/blocking.py signature is:
    #
    # validate_candidate_ids(
    #     candidate_pairs,
    #     source1,
    #     source2,
    #     source3,
    # )
    #
    # Therefore we pass the exact parameter names here.

    validate_candidate_ids(
        candidate_pairs=candidates,
        source1=source1,
        source2=source2,
        source3=source3,
    )

    validate_seconds = (
        time.perf_counter()
        - validate_start
    )

    print(
        "Candidate IDs : VALID"
    )

    print(
        "Candidate sources (S2/S3) : VALID"
    )

    print(
        f"Validation time: "
        f"{validate_seconds:.2f} seconds"
    )

    # --------------------------------------------------------
    # 7. CANDIDATE DISTRIBUTION
    # --------------------------------------------------------

    print_candidate_distribution(
        candidates,
        source1,
    )

    # --------------------------------------------------------
    # 8. BLOCKER PROVENANCE
    # --------------------------------------------------------

    print_blocker_usage(
        candidates
    )

    # --------------------------------------------------------
    # 9. EXISTING CANDIDATE SUMMARY
    # --------------------------------------------------------

    print_candidate_summary(
        candidates
    )

    # --------------------------------------------------------
    # 10. OUTPUT CONTRACT VALIDATION
    # --------------------------------------------------------

    validate_candidate_contract(
        candidates
    )

    # --------------------------------------------------------
    # 11. RUNTIME SUMMARY
    # --------------------------------------------------------

    total_seconds = (
        load_seconds
        + sample_seconds
        + preprocess_seconds
        + blocking_seconds
        + validate_seconds
    )

    print_header(
        "RUNTIME"
    )

    print(
        f"Load time         : "
        f"{load_seconds:,.2f} sec"
    )

    print(
        f"Sampling time     : "
        f"{sample_seconds:,.2f} sec"
    )

    print(
        f"Preprocessing     : "
        f"{preprocess_seconds:,.2f} sec"
    )

    print(
        f"Blocking + fusion : "
        f"{blocking_seconds:,.2f} sec"
    )

    print(
        f"Validation        : "
        f"{validate_seconds:,.2f} sec"
    )

    print(
        f"Total benchmark   : "
        f"{total_seconds:,.2f} sec"
    )

    # --------------------------------------------------------
    # 12. FINAL STATUS
    # --------------------------------------------------------

    print_header(
        "BENCHMARK STATUS"
    )

    print(
        "CONTROLLED BLOCKING BENCHMARK: PASS"
    )

    print()
    print(
        "Ground truth used      : NO"
    )

    print(
        "ML matcher used        : NO"
    )

    print(
        "Candidate fusion       : YES"
    )

    print(
        "Candidate validation   : YES"
    )

    print(
        "Output contract check  : PASS"
    )


if __name__ == "__main__":
    main()