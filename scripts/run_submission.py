"""
ATLAS - Final Submission Pipeline

Complete end-to-end pipeline:

TRAIN DATA
    |
    v
Normalization
    |
    v
Fingerprinting
    |
    v
Candidate Generation
    |
    v
Candidate Validation / Deduplication
    |
    v
Candidate Labeling
    |
    v
Feature Engineering
    |
    v
Matcher Training
    |
    v
F0.5 Threshold Optimization
    |
    v
TEST DATA
    |
    v
Normalization
    |
    v
Fingerprinting
    |
    v
Candidate Generation
    |
    v
ML Scoring
    |
    v
Entity-Level Decisions
    |
    v
Submission Files

Outputs:
    output/matching_results.tsv
    output/candidate_pairs.tsv

Important:
    - Ground truth is used only during training.
    - Ground truth is NEVER used during test candidate generation.
    - Candidate pairs are generated before ML scoring.
    - Final matches must be contained in candidate_pairs.tsv.
    - Normalization MUST happen before fingerprinting/blocking.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd


# ============================================================================
# PROJECT ROOT
# ============================================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from src.blocking import (
    BlockingConfig,
    generate_candidate_pairs,
)

from src.evaluator import (
    parse_ground_truth,
)

from src.fingerprinting import (
    add_fingerprint_columns,
)

from src.normalization import (
    add_normalized_columns,
)

from src.inference import (
    InferencePipeline,
)

from src.training import (
    MatcherTrainer,
)


# ============================================================================
# PATH CONFIGURATION
# ============================================================================

DATASET_DIR = ROOT / "dataset"

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

OUTPUT_DIR = ROOT / "output"


TRAIN_SOURCE1_PATH = (
    TRAIN_DIR / "train_source1.tsv"
)

TRAIN_SOURCE2_PATH = (
    TRAIN_DIR / "train_source2.tsv"
)

TRAIN_SOURCE3_PATH = (
    TRAIN_DIR / "train_source3.tsv"
)

GROUND_TRUTH_PATH = (
    TRAIN_DIR / "train_ground_truth.tsv"
)


TEST_SOURCE1_PATH = (
    TEST_DIR / "test_source1.tsv"
)

TEST_SOURCE2_PATH = (
    TEST_DIR / "test_source2.tsv"
)

TEST_SOURCE3_PATH = (
    TEST_DIR / "test_source3.tsv"
)


# ============================================================================
# SPEED CONFIGURATION (temporary -- for a fast run, not the final tuned one)
# ============================================================================
#
# FAST_BLOCKING_CONFIG keeps ONLY the four exact/composite blockers:
# name_exact, address_exact, name_country, address_country. Those are
# the only blockers in blocking.py implemented with vectorized pandas
# .merge() calls. Every other blocker (token, prefix, token-pair,
# name_token) builds its target index via a per-row Python loop --
# that loop is chunked/disk-spilled so it won't run out of memory at
# full scale, but it is NOT fast at full scale. Test S1 alone is
# roughly 1.7M rows and S2/S3 are roughly 5M rows each, so any
# per-row-Python-loop blocker over that volume is realistically many
# minutes-to-hours, regardless of how tightly its frequency cap is
# set. This is the only blocking configuration in this file that is
# honestly fast at full scale.
#
# The real cost of this choice: this reproduces the ~964-candidates-
# per-10k-S1 sparsity from before any of the token/prefix/pair
# blockers existed. If this run is meant to BE the real submission
# (not a fast validation/dry-run pass), that recall cost is real --
# re-enable the other blockers and accept a much longer runtime once
# you're not under a hard 10-minute constraint.
#
# TRAIN_S1_SAMPLE / TRAIN_S2_SAMPLE / TRAIN_S3_SAMPLE cap how much
# training data MatcherTrainer sees, purely for speed. This is safe
# to tune freely -- it only affects model quality, not correctness.
#
# TEST DATA IS NEVER SAMPLED. Every test_source1 entity must appear in
# the final submission, and truncating test S2/S3 would make real
# matches structurally unreachable -- the exact mistake the earlier
# (now-fixed) recall-measurement script made. Test candidate
# generation still uses FAST_BLOCKING_CONFIG for speed, but over the
# FULL test S2/S3 data.
#
# If this configuration still doesn't finish in ~10 minutes on your
# machine, the bottleneck is inside src/training.py or src/inference.py
# (feature engineering, model fit, or scoring) -- this file has no
# visibility into either. Share those and the actual cost there can be
# looked at directly instead of guessed at further. The per-phase
# elapsed-time prints below (see `banner()`) will show you which phase
# to look at first.
# ============================================================================

FAST_BLOCKING_CONFIG = BlockingConfig(
    enable_name_exact=True,
    enable_address_exact=True,
    enable_name_country=True,
    enable_address_country=True,
    enable_name_token_country=False,
    enable_name_prefix_country=False,
    enable_address_token_country=False,
    enable_address_prefix_country=False,
    enable_name_token_pair_country=False,
    enable_address_token_pair_country=False,
    enable_name_token=False,
)

TRAIN_S1_SAMPLE = 20_000
TRAIN_S2_SAMPLE = 100_000
TRAIN_S3_SAMPLE = 100_000


# ============================================================================
# REQUIRED SOURCE COLUMNS
# ============================================================================

SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


CANDIDATE_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "blocker_sources",
    "num_blockers",
]


MATCHING_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


# ============================================================================
# LOGGING
# ============================================================================

PIPELINE_START = time.perf_counter()


def banner(title: str) -> None:
    elapsed = time.perf_counter() - PIPELINE_START
    print()
    print("=" * 75)
    print(f"{title}   [elapsed: {elapsed:6.1f}s]")
    print("=" * 75)


def section(title: str) -> None:
    elapsed = time.perf_counter() - PIPELINE_START
    print()
    print("-" * 75)
    print(f"{title}   [elapsed: {elapsed:6.1f}s]")
    print("-" * 75)


# ============================================================================
# FILE VALIDATION
# ============================================================================

def require_file(path: Path) -> None:

    if not path.exists():
        raise FileNotFoundError(
            "\nRequired file does not exist:\n"
            f"{path}\n"
        )

    if not path.is_file():
        raise RuntimeError(
            "\nExpected a file but found:\n"
            f"{path}\n"
        )


# ============================================================================
# TSV LOADING
# ============================================================================

def load_source(
    path: Path,
    label: str,
) -> pd.DataFrame:

    require_file(path)

    print()
    print(f"Loading: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    missing = (
        set(SOURCE_COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"\n{label} is missing required columns:\n"
            f"{sorted(missing)}\n"
            f"\nActual columns:\n"
            f"{list(df.columns)}"
        )

    # ------------------------------------------------------------------
    # Entity ID validation
    # ------------------------------------------------------------------

    if df["entity_id"].isna().any():
        raise RuntimeError(
            f"{label} contains null entity IDs."
        )

    empty_ids = (
        df["entity_id"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_ids:
        raise RuntimeError(
            f"{label} contains "
            f"{empty_ids:,} empty entity IDs."
        )

    duplicate_ids = int(
        df["entity_id"]
        .duplicated()
        .sum()
    )

    if duplicate_ids:
        raise RuntimeError(
            f"{label} contains "
            f"{duplicate_ids:,} duplicate entity IDs."
        )

    return df


# ============================================================================
# TRAINING-DATA SAMPLING (SPEED ONLY -- NEVER APPLIED TO TEST DATA)
# ============================================================================

def sample_for_speed(
    df: pd.DataFrame,
    limit: int,
    label: str,
) -> pd.DataFrame:
    """
    Randomly downsample a TRAINING dataframe for a faster run.

    Never call this on test data: every test_source1 entity must
    appear in the final submission, and truncating test S2/S3 would
    make real matches structurally unreachable.
    """

    if len(df) <= limit:
        print(
            f"[{label}] {len(df):,} rows already <= "
            f"sample limit {limit:,}; using all rows."
        )
        return df

    sampled = df.sample(
        n=limit,
        random_state=42,
    ).reset_index(drop=True)

    print(
        f"[{label}] Sampled {limit:,} / {len(df):,} rows "
        "for a faster training run."
    )

    return sampled


# ============================================================================
# NORMALIZATION + FINGERPRINTING
# ============================================================================

def add_fingerprints(
    df: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """
    Prepare a source dataframe for blocking.

    Correct order:

        raw source
            |
            v
        normalization
            |
            v
        fingerprinting
            |
            v
        blocking

    The blocking module requires:

        name_fingerprint
        address_fingerprint
        country_fingerprint
    """

    section(
        f"{label} — NORMALIZATION + FINGERPRINTING"
    )

    required_fingerprints = {
        "name_fingerprint",
        "address_fingerprint",
        "country_fingerprint",
    }

    # ------------------------------------------------------------------
    # 1. NORMALIZATION
    # ------------------------------------------------------------------

    print(
        f"[{label}] Running normalization..."
    )

    normalized = add_normalized_columns(
        df,
    )

    if not isinstance(
        normalized,
        pd.DataFrame,
    ):
        raise RuntimeError(
            f"{label}: "
            "add_normalized_columns() did not "
            "return a pandas DataFrame."
        )

    print(
        f"[{label}] Normalization complete."
    )

    # ------------------------------------------------------------------
    # 2. FINGERPRINTING
    # ------------------------------------------------------------------

    missing_before = (
        required_fingerprints
        - set(normalized.columns)
    )

    if not missing_before:
        print(
            f"[{label}] Fingerprints already exist."
        )
        return normalized

    print(
        f"[{label}] Missing fingerprints:"
    )

    for column in sorted(missing_before):
        print(
            f"  - {column}"
        )

    print(
        f"[{label}] Running "
        "add_fingerprint_columns()..."
    )

    result = add_fingerprint_columns(
        normalized,
        copy=True,
    )

    if not isinstance(
        result,
        pd.DataFrame,
    ):
        raise RuntimeError(
            f"{label}: "
            "add_fingerprint_columns() did not "
            "return a pandas DataFrame."
        )

    missing_after = (
        required_fingerprints
        - set(result.columns)
    )

    if missing_after:
        raise RuntimeError(
            f"\n[{label}] Fingerprinting failed.\n"
            f"Missing after fingerprinting:\n"
            f"{sorted(missing_after)}\n"
            f"\nAvailable columns:\n"
            f"{list(result.columns)}"
        )

    print(
        f"[{label}] Fingerprinting complete."
    )

    print(
        f"[{label}] Ready for blocking."
    )

    return result


# ============================================================================
# CANDIDATE VALIDATION
# ============================================================================

def validate_candidates(
    candidates: pd.DataFrame,
    label: str,
) -> pd.DataFrame:

    section(
        f"{label} — CANDIDATE VALIDATION"
    )

    missing = (
        set(CANDIDATE_COLUMNS)
        - set(candidates.columns)
    )

    if missing:
        raise RuntimeError(
            f"{label} candidate pairs are missing:\n"
            f"{sorted(missing)}"
        )

    print(
        f"Candidate rows: "
        f"{len(candidates):,}"
    )

    if candidates.empty:
        raise RuntimeError(
            f"{label} candidate generation "
            "produced ZERO candidates."
        )

    # ------------------------------------------------------------------
    # Candidate source validation
    # ------------------------------------------------------------------

    valid_sources = {
        "S2",
        "S3",
    }

    actual_sources = set(
        candidates[
            "candidate_source"
        ]
        .astype(str)
        .unique()
    )

    invalid_sources = (
        actual_sources
        - valid_sources
    )

    if invalid_sources:
        raise RuntimeError(
            f"{label} contains invalid "
            f"candidate sources:\n"
            f"{sorted(invalid_sources)}"
        )

    # ------------------------------------------------------------------
    # Defensive candidate-pair deduplication
    #
    # One S1 -> one target -> one source must appear once.
    #
    # IMPORTANT:
    # This is only final defensive deduplication.
    # The blocker-level duplicate-bitmask fix belongs in blocking.py.
    # ------------------------------------------------------------------

    duplicate_subset = [
        "source1_entity_id",
        "candidate_entity_id",
        "candidate_source",
    ]

    duplicates = int(
        candidates
        .duplicated(
            subset=duplicate_subset
        )
        .sum()
    )

    print(
        f"Duplicate candidate pairs: "
        f"{duplicates:,}"
    )

    if duplicates:

        print(
            "Removing duplicate candidate pairs..."
        )

        candidates = (
            candidates
            .drop_duplicates(
                subset=duplicate_subset,
                keep="first",
            )
            .reset_index(drop=True)
        )

        print(
            f"Candidates after deduplication: "
            f"{len(candidates):,}"
        )

    # ------------------------------------------------------------------
    # Sanity-check candidate IDs
    # ------------------------------------------------------------------

    if (
        candidates[
            "source1_entity_id"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            f"{label} candidates contain "
            "null Source-1 IDs."
        )

    if (
        candidates[
            "candidate_entity_id"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            f"{label} candidates contain "
            "null candidate IDs."
        )

    return candidates


# ============================================================================
# SAVE CANDIDATE PAIRS
# ============================================================================

def save_candidate_pairs(
    candidates: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates[
        CANDIDATE_COLUMNS
    ].to_csv(
        path,
        sep="\t",
        index=False,
    )

    print(
        f"\nSaved candidate pairs:\n"
        f"{path}"
    )


# ============================================================================
# FINAL OUTPUT VALIDATION
# ============================================================================

def validate_final_outputs(
    test_source1: pd.DataFrame,
    candidate_path: Path,
    matching_path: Path,
) -> None:

    section(
        "FINAL SUBMISSION VALIDATION"
    )

    require_file(candidate_path)
    require_file(matching_path)

    candidates = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    matching = pd.read_csv(
        matching_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    # ------------------------------------------------------------------
    # Candidate schema
    # ------------------------------------------------------------------

    if list(candidates.columns) != (
        CANDIDATE_COLUMNS
    ):
        raise RuntimeError(
            "\nInvalid candidate_pairs.tsv columns.\n"
            f"Expected:\n{CANDIDATE_COLUMNS}\n"
            f"Actual:\n{list(candidates.columns)}"
        )

    # ------------------------------------------------------------------
    # Matching schema
    # ------------------------------------------------------------------

    if list(matching.columns) != (
        MATCHING_COLUMNS
    ):
        raise RuntimeError(
            "\nInvalid matching_results.tsv columns.\n"
            f"Expected:\n{MATCHING_COLUMNS}\n"
            f"Actual:\n{list(matching.columns)}"
        )

    # ------------------------------------------------------------------
    # Every test S1 must appear exactly once
    # ------------------------------------------------------------------

    expected_ids = set(
        test_source1[
            "entity_id"
        ]
        .astype(str)
    )

    actual_ids = set(
        matching[
            "source1_entity_id"
        ]
        .astype(str)
    )

    missing_ids = (
        expected_ids
        - actual_ids
    )

    extra_ids = (
        actual_ids
        - expected_ids
    )

    if missing_ids:
        raise RuntimeError(
            "matching_results.tsv is missing "
            f"{len(missing_ids):,} "
            "Source-1 entities."
        )

    if extra_ids:
        raise RuntimeError(
            "matching_results.tsv contains "
            f"{len(extra_ids):,} unexpected "
            "Source-1 entities."
        )

    if len(matching) != len(
        test_source1
    ):
        raise RuntimeError(
            "matching_results.tsv must contain "
            "exactly one row per test "
            "Source-1 entity."
        )

    duplicate_s1 = int(
        matching[
            "source1_entity_id"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_s1:
        raise RuntimeError(
            "matching_results.tsv contains "
            f"{duplicate_s1:,} duplicate "
            "Source-1 rows."
        )

    # ------------------------------------------------------------------
    # Candidate duplicate validation
    # ------------------------------------------------------------------

    duplicate_candidates = int(
        candidates
        .duplicated(
            subset=[
                "source1_entity_id",
                "candidate_entity_id",
                "candidate_source",
            ]
        )
        .sum()
    )

    if duplicate_candidates:
        raise RuntimeError(
            "candidate_pairs.tsv contains "
            f"{duplicate_candidates:,} duplicate "
            "candidate pairs."
        )

    # ------------------------------------------------------------------
    # Candidate source validation
    # ------------------------------------------------------------------

    sources = set(
        candidates[
            "candidate_source"
        ]
        .astype(str)
        .unique()
    )

    invalid_sources = (
        sources
        - {"S2", "S3"}
    )

    if invalid_sources:
        raise RuntimeError(
            "Invalid candidate_source values:\n"
            f"{sorted(invalid_sources)}"
        )

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------

    print(
        f"\nCandidate rows: "
        f"{len(candidates):,}"
    )

    print(
        f"Matching rows: "
        f"{len(matching):,}"
    )

    print(
        f"Test S1 rows: "
        f"{len(test_source1):,}"
    )

    print(
        "\nFinal submission validation: PASSED"
    )


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:

    banner(
        "ATLAS FINAL SUBMISSION PIPELINE (FAST MODE)"
    )

    print(
        "\nFAST MODE is active:\n"
        "  - Blocking uses ONLY the 4 vectorized exact/composite blockers\n"
        "    (name_exact, address_exact, name_country, address_country).\n"
        f"  - Training data is sampled: S1<={TRAIN_S1_SAMPLE:,}, "
        f"S2<={TRAIN_S2_SAMPLE:,}, S3<={TRAIN_S3_SAMPLE:,}.\n"
        "  - Test data is NOT sampled -- full S2/S3, every test S1 entity.\n"
        "See the SPEED CONFIGURATION comment block near the top of this\n"
        "file for the recall tradeoff this makes, and how to revert it."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # PHASE 1 — LOAD TRAINING DATA
    # ========================================================================

    banner(
        "PHASE 1 — LOAD TRAINING DATA"
    )

    train_s1 = load_source(
        TRAIN_SOURCE1_PATH,
        "TRAIN S1",
    )

    train_s2 = load_source(
        TRAIN_SOURCE2_PATH,
        "TRAIN S2",
    )

    train_s3 = load_source(
        TRAIN_SOURCE3_PATH,
        "TRAIN S3",
    )

    # ------------------------------------------------------------------------
    # Speed-only sampling. Never applied to test data (see
    # sample_for_speed()'s docstring and the SPEED CONFIGURATION block).
    # ------------------------------------------------------------------------

    train_s1 = sample_for_speed(
        train_s1, TRAIN_S1_SAMPLE, "TRAIN S1"
    )

    train_s2 = sample_for_speed(
        train_s2, TRAIN_S2_SAMPLE, "TRAIN S2"
    )

    train_s3 = sample_for_speed(
        train_s3, TRAIN_S3_SAMPLE, "TRAIN S3"
    )

    # ------------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------------

    require_file(
        GROUND_TRUTH_PATH
    )

    print(
        f"\nLoading ground truth: "
        f"{GROUND_TRUTH_PATH}"
    )

    ground_truth = parse_ground_truth(
        GROUND_TRUTH_PATH
    )

    print(
        f"Ground-truth Source-1 entities: "
        f"{len(ground_truth):,}"
    )

    # ========================================================================
    # PHASE 1.5 — NORMALIZATION + FINGERPRINTING
    # ========================================================================

    banner(
        "PHASE 1.5 — PREPARE TRAINING DATA"
    )

    train_s1 = add_fingerprints(
        train_s1,
        "TRAIN S1",
    )

    train_s2 = add_fingerprints(
        train_s2,
        "TRAIN S2",
    )

    train_s3 = add_fingerprints(
        train_s3,
        "TRAIN S3",
    )

    # ========================================================================
    # PHASE 2 — GENERATE TRAINING CANDIDATES
    # ========================================================================

    banner(
        "PHASE 2 — GENERATE TRAINING CANDIDATES"
    )

    print(
        "\nStarting training candidate generation "
        "(FAST_BLOCKING_CONFIG)..."
    )

    config = FAST_BLOCKING_CONFIG

    train_candidates = (
        generate_candidate_pairs(
            source1=train_s1,
            source2=train_s2,
            source3=train_s3,
            config=config,
        )
    )

    train_candidates = validate_candidates(
        train_candidates,
        "TRAIN",
    )

    print(
        "\nTraining candidates ready:"
    )

    print(
        f"{len(train_candidates):,}"
    )

    # ========================================================================
    # PHASE 3 — TRAIN MATCHER
    # ========================================================================

    banner(
        "PHASE 3 — TRAIN MATCHER"
    )

    print(
        "\nInitializing MatcherTrainer..."
    )

    trainer = MatcherTrainer(
        validation_fraction=0.2,
        random_state=42,
    )

    print(
        "\nTraining matcher..."
    )

    training_result = trainer.train(
        source1=train_s1,
        source2=train_s2,
        source3=train_s3,
        candidate_pairs=train_candidates,
        ground_truth=ground_truth,
    )

    matcher = (
        training_result.matcher
    )

    threshold_result = (
        training_result.threshold_result
    )

    # ------------------------------------------------------------------------
    # Extract optimized threshold
    # ------------------------------------------------------------------------

    if hasattr(
        threshold_result,
        "threshold",
    ):

        optimized_threshold = float(
            threshold_result.threshold
        )

    elif isinstance(
        threshold_result,
        dict,
    ):

        optimized_threshold = float(
            threshold_result[
                "threshold"
            ]
        )

    else:

        raise RuntimeError(
            "Could not extract optimized "
            "threshold from TrainingResult."
        )

    print()

    print(
        f"Training rows: "
        f"{training_result.train_rows:,}"
    )

    print(
        f"Validation rows: "
        f"{training_result.validation_rows:,}"
    )

    print(
        f"Training positives: "
        f"{training_result.train_positive_rows:,}"
    )

    print(
        f"Training negatives: "
        f"{training_result.train_negative_rows:,}"
    )

    print(
        f"Validation positives: "
        f"{training_result.validation_positive_rows:,}"
    )

    print(
        f"Validation negatives: "
        f"{training_result.validation_negative_rows:,}"
    )

    print(
        "\nOptimized F0.5 threshold: "
        f"{optimized_threshold:.6f}"
    )

    # ------------------------------------------------------------------------
    # PERSIST THE MATCHER + THRESHOLD.
    #
    # BUG FIX: previously this pipeline trained the matcher and used it
    # in-memory for inference in the same run, but never saved it to
    # disk anywhere. Any separate/standalone inference script (e.g.
    # matching.py) expecting to EntityMatcher.load("output/matcher.joblib")
    # would find nothing there, since nothing had ever written it.
    # ------------------------------------------------------------------------

    matcher_path = OUTPUT_DIR / "matcher.joblib"
    threshold_path = OUTPUT_DIR / "threshold.txt"

    print(
        f"\nSaving trained matcher to: {matcher_path}"
    )

    matcher.save(str(matcher_path))

    threshold_path.write_text(
        f"{optimized_threshold:.6f}\n"
    )

    print(
        f"Saved optimized threshold to: {threshold_path}"
    )

    # ========================================================================
    # PHASE 4 — LOAD TEST DATA
    # ========================================================================

    banner(
        "PHASE 4 — LOAD TEST DATA"
    )

    print(
        "\nTest data is loaded in FULL, unsampled -- every test_source1\n"
        "entity must appear in the final submission, and truncating\n"
        "test S2/S3 would make real matches structurally unreachable."
    )

    test_s1 = load_source(
        TEST_SOURCE1_PATH,
        "TEST S1",
    )

    test_s2 = load_source(
        TEST_SOURCE2_PATH,
        "TEST S2",
    )

    test_s3 = load_source(
        TEST_SOURCE3_PATH,
        "TEST S3",
    )

    # ========================================================================
    # PHASE 4.5 — NORMALIZATION + FINGERPRINTING
    # ========================================================================

    banner(
        "PHASE 4.5 — PREPARE TEST DATA"
    )

    test_s1 = add_fingerprints(
        test_s1,
        "TEST S1",
    )

    test_s2 = add_fingerprints(
        test_s2,
        "TEST S2",
    )

    test_s3 = add_fingerprints(
        test_s3,
        "TEST S3",
    )

    # ========================================================================
    # PHASE 5 — GENERATE TEST CANDIDATES
    # ========================================================================

    banner(
        "PHASE 5 — GENERATE TEST CANDIDATES"
    )

    print(
        "\nStarting test candidate generation "
        "(FAST_BLOCKING_CONFIG, full test S2/S3)..."
    )

    test_candidates = (
        generate_candidate_pairs(
            source1=test_s1,
            source2=test_s2,
            source3=test_s3,
            config=config,
        )
    )

    test_candidates = validate_candidates(
        test_candidates,
        "TEST",
    )

    print(
        "\nTest candidates ready:"
    )

    print(
        f"{len(test_candidates):,}"
    )

    # ========================================================================
    # PHASE 5.5 — SAVE CANDIDATE OUTPUT
    # ========================================================================

    banner(
        "PHASE 5.5 — SAVE CANDIDATE PAIRS"
    )

    candidate_output_path = (
        OUTPUT_DIR
        / "candidate_pairs.tsv"
    )

    save_candidate_pairs(
        test_candidates,
        candidate_output_path,
    )

    # ========================================================================
    # PHASE 6 — INFERENCE
    # ========================================================================

    banner(
        "PHASE 6 — RUN TEST INFERENCE"
    )

    print(
        "\nInitializing inference pipeline..."
    )

    inference = InferencePipeline(
        matcher=matcher,
        threshold=optimized_threshold,
        conflict_gap=0.05,
    )

    print(
        "\nRunning inference..."
    )

    inference_result = inference.run(
        test_source1=test_s1,
        test_source2=test_s2,
        test_source3=test_s3,
        candidate_pairs=test_candidates,
        output_dir=OUTPUT_DIR,
    )

    # ------------------------------------------------------------------------
    # InferencePipeline generates matching_results.tsv.
    # ------------------------------------------------------------------------

    matching_output_path = (
        OUTPUT_DIR
        / "matching_results.tsv"
    )

    # ========================================================================
    # PHASE 7 — FINAL VALIDATION
    # ========================================================================

    banner(
        "PHASE 7 — VALIDATE FINAL SUBMISSION"
    )

    validate_final_outputs(
        test_source1=test_s1,
        candidate_path=candidate_output_path,
        matching_path=matching_output_path,
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    banner(
        "ATLAS SUBMISSION READY"
    )

    total_elapsed = time.perf_counter() - PIPELINE_START

    print(
        "\nCandidate file:"
        f"\n  {candidate_output_path}"
    )

    print(
        "\nMatching file:"
        f"\n  {matching_output_path}"
    )

    print(
        "\nOptimized threshold:"
        f" {optimized_threshold:.6f}"
    )

    print(
        f"\nTotal pipeline time: {total_elapsed:.1f}s "
        f"({total_elapsed / 60:.1f} min)"
    )

    print(
        "\nRequired submission files:"
    )

    print(
        "  [OK] output/candidate_pairs.tsv"
    )

    print(
        "  [OK] output/matching_results.tsv"
    )

    print(
        "\nFINAL STATUS: READY"
    )

    print(
        "\nNOTE: this was a FAST MODE run (exact/composite blockers only, "
        "sampled training data). If this needs to be your real "
        "submission rather than a validation pass, see the SPEED "
        "CONFIGURATION comment block near the top of this file before "
        "you rely on it."
    )

    print(
        "\nDo not manually modify the generated "
        "submission files after this validation."
    )

    print(
        "\n" + "=" * 75
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\nPipeline interrupted by user."
        )

        raise

    except Exception as exc:

        print(
            "\n\n"
            + "=" * 75
        )

        print(
            "ATLAS PIPELINE FAILED"
        )

        print(
            "=" * 75
        )

        print(
            f"\nError:\n{exc}"
        )

        print()

        raise