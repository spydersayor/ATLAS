"""
ATLAS - Feature Engineering Profiler

Profiles the current FeatureEngineer implementation to identify
where computation time is being spent.

IMPORTANT:
    This file does NOT modify src/features.py.

Run:
    python -m tests.profile_features
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import pandas as pd

from src.features import FeatureEngineer
from tests.benchmark_features import (
    build_candidate_pairs,
    build_source_tables,
)


# ============================================================
# CONFIG
# ============================================================

PROFILE_ROWS = 100_000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ATLAS FEATURE ENGINEERING PROFILER")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Build synthetic data
    # --------------------------------------------------------

    print("Generating synthetic source tables...")

    source1, source2, source3 = build_source_tables()

    print(f"Source 1 rows : {len(source1):,}")
    print(f"Source 2 rows : {len(source2):,}")
    print(f"Source 3 rows : {len(source3):,}")
    print()

    print(
        f"Generating {PROFILE_ROWS:,} candidate pairs..."
    )

    candidate_pairs = build_candidate_pairs(
        source1=source1,
        source2=source2,
        source3=source3,
        num_candidates=PROFILE_ROWS,
    )

    print(
        f"Candidate pairs: {len(candidate_pairs):,}"
    )
    print()

    # --------------------------------------------------------
    # Create feature engineer
    # --------------------------------------------------------

    feature_engineer = FeatureEngineer()

    # --------------------------------------------------------
    # Warm-up
    #
    # This removes some first-call overhead from the profile.
    # --------------------------------------------------------

    print("Running warm-up...")

    warmup_candidates = candidate_pairs.head(100)

    feature_engineer.transform(
        source1=source1,
        source2=source2,
        source3=source3,
        candidate_pairs=warmup_candidates,
    )

    print("Warm-up complete.")
    print()

    # --------------------------------------------------------
    # cProfile
    # --------------------------------------------------------

    profiler = cProfile.Profile()

    print(
        f"Profiling {PROFILE_ROWS:,} candidate pairs..."
    )

    start = time.perf_counter()

    profiler.enable()

    result = feature_engineer.transform(
        source1=source1,
        source2=source2,
        source3=source3,
        candidate_pairs=candidate_pairs,
    )

    profiler.disable()

    elapsed = time.perf_counter() - start

    # --------------------------------------------------------
    # Basic result validation
    # --------------------------------------------------------

    assert len(result) == PROFILE_ROWS

    print()
    print("Profiling complete.")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Rows processed : {PROFILE_ROWS:,}"
    )

    print(
        f"Runtime        : {elapsed:.3f} sec"
    )

    print(
        f"Throughput     : "
        f"{PROFILE_ROWS / elapsed:,.0f} rows/sec"
    )

    print(
        f"Output memory  : "
        f"{result.memory_usage(index=True, deep=True).sum() / (1024 ** 2):.2f} MB"
    )

    print()

    # --------------------------------------------------------
    # Print top cumulative functions
    # --------------------------------------------------------

    print("=" * 70)
    print("TOP FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 70)

    stream = io.StringIO()

    stats = pstats.Stats(
        profiler,
        stream=stream,
    )

    stats.strip_dirs()
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    print(stream.getvalue())

    # --------------------------------------------------------
    # Print top functions by internal/self time
    # --------------------------------------------------------

    print("=" * 70)
    print("TOP FUNCTIONS BY INTERNAL TIME")
    print("=" * 70)

    stream_internal = io.StringIO()

    stats_internal = pstats.Stats(
        profiler,
        stream=stream_internal,
    )

    stats_internal.strip_dirs()
    stats_internal.sort_stats("tottime")
    stats_internal.print_stats(30)

    print(stream_internal.getvalue())

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del result
    del candidate_pairs
    del source1
    del source2
    del source3


if __name__ == "__main__":
    main()