"""
ATLAS - Feature Engineering Scalability Benchmark

Measures the current FeatureEngineer implementation on synthetic
candidate-pair workloads.

IMPORTANT:
    This is a benchmark only.
    It does not modify the feature implementation.

Measures:
    - candidate-pair count
    - runtime
    - rows/second
    - approximate output memory
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd

from src.features import FeatureEngineer


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42

# Start small.
# Increase later only if these complete comfortably.
BENCHMARK_SIZES = [
    5_000,
    10_000,
    50_000,
    100_000,
]


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

def build_source_tables(
    num_source1: int = 5_000,
    num_source2: int = 10_000,
    num_source3: int = 10_000,
):
    """
    Create synthetic source tables with the same schema expected
    by FeatureEngineer.
    """

    rng = np.random.default_rng(RANDOM_STATE)

    def make_source(
        prefix: str,
        size: int,
        address_missing_rate: float = 0.20,
    ) -> pd.DataFrame:

        names = np.array(
            [
                "ABC Electronics",
                "XYZ Foods",
                "Global Tech",
                "Sharma Traders",
                "Bangalore Services",
                "Delhi Enterprises",
                "Mumbai Retail",
                "International Business",
                "Green Valley Stores",
                "Smart Solutions",
            ],
            dtype=object,
        )

        addresses = np.array(
            [
                "MG Road Bangalore",
                "Delhi Main Road",
                "Paris France",
                "Mumbai Central",
                "Kolkata West Bengal",
                "Bangalore Whitefield",
                "Chennai Main Street",
                "Hyderabad Hitech City",
                "Pune Maharashtra",
                "Noida Sector 18",
            ],
            dtype=object,
        )

        countries = np.array(
            [
                "India",
                "France",
                "United States",
                "United Kingdom",
            ],
            dtype=object,
        )

        name_values = rng.choice(names, size=size)
        address_values = rng.choice(addresses, size=size)
        country_values = rng.choice(countries, size=size)

        # Introduce missing addresses.
        missing_mask = rng.random(size) < address_missing_rate

        address_values = address_values.astype(object)
        address_values[missing_mask] = None

        return pd.DataFrame(
            {
                "entity_id": [
                    f"{prefix}-{i:08d}"
                    for i in range(size)
                ],
                "business_name": name_values,
                "business_address": address_values,
                "country": country_values,
            }
        )

    source1 = make_source(
        "S1",
        num_source1,
        address_missing_rate=0.05,
    )

    source2 = make_source(
        "S2",
        num_source2,
        address_missing_rate=0.20,
    )

    source3 = make_source(
        "S3",
        num_source3,
        address_missing_rate=0.20,
    )

    return source1, source2, source3


def build_candidate_pairs(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    num_candidates: int,
) -> pd.DataFrame:
    """
    Generate synthetic candidate pairs.

    This is ONLY for benchmarking FeatureEngineer.

    It is NOT a replacement for Satyam's blocking system.
    """

    rng = np.random.default_rng(RANDOM_STATE + num_candidates)

    s1_ids = source1["entity_id"].to_numpy()
    s2_ids = source2["entity_id"].to_numpy()
    s3_ids = source3["entity_id"].to_numpy()

    s1_selected = rng.choice(
        s1_ids,
        size=num_candidates,
        replace=True,
    )

    candidate_source = rng.choice(
        ["S2", "S3"],
        size=num_candidates,
        replace=True,
    )

    s2_mask = candidate_source == "S2"
    s3_mask = ~s2_mask

    candidate_ids = np.empty(
        num_candidates,
        dtype=object,
    )

    candidate_ids[s2_mask] = rng.choice(
        s2_ids,
        size=int(s2_mask.sum()),
        replace=True,
    )

    candidate_ids[s3_mask] = rng.choice(
        s3_ids,
        size=int(s3_mask.sum()),
        replace=True,
    )

    blocker_options = [
        ["name_exact"],
        ["address_exact"],
        ["country_exact"],
        ["name_exact", "address_exact"],
        ["name_exact", "country_exact"],
        ["address_exact", "country_exact"],
        [
            "name_exact",
            "address_exact",
            "country_exact",
        ],
    ]

    blocker_indices = rng.integers(
        0,
        len(blocker_options),
        size=num_candidates,
    )

    blockers = [
        blocker_options[index]
        for index in blocker_indices
    ]

    num_blockers = [
        len(blocker_list)
        for blocker_list in blockers
    ]

    return pd.DataFrame(
        {
            "source1_entity_id": s1_selected,
            "candidate_entity_id": candidate_ids,
            "candidate_source": candidate_source,
            "blocker_sources": blockers,
            "num_blockers": num_blockers,
        }
    )


# ============================================================
# BENCHMARK
# ============================================================

def benchmark_size(
    feature_engineer: FeatureEngineer,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
):
    """
    Benchmark one candidate-pair workload.
    """

    num_rows = len(candidate_pairs)

    # Force garbage collection before measurement.
    gc.collect()

    start = time.perf_counter()

    result = feature_engineer.transform(
        source1=source1,
        source2=source2,
        source3=source3,
        candidate_pairs=candidate_pairs,
    )

    elapsed = time.perf_counter() - start

    rows_per_second = (
        num_rows / elapsed
        if elapsed > 0
        else float("inf")
    )

    memory_mb = (
        result.memory_usage(
            index=True,
            deep=True,
        ).sum()
        / (1024 ** 2)
    )

    return {
        "candidate_pairs": num_rows,
        "output_rows": len(result),
        "output_columns": len(result.columns),
        "runtime_seconds": elapsed,
        "rows_per_second": rows_per_second,
        "output_memory_mb": memory_mb,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ATLAS FEATURE ENGINEERING SCALABILITY BENCHMARK")
    print("=" * 70)
    print()

    print("Generating synthetic source tables...")

    source1, source2, source3 = build_source_tables()

    print(f"Source 1 rows : {len(source1):,}")
    print(f"Source 2 rows : {len(source2):,}")
    print(f"Source 3 rows : {len(source3):,}")
    print()

    feature_engineer = FeatureEngineer()

    results = []

    for size in BENCHMARK_SIZES:

        print("-" * 70)
        print(f"Benchmarking {size:,} candidate pairs...")
        print("-" * 70)

        candidate_pairs = build_candidate_pairs(
            source1=source1,
            source2=source2,
            source3=source3,
            num_candidates=size,
        )

        result = benchmark_size(
            feature_engineer=feature_engineer,
            source1=source1,
            source2=source2,
            source3=source3,
            candidate_pairs=candidate_pairs,
        )

        results.append(result)

        print(
            f"Candidate pairs : "
            f"{result['candidate_pairs']:,}"
        )

        print(
            f"Output rows     : "
            f"{result['output_rows']:,}"
        )

        print(
            f"Output columns  : "
            f"{result['output_columns']}"
        )

        print(
            f"Runtime         : "
            f"{result['runtime_seconds']:.3f} sec"
        )

        print(
            f"Throughput      : "
            f"{result['rows_per_second']:,.0f} rows/sec"
        )

        print(
            f"Output memory   : "
            f"{result['output_memory_mb']:.2f} MB"
        )

        print()

        # Cleanup before next workload.
        del candidate_pairs
        gc.collect()

    # ========================================================
    # Summary
    # ========================================================

    summary = pd.DataFrame(results)

    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            formatters={
                "runtime_seconds": "{:.3f}".format,
                "rows_per_second": "{:,.0f}".format,
                "output_memory_mb": "{:.2f}".format,
            },
        )
    )

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
    