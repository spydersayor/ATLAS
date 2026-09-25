from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Iterable

import numpy as np


GroundTruth = Dict[str, Set[str]]


def parse_ground_truth(file_path: str | Path) -> GroundTruth:
    """
    Parse train_ground_truth.tsv into:

        {
            "S1-123": {"S2-456", "S3-789"},
            "S1-124": {"S2-999"},
            "S1-125": set(),
        }

    Supports zero, one, and multiple true matches.
    """

    ground_truth: GroundTruth = {}

    file_path = Path(file_path)

    with file_path.open("r", encoding="utf-8", newline="") as file:
        header = file.readline().rstrip("\r\n")

        if not header:
            raise ValueError("Ground-truth file is empty.")

        columns = header.split("\t")

        if columns != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(
                "Unexpected ground-truth header. "
                f"Expected ['source1_entity_id', 'matched_entity_ids'], "
                f"got {columns}"
            )

        for line_number, line in enumerate(file, start=2):
            line = line.rstrip("\r\n")

            if not line:
                continue

            parts = line.split("\t", 1)

            if len(parts) != 2:
                raise ValueError(
                    f"Malformed ground-truth row at line {line_number}: {line!r}"
                )

            source1_id = parts[0].strip()
            matched_ids_raw = parts[1].strip()

            if not source1_id:
                raise ValueError(
                    f"Missing source1_entity_id at line {line_number}."
                )

            if not matched_ids_raw:
                matches: Set[str] = set()
            else:
                matches = {
                    entity_id.strip()
                    for entity_id in matched_ids_raw.split(",")
                    if entity_id.strip()
                }

            ground_truth[source1_id] = matches

    return ground_truth


def get_true_matches(
    ground_truth: GroundTruth,
    source1_entity_id: str,
) -> Set[str]:
    """Return true matches for a Source-1 entity."""

    return ground_truth.get(source1_entity_id, set())


def is_true_match(
    ground_truth: GroundTruth,
    source1_entity_id: str,
    candidate_entity_id: str,
) -> bool:
    """Check whether a candidate is a ground-truth match."""

    return candidate_entity_id in ground_truth.get(source1_entity_id, set())


def calculate_entity_metrics(
    true_matches: set[str],
    predicted_matches: set[str],
) -> dict[str, float]:
    """Calculate precision, recall, and F0.5 for one Source-1 entity."""

    true_matches = set(true_matches)
    predicted_matches = set(predicted_matches)

    true_positive = len(true_matches & predicted_matches)

    if not true_matches and not predicted_matches:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f0.5": 1.0,
        }

    if not true_matches:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
        }

    if not predicted_matches:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
        }

    precision = true_positive / len(predicted_matches)
    recall = true_positive / len(true_matches)

    beta = 0.5
    denominator = (beta ** 2 * precision) + recall

    if denominator == 0:
        f0_5 = 0.0
    else:
        f0_5 = (
            (1 + beta ** 2)
            * precision
            * recall
            / denominator
        )

    return {
        "precision": precision,
        "recall": recall,
        "f0.5": f0_5,
    }


def calculate_macro_metrics(
    ground_truth: dict[str, set[str]],
    predictions: dict[str, set[str]],
) -> dict[str, float]:
    """Calculate macro precision, recall, and F0.5."""

    entity_ids = set(ground_truth) | set(predictions)

    if not entity_ids:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
        }

    precision_sum = 0.0
    recall_sum = 0.0
    f0_5_sum = 0.0

    for entity_id in entity_ids:
        metrics = calculate_entity_metrics(
            ground_truth.get(entity_id, set()),
            predictions.get(entity_id, set()),
        )

        precision_sum += metrics["precision"]
        recall_sum += metrics["recall"]
        f0_5_sum += metrics["f0.5"]

    count = len(entity_ids)

    return {
        "precision": precision_sum / count,
        "recall": recall_sum / count,
        "f0.5": f0_5_sum / count,
    }


def calculate_candidate_recall(
    ground_truth: dict[str, set[str]],
    candidates: dict[str, set[str]],
) -> dict[str, float]:
    """Measure true-match retrieval recall."""

    total_true_matches = 0
    retrieved_true_matches = 0

    for source1_id, true_matches in ground_truth.items():
        true_matches = set(true_matches)
        candidate_matches = candidates.get(source1_id, set())

        total_true_matches += len(true_matches)
        retrieved_true_matches += len(
            true_matches & candidate_matches
        )

    if total_true_matches == 0:
        candidate_recall = 1.0
    else:
        candidate_recall = (
            retrieved_true_matches / total_true_matches
        )

    return {
        "total_true_matches": total_true_matches,
        "retrieved_true_matches": retrieved_true_matches,
        "candidate_recall": candidate_recall,
    }


def calculate_candidate_distribution(
    candidates: dict[str, set[str]],
) -> dict[str, float]:
    """Calculate candidate-count distribution per Source-1 entity."""

    candidate_counts = np.array(
        [
            len(candidate_ids)
            for candidate_ids in candidates.values()
        ],
        dtype=np.int64,
    )

    if len(candidate_counts) == 0:
        return {
            "total_candidate_pairs": 0,
            "num_source1_entities": 0,
            "avg_candidates_per_s1": 0.0,
            "median_candidates_per_s1": 0.0,
            "p90_candidates_per_s1": 0.0,
            "p95_candidates_per_s1": 0.0,
            "p99_candidates_per_s1": 0.0,
            "max_candidates_per_s1": 0.0,
        }

    return {
        "total_candidate_pairs": int(candidate_counts.sum()),
        "num_source1_entities": int(len(candidate_counts)),
        "avg_candidates_per_s1": float(
            np.mean(candidate_counts)
        ),
        "median_candidates_per_s1": float(
            np.percentile(candidate_counts, 50)
        ),
        "p90_candidates_per_s1": float(
            np.percentile(candidate_counts, 90)
        ),
        "p95_candidates_per_s1": float(
            np.percentile(candidate_counts, 95)
        ),
        "p99_candidates_per_s1": float(
            np.percentile(candidate_counts, 99)
        ),
        "max_candidates_per_s1": float(
            np.max(candidate_counts)
        ),
    }


def label_candidate_pairs(
    candidate_pairs: Iterable[tuple[str, str]],
    ground_truth: GroundTruth,
) -> list[dict[str, str | int]]:
    """
    Label generated candidate pairs.

    A candidate matching ground truth receives label=1.
    A candidate not in ground truth receives label=0.

    A ground-truth match that was never generated is NOT converted
    into an ML negative. It is a retrieval/blocking failure.
    """

    labeled_pairs: list[dict[str, str | int]] = []

    for source1_id, candidate_id in candidate_pairs:
        true_matches = ground_truth.get(source1_id, set())

        label = 1 if candidate_id in true_matches else 0

        labeled_pairs.append(
            {
                "source1_entity_id": source1_id,
                "candidate_entity_id": candidate_id,
                "label": label,
            }
        )

    return labeled_pairs


def calculate_retrieval_report(
    ground_truth: GroundTruth,
    candidate_pairs: Iterable[tuple[str, str]],
) -> dict:
    """
    Produce a complete candidate-generation/retrieval report.

    Candidate pairs use the neutral representation:

        (source1_entity_id, candidate_entity_id)

    The function does not assume a one-to-one relationship.

    It distinguishes:
    - retrieved true matches
    - missed true matches
    - no-match Source-1 entities
    - multi-match Source-1 entities
    - S2 vs S3 retrieval
    - candidate-count distribution

    Candidate pairs are expected to be unique.
    """

    # Candidate count for every S1 in ground truth.
    # This means an S1 with zero candidates contributes a zero.
    candidate_counts: dict[str, int] = {
        source1_id: 0
        for source1_id in ground_truth
    }

    # Retrieved true matches for each S1.
    retrieved_matches: dict[str, set[str]] = {
        source1_id: set()
        for source1_id in ground_truth
    }

    total_candidate_pairs = 0

    # Process candidate pairs without requiring a giant list copy.
    for source1_id, candidate_id in candidate_pairs:
        total_candidate_pairs += 1

        if source1_id not in candidate_counts:
            candidate_counts[source1_id] = 0
            retrieved_matches[source1_id] = set()

        candidate_counts[source1_id] += 1

        true_matches = ground_truth.get(source1_id, set())

        if candidate_id in true_matches:
            retrieved_matches[source1_id].add(candidate_id)

    # Overall retrieval statistics.
    total_true_matches = 0
    true_matches_retrieved = 0

    completely_missed_s1: list[str] = []

    for source1_id, true_matches in ground_truth.items():
        true_matches = set(true_matches)

        total_true_matches += len(true_matches)

        retrieved = retrieved_matches.get(
            source1_id,
            set(),
        )

        true_matches_retrieved += len(
            true_matches & retrieved
        )

        if true_matches and not (true_matches & retrieved):
            completely_missed_s1.append(source1_id)

    true_matches_missed = (
        total_true_matches - true_matches_retrieved
    )

    if total_true_matches == 0:
        candidate_recall = 1.0
    else:
        candidate_recall = (
            true_matches_retrieved / total_true_matches
        )

    # Candidate-count distribution across all ground-truth S1 entities.
    counts = np.array(
        list(candidate_counts.values()),
        dtype=np.int64,
    )

    if len(counts) == 0:
        candidate_count_mean = 0.0
        candidate_count_median = 0.0
        candidate_count_p90 = 0.0
        candidate_count_p95 = 0.0
        candidate_count_p99 = 0.0
        candidate_count_max = 0
    else:
        candidate_count_mean = float(np.mean(counts))
        candidate_count_median = float(np.percentile(counts, 50))
        candidate_count_p90 = float(np.percentile(counts, 90))
        candidate_count_p95 = float(np.percentile(counts, 95))
        candidate_count_p99 = float(np.percentile(counts, 99))
        candidate_count_max = int(np.max(counts))

    # S2 / S3 retrieval statistics.
    s2_true = 0
    s2_retrieved = 0
    s3_true = 0
    s3_retrieved = 0

    for source1_id, true_matches in ground_truth.items():
        retrieved = retrieved_matches.get(
            source1_id,
            set(),
        )

        for entity_id in true_matches:
            if entity_id.startswith("S2-"):
                s2_true += 1
                if entity_id in retrieved:
                    s2_retrieved += 1

            elif entity_id.startswith("S3-"):
                s3_true += 1
                if entity_id in retrieved:
                    s3_retrieved += 1

    s2_recall = (
        s2_retrieved / s2_true
        if s2_true
        else 1.0
    )

    s3_recall = (
        s3_retrieved / s3_true
        if s3_true
        else 1.0
    )

    # No-match statistics.
    no_match_s1 = [
        source1_id
        for source1_id, true_matches in ground_truth.items()
        if not true_matches
    ]

    no_match_with_candidates = sum(
        1
        for source1_id in no_match_s1
        if candidate_counts.get(source1_id, 0) > 0
    )

    # Multi-match statistics.
    multi_match_s1 = [
        source1_id
        for source1_id, true_matches in ground_truth.items()
        if len(true_matches) > 1
    ]

    multi_match_true_matches = sum(
        len(ground_truth[source1_id])
        for source1_id in multi_match_s1
    )

    multi_match_fully_retrieved = sum(
        1
        for source1_id in multi_match_s1
        if ground_truth[source1_id].issubset(
            retrieved_matches.get(source1_id, set())
        )
    )

    return {
        "total_true_matches": total_true_matches,
        "true_matches_retrieved": true_matches_retrieved,
        "true_matches_missed": true_matches_missed,
        "candidate_recall": candidate_recall,

        "s1_entities_with_completely_missed_true_matches": (
            completely_missed_s1
        ),

        "candidate_count_mean": candidate_count_mean,
        "candidate_count_median": candidate_count_median,
        "candidate_count_p90": candidate_count_p90,
        "candidate_count_p95": candidate_count_p95,
        "candidate_count_p99": candidate_count_p99,
        "candidate_count_max": candidate_count_max,
        "total_candidate_pairs": total_candidate_pairs,

        "s2": {
            "true_matches": s2_true,
            "retrieved": s2_retrieved,
            "missed": s2_true - s2_retrieved,
            "recall": s2_recall,
        },

        "s3": {
            "true_matches": s3_true,
            "retrieved": s3_retrieved,
            "missed": s3_true - s3_retrieved,
            "recall": s3_recall,
        },

        "no_match": {
            "s1_entities": len(no_match_s1),
            "s1_entities_with_candidates": (
                no_match_with_candidates
            ),
        },

        "multi_match": {
            "s1_entities": len(multi_match_s1),
            "true_matches": multi_match_true_matches,
            "fully_retrieved_entities": (
                multi_match_fully_retrieved
            ),
        },
    }