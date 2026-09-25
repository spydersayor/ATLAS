from __future__ import annotations

from pathlib import Path
from typing import Dict, Set

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

    The parser is streaming/memory-conscious and does not load the
    entire TSV into a pandas DataFrame.
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
    """
    Return the set of true matches for a Source-1 entity.

    Unknown Source-1 entities are treated as having no matches.
    """

    return ground_truth.get(source1_entity_id, set())


def is_true_match(
    ground_truth: GroundTruth,
    source1_entity_id: str,
    candidate_entity_id: str,
) -> bool:
    """
    Check whether a candidate entity is a true match
    for a Source-1 entity.
    """

    return candidate_entity_id in ground_truth.get(source1_entity_id, set())


def calculate_entity_metrics(
    true_matches: set[str],
    predicted_matches: set[str],
) -> dict[str, float]:
    """
    Calculate precision, recall, and F0.5 for one Source-1 entity.

    Special cases:
    - true={} and predicted={} -> all metrics are 1.0
    - true={} and predicted!= {} -> all metrics are 0.0
    - true!={} and predicted={} -> all metrics are 0.0
    """

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
    """
    Calculate macro-averaged precision, recall, and F0.5
    across Source-1 entities.

    Every Source-1 entity appearing in either ground truth
    or predictions is evaluated.
    """

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
    """
    Measure how many ground-truth true matches were retrieved
    by the candidate-generation stage.

    Candidate recall:

        retrieved true matches
        -----------------------
        total true matches
    """

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
    """
    Calculate the distribution of candidate counts per Source-1 entity.

    Returns:
    - total_candidate_pairs
    - num_source1_entities
    - avg_candidates_per_s1
    - median_candidates_per_s1
    - p90_candidates_per_s1
    - p95_candidates_per_s1
    - p99_candidates_per_s1
    - max_candidates_per_s1
    """

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
    