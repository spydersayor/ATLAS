from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Iterable

import numpy as np


GroundTruth = Dict[str, Set[str]]


def parse_ground_truth(file_path: str | Path) -> GroundTruth:
    """
    Parse the ground-truth TSV file.

    Expected columns:
        source1_entity_id
        matched_entity_ids

    matched_entity_ids may contain multiple comma-separated IDs.
    An empty value means the Source-1 entity has no true matches.
    """
    ground_truth: GroundTruth = {}

    with open(file_path, "r", encoding="utf-8") as file:
        header = file.readline().strip().split("\t")

        if "source1_entity_id" not in header:
            raise ValueError("Missing source1_entity_id column.")

        if "matched_entity_ids" not in header:
            raise ValueError("Missing matched_entity_ids column.")

        source1_index = header.index("source1_entity_id")
        matches_index = header.index("matched_entity_ids")

        for line_number, line in enumerate(file, start=2):
            line = line.rstrip("\n\r")

            if not line:
                continue

            columns = line.split("\t")

            if len(columns) <= max(source1_index, matches_index):
                raise ValueError(
                    f"Malformed row at line {line_number}."
                )

            source1_entity_id = columns[source1_index].strip()
            matched_value = columns[matches_index].strip()

            if not source1_entity_id:
                continue

            if matched_value:
                matched_ids = {
                    entity_id.strip()
                    for entity_id in matched_value.split(",")
                    if entity_id.strip()
                }
            else:
                matched_ids = set()

            ground_truth[source1_entity_id] = matched_ids

    return ground_truth


def get_true_matches(
    ground_truth: GroundTruth,
    source1_entity_id: str,
) -> Set[str]:
    """
    Return the true matched entity IDs for a Source-1 entity.
    """
    return ground_truth.get(source1_entity_id, set())


def is_true_match(
    ground_truth: GroundTruth,
    source1_entity_id: str,
    candidate_entity_id: str,
) -> bool:
    """
    Return True if candidate_entity_id is a ground-truth match
    for source1_entity_id.
    """
    return candidate_entity_id in get_true_matches(
        ground_truth,
        source1_entity_id,
    )


def calculate_entity_metrics(
    true_matches: Set[str],
    predicted_matches: Set[str],
) -> dict[str, float]:
    """
    Calculate precision, recall and F0.5 for one Source-1 entity.

    Empty true set + empty prediction is treated as a correct
    no-match decision with precision, recall and F0.5 equal to 1.
    """
    true_matches = set(true_matches)
    predicted_matches = set(predicted_matches)

    true_positive = len(true_matches & predicted_matches)
    false_positive = len(predicted_matches - true_matches)
    false_negative = len(true_matches - predicted_matches)

    if not true_matches and not predicted_matches:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f0.5": 1.0,
        }

    if not predicted_matches:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
        }

    if not true_matches:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
        }

    precision = (
        true_positive
        / (true_positive + false_positive)
    )

    recall = (
        true_positive
        / (true_positive + false_negative)
    )

    beta = 0.5
    beta_squared = beta ** 2

    if precision == 0.0 and recall == 0.0:
        f05 = 0.0
    else:
        f05 = (
            (1 + beta_squared)
            * precision
            * recall
            / ((beta_squared * precision) + recall)
        )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f0.5": float(f05),
    }


def calculate_macro_metrics(
    ground_truth: GroundTruth,
    predictions: Dict[str, Set[str]],
) -> dict[str, float]:
    """
    Calculate macro-averaged precision, recall and F0.5
    across Source-1 entities.
    """
    entity_metrics = []

    for source1_entity_id, true_matches in ground_truth.items():
        predicted_matches = predictions.get(
            source1_entity_id,
            set(),
        )

        metrics = calculate_entity_metrics(
            true_matches,
            predicted_matches,
        )

        entity_metrics.append(metrics)

    if not entity_metrics:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
        }

    return {
        "precision": float(
            np.mean(
                [metrics["precision"] for metrics in entity_metrics]
            )
        ),
        "recall": float(
            np.mean(
                [metrics["recall"] for metrics in entity_metrics]
            )
        ),
        "f0.5": float(
            np.mean(
                [metrics["f0.5"] for metrics in entity_metrics]
            )
        ),
    }


def calculate_candidate_recall(
    ground_truth: GroundTruth,
    candidates: Dict[str, Set[str]],
) -> dict[str, float]:
    """
    Calculate candidate recall.

    A true match that never appears in the candidate set is a
    retrieval/blocking failure, not an ML false negative.

    If there are no true matches, recall is defined as 1.0.
    """
    total_true_matches = 0
    retrieved_true_matches = 0

    for source1_entity_id, true_matches in ground_truth.items():
        total_true_matches += len(true_matches)

        candidate_set = candidates.get(
            source1_entity_id,
            set(),
        )

        retrieved_true_matches += len(
            true_matches & candidate_set
        )

    if total_true_matches == 0:
        candidate_recall = 1.0
    else:
        candidate_recall = (
            retrieved_true_matches / total_true_matches
        )

    return {
        "total_true_matches": float(total_true_matches),
        "retrieved_true_matches": float(retrieved_true_matches),
        "candidate_recall": float(candidate_recall),
    }


def calculate_candidate_distribution(
    candidates: Dict[str, Set[str]],
) -> dict[str, float]:
    """
    Calculate candidate-set size statistics per Source-1 entity.
    """
    candidate_counts = [
        len(candidate_set)
        for candidate_set in candidates.values()
    ]

    if not candidate_counts:
        return {
            "total_candidate_pairs": 0.0,
            "num_source1_entities": 0.0,
            "avg_candidates_per_s1": 0.0,
            "median_candidates_per_s1": 0.0,
            "p90_candidates_per_s1": 0.0,
            "p95_candidates_per_s1": 0.0,
            "p99_candidates_per_s1": 0.0,
            "max_candidates_per_s1": 0.0,
        }

    counts = np.asarray(candidate_counts, dtype=float)

    return {
        "total_candidate_pairs": float(np.sum(counts)),
        "num_source1_entities": float(len(counts)),
        "avg_candidates_per_s1": float(np.mean(counts)),
        "median_candidates_per_s1": float(np.median(counts)),
        "p90_candidates_per_s1": float(np.percentile(counts, 90)),
        "p95_candidates_per_s1": float(np.percentile(counts, 95)),
        "p99_candidates_per_s1": float(np.percentile(counts, 99)),
        "max_candidates_per_s1": float(np.max(counts)),
    }


def label_candidate_pairs(
    candidate_pairs: Iterable[tuple[str, str]],
    ground_truth: GroundTruth,
) -> list[dict[str, str | int]]:
    """
    Label retrieved candidate pairs using ground truth.

    Only candidates that actually exist in candidate_pairs are
    labeled. True matches absent from candidate_pairs are NOT
    converted into negative examples.
    """
    labeled_pairs = []

    for source1_entity_id, candidate_entity_id in candidate_pairs:
        label = int(
            is_true_match(
                ground_truth,
                source1_entity_id,
                candidate_entity_id,
            )
        )

        labeled_pairs.append(
            {
                "source1_entity_id": source1_entity_id,
                "candidate_entity_id": candidate_entity_id,
                "label": label,
            }
        )

    return labeled_pairs


def calculate_retrieval_report(
    ground_truth: GroundTruth,
    candidate_pairs: Iterable[tuple[str, str]],
) -> dict:
    """
    Calculate a complete candidate-retrieval report.

    Includes:
        - total true matches
        - retrieved true matches
        - missed true matches
        - candidate recall
        - completely missed Source-1 entities
        - candidate count distribution
        - S2 vs S3 retrieval statistics
        - no-match statistics
        - multi-match statistics
    """
    candidate_counts = {
        source1_entity_id: 0
        for source1_entity_id in ground_truth
    }

    retrieved_matches = {
        source1_entity_id: set()
        for source1_entity_id in ground_truth
    }

    total_candidate_pairs = 0

    for source1_entity_id, candidate_entity_id in candidate_pairs:
        total_candidate_pairs += 1

        if source1_entity_id in candidate_counts:
            candidate_counts[source1_entity_id] += 1

        if source1_entity_id in ground_truth:
            if candidate_entity_id in ground_truth[source1_entity_id]:
                retrieved_matches[source1_entity_id].add(
                    candidate_entity_id
                )

    total_true_matches = sum(
        len(matches)
        for matches in ground_truth.values()
    )

    true_matches_retrieved = sum(
        len(retrieved_matches[source1_entity_id])
        for source1_entity_id in ground_truth
    )

    true_matches_missed = (
        total_true_matches
        - true_matches_retrieved
    )

    if total_true_matches == 0:
        candidate_recall = 1.0
    else:
        candidate_recall = (
            true_matches_retrieved
            / total_true_matches
        )

    completely_missed_entities = [
        source1_entity_id
        for source1_entity_id, true_matches in ground_truth.items()
        if true_matches
        and not retrieved_matches[source1_entity_id]
    ]

    candidate_count_values = np.asarray(
        list(candidate_counts.values()),
        dtype=float,
    )

    if len(candidate_count_values) == 0:
        candidate_count_mean = 0.0
        candidate_count_median = 0.0
        candidate_count_p90 = 0.0
        candidate_count_p95 = 0.0
        candidate_count_p99 = 0.0
        candidate_count_max = 0.0
    else:
        candidate_count_mean = float(
            np.mean(candidate_count_values)
        )
        candidate_count_median = float(
            np.median(candidate_count_values)
        )
        candidate_count_p90 = float(
            np.percentile(candidate_count_values, 90)
        )
        candidate_count_p95 = float(
            np.percentile(candidate_count_values, 95)
        )
        candidate_count_p99 = float(
            np.percentile(candidate_count_values, 99)
        )
        candidate_count_max = float(
            np.max(candidate_count_values)
        )

    s2_true_matches = set()
    s3_true_matches = set()

    for true_matches in ground_truth.values():
        for entity_id in true_matches:
            if entity_id.startswith("S2-"):
                s2_true_matches.add(entity_id)
            elif entity_id.startswith("S3-"):
                s3_true_matches.add(entity_id)

    s2_retrieved = set()
    s3_retrieved = set()

    for matches in retrieved_matches.values():
        for entity_id in matches:
            if entity_id.startswith("S2-"):
                s2_retrieved.add(entity_id)
            elif entity_id.startswith("S3-"):
                s3_retrieved.add(entity_id)

    s2_true_count = len(s2_true_matches)
    s3_true_count = len(s3_true_matches)

    s2_retrieved_count = len(s2_retrieved)
    s3_retrieved_count = len(s3_retrieved)

    s2_missed = s2_true_count - s2_retrieved_count
    s3_missed = s3_true_count - s3_retrieved_count

    s2_recall = (
        1.0
        if s2_true_count == 0
        else s2_retrieved_count / s2_true_count
    )

    s3_recall = (
        1.0
        if s3_true_count == 0
        else s3_retrieved_count / s3_true_count
    )

    no_match_entities = [
        source1_entity_id
        for source1_entity_id, true_matches in ground_truth.items()
        if not true_matches
    ]

    no_match_entities_with_candidates = [
        source1_entity_id
        for source1_entity_id in no_match_entities
        if candidate_counts[source1_entity_id] > 0
    ]

    multi_match_entities = [
        source1_entity_id
        for source1_entity_id, true_matches in ground_truth.items()
        if len(true_matches) > 1
    ]

    multi_match_true_matches = sum(
        len(ground_truth[source1_entity_id])
        for source1_entity_id in multi_match_entities
    )

    fully_retrieved_multi_match_entities = sum(
        1
        for source1_entity_id in multi_match_entities
        if (
            retrieved_matches[source1_entity_id]
            == ground_truth[source1_entity_id]
        )
    )

    return {
        "total_true_matches": total_true_matches,
        "true_matches_retrieved": true_matches_retrieved,
        "true_matches_missed": true_matches_missed,
        "candidate_recall": float(candidate_recall),
        "s1_entities_with_completely_missed_true_matches":
            completely_missed_entities,
        "candidate_count_mean": candidate_count_mean,
        "candidate_count_median": candidate_count_median,
        "candidate_count_p90": candidate_count_p90,
        "candidate_count_p95": candidate_count_p95,
        "candidate_count_p99": candidate_count_p99,
        "candidate_count_max": candidate_count_max,
        "total_candidate_pairs": total_candidate_pairs,
        "s2": {
            "true_matches": s2_true_count,
            "retrieved": s2_retrieved_count,
            "missed": s2_missed,
            "recall": float(s2_recall),
        },
        "s3": {
            "true_matches": s3_true_count,
            "retrieved": s3_retrieved_count,
            "missed": s3_missed,
            "recall": float(s3_recall),
        },
        "no_match": {
            "s1_entities": len(no_match_entities),
            "s1_entities_with_candidates":
                len(no_match_entities_with_candidates),
        },
        "multi_match": {
            "s1_entities": len(multi_match_entities),
            "true_matches": multi_match_true_matches,
            "fully_retrieved_entities":
                fully_retrieved_multi_match_entities,
        },
    }


def calculate_binary_metrics(
    y_true: Iterable[int],
    y_scores: Iterable[float],
    threshold: float,
) -> dict[str, float | int]:
    """
    Calculate candidate-level binary classification metrics.

    Scores >= threshold are predicted as positive.
    """
    y_true_array = np.asarray(
        list(y_true),
        dtype=int,
    )

    y_scores_array = np.asarray(
        list(y_scores),
        dtype=float,
    )

    if y_true_array.shape != y_scores_array.shape:
        raise ValueError(
            "y_true and y_scores must have the same length."
        )

    if y_true_array.ndim != 1:
        raise ValueError(
            "y_true and y_scores must be one-dimensional."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if not np.all(
        np.isin(y_true_array, [0, 1])
    ):
        raise ValueError(
            "y_true must contain only 0 and 1."
        )

    if not np.all(
        np.isfinite(y_scores_array)
    ):
        raise ValueError(
            "y_scores must contain only finite values."
        )

    predicted = (
        y_scores_array >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (y_true_array == 1)
            & (predicted == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true_array == 0)
            & (predicted == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true_array == 1)
            & (predicted == 0)
        )
    )

    tn = int(
        np.sum(
            (y_true_array == 0)
            & (predicted == 0)
        )
    )

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    beta = 0.5
    beta_squared = beta ** 2

    if precision == 0.0 and recall == 0.0:
        f05 = 0.0
    else:
        f05 = (
            (1 + beta_squared)
            * precision
            * recall
            / (
                (beta_squared * precision)
                + recall
            )
        )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f0.5": float(f05),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def classify_candidate_errors(
    candidate_pairs: Iterable[tuple[str, str]],
    ground_truth: GroundTruth,
    predictions: Dict[tuple[str, str], int],
) -> dict:
    """
    Classify candidate-level and retrieval-level errors.

    Categories:

        TP
        FP
        FN_WITH_CANDIDATE
        BLOCKING_FAILURE

    A true match that was never retrieved is a blocking failure,
    not a classification false negative.

    Parameters
    ----------
    candidate_pairs:
        Retrieved candidate pairs represented as:
            (source1_entity_id, candidate_entity_id)

    ground_truth:
        Mapping from Source-1 entity IDs to true matched IDs.

    predictions:
        Mapping from candidate pair to binary model prediction:
            1 = predicted match
            0 = predicted non-match
    """
    candidate_set = set(candidate_pairs)

    true_positives = []
    false_positives = []
    false_negatives_with_candidate = []
    blocking_failures = []

    for pair in candidate_set:
        source1_entity_id, candidate_entity_id = pair

        is_true = is_true_match(
            ground_truth,
            source1_entity_id,
            candidate_entity_id,
        )

        if pair not in predictions:
            raise ValueError(
                "Missing prediction for candidate pair: "
                f"{pair}"
            )

        prediction = predictions[pair]

        if prediction not in (0, 1):
            raise ValueError(
                "Predictions must contain only 0 and 1."
            )

        if is_true and prediction == 1:
            true_positives.append(pair)

        elif not is_true and prediction == 1:
            false_positives.append(pair)

        elif is_true and prediction == 0:
            false_negatives_with_candidate.append(pair)

    for source1_entity_id, true_matches in ground_truth.items():
        for candidate_entity_id in true_matches:
            pair = (
                source1_entity_id,
                candidate_entity_id,
            )

            if pair not in candidate_set:
                blocking_failures.append(pair)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives_with_candidate":
            false_negatives_with_candidate,
        "blocking_failures": blocking_failures,
        "counts": {
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "false_negatives_with_candidate":
                len(false_negatives_with_candidate),
            "blocking_failures": len(blocking_failures),
        },
    }