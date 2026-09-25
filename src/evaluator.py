from __future__ import annotations

from pathlib import Path
from typing import Dict, Set, Iterable

import numpy as np


GroundTruth = Dict[str, Set[str]]


def parse_ground_truth(
    file_path: str | Path,
) -> GroundTruth:
    """
    Parse the ground-truth TSV file.

    Expected columns:
        source1_entity_id
        matched_entity_ids

    Empty matched_entity_ids represents a no-match entity.
    """
    ground_truth: GroundTruth = {}

    with open(
        file_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        header = file.readline().rstrip("\n\r")

        if not header:
            raise ValueError("Ground-truth file is empty.")

        columns = header.split("\t")

        if columns != [
            "source1_entity_id",
            "matched_entity_ids",
        ]:
            raise ValueError(
                "Ground-truth file must contain columns "
                "'source1_entity_id' and 'matched_entity_ids'."
            )

        for line_number, line in enumerate(
            file,
            start=2,
        ):
            line = line.rstrip("\n\r")

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 2:
                raise ValueError(
                    f"Malformed ground-truth row at line "
                    f"{line_number}."
                )

            source1_entity_id = parts[0].strip()
            matched_entity_ids_raw = parts[1].strip()

            if not source1_entity_id:
                raise ValueError(
                    f"Missing Source-1 entity ID at line "
                    f"{line_number}."
                )

            if matched_entity_ids_raw:
                matched_entity_ids = {
                    entity_id.strip()
                    for entity_id in matched_entity_ids_raw.split(",")
                    if entity_id.strip()
                }
            else:
                matched_entity_ids = set()

            ground_truth[source1_entity_id] = matched_entity_ids

    return ground_truth


def get_true_matches(
    ground_truth: GroundTruth,
    source1_entity_id: str,
) -> Set[str]:
    """Return the set of true matches for a Source-1 entity."""
    return set(
        ground_truth.get(
            source1_entity_id,
            set(),
        )
    )


def is_true_match(
    ground_truth: GroundTruth,
    source1_entity_id: str,
    candidate_entity_id: str,
) -> bool:
    """Check whether a candidate entity is a ground-truth match."""
    return candidate_entity_id in ground_truth.get(
        source1_entity_id,
        set(),
    )


def calculate_entity_metrics(
    true_matches: Set[str],
    predicted_matches: Set[str],
) -> dict[str, float | int]:
    """
    Calculate entity-level precision, recall and F0.5.

    Supports:
        - zero matches
        - one match
        - multiple matches

    No top-1 assumption is made.
    """
    true_matches = set(true_matches)
    predicted_matches = set(predicted_matches)

    true_positive = len(
        true_matches & predicted_matches
    )

    false_positive = len(
        predicted_matches - true_matches
    )

    false_negative = len(
        true_matches - predicted_matches
    )

    # Correct no-match prediction.
    if not true_matches and not predicted_matches:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f0.5": 1.0,
        }

    # No predictions when true matches exist.
    if not predicted_matches:
        return {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": 0.0,
            "recall": 0.0,
            "f0.5": 0.0,
        }

    # Predictions exist but there are no true matches.
    if not true_matches:
        return {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
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
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f0.5": float(f05),
    }


def calculate_macro_metrics(
    ground_truth: GroundTruth,
    predictions: Dict[str, Set[str]],
) -> dict[str, float]:
    """Calculate macro-averaged entity metrics."""
    entity_metrics = []

    for source1_entity_id in ground_truth:
        metrics = calculate_entity_metrics(
            ground_truth[source1_entity_id],
            predictions.get(
                source1_entity_id,
                set(),
            ),
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
                [
                    metrics["precision"]
                    for metrics in entity_metrics
                ]
            )
        ),
        "recall": float(
            np.mean(
                [
                    metrics["recall"]
                    for metrics in entity_metrics
                ]
            )
        ),
        "f0.5": float(
            np.mean(
                [
                    metrics["f0.5"]
                    for metrics in entity_metrics
                ]
            )
        ),
    }


def calculate_candidate_recall(
    ground_truth: GroundTruth,
    candidates: Dict[str, Set[str]],
) -> dict[str, float]:
    """Calculate candidate retrieval recall."""
    total_true_matches = 0
    retrieved_true_matches = 0

    for source1_entity_id, true_matches in ground_truth.items():
        candidate_set = candidates.get(
            source1_entity_id,
            set(),
        )

        total_true_matches += len(true_matches)

        retrieved_true_matches += len(
            true_matches & candidate_set
        )

    if total_true_matches == 0:
        recall = 1.0
    else:
        recall = (
            retrieved_true_matches
            / total_true_matches
        )

    return {
        "total_true_matches": total_true_matches,
        "retrieved_true_matches": retrieved_true_matches,
        "candidate_recall": float(recall),
    }


def calculate_candidate_distribution(
    candidates: Dict[str, Set[str]],
) -> dict[str, float | int]:
    """
    Calculate candidate-count distribution per Source-1 entity.

    Includes entities with zero candidates.
    """
    candidate_counts = [
        len(candidate_set)
        for candidate_set in candidates.values()
    ]

    if not candidate_counts:
        return {
            "num_source1_entities": 0,
            "total_candidate_pairs": 0,
            "avg_candidates_per_s1": 0.0,
            "median_candidates_per_s1": 0.0,
            "p90_candidates_per_s1": 0.0,
            "p95_candidates_per_s1": 0.0,
            "p99_candidates_per_s1": 0.0,
            "max_candidates_per_s1": 0,
        }

    counts_array = np.array(
        candidate_counts,
        dtype=float,
    )

    return {
        "num_source1_entities": len(candidate_counts),
        "total_candidate_pairs": int(
            sum(candidate_counts)
        ),
        "avg_candidates_per_s1": float(
            np.mean(counts_array)
        ),
        "median_candidates_per_s1": float(
            np.median(counts_array)
        ),
        "p90_candidates_per_s1": float(
            np.percentile(counts_array, 90)
        ),
        "p95_candidates_per_s1": float(
            np.percentile(counts_array, 95)
        ),
        "p99_candidates_per_s1": float(
            np.percentile(counts_array, 99)
        ),
        "max_candidates_per_s1": int(
            np.max(counts_array)
        ),
    }


def label_candidate_pairs(
    candidate_pairs: Iterable[tuple[str, str]],
    ground_truth: GroundTruth,
) -> list[dict[str, str | int]]:
    """
    Label candidate pairs using ground truth.

    label = 1 only when the candidate entity is a true match.

    Missing true matches are not created as negatives because
    only retrieved candidate pairs are labeled here.
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
    """Produce a detailed candidate retrieval report."""
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

            if is_true_match(
                ground_truth,
                source1_entity_id,
                candidate_entity_id,
            ):
                retrieved_matches[
                    source1_entity_id
                ].add(candidate_entity_id)

    total_true_matches = sum(
        len(matches)
        for matches in ground_truth.values()
    )

    true_matches_retrieved = sum(
        len(matches)
        for matches in retrieved_matches.values()
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

    completely_missed = [
        source1_entity_id
        for source1_entity_id, true_matches
        in ground_truth.items()
        if true_matches
        and not retrieved_matches[source1_entity_id]
    ]

    candidate_count_values = np.array(
        list(candidate_counts.values()),
        dtype=float,
    )

    if len(candidate_count_values) == 0:
        candidate_count_mean = 0.0
        candidate_count_median = 0.0
        candidate_count_p90 = 0.0
        candidate_count_p95 = 0.0
        candidate_count_p99 = 0.0
        candidate_count_max = 0
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
        candidate_count_max = int(
            np.max(candidate_count_values)
        )

    def source_stats(
        prefix: str,
    ) -> dict[str, int | float]:
        total = 0
        retrieved = 0

        for source1_entity_id, true_matches in ground_truth.items():
            for candidate_entity_id in true_matches:
                if candidate_entity_id.startswith(prefix):
                    total += 1

                    if (
                        candidate_entity_id
                        in retrieved_matches[source1_entity_id]
                    ):
                        retrieved += 1

        missed = total - retrieved

        if total == 0:
            recall = 1.0
        else:
            recall = retrieved / total

        return {
            "true_matches": total,
            "retrieved": retrieved,
            "missed": missed,
            "recall": float(recall),
        }

    no_match_source1 = [
        source1_entity_id
        for source1_entity_id, true_matches
        in ground_truth.items()
        if not true_matches
    ]

    no_match_with_candidates = [
        source1_entity_id
        for source1_entity_id in no_match_source1
        if candidate_counts[source1_entity_id] > 0
    ]

    multi_match_source1 = {
        source1_entity_id: true_matches
        for source1_entity_id, true_matches
        in ground_truth.items()
        if len(true_matches) > 1
    }

    multi_match_true_matches = sum(
        len(true_matches)
        for true_matches in multi_match_source1.values()
    )

    multi_match_fully_retrieved = sum(
        1
        for source1_entity_id in multi_match_source1
        if multi_match_source1[source1_entity_id]
        <= retrieved_matches[source1_entity_id]
    )

    return {
        "total_true_matches": total_true_matches,
        "true_matches_retrieved": true_matches_retrieved,
        "true_matches_missed": true_matches_missed,
        "candidate_recall": float(candidate_recall),
        "s1_entities_with_completely_missed_true_matches":
            completely_missed,
        "candidate_count_mean": candidate_count_mean,
        "candidate_count_median": candidate_count_median,
        "candidate_count_p90": candidate_count_p90,
        "candidate_count_p95": candidate_count_p95,
        "candidate_count_p99": candidate_count_p99,
        "candidate_count_max": candidate_count_max,
        "total_candidate_pairs": total_candidate_pairs,
        "s2": source_stats("S2-"),
        "s3": source_stats("S3-"),
        "no_match": {
            "s1_entities": len(no_match_source1),
            "s1_entities_with_candidates":
                len(no_match_with_candidates),
        },
        "multi_match": {
            "s1_entities": len(multi_match_source1),
            "true_matches": multi_match_true_matches,
            "fully_retrieved_entities":
                multi_match_fully_retrieved,
        },
    }


def calculate_binary_metrics(
    y_true: Iterable[int],
    y_scores: Iterable[float],
    threshold: float,
) -> dict[str, float | int]:
    """Calculate candidate-level binary classification metrics."""
    y_true_array = np.asarray(
        list(y_true),
        dtype=int,
    )

    y_scores_array = np.asarray(
        list(y_scores),
        dtype=float,
    )

    if y_true_array.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    if y_scores_array.ndim != 1:
        raise ValueError("y_scores must be one-dimensional.")

    if len(y_true_array) != len(y_scores_array):
        raise ValueError(
            "y_true and y_scores must have the same length."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if not np.all(
        np.isin(y_true_array, [0, 1])
    ):
        raise ValueError(
            "y_true must contain only 0 and 1."
        )

    if not np.all(np.isfinite(y_scores_array)):
        raise ValueError(
            "y_scores must contain only finite values."
        )

    y_pred = (
        y_scores_array >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (y_true_array == 1)
            & (y_pred == 1)
        )
    )

    fp = int(
        np.sum(
            (y_true_array == 0)
            & (y_pred == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true_array == 1)
            & (y_pred == 0)
        )
    )

    tn = int(
        np.sum(
            (y_true_array == 0)
            & (y_pred == 0)
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
            / ((beta_squared * precision) + recall)
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