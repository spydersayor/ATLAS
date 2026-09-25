"""
ATLAS - Threshold Optimization

Responsible for:
    Validation scores -> threshold selection

Optimizes the decision threshold for binary entity matching
using F0.5, which weights precision more heavily than recall.

Does NOT handle:
    - Candidate generation
    - Feature engineering
    - Model training
    - Entity-level conflict resolution
    - Final submission generation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score


@dataclass(frozen=True)
class ThresholdResult:
    """Metrics associated with one decision threshold."""

    threshold: float
    precision: float
    recall: float
    f05: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int


class ThresholdOptimizer:
    """
    Find the threshold that maximizes validation F0.5.

    F0.5 gives more importance to precision than recall:

        F_beta = (1 + beta^2) * P * R
                 -----------------------
                 beta^2 * P + R

    With beta=0.5, false positives are penalized more heavily
    than false negatives.
    """

    def __init__(
        self,
        beta: float = 0.5,
        min_threshold: float = 0.01,
        max_threshold: float = 0.99,
        num_thresholds: int = 99,
    ) -> None:

        if beta <= 0:
            raise ValueError("beta must be greater than 0.")

        if not 0.0 <= min_threshold <= 1.0:
            raise ValueError("min_threshold must be between 0 and 1.")

        if not 0.0 <= max_threshold <= 1.0:
            raise ValueError("max_threshold must be between 0 and 1.")

        if min_threshold > max_threshold:
            raise ValueError(
                "min_threshold cannot be greater than max_threshold."
            )

        if num_thresholds < 2:
            raise ValueError("num_thresholds must be at least 2.")

        self.beta = beta
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.num_thresholds = num_thresholds

        self.best_result: ThresholdResult | None = None

    def optimize(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> ThresholdResult:
        """
        Search thresholds and return the one with maximum F0.5.
        """

        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        self._validate_inputs(y_true, y_scores)

        thresholds = np.linspace(
            self.min_threshold,
            self.max_threshold,
            self.num_thresholds,
        )

        results = [
            self.evaluate_threshold(
                y_true=y_true,
                y_scores=y_scores,
                threshold=threshold,
            )
            for threshold in thresholds
        ]

        # Primary objective: maximize F0.5.
        #
        # Tie-breaker 1:
        #     prefer higher precision.
        #
        # Tie-breaker 2:
        #     prefer the higher threshold.
        #
        # This keeps the optimizer conservative when two thresholds
        # produce effectively identical F0.5 values.
        best = max(
            results,
            key=lambda result: (
                result.f05,
                result.precision,
                result.threshold,
            ),
        )

        self.best_result = best

        return best

    def evaluate_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float,
    ) -> ThresholdResult:
        """
        Evaluate precision, recall, F0.5 and confusion-matrix counts
        for one threshold.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")

        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)

        self._validate_inputs(y_true, y_scores)

        y_pred = (y_scores >= threshold).astype(np.int8)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        precision = float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        recall = float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        f05 = self._f_beta(
            precision=precision,
            recall=recall,
        )

        return ThresholdResult(
            threshold=float(threshold),
            precision=precision,
            recall=recall,
            f05=f05,
            false_positives=int(fp),
            false_negatives=int(fn),
            true_positives=int(tp),
            true_negatives=int(tn),
        )

    def get_best_result(self) -> ThresholdResult:
        """Return the previously optimized result."""

        if self.best_result is None:
            raise RuntimeError(
                "No threshold has been optimized yet. "
                "Call optimize() first."
            )

        return self.best_result

    def _f_beta(
        self,
        precision: float,
        recall: float,
    ) -> float:
        """Calculate F-beta."""

        beta_squared = self.beta ** 2

        denominator = (
            beta_squared * precision
            + recall
        )

        if denominator == 0:
            return 0.0

        return (
            (1 + beta_squared)
            * precision
            * recall
            / denominator
        )

    @staticmethod
    def _validate_inputs(
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> None:

        if y_true.ndim != 1:
            raise ValueError(
                f"y_true must be 1D. Got shape {y_true.shape}."
            )

        if y_scores.ndim != 1:
            raise ValueError(
                f"y_scores must be 1D. Got shape {y_scores.shape}."
            )

        if len(y_true) != len(y_scores):
            raise ValueError(
                "y_true and y_scores must have the same length. "
                f"Got {len(y_true)} and {len(y_scores)}."
            )

        if len(y_true) == 0:
            raise ValueError(
                "y_true and y_scores cannot be empty."
            )

        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError(
                "y_true must contain only binary labels 0 and 1."
            )

        if not np.all(np.isfinite(y_scores)):
            raise ValueError(
                "y_scores must contain only finite values."
            )

        if np.any(y_scores < 0) or np.any(y_scores > 1):
            raise ValueError(
                "y_scores must be probabilities between 0 and 1."
            )

        if len(np.unique(y_true)) < 2:
            raise ValueError(
                "y_true must contain both positive and negative examples."
            )