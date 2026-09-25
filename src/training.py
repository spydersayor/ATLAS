"""
ATLAS - Matcher Training Pipeline

Responsible for:
    Candidate labeling -> entity-level train/validation split
    -> pairwise feature engineering -> model training
    -> validation scoring -> F0.5 threshold optimization

Does NOT handle:
    - Candidate generation
    - Blocking
    - Candidate fusion
    - Entity-level conflict resolution
    - Final submission generation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluator import GroundTruth, label_candidate_pairs
from src.features import FeatureEngineer, get_feature_columns
from src.matcher import EntityMatcher
from src.splitter import split_source1_entities
from src.threshold_optimizer import ThresholdOptimizer, ThresholdResult


@dataclass
class TrainingResult:
    """Artifacts produced by the matcher training workflow."""

    matcher: EntityMatcher
    threshold_result: ThresholdResult

    train_rows: int
    validation_rows: int

    train_source1_entities: int
    validation_source1_entities: int

    train_positive_rows: int
    train_negative_rows: int

    validation_positive_rows: int
    validation_negative_rows: int

    validation_scores: np.ndarray
    validation_labels: np.ndarray


class MatcherTrainer:
    """
    Train the ATLAS pairwise entity matcher.

    The split is performed at Source-1 entity level to prevent
    candidate-row leakage between training and validation.
    """

    def __init__(
        self,
        matcher: EntityMatcher | None = None,
        threshold_optimizer: ThresholdOptimizer | None = None,
        feature_engineer: FeatureEngineer | None = None,
        validation_fraction: float = 0.2,
        random_state: int = 42,
    ) -> None:

        if not 0.0 < validation_fraction < 1.0:
            raise ValueError(
                "validation_fraction must be between 0 and 1."
            )

        self.matcher = matcher or EntityMatcher(
            random_state=random_state,
        )

        self.threshold_optimizer = (
            threshold_optimizer or ThresholdOptimizer()
        )

        self.feature_engineer = (
            feature_engineer or FeatureEngineer()
        )

        self.validation_fraction = validation_fraction
        self.random_state = random_state

    def train(
        self,
        source1: pd.DataFrame,
        source2: pd.DataFrame,
        source3: pd.DataFrame,
        candidate_pairs: pd.DataFrame,
        ground_truth: GroundTruth,
    ) -> TrainingResult:
        """
        Execute the complete matcher training workflow.

        Parameters
        ----------
        source1:
            Source-1 entity table.

        source2:
            Source-2 entity table.

        source3:
            Source-3 entity table.

        candidate_pairs:
            Candidate pairs following the ATLAS candidate contract.

        ground_truth:
            Ground-truth mapping owned by the evaluation layer.

        Returns
        -------
        TrainingResult
            Trained matcher, optimized threshold, validation scores,
            labels, and training/validation statistics.
        """

        # --------------------------------------------------------
        # 0. Validate candidate input
        # --------------------------------------------------------

        self._validate_candidate_pairs(candidate_pairs)

        if candidate_pairs.empty:
            raise ValueError(
                "candidate_pairs cannot be empty."
            )

        # --------------------------------------------------------
        # 1. Split Source-1 entities
        # --------------------------------------------------------
        #
        # IMPORTANT:
        # We split by Source-1 entity rather than candidate rows.
        #
        # This prevents candidates belonging to the same S1 entity
        # from appearing in both training and validation.
        # --------------------------------------------------------

        source1_ids = (
            candidate_pairs["source1_entity_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        train_ids, validation_ids = split_source1_entities(
            source1_ids,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )

        train_ids = set(train_ids)
        validation_ids = set(validation_ids)

        if train_ids & validation_ids:
            raise RuntimeError(
                "Training and validation Source-1 entity sets overlap."
            )

        # --------------------------------------------------------
        # 2. Label candidate pairs
        # --------------------------------------------------------
        #
        # Vijay's labeling API returns:
        #
        # [
        #     {
        #         "source1_entity_id": "...",
        #         "candidate_entity_id": "...",
        #         "label": 0 or 1,
        #     },
        #     ...
        # ]
        #
        # We preserve that API and extract only the label column.
        # --------------------------------------------------------

        candidate_records = list(
            candidate_pairs[
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            ].itertuples(index=False, name=None)
        )

        labeled_records = label_candidate_pairs(
            candidate_records,
            ground_truth,
        )

        if len(labeled_records) != len(candidate_pairs):
            raise RuntimeError(
                "Candidate labeling returned an unexpected number "
                "of labeled records."
            )

        labeled_pairs = candidate_pairs.copy()

        labeled_pairs["_label"] = np.asarray(
            [
                record["label"]
                for record in labeled_records
            ],
            dtype=np.int8,
        )

        # --------------------------------------------------------
        # 3. Split candidate rows by Source-1 entity
        # --------------------------------------------------------

        train_mask = (
            labeled_pairs["source1_entity_id"]
            .astype(str)
            .isin(train_ids)
        )

        validation_mask = (
            labeled_pairs["source1_entity_id"]
            .astype(str)
            .isin(validation_ids)
        )

        train_pairs = labeled_pairs.loc[
            train_mask
        ].copy()

        validation_pairs = labeled_pairs.loc[
            validation_mask
        ].copy()

        if train_pairs.empty:
            raise ValueError(
                "Training candidate set is empty."
            )

        if validation_pairs.empty:
            raise ValueError(
                "Validation candidate set is empty."
            )

        # --------------------------------------------------------
        # 4. Validate binary labels
        # --------------------------------------------------------

        y_train = train_pairs["_label"].to_numpy(
            dtype=np.int8,
        )

        y_validation = validation_pairs["_label"].to_numpy(
            dtype=np.int8,
        )

        self._validate_labels(
            y_train,
            "training",
        )

        self._validate_labels(
            y_validation,
            "validation",
        )

        # --------------------------------------------------------
        # 5. Generate pairwise features
        # --------------------------------------------------------

        train_features = self.feature_engineer.transform(
            source1=source1,
            source2=source2,
            source3=source3,
            candidate_pairs=train_pairs,
        )

        validation_features = self.feature_engineer.transform(
            source1=source1,
            source2=source2,
            source3=source3,
            candidate_pairs=validation_pairs,
        )

        # --------------------------------------------------------
        # 6. Select exact model feature schema
        # --------------------------------------------------------

        feature_columns = get_feature_columns()

        X_train = train_features[
            feature_columns
        ].to_numpy(
            dtype=np.float64,
        )

        X_validation = validation_features[
            feature_columns
        ].to_numpy(
            dtype=np.float64,
        )

        self._validate_feature_matrix(
            X_train,
            "training",
        )

        self._validate_feature_matrix(
            X_validation,
            "validation",
        )

        # --------------------------------------------------------
        # 7. Train matcher
        # --------------------------------------------------------

        self.matcher.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------------
        # 8. Score validation candidates
        # --------------------------------------------------------

        validation_scores = self.matcher.predict_scores(
            X_validation,
        )

        if not np.all(
            np.isfinite(validation_scores)
        ):
            raise ValueError(
                "Validation scores contain non-finite values."
            )

        if np.any(validation_scores < 0) or np.any(
            validation_scores > 1
        ):
            raise ValueError(
                "Validation scores must be probabilities in [0, 1]."
            )

        # --------------------------------------------------------
        # 9. Optimize F0.5 threshold
        # --------------------------------------------------------

        threshold_result = self.threshold_optimizer.optimize(
            y_true=y_validation,
            y_scores=validation_scores,
        )

        # --------------------------------------------------------
        # 10. Return reusable training artifacts
        # --------------------------------------------------------

        return TrainingResult(
            matcher=self.matcher,
            threshold_result=threshold_result,
            train_rows=len(train_pairs),
            validation_rows=len(validation_pairs),
            train_source1_entities=len(train_ids),
            validation_source1_entities=len(validation_ids),
            train_positive_rows=int(
                y_train.sum()
            ),
            train_negative_rows=int(
                (y_train == 0).sum()
            ),
            validation_positive_rows=int(
                y_validation.sum()
            ),
            validation_negative_rows=int(
                (y_validation == 0).sum()
            ),
            validation_scores=validation_scores,
            validation_labels=y_validation,
        )

    # ============================================================
    # VALIDATION HELPERS
    # ============================================================

    @staticmethod
    def _validate_candidate_pairs(
        candidate_pairs: pd.DataFrame,
    ) -> None:
        """Validate the minimum candidate-pair contract."""

        required = {
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "blocker_sources",
            "num_blockers",
        }

        missing = required - set(
            candidate_pairs.columns
        )

        if missing:
            raise ValueError(
                "Candidate-pair contract violation. "
                f"Missing columns: {sorted(missing)}"
            )

    @staticmethod
    def _validate_labels(
        labels: np.ndarray,
        split_name: str,
    ) -> None:
        """Validate that a split contains both binary classes."""

        if labels.ndim != 1:
            raise ValueError(
                f"{split_name} labels must be 1D."
            )

        if len(labels) == 0:
            raise ValueError(
                f"{split_name} labels cannot be empty."
            )

        if not np.all(
            np.isin(labels, [0, 1])
        ):
            raise ValueError(
                f"{split_name} labels must contain only 0 and 1."
            )

        if len(np.unique(labels)) < 2:
            raise ValueError(
                f"{split_name} split must contain both "
                "positive and negative examples."
            )

    @staticmethod
    def _validate_feature_matrix(
        X: np.ndarray,
        split_name: str,
    ) -> None:
        """Validate the numerical model matrix."""

        if X.ndim != 2:
            raise ValueError(
                f"{split_name} feature matrix must be 2D."
            )

        expected_features = len(
            get_feature_columns()
        )

        if X.shape[1] != expected_features:
            raise ValueError(
                f"{split_name} feature matrix has "
                f"{X.shape[1]} columns; expected "
                f"{expected_features}."
            )

        if len(X) == 0:
            raise ValueError(
                f"{split_name} feature matrix cannot be empty."
            )

        if not np.all(
            np.isfinite(X)
        ):
            raise ValueError(
                f"{split_name} feature matrix contains "
                "non-finite values."
            )