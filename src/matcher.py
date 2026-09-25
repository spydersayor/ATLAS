"""
ATLAS - Entity Matching Model

Responsible for:
    Candidate features -> ML model -> match probability

Does NOT handle:
    - Candidate generation
    - Threshold optimization
    - Entity-level conflict resolution
    - Final submission generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class EntityMatcher:
    """
    ML matcher for candidate entity pairs.

    The model learns:
        P(candidate pair is a true match)

    Threshold selection is intentionally kept outside this class.
    """

    def __init__(
        self,
        random_state: int = 42,
        max_iter: int = 200,
        learning_rate: float = 0.08,
        max_leaf_nodes: int = 31,
        l2_regularization: float = 1.0,
    ) -> None:
        self.random_state = random_state

        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization,
            random_state=random_state,
        )

        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "EntityMatcher":
        """
        Train the matcher.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features).

        y:
            Binary labels:
                1 = true match
                0 = non-match
        """

        X = np.asarray(X)
        y = np.asarray(y)

        self._validate_training_data(X, y)

        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict_scores(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Return P(match) for every candidate pair.

        No threshold is applied here.
        """

        self._check_fitted()

        X = np.asarray(X)

        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2D feature matrix. Got shape {X.shape}."
            )

        return self.model.predict_proba(X)[:, 1]

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Convert match probabilities into binary predictions.

        Threshold optimization will eventually determine the
        appropriate threshold on validation data.
        """

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be between 0 and 1. Got {threshold}."
            )

        scores = self.predict_scores(X)

        return (scores >= threshold).astype(np.int8)

    def save(self, path: str | Path) -> None:
        """
        Save the trained matcher to disk.
        """

        self._check_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "EntityMatcher":
        """
        Load a previously trained matcher.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Matcher file not found: {path}"
            )

        matcher = joblib.load(path)

        if not isinstance(matcher, EntityMatcher):
            raise TypeError(
                "Loaded object is not an EntityMatcher."
            )

        return matcher

    def _check_fitted(self) -> None:
        """Ensure the model has been trained."""

        if not self.is_fitted:
            raise RuntimeError(
                "EntityMatcher has not been fitted yet. "
                "Call fit(X, y) first."
            )

    @staticmethod
    def _validate_training_data(
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """Validate training matrix and labels."""

        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2D feature matrix. Got shape {X.shape}."
            )

        if y.ndim != 1:
            raise ValueError(
                f"y must be a 1D label array. Got shape {y.shape}."
            )

        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same number of rows. "
                f"Got {len(X)} and {len(y)}."
            )

        if len(X) == 0:
            raise ValueError("Training data cannot be empty.")

        unique_labels = np.unique(y)

        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError(
                f"y must contain only binary labels 0 and 1. "
                f"Got {unique_labels}."
            )

        if len(unique_labels) < 2:
            raise ValueError(
                "Training data must contain both positive (1) "
                "and negative (0) examples."
            )