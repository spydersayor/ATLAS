"""
ATLAS - Entity-Level Conflict Engine

Responsible for:
    Pair scores -> entity-level matching decisions

This module:
    - Applies a supplied matching threshold
    - Groups candidates by source1_entity_id
    - Preserves zero, one, and multiple matches
    - Orders accepted matches by score
    - Computes entity-level score statistics
    - Detects score conflicts / ambiguity

Does NOT handle:
    - Candidate generation
    - Feature engineering
    - Model training
    - Threshold optimization
    - Final file generation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "score",
}


@dataclass(frozen=True)
class EntityDecision:
    """Decision summary for one source1 entity."""

    source1_entity_id: str
    candidate_count: int
    accepted_count: int
    best_score: float
    second_best_score: float
    score_gap: float
    has_conflict: bool
    matched_entity_ids: tuple[str, ...]


class ConflictEngine:
    """
    Convert pair-level model scores into entity-level decisions.

    Important:
        The engine does NOT force one-to-one matching.

    A source1 entity may therefore produce:
        - zero matches
        - one match
        - multiple matches
    """

    def __init__(
        self,
        threshold: float,
        conflict_gap: float = 0.05,
    ) -> None:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        if conflict_gap < 0.0:
            raise ValueError(
                "conflict_gap must be non-negative."
            )

        self.threshold = float(threshold)
        self.conflict_gap = float(conflict_gap)

    def decide(
        self,
        scored_candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce one entity-level decision row per source1 entity.

        Input columns:
            source1_entity_id
            candidate_entity_id
            candidate_source
            score

        Output columns:
            source1_entity_id
            candidate_count
            accepted_count
            best_score
            second_best_score
            score_gap
            has_conflict
            matched_entity_ids
        """

        self._validate_input(scored_candidates)

        if scored_candidates.empty:
            return self._empty_result()

        working = scored_candidates.copy()

        working["score"] = working["score"].astype(float)

        # Deterministic ordering:
        # highest score first, then candidate ID.
        working = working.sort_values(
            by=[
                "source1_entity_id",
                "score",
                "candidate_entity_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="mergesort",
        )

        working["is_accepted"] = (
            working["score"] >= self.threshold
        )

        decisions = []

        for source1_id, group in working.groupby(
            "source1_entity_id",
            sort=False,
        ):
            decisions.append(
                self._build_entity_decision(
                    source1_entity_id=str(source1_id),
                    group=group,
                )
            )

        return pd.DataFrame(decisions)

    def get_match_table(
        self,
        scored_candidates: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return accepted candidate pairs only.

        This is useful before final submission generation.
        """

        self._validate_input(scored_candidates)

        if scored_candidates.empty:
            return scored_candidates.copy()

        matches = scored_candidates[
            scored_candidates["score"] >= self.threshold
        ].copy()

        matches = matches.sort_values(
            by=[
                "source1_entity_id",
                "score",
                "candidate_entity_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
            kind="mergesort",
        )

        matches.reset_index(drop=True, inplace=True)

        return matches

    def _build_entity_decision(
        self,
        source1_entity_id: str,
        group: pd.DataFrame,
    ) -> dict:
        """Build a decision summary for one source1 entity."""

        scores = group["score"].to_numpy(dtype=float)

        candidate_count = len(group)

        accepted = group[
            group["score"] >= self.threshold
        ]

        accepted_count = len(accepted)

        if candidate_count >= 1:
            best_score = float(scores[0])
        else:
            best_score = 0.0

        if candidate_count >= 2:
            second_best_score = float(scores[1])
        else:
            second_best_score = 0.0

        if candidate_count >= 2:
            score_gap = best_score - second_best_score
        else:
            score_gap = best_score

        # A conflict is specifically an ambiguity among strong
        # candidates, not simply the existence of multiple candidates.
        #
        # Two candidates are considered potentially conflicting when:
        #   1. both pass the matching threshold
        #   2. their scores are close enough according to conflict_gap
        has_conflict = self._has_conflict(
            group=group,
            score_gap=score_gap,
        )

        matched_entity_ids = tuple(
            accepted["candidate_entity_id"]
            .astype(str)
            .tolist()
        )

        return {
            "source1_entity_id": source1_entity_id,
            "candidate_count": int(candidate_count),
            "accepted_count": int(accepted_count),
            "best_score": best_score,
            "second_best_score": second_best_score,
            "score_gap": score_gap,
            "has_conflict": bool(has_conflict),
            "matched_entity_ids": matched_entity_ids,
        }

    def _has_conflict(
        self,
        group: pd.DataFrame,
        score_gap: float,
    ) -> bool:
        """
        Detect ambiguity between strong candidates.

        This does NOT reject either candidate.

        It only marks the entity as potentially conflicting so that
        downstream evaluation/error analysis can inspect it.
        """

        accepted_count = int(
            (group["score"] >= self.threshold).sum()
        )

        if accepted_count < 2:
            return False

        return score_gap <= self.conflict_gap

    @staticmethod
    def _validate_input(
        scored_candidates: pd.DataFrame,
    ) -> None:

        if not isinstance(
            scored_candidates,
            pd.DataFrame,
        ):
            raise TypeError(
                "scored_candidates must be a pandas DataFrame."
            )

        missing = (
            REQUIRED_COLUMNS
            - set(scored_candidates.columns)
        )

        if missing:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing)}"
            )

        if scored_candidates["score"].isna().any():
            raise ValueError(
                "score cannot contain NaN values."
            )

        scores = scored_candidates["score"].to_numpy(
            dtype=float
        )

        if not np.all(np.isfinite(scores)):
            raise ValueError(
                "score must contain only finite values."
            )

        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError(
                "score must contain probabilities between 0 and 1."
            )

        if scored_candidates[
            "source1_entity_id"
        ].isna().any():
            raise ValueError(
                "source1_entity_id cannot contain missing values."
            )

        if scored_candidates[
            "candidate_entity_id"
        ].isna().any():
            raise ValueError(
                "candidate_entity_id cannot contain missing values."
            )

    @staticmethod
    def _empty_result() -> pd.DataFrame:
        """Return an empty decision table with the correct schema."""

        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_count",
                "accepted_count",
                "best_score",
                "second_best_score",
                "score_gap",
                "has_conflict",
                "matched_entity_ids",
            ]
        )