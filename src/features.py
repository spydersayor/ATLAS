"""
ATLAS - Pairwise Feature Engineering

Owner:
    Sayor - ML + Integration

Purpose:
    Convert candidate pairs into interpretable pairwise evidence features.

Input:
    - Source 1 records
    - Source 2 records
    - Source 3 records
    - Candidate pairs produced by the blocking layer

Output:
    One feature row per candidate pair.

Important:
    This module does NOT perform candidate generation or blocking.
    It consumes the candidate contract provided by the retrieval layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from rapidfuzz import fuzz


# ============================================================
# CONTRACT
# ============================================================

REQUIRED_CANDIDATE_COLUMNS = {
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "blocker_sources",
    "num_blockers",
}

REQUIRED_SOURCE_COLUMNS = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}


# ============================================================
# LOW-LEVEL TEXT UTILITIES
# ============================================================

def _safe_text(value) -> str:
    """
    Convert a value into a safe string.

    Missing values become an empty string.
    Unicode is preserved.
    """
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def _tokenize(value: str) -> set[str]:
    """
    Lightweight whitespace tokenization.

    We deliberately do not perform normalization here.
    Normalization belongs to the upstream data-preparation layer.
    """
    text = _safe_text(value)

    if not text:
        return set()

    return set(text.split())


def _jaccard_similarity(a: str, b: str) -> float:
    """
    Token-level Jaccard similarity.

    J(A,B) = |A ∩ B| / |A ∪ B|

    Returns 0 when either side has no tokens.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)

    if not tokens_a or not tokens_b:
        return 0.0

    union = tokens_a | tokens_b

    if not union:
        return 0.0

    return len(tokens_a & tokens_b) / len(union)


def _ratio(a: str, b: str) -> float:
    """
    RapidFuzz ratio normalized to [0, 1].
    """
    a = _safe_text(a)
    b = _safe_text(b)

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def _levenshtein_similarity(a: str, b: str) -> float:
    """
    Normalized Levenshtein similarity.

    1.0 = identical
    0.0 = maximally different

    This uses RapidFuzz's normalized edit-distance similarity.
    """
    a = _safe_text(a)
    b = _safe_text(b)

    if not a or not b:
        return 0.0

    return fuzz.ratio(a, b) / 100.0


def _token_similarity(a: str, b: str) -> float:
    """
    Token-set similarity.

    Useful when token order changes.
    """
    a = _safe_text(a)
    b = _safe_text(b)

    if not a or not b:
        return 0.0

    return fuzz.token_set_ratio(a, b) / 100.0


def _length_difference(a: str, b: str) -> int:
    """
    Absolute character-length difference.
    """
    return abs(len(_safe_text(a)) - len(_safe_text(b)))


# ============================================================
# FEATURE ENGINEER
# ============================================================

@dataclass
class FeatureEngineer:
    """
    Pairwise feature generator for S1 ↔ S2/S3 candidate pairs.
    """

    def validate_candidate_contract(
        self,
        candidate_pairs: pd.DataFrame,
    ) -> None:
        """
        Validate the candidate-pair interface.
        """
        missing = REQUIRED_CANDIDATE_COLUMNS - set(candidate_pairs.columns)

        if missing:
            raise ValueError(
                "Candidate-pair contract violation. "
                f"Missing columns: {sorted(missing)}"
            )

        valid_sources = {"S2", "S3"}

        actual_sources = set(
            candidate_pairs["candidate_source"]
            .dropna()
            .astype(str)
            .unique()
        )

        invalid_sources = actual_sources - valid_sources

        if invalid_sources:
            raise ValueError(
                "Invalid candidate_source values: "
                f"{sorted(invalid_sources)}. "
                "Expected only S2 or S3."
            )

    def validate_source_contract(
        self,
        source_df: pd.DataFrame,
        source_name: str,
    ) -> None:
        """
        Validate a source table.
        """
        missing = REQUIRED_SOURCE_COLUMNS - set(source_df.columns)

        if missing:
            raise ValueError(
                f"{source_name} contract violation. "
                f"Missing columns: {sorted(missing)}"
            )

    # --------------------------------------------------------
    # Pair-level feature generation
    # --------------------------------------------------------

    def _compute_name_features(
        self,
        name_1: str,
        name_2: str,
    ) -> dict:
        """
        Generate all name-related features.
        """
        name_1 = _safe_text(name_1)
        name_2 = _safe_text(name_2)

        return {
            "name_exact": float(
                bool(name_1) and bool(name_2) and name_1 == name_2
            ),
            "name_ratio": _ratio(name_1, name_2),
            "name_levenshtein": _levenshtein_similarity(
                name_1,
                name_2,
            ),
            "name_jaccard": _jaccard_similarity(
                name_1,
                name_2,
            ),
            "name_token_similarity": _token_similarity(
                name_1,
                name_2,
            ),
            "name_length_difference": _length_difference(
                name_1,
                name_2,
            ),
        }

    def _compute_address_features(
        self,
        address_1: str,
        address_2: str,
        candidate_source: str,
    ) -> dict:
        """
        Generate address-related features.

        Missing addresses are explicitly represented.
        Missing address does not automatically become a negative
        matching signal.
        """
        address_1 = _safe_text(address_1)
        address_2 = _safe_text(address_2)

        address_1_available = bool(address_1)
        address_2_available = bool(address_2)

        return {
            "address_exact": float(
                address_1_available
                and address_2_available
                and address_1 == address_2
            ),
            "address_ratio": _ratio(
                address_1,
                address_2,
            ),
            "address_jaccard": _jaccard_similarity(
                address_1,
                address_2,
            ),
            "address_token_similarity": _token_similarity(
                address_1,
                address_2,
            ),
            "address_length_difference": _length_difference(
                address_1,
                address_2,
            ),
            "address_available_both": float(
                address_1_available and address_2_available
            ),
            "address_missing_source2": float(
                candidate_source == "S2"
                and not address_2_available
            ),
            "address_missing_source3": float(
                candidate_source == "S3"
                and not address_2_available
            ),
        }

    def _compute_country_features(
        self,
        country_1: str,
        country_2: str,
    ) -> dict:
        """
        Country agreement feature.

        No country values are hard-coded.
        """
        country_1 = _safe_text(country_1)
        country_2 = _safe_text(country_2)

        return {
            "country_exact": float(
                bool(country_1)
                and bool(country_2)
                and country_1 == country_2
            ),
        }

    def _compute_cross_field_features(
        self,
        name_features: dict,
        address_features: dict,
    ) -> dict:
        """
        Combine name and address evidence.

        These are evidence features, NOT final decision rules.
        """
        name_score = name_features["name_token_similarity"]
        address_score = address_features["address_token_similarity"]

        both_available = address_features["address_available_both"]

        agreement = (
            min(name_score, address_score)
            if both_available
            else 0.0
        )

        gap = (
            abs(name_score - address_score)
            if both_available
            else 0.0
        )

        combined = (
            (name_score + address_score) / 2.0
            if both_available
            else name_score
        )

        return {
            "name_address_agreement": agreement,
            "name_address_gap": gap,
            "combined_name_address_score": combined,
        }

    def _compute_blocking_features(
        self,
        candidate_row: pd.Series,
    ) -> dict:
        """
        Convert blocker provenance into numeric model features.
        """
        blocker_sources = candidate_row["blocker_sources"]

        if blocker_sources is None or (
            isinstance(blocker_sources, float)
            and pd.isna(blocker_sources)
        ):
            blockers = []
        elif isinstance(blocker_sources, str):
            blockers = [
                x.strip()
                for x in blocker_sources.split(",")
                if x.strip()
            ]
        elif isinstance(blocker_sources, Iterable):
            blockers = list(blocker_sources)
        else:
            blockers = []

        normalized = {
            str(blocker).strip().lower()
            for blocker in blockers
        }

        return {
            "was_name_block": float(
                any("name" in blocker for blocker in normalized)
            ),
            "was_address_block": float(
                any("address" in blocker for blocker in normalized)
            ),
            "was_country_block": float(
                any("country" in blocker for blocker in normalized)
            ),
            "has_exact_name_block": float(
                "name_exact" in normalized
            ),
            "has_exact_address_block": float(
                "address_exact" in normalized
            ),
            "multiple_independent_blockers": float(
                len(normalized) >= 2
            ),
            "num_blockers": float(
                candidate_row["num_blockers"]
            ),
        }

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def transform(
        self,
        source1: pd.DataFrame,
        source2: pd.DataFrame,
        source3: pd.DataFrame,
        candidate_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate one feature row per candidate pair.

        Parameters
        ----------
        source1:
            Source 1 records.

        source2:
            Source 2 records.

        source3:
            Source 3 records.

        candidate_pairs:
            Candidate-pair DataFrame following the ATLAS contract.

        Returns
        -------
        pd.DataFrame
            Candidate identifiers + numeric model features.
        """

        self.validate_source_contract(source1, "source1")
        self.validate_source_contract(source2, "source2")
        self.validate_source_contract(source3, "source3")
        self.validate_candidate_contract(candidate_pairs)

        # ----------------------------------------------------
        # Index source tables for O(1)-style lookup.
        # ----------------------------------------------------

        s1_lookup = source1.set_index("entity_id", drop=False)
        s2_lookup = source2.set_index("entity_id", drop=False)
        s3_lookup = source3.set_index("entity_id", drop=False)

        feature_rows = []

        for _, candidate in candidate_pairs.iterrows():

            s1_id = candidate["source1_entity_id"]
            candidate_id = candidate["candidate_entity_id"]
            candidate_source = str(
                candidate["candidate_source"]
            )

            # ----------------------------------------------
            # Retrieve S1
            # ----------------------------------------------

            if s1_id not in s1_lookup.index:
                raise KeyError(
                    f"Source1 entity_id not found: {s1_id}"
                )

            s1 = s1_lookup.loc[s1_id]

            # ----------------------------------------------
            # Retrieve candidate from S2/S3
            # ----------------------------------------------

            if candidate_source == "S2":
                lookup = s2_lookup
            elif candidate_source == "S3":
                lookup = s3_lookup
            else:
                raise ValueError(
                    f"Unsupported candidate source: "
                    f"{candidate_source}"
                )

            if candidate_id not in lookup.index:
                raise KeyError(
                    f"{candidate_source} entity_id not found: "
                    f"{candidate_id}"
                )

            candidate_entity = lookup.loc[candidate_id]

            # ----------------------------------------------
            # Name
            # ----------------------------------------------

            name_features = self._compute_name_features(
                s1["business_name"],
                candidate_entity["business_name"],
            )

            # ----------------------------------------------
            # Address
            # ----------------------------------------------

            address_features = self._compute_address_features(
                s1["business_address"],
                candidate_entity["business_address"],
                candidate_source,
            )

            # ----------------------------------------------
            # Country
            # ----------------------------------------------

            country_features = self._compute_country_features(
                s1["country"],
                candidate_entity["country"],
            )

            # ----------------------------------------------
            # Cross-field evidence
            # ----------------------------------------------

            cross_features = self._compute_cross_field_features(
                name_features,
                address_features,
            )

            # ----------------------------------------------
            # Blocking provenance
            # ----------------------------------------------

            blocking_features = self._compute_blocking_features(
                candidate
            )

            # ----------------------------------------------
            # Assemble feature row
            # ----------------------------------------------

            row = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": candidate_id,
                "candidate_source": candidate_source,
                **name_features,
                **address_features,
                **country_features,
                **cross_features,
                **blocking_features,
            }

            feature_rows.append(row)

        result = pd.DataFrame(feature_rows)

        return result


# ============================================================
# FEATURE COLUMN CONTRACT
# ============================================================

FEATURE_COLUMNS = [
    # Name
    "name_exact",
    "name_ratio",
    "name_levenshtein",
    "name_jaccard",
    "name_token_similarity",
    "name_length_difference",

    # Address
    "address_exact",
    "address_ratio",
    "address_jaccard",
    "address_token_similarity",
    "address_length_difference",
    "address_available_both",
    "address_missing_source2",
    "address_missing_source3",

    # Country
    "country_exact",

    # Cross-field
    "name_address_agreement",
    "name_address_gap",
    "combined_name_address_score",

    # Blocking provenance
    "was_name_block",
    "was_address_block",
    "was_country_block",
    "has_exact_name_block",
    "has_exact_address_block",
    "multiple_independent_blockers",
    "num_blockers",
]


def get_feature_columns() -> list[str]:
    """
    Return the ordered model-feature schema.

    IDs/source metadata are intentionally excluded.
    """
    return FEATURE_COLUMNS.copy()