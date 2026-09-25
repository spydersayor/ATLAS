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

    Preserves the existing ATLAS implementation semantics.
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
# FAST INTERNAL HELPERS
# ============================================================

def _prepare_text(value) -> str:
    """
    Fast text preparation for the hot path.

    This intentionally preserves _safe_text semantics.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if pd.isna(value):
        return ""

    return str(value).strip()


def _build_source_lookup(source_df: pd.DataFrame) -> dict:
    """
    Build a lightweight Python dictionary for O(1)-style
    entity lookup without creating pandas Series objects.

    Values are compact tuples:
        (business_name, business_address, country)
    """
    lookup = {}

    entity_ids = source_df["entity_id"].to_numpy(copy=False)
    names = source_df["business_name"].to_numpy(copy=False)
    addresses = source_df["business_address"].to_numpy(copy=False)
    countries = source_df["country"].to_numpy(copy=False)

    for entity_id, name, address, country in zip(
        entity_ids,
        names,
        addresses,
        countries,
    ):
        lookup[entity_id] = (
            name,
            address,
            country,
        )

    return lookup


def _compute_text_pair_features(
    text_1: str,
    text_2: str,
) -> tuple[float, float, float, float, int]:
    """
    Compute the shared fuzzy/text features for one field.

    Returns:
        ratio,
        levenshtein,
        jaccard,
        token_similarity,
        length_difference

    Important:
        name_ratio and name_levenshtein currently have identical
        semantics in the existing ATLAS implementation, so the
        RapidFuzz ratio is computed once and reused.
    """
    if not text_1 or not text_2:
        return 0.0, 0.0, 0.0, 0.0, abs(len(text_1) - len(text_2))

    ratio = fuzz.ratio(text_1, text_2) / 100.0

    tokens_1 = set(text_1.split())
    tokens_2 = set(text_2.split())

    if tokens_1 and tokens_2:
        union = tokens_1 | tokens_2
        jaccard = len(tokens_1 & tokens_2) / len(union)
    else:
        jaccard = 0.0

    token_similarity = fuzz.token_set_ratio(
        text_1,
        text_2,
    ) / 100.0

    return (
        ratio,
        ratio,
        jaccard,
        token_similarity,
        abs(len(text_1) - len(text_2)),
    )


def _compute_blocking_features_fast(
    blocker_sources,
    num_blockers,
) -> dict:
    """
    Fast blocker provenance feature computation.

    Preserves the existing accepted blocker representations:
        - None / NaN
        - comma-separated string
        - iterable
    """
    if blocker_sources is None:
        blockers = []
    elif isinstance(blocker_sources, float) and pd.isna(blocker_sources):
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
        "num_blockers": float(num_blockers),
    }


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

        ratio, levenshtein, jaccard, token_similarity, length_difference = (
            _compute_text_pair_features(
                name_1,
                name_2,
            )
        )

        return {
            "name_exact": float(
                bool(name_1)
                and bool(name_2)
                and name_1 == name_2
            ),
            "name_ratio": ratio,
            "name_levenshtein": levenshtein,
            "name_jaccard": jaccard,
            "name_token_similarity": token_similarity,
            "name_length_difference": length_difference,
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

        (
            ratio,
            _levenshtein,
            jaccard,
            token_similarity,
            length_difference,
        ) = _compute_text_pair_features(
            address_1,
            address_2,
        )

        return {
            "address_exact": float(
                address_1_available
                and address_2_available
                and address_1 == address_2
            ),
            "address_ratio": ratio,
            "address_jaccard": jaccard,
            "address_token_similarity": token_similarity,
            "address_length_difference": length_difference,
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

        Kept for compatibility with callers/tests that may directly
        invoke this helper.
        """
        return _compute_blocking_features_fast(
            candidate_row["blocker_sources"],
            candidate_row["num_blockers"],
        )

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
        # FAST SOURCE LOOKUPS
        #
        # Convert source tables into compact Python dictionaries.
        # This avoids pandas .loc / Series creation inside the
        # candidate loop.
        # ----------------------------------------------------

        s1_lookup = _build_source_lookup(source1)
        s2_lookup = _build_source_lookup(source2)
        s3_lookup = _build_source_lookup(source3)

        # ----------------------------------------------------
        # PRE-EXTRACT CANDIDATE COLUMNS
        #
        # Avoid iterrows(), which creates one pandas Series per
        # candidate row.
        # ----------------------------------------------------

        s1_ids = candidate_pairs["source1_entity_id"].to_numpy(
            copy=False
        )

        candidate_ids = candidate_pairs[
            "candidate_entity_id"
        ].to_numpy(
            copy=False
        )

        candidate_sources = candidate_pairs[
            "candidate_source"
        ].to_numpy(
            copy=False
        )

        blocker_sources = candidate_pairs[
            "blocker_sources"
        ].to_numpy(
            copy=False
        )

        num_blockers = candidate_pairs[
            "num_blockers"
        ].to_numpy(
            copy=False
        )

        feature_rows = []

        # ----------------------------------------------------
        # HOT LOOP
        # ----------------------------------------------------

        for (
            s1_id,
            candidate_id,
            candidate_source_raw,
            blockers,
            blocker_count,
        ) in zip(
            s1_ids,
            candidate_ids,
            candidate_sources,
            blocker_sources,
            num_blockers,
        ):

            candidate_source = str(candidate_source_raw)

            # ----------------------------------------------
            # Retrieve S1
            # ----------------------------------------------

            try:
                (
                    s1_name_raw,
                    s1_address_raw,
                    s1_country_raw,
                ) = s1_lookup[s1_id]
            except KeyError:
                raise KeyError(
                    f"Source1 entity_id not found: {s1_id}"
                ) from None

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

            try:
                (
                    candidate_name_raw,
                    candidate_address_raw,
                    candidate_country_raw,
                ) = lookup[candidate_id]
            except KeyError:
                raise KeyError(
                    f"{candidate_source} entity_id not found: "
                    f"{candidate_id}"
                ) from None

            # ----------------------------------------------
            # Prepare text once
            # ----------------------------------------------

            s1_name = _prepare_text(s1_name_raw)
            candidate_name = _prepare_text(candidate_name_raw)

            s1_address = _prepare_text(s1_address_raw)
            candidate_address = _prepare_text(candidate_address_raw)

            s1_country = _prepare_text(s1_country_raw)
            candidate_country = _prepare_text(candidate_country_raw)

            # ----------------------------------------------
            # NAME FEATURES
            # ----------------------------------------------

            (
                name_ratio,
                name_levenshtein,
                name_jaccard,
                name_token_similarity,
                name_length_difference,
            ) = _compute_text_pair_features(
                s1_name,
                candidate_name,
            )

            name_features = {
                "name_exact": float(
                    bool(s1_name)
                    and bool(candidate_name)
                    and s1_name == candidate_name
                ),
                "name_ratio": name_ratio,
                "name_levenshtein": name_levenshtein,
                "name_jaccard": name_jaccard,
                "name_token_similarity": name_token_similarity,
                "name_length_difference": name_length_difference,
            }

            # ----------------------------------------------
            # ADDRESS FEATURES
            # ----------------------------------------------

            address_1_available = bool(s1_address)
            address_2_available = bool(candidate_address)

            (
                address_ratio,
                _address_levenshtein,
                address_jaccard,
                address_token_similarity,
                address_length_difference,
            ) = _compute_text_pair_features(
                s1_address,
                candidate_address,
            )

            address_features = {
                "address_exact": float(
                    address_1_available
                    and address_2_available
                    and s1_address == candidate_address
                ),
                "address_ratio": address_ratio,
                "address_jaccard": address_jaccard,
                "address_token_similarity": address_token_similarity,
                "address_length_difference": address_length_difference,
                "address_available_both": float(
                    address_1_available
                    and address_2_available
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

            # ----------------------------------------------
            # COUNTRY
            # ----------------------------------------------

            country_features = {
                "country_exact": float(
                    bool(s1_country)
                    and bool(candidate_country)
                    and s1_country == candidate_country
                )
            }

            # ----------------------------------------------
            # CROSS-FIELD EVIDENCE
            # ----------------------------------------------

            name_score = name_token_similarity
            address_score = address_token_similarity

            both_available = (
                address_1_available
                and address_2_available
            )

            if both_available:
                agreement = min(
                    name_score,
                    address_score,
                )

                gap = abs(
                    name_score - address_score
                )

                combined = (
                    name_score + address_score
                ) / 2.0
            else:
                agreement = 0.0
                gap = 0.0
                combined = name_score

            cross_features = {
                "name_address_agreement": agreement,
                "name_address_gap": gap,
                "combined_name_address_score": combined,
            }

            # ----------------------------------------------
            # BLOCKING PROVENANCE
            # ----------------------------------------------

            blocking_features = _compute_blocking_features_fast(
                blockers,
                blocker_count,
            )

            # ----------------------------------------------
            # ASSEMBLE
            # ----------------------------------------------

            feature_rows.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "candidate_source": candidate_source,
                    **name_features,
                    **address_features,
                    **country_features,
                    **cross_features,
                    **blocking_features,
                }
            )

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