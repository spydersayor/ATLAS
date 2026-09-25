"""
ATLAS - Entity Fingerprinting

Owner:
    Satyam - Fingerprinting + Blocking

Purpose:
    Create deterministic retrieval fingerprints from normalized
    business names, addresses, and countries.

Fingerprinting is used only for candidate retrieval/blocking.
It must NOT make final entity-match decisions.

Design principles:
    - Deterministic
    - Unicode-safe
    - Preserve multilingual scripts
    - Preserve Unicode combining marks
    - Remove punctuation, symbols, and whitespace
    - Never modify raw columns
    - Never use ground truth
    - Missing/empty values produce empty fingerprints
    - Keep the implementation lightweight for large datasets
"""

from __future__ import annotations

import unicodedata

import pandas as pd


# ---------------------------------------------------------------------------
# Required normalized input columns -> generated fingerprint columns
# ---------------------------------------------------------------------------

FINGERPRINT_COLUMNS = {
    "business_name_normalized": "name_fingerprint",
    "business_address_normalized": "address_fingerprint",
    "country_normalized": "country_fingerprint",
}


# ---------------------------------------------------------------------------
# Generic fingerprinting
# ---------------------------------------------------------------------------

def fingerprint_text(value: object) -> str:
    """
    Create a compact deterministic fingerprint from normalized text.

    Transformation:
        1. Missing values -> ""
        2. Convert to string
        3. Remove Unicode punctuation
        4. Remove Unicode symbols
        5. Remove whitespace
        6. Preserve Unicode letters
        7. Preserve Unicode combining marks
        8. Preserve Unicode numbers
        9. Preserve original character order

    Examples:
        "café déjà vu pvt ltd"
            -> "cafédéjàvupvtltd"

        "123 main street"
            -> "123mainstreet"

        "भारत कंपनी"
            -> "भारतकंपनी"

        "株式会社テスト"
            -> "株式会社テスト"

    Important:
        We intentionally do NOT use:
            [A-Za-z0-9]
        or:
            str.isalnum()

        because those approaches can lose Unicode combining marks
        required by scripts such as Devanagari.
    """

    # Missing Python value.
    if value is None:
        return ""

    # Missing pandas value.
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if not text:
        return ""

    fingerprint_chars: list[str] = []

    for char in text:
        category = unicodedata.category(char)

        # Unicode categories:
        #
        # L = Letter
        # M = Mark / combining mark
        # N = Number
        #
        # Keeping M is important for scripts such as Devanagari.
        if category.startswith(("L", "M", "N")):
            fingerprint_chars.append(char)

    return "".join(fingerprint_chars)


# ---------------------------------------------------------------------------
# Field-specific fingerprints
# ---------------------------------------------------------------------------

def fingerprint_name(value: object) -> str:
    """
    Create a fingerprint for a normalized business name.
    """
    return fingerprint_text(value)


def fingerprint_address(value: object) -> str:
    """
    Create a fingerprint for a normalized business address.
    """
    return fingerprint_text(value)


def fingerprint_country(value: object) -> str:
    """
    Create a fingerprint for a normalized country value.

    Country handling remains open-set. No country whitelist is applied.
    """
    return fingerprint_text(value)


# ---------------------------------------------------------------------------
# DataFrame integration
# ---------------------------------------------------------------------------

def add_fingerprint_columns(
    df: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add fingerprint columns to a normalized source DataFrame.

    Required input columns:
        business_name_normalized
        business_address_normalized
        country_normalized

    Generated columns:
        name_fingerprint
        address_fingerprint
        country_fingerprint

    The original raw and normalized columns are preserved.

    Args:
        df:
            Input DataFrame containing normalized source fields.

        copy:
            If True, return a copied DataFrame.
            If False, modify the supplied DataFrame in place.

    Returns:
        DataFrame containing the original columns plus fingerprint columns.

    Raises:
        TypeError:
            If df is not a pandas DataFrame.

        ValueError:
            If required normalized columns are missing.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = set(FINGERPRINT_COLUMNS.keys())

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required normalized columns for fingerprinting: "
            f"{sorted(missing_columns)}"
        )

    result = df.copy() if copy else df

    result["name_fingerprint"] = (
        result["business_name_normalized"]
        .map(fingerprint_name)
    )

    result["address_fingerprint"] = (
        result["business_address_normalized"]
        .map(fingerprint_address)
    )

    result["country_fingerprint"] = (
        result["country_normalized"]
        .map(fingerprint_country)
    )

    return result


# ---------------------------------------------------------------------------
# Fingerprint validation helpers
# ---------------------------------------------------------------------------

def validate_fingerprint_columns(df: pd.DataFrame) -> None:
    """
    Validate that all expected fingerprint columns exist.

    This does not validate whether a fingerprint is a correct match.
    It only validates the structural fingerprinting contract.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    missing_columns = [
        column
        for column in FINGERPRINT_COLUMNS.values()
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing fingerprint columns: "
            f"{missing_columns}"
        )


def non_empty_fingerprint_mask(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return a boolean mask identifying usable fingerprints.

    Empty fingerprints are excluded from blocking because indexing
    every record under an empty key could create extremely large
    candidate sets.
    """

    if column not in df.columns:
        raise ValueError(
            f"Fingerprint column not found: {column}"
        )

    return df[column].fillna("").astype(str).str.len().gt(0)