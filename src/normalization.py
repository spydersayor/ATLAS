"""
ATLAS - Text Normalization

Owner:
    Satyam - Data Loading + Retrieval

Purpose:
    Normalize business names, addresses, and countries before
    fingerprinting and blocking.

Design principles:
    - Preserve Unicode characters.
    - Preserve Indic and other non-Latin scripts.
    - Never modify the original raw columns.
    - Convert missing values to empty strings.
    - Apply deterministic normalization.
    - Keep normalization lightweight for large datasets.
"""

from __future__ import annotations

import unicodedata

import pandas as pd


NORMALIZED_COLUMNS = {
    "business_name": "business_name_normalized",
    "business_address": "business_address_normalized",
    "country": "country_normalized",
}


def normalize_text(value: object) -> str:
    """
    Generic text normalization.

    Steps:
        1. Convert missing values to empty string.
        2. Convert value to string.
        3. Unicode-normalize using NFKC.
        4. Apply Unicode-aware case folding.
        5. Replace punctuation and symbols with spaces.
        6. Collapse repeated whitespace.
        7. Strip leading/trailing whitespace.

    Important:
        Unicode letters and combining marks are preserved.
        This is required for multilingual business names and
        addresses.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    text = str(value)

    # Unicode compatibility normalization.
    text = unicodedata.normalize("NFKC", text)

    # Unicode-aware case normalization.
    text = text.casefold()

    # Replace punctuation and symbols with spaces.
    #
    # We intentionally do NOT use a regex such as [^\w\s],
    # because that can remove Unicode combining marks used by
    # scripts such as Devanagari.
    normalized_chars = []

    for char in text:
        category = unicodedata.category(char)

        if category.startswith(("P", "S")) or char.isspace():
            normalized_chars.append(" ")
        else:
            normalized_chars.append(char)

    text = "".join(normalized_chars)

    # Collapse repeated whitespace.
    text = " ".join(text.split())

    return text.strip()


def normalize_business_name(value: object) -> str:
    """Normalize a business name."""

    return normalize_text(value)


def normalize_business_address(value: object) -> str:
    """Normalize a business address."""

    return normalize_text(value)


def normalize_country(value: object) -> str:
    """
    Normalize a country value.

    No fixed country whitelist is used because country handling
    must remain open-set.
    """

    return normalize_text(value)


def add_normalized_columns(
    df: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Add normalized columns to a source DataFrame.

    Original raw columns remain unchanged.

    Adds:
        business_name_normalized
        business_address_normalized
        country_normalized

    Args:
        df:
            Source DataFrame containing the raw fields.

        copy:
            If True, return a copy of the DataFrame.
            If False, modify the supplied DataFrame in place.

    Returns:
        DataFrame containing the original fields plus normalized fields.
    """

    required_columns = {
        "business_name",
        "business_address",
        "country",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns for normalization: "
            f"{sorted(missing_columns)}"
        )

    result = df.copy() if copy else df

    result["business_name_normalized"] = (
        result["business_name"].map(normalize_business_name)
    )

    result["business_address_normalized"] = (
        result["business_address"].map(normalize_business_address)
    )

    result["country_normalized"] = (
        result["country"].map(normalize_country)
    )

    return result