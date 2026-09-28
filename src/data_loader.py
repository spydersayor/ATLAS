"""
ATLAS - Data Loading

Owner:
    Satyam - Data Loading + Retrieval

Purpose:
    Load and validate ATLAS source TSV files.

Expected schema:
    entity_id
    business_name
    business_address
    country

This module does not perform:
    - normalization
    - fingerprinting
    - blocking
    - candidate generation
    - ground-truth-based retrieval
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def load_source(path: str | Path) -> pd.DataFrame:
    """
    Load one ATLAS source TSV file.

    Parameters
    ----------
    path:
        Path to the TSV file.

    Returns
    -------
    pd.DataFrame
        Loaded source table with the required schema.

    Raises
    ------
    FileNotFoundError
        If the source file does not exist.

    ValueError
        If required columns are missing or entity IDs are invalid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"ATLAS source file not found: {path}"
        )

    if path.suffix.lower() != ".tsv":
        raise ValueError(
            f"Expected a TSV file, got: {path}"
        )

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    validate_source(df, path.name)

    return df


def validate_source(
    df: pd.DataFrame,
    source_name: str = "source",
) -> None:
    """
    Validate an ATLAS source DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{source_name} must be a pandas DataFrame."
        )

    missing_columns = [
        column
        for column in REQUIRED_SOURCE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing required columns: "
            f"{missing_columns}"
        )

    if df["entity_id"].isna().any():
        raise ValueError(
            f"{source_name} contains missing entity_id values."
        )

    if (df["entity_id"].astype(str).str.strip() == "").any():
        raise ValueError(
            f"{source_name} contains empty entity_id values."
        )

    if df["entity_id"].duplicated().any():
        duplicate_count = int(
            df["entity_id"].duplicated().sum()
        )

        raise ValueError(
            f"{source_name} contains "
            f"{duplicate_count} duplicate entity_id values."
        )


def load_training_sources(
    dataset_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three ATLAS training sources.

    Ground truth is intentionally NOT loaded here because
    candidate generation must not depend on ground truth.
    """
    root = Path(dataset_root)

    source1 = load_source(
        root / "train" / "train_source1.tsv"
    )

    source2 = load_source(
        root / "train" / "train_source2.tsv"
    )

    source3 = load_source(
        root / "train" / "train_source3.tsv"
    )

    return source1, source2, source3


def load_test_sources(
    dataset_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three ATLAS test sources.

    No ground truth is used or required.
    """
    root = Path(dataset_root)

    source1 = load_source(
        root / "test" / "test_source1.tsv"
    )

    source2 = load_source(
        root / "test" / "test_source2.tsv"
    )

    source3 = load_source(
        root / "test" / "test_source3.tsv"
    )

    return source1, source2, source3