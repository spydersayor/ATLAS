"""
ATLAS - Submission Output Generator

Responsible for:
    Entity-level decisions + candidate pairs -> submission TSV files

Outputs:
    output/matching_results.tsv
    output/candidate_pairs.tsv

Important invariants:
    - Every test source1 entity appears exactly once in matching_results.
    - Zero, one, or multiple matches are allowed.
    - Final matches must be a subset of candidate pairs.
    - Candidate pairs are preserved exactly as supplied.
    - Only S2/S3 candidate IDs may appear in final matches.

Does NOT handle:
    - Candidate generation
    - Feature engineering
    - Model training
    - Threshold optimization
    - Conflict resolution
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


CANDIDATE_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "blocker_sources",
    "num_blockers",
]

DECISION_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]

MATCHING_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]


class OutputGenerator:
    """
    Generate and validate ATLAS submission files.
    """

    def __init__(
        self,
        output_dir: str | Path = "output",
    ) -> None:
        self.output_dir = Path(output_dir)

    def generate(
        self,
        test_source1: pd.DataFrame,
        candidate_pairs: pd.DataFrame,
        decisions: pd.DataFrame,
    ) -> tuple[Path, Path]:
        """
        Generate both required submission files.

        Parameters
        ----------
        test_source1:
            Complete test Source 1 dataframe.

        candidate_pairs:
            Candidate pairs supplied by the blocking pipeline.

        decisions:
            Entity-level decisions from ConflictEngine.

        Returns
        -------
        tuple[Path, Path]
            Paths to:
                matching_results.tsv
                candidate_pairs.tsv
        """

        self._validate_test_source1(
            test_source1
        )

        self._validate_candidate_pairs(
            candidate_pairs
        )

        self._validate_decisions(
            decisions
        )

        matching_results = (
            self.build_matching_results(
                test_source1=test_source1,
                decisions=decisions,
                candidate_pairs=candidate_pairs,
            )
        )

        candidate_output = (
            self.build_candidate_pairs(
                candidate_pairs
            )
        )

        self._validate_matching_results(
            matching_results=matching_results,
            test_source1=test_source1,
            candidate_pairs=candidate_pairs,
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        matching_path = (
            self.output_dir
            / "matching_results.tsv"
        )

        candidate_path = (
            self.output_dir
            / "candidate_pairs.tsv"
        )

        matching_results.to_csv(
            matching_path,
            sep="\t",
            index=False,
        )

        candidate_output.to_csv(
            candidate_path,
            sep="\t",
            index=False,
        )

        return (
            matching_path,
            candidate_path,
        )

    def build_matching_results(
        self,
        test_source1: pd.DataFrame,
        decisions: pd.DataFrame,
        candidate_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build one matching-results row for every test S1 entity.

        Entities without accepted matches receive an empty
        matched_entity_ids field.

        The final match set is always restricted to candidate pairs.
        """

        self._validate_test_source1(
            test_source1
        )

        self._validate_candidate_pairs(
            candidate_pairs
        )

        self._validate_decisions(
            decisions
        )

        test_ids = (
            test_source1["entity_id"]
            .astype(str)
            .drop_duplicates()
        )

        candidate_lookup = (
            self._build_candidate_lookup(
                candidate_pairs
            )
        )

        decision_lookup: dict[
            str,
            list[str],
        ] = {}

        for _, row in decisions.iterrows():

            source1_id = str(
                row["source1_entity_id"]
            )

            matched_ids = (
                self._parse_match_ids(
                    row["matched_entity_ids"]
                )
            )

            decision_lookup[
                source1_id
            ] = matched_ids

        output_rows = []

        for source1_id in test_ids:

            source1_id = str(
                source1_id
            )

            allowed_candidates = (
                candidate_lookup.get(
                    source1_id,
                    set(),
                )
            )

            requested_matches = (
                decision_lookup.get(
                    source1_id,
                    [],
                )
            )

            # Critical invariant:
            #
            # final_matches ⊆ candidate_pairs
            #
            valid_matches = [
                entity_id
                for entity_id in requested_matches
                if entity_id in allowed_candidates
            ]

            # Remove accidental duplicates while
            # preserving deterministic order.
            valid_matches = list(
                dict.fromkeys(
                    valid_matches
                )
            )

            output_rows.append(
                {
                    "source1_entity_id": source1_id,
                    "matched_entity_ids": ",".join(
                        valid_matches
                    ),
                }
            )

        return pd.DataFrame(
            output_rows,
            columns=MATCHING_COLUMNS,
        )

    @staticmethod
    def build_candidate_pairs(
        candidate_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return candidate pairs in the required output schema.

        No candidate generation or filtering happens here.

        The complete candidate contract is validated before
        the dataframe is returned.
        """

        OutputGenerator._validate_candidate_pairs(
            candidate_pairs
        )

        return candidate_pairs[
            CANDIDATE_COLUMNS
        ].copy()

    @staticmethod
    def _build_candidate_lookup(
        candidate_pairs: pd.DataFrame,
    ) -> dict[str, set[str]]:
        """
        Build:

            source1_entity_id
                -> set(candidate_entity_id)
        """

        lookup: dict[
            str,
            set[str],
        ] = {}

        for (
            source1_id,
            group,
        ) in candidate_pairs.groupby(
            "source1_entity_id",
            sort=False,
        ):

            lookup[
                str(source1_id)
            ] = set(
                group[
                    "candidate_entity_id"
                ]
                .astype(str)
            )

        return lookup

    @staticmethod
    def _parse_match_ids(
        value,
    ) -> list[str]:
        """
        Convert supported matched_entity_ids representations
        into a clean list of entity IDs.

        Supported:
            tuple
            list
            set
            numpy array
            comma-separated string
            empty string
            None
            NaN
        """

        if value is None:
            return []

        if isinstance(
            value,
            float,
        ) and pd.isna(value):
            return []

        if isinstance(
            value,
            str,
        ):

            value = value.strip()

            if not value:
                return []

            return [
                item.strip()
                for item in value.split(",")
                if item.strip()
            ]

        if isinstance(
            value,
            (
                tuple,
                list,
                set,
            ),
        ):

            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        # Handle numpy arrays and similar iterables.
        try:

            return [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]

        except TypeError:

            return [
                str(value).strip()
            ]

    @staticmethod
    def _validate_test_source1(
        test_source1: pd.DataFrame,
    ) -> None:

        if not isinstance(
            test_source1,
            pd.DataFrame,
        ):
            raise TypeError(
                "test_source1 must be a pandas DataFrame."
            )

        if "entity_id" not in test_source1.columns:
            raise ValueError(
                "test_source1 must contain "
                "'entity_id'."
            )

        if test_source1[
            "entity_id"
        ].isna().any():

            raise ValueError(
                "test_source1 entity_id cannot contain "
                "missing values."
            )

    @staticmethod
    def _validate_candidate_pairs(
        candidate_pairs: pd.DataFrame,
    ) -> None:

        if not isinstance(
            candidate_pairs,
            pd.DataFrame,
        ):
            raise TypeError(
                "candidate_pairs must be a pandas DataFrame."
            )

        missing = (
            set(CANDIDATE_COLUMNS)
            - set(candidate_pairs.columns)
        )

        if missing:
            raise ValueError(
                "Candidate pairs are missing required "
                f"columns: {sorted(missing)}"
            )

        if candidate_pairs[
            "source1_entity_id"
        ].isna().any():

            raise ValueError(
                "source1_entity_id cannot contain "
                "missing values."
            )

        if candidate_pairs[
            "candidate_entity_id"
        ].isna().any():

            raise ValueError(
                "candidate_entity_id cannot contain "
                "missing values."
            )

        invalid_sources = (
            set(
                candidate_pairs[
                    "candidate_source"
                ].astype(str)
            )
            - {"S2", "S3"}
        )

        if invalid_sources:

            raise ValueError(
                "candidate_source contains invalid values: "
                f"{sorted(invalid_sources)}"
            )

    @staticmethod
    def _validate_decisions(
        decisions: pd.DataFrame,
    ) -> None:

        if not isinstance(
            decisions,
            pd.DataFrame,
        ):
            raise TypeError(
                "decisions must be a pandas DataFrame."
            )

        required = {
            "source1_entity_id",
            "matched_entity_ids",
        }

        missing = (
            required
            - set(decisions.columns)
        )

        if missing:

            raise ValueError(
                "Decisions are missing required "
                f"columns: {sorted(missing)}"
            )

        if decisions[
            "source1_entity_id"
        ].isna().any():

            raise ValueError(
                "Decision source1_entity_id cannot "
                "contain missing values."
            )

    @staticmethod
    def _validate_matching_results(
        matching_results: pd.DataFrame,
        test_source1: pd.DataFrame,
        candidate_pairs: pd.DataFrame,
    ) -> None:
        """
        Validate final matching-results invariants.
        """

        expected_ids = (
            test_source1[
                "entity_id"
            ]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        actual_ids = (
            matching_results[
                "source1_entity_id"
            ]
            .astype(str)
            .tolist()
        )

        # Every test S1 exactly once.
        if len(actual_ids) != len(
            set(actual_ids)
        ):

            raise ValueError(
                "matching_results contains duplicate "
                "source1_entity_id values."
            )

        if set(actual_ids) != set(
            expected_ids
        ):

            raise ValueError(
                "matching_results must contain exactly "
                "one row for every test Source 1 entity."
            )

        candidate_lookup = (
            OutputGenerator._build_candidate_lookup(
                candidate_pairs
            )
        )

        # Every final match must be a candidate.
        for _, row in (
            matching_results.iterrows()
        ):

            source1_id = str(
                row["source1_entity_id"]
            )

            matches = (
                OutputGenerator._parse_match_ids(
                    row["matched_entity_ids"]
                )
            )

            allowed = candidate_lookup.get(
                source1_id,
                set(),
            )

            invalid_matches = (
                set(matches) - allowed
            )

            if invalid_matches:

                raise ValueError(
                    f"Source1 entity {source1_id} "
                    "contains matches that are not "
                    "candidate pairs: "
                    f"{sorted(invalid_matches)}"
                )

            # Only S2/S3 IDs are permitted.
            for match_id in matches:

                if not (
                    match_id.startswith("S2-")
                    or match_id.startswith("S3-")
                ):

                    raise ValueError(
                        f"Invalid matched entity ID: "
                        f"{match_id}"
                    )