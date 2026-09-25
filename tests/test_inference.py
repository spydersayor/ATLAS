"""
Integration test for the ATLAS inference pipeline.

This test verifies the complete downstream flow:

    Candidate pairs
        -> Feature engineering
        -> ML scoring
        -> Entity-level decisions
        -> Submission output generation

The matcher used here is synthetic and exists only to test
pipeline integration. It is NOT the real ATLAS training process.
"""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.features import FeatureEngineer, get_feature_columns
from src.inference import InferencePipeline
from src.matcher import EntityMatcher


def build_test_sources():
    """
    Build small synthetic Source 1, Source 2 and Source 3 datasets.
    """

    source1 = pd.DataFrame(
        [
            {
                "entity_id": "S1-001",
                "business_name": "ABC Electronics",
                "business_address": "MG Road Bangalore",
                "country": "India",
            },
            {
                "entity_id": "S1-002",
                "business_name": "XYZ Foods",
                "business_address": "Delhi Main Road",
                "country": "India",
            },
            {
                "entity_id": "S1-003",
                "business_name": "Global Tech",
                "business_address": "Paris France",
                "country": "France",
            },
        ]
    )

    source2 = pd.DataFrame(
        [
            {
                "entity_id": "S2-001",
                "business_name": "ABC Electronics",
                "business_address": "MG Road Bangalore",
                "country": "India",
            },
            {
                "entity_id": "S2-002",
                "business_name": "XYZ Foods",
                "business_address": "Delhi Main Road",
                "country": "India",
            },
            {
                "entity_id": "S2-003",
                "business_name": "Completely Different",
                "business_address": "Mumbai",
                "country": "India",
            },
        ]
    )

    source3 = pd.DataFrame(
        [
            {
                "entity_id": "S3-001",
                "business_name": "Global Tech",
                "business_address": "Paris France",
                "country": "France",
            },
            {
                "entity_id": "S3-002",
                "business_name": "Random Business",
                "business_address": "London UK",
                "country": "UK",
            },
        ]
    )

    return source1, source2, source3


def build_candidate_pairs():
    """
    Build synthetic candidate pairs using the exact candidate
    contract expected from the blocking/candidate-generation module.
    """

    return pd.DataFrame(
        [
            {
                "source1_entity_id": "S1-001",
                "candidate_entity_id": "S2-001",
                "candidate_source": "S2",
                "blocker_sources": ["name_exact", "address_exact"],
                "num_blockers": 2,
            },
            {
                "source1_entity_id": "S1-001",
                "candidate_entity_id": "S2-003",
                "candidate_source": "S2",
                "blocker_sources": ["country_exact"],
                "num_blockers": 1,
            },
            {
                "source1_entity_id": "S1-002",
                "candidate_entity_id": "S2-002",
                "candidate_source": "S2",
                "blocker_sources": ["name_exact", "address_exact"],
                "num_blockers": 2,
            },
            {
                "source1_entity_id": "S1-003",
                "candidate_entity_id": "S3-001",
                "candidate_source": "S3",
                "blocker_sources": ["name_exact", "address_exact"],
                "num_blockers": 2,
            },
            {
                "source1_entity_id": "S1-003",
                "candidate_entity_id": "S3-002",
                "candidate_source": "S3",
                "blocker_sources": ["country_exact"],
                "num_blockers": 1,
            },
        ]
    )


def build_trained_matcher():
    """
    Train a small synthetic matcher for integration testing.

    IMPORTANT:
        This is NOT the real ATLAS training procedure.

    It only proves that:
        feature matrix -> matcher -> probabilities
    works correctly.
    """

    rng = np.random.default_rng(42)

    # Create synthetic feature matrix.
    X = rng.random(
        (300, 25),
        dtype=np.float32,
    )

    # Create synthetic binary labels.
    y = (
        (
            X[:, 0]
            + X[:, 1]
            + X[:, 2]
            + X[:, 3]
        )
        > 2.0
    ).astype(np.int8)

    # Make sure both classes exist.
    assert len(np.unique(y)) == 2

    matcher = EntityMatcher(
        random_state=42,
        max_iter=100,
    )

    matcher.fit(X, y)

    return matcher


def main():
    """
    Run the complete inference integration test.
    """

    # =========================================================
    # 1. Build synthetic input data
    # =========================================================

    source1, source2, source3 = build_test_sources()

    candidate_pairs = build_candidate_pairs()

    print("Synthetic input created")
    print(f"Source 1 rows      : {len(source1)}")
    print(f"Source 2 rows      : {len(source2)}")
    print(f"Source 3 rows      : {len(source3)}")
    print(f"Candidate pairs    : {len(candidate_pairs)}")
    print()

    # =========================================================
    # 2. Train synthetic matcher
    # =========================================================

    matcher = build_trained_matcher()

    print("[PASS] Synthetic matcher training")
    print()

    # =========================================================
    # 3. Create inference pipeline
    # =========================================================

    with tempfile.TemporaryDirectory() as tmp_dir:

        pipeline = InferencePipeline(
            feature_engineer=FeatureEngineer(),
            matcher=matcher,
            threshold=0.5,
            conflict_gap=0.05,
        )

        # =====================================================
        # 4. Run complete inference pipeline
        # =====================================================

        result = pipeline.run(
            test_source1=source1,
            test_source2=source2,
            test_source3=source3,
            candidate_pairs=candidate_pairs,
            output_dir=tmp_dir,
        )

        # =====================================================
        # 5. Extract pipeline results
        # =====================================================

        feature_table = result["feature_table"]
        scored_candidates = result["scored_candidates"]
        decisions = result["decisions"]

        matching_result_path = result["matching_results"]
        candidate_output_path = result["candidate_output"]

        # =====================================================
        # 6. Feature generation validation
        # =====================================================

        assert len(feature_table) == len(candidate_pairs)

        feature_columns = get_feature_columns()

        missing_features = [
            column
            for column in feature_columns
            if column not in feature_table.columns
        ]

        assert not missing_features, (
            f"Missing feature columns: {missing_features}"
        )

        model_feature_matrix = feature_table[
            feature_columns
        ]

        assert model_feature_matrix.shape == (
            len(candidate_pairs),
            len(feature_columns),
        )

        assert np.isfinite(
            model_feature_matrix.to_numpy()
        ).all()

        print("[PASS] Feature generation")

        # =====================================================
        # 7. Candidate scoring validation
        # =====================================================

        assert len(scored_candidates) == len(candidate_pairs)

        required_score_columns = [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "score",
        ]

        assert list(scored_candidates.columns) == (
            required_score_columns
        )

        assert scored_candidates["score"].between(
            0.0,
            1.0,
        ).all()

        assert np.isfinite(
            scored_candidates["score"].to_numpy()
        ).all()

        print("[PASS] Candidate scoring")

        # =====================================================
        # 8. Entity-level decision validation
        # =====================================================

        assert len(decisions) == len(source1)

        assert set(
            decisions["source1_entity_id"]
        ) == set(
            source1["entity_id"]
        )

        required_decision_columns = [
            "source1_entity_id",
            "candidate_count",
            "accepted_count",
            "best_score",
            "second_best_score",
            "score_gap",
            "has_conflict",
            "matched_entity_ids",
        ]

        for column in required_decision_columns:
            assert column in decisions.columns

        print("[PASS] Entity-level decisions")

        # =====================================================
        # 9. Output file validation
        # =====================================================

        matching_path = Path(matching_result_path)
        candidate_path = Path(candidate_output_path)

        assert matching_path.exists()
        assert candidate_path.exists()

        print("[PASS] Submission files generated")

        # =====================================================
        # 10. Read generated matching file
        # =====================================================

        matching_df = pd.read_csv(
            matching_path,
            sep="\t",
            dtype=str,
        )

        # Every Source 1 entity must appear exactly once.
        assert len(matching_df) == len(source1)

        assert set(
            matching_df["source1_entity_id"]
        ) == set(
            source1["entity_id"]
        )

        assert (
            matching_df["source1_entity_id"]
            .duplicated()
            .sum()
            == 0
        )

        assert list(matching_df.columns) == [
            "source1_entity_id",
            "matched_entity_ids",
        ]

        print("[PASS] Matching results")

        # =====================================================
        # 11. Read generated candidate file
        # =====================================================

        candidate_df = pd.read_csv(
            candidate_path,
            sep="\t",
            dtype=str,
        )

        assert len(candidate_df) == len(candidate_pairs)

        assert list(candidate_df.columns) == [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "blocker_sources",
            "num_blockers",
        ]

        # Make sure both candidate sources are preserved.
        assert set(
            candidate_df["candidate_source"]
        ).issubset({"S2", "S3"})

        print("[PASS] Candidate output")

        # =====================================================
        # 12. Candidate-subset invariant
        # =====================================================
        #
        # Every final match MUST have originally existed in
        # candidate_pairs.
        #
        # This is one of the most important ATLAS invariants.
        # =====================================================

        candidate_ids_by_source1 = {}

        for _, row in candidate_pairs.iterrows():

            s1_id = row["source1_entity_id"]
            candidate_id = row["candidate_entity_id"]

            candidate_ids_by_source1.setdefault(
                s1_id,
                set(),
            ).add(candidate_id)

        for _, row in matching_df.iterrows():

            s1_id = row["source1_entity_id"]
            matched_ids = row["matched_entity_ids"]

            # Empty / NaN means no match.
            if pd.isna(matched_ids):
                continue

            if not matched_ids:
                continue

            allowed_candidates = candidate_ids_by_source1.get(
                s1_id,
                set(),
            )

            for candidate_id in matched_ids.split(","):

                candidate_id = candidate_id.strip()

                assert candidate_id in allowed_candidates, (
                    f"Invalid final match: {s1_id} -> "
                    f"{candidate_id}. "
                    "Candidate was not present in candidate_pairs."
                )

        print(
            "[PASS] Final matches are "
            "candidate-subset constrained"
        )

        # =====================================================
        # 13. Candidate row preservation
        # =====================================================

        original_pairs = set(
            zip(
                candidate_pairs["source1_entity_id"],
                candidate_pairs["candidate_entity_id"],
            )
        )

        generated_pairs = set(
            zip(
                candidate_df["source1_entity_id"],
                candidate_df["candidate_entity_id"],
            )
        )

        assert original_pairs == generated_pairs

        print("[PASS] Candidate pairs preserved")

        # =====================================================
        # 14. Temporary output directory cleanup
        # =====================================================
        #
        # tempfile.TemporaryDirectory automatically removes
        # everything after the test finishes.
        # =========================================================


    print()
    print("==========================================")
    print("ALL INFERENCE PIPELINE TESTS PASSED")
    print("==========================================")


if __name__ == "__main__":
    main()