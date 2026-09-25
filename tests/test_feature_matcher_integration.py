"""
ATLAS - Feature Engineering + Matcher Integration Test

Verifies that:

    candidate pairs
        -> FeatureEngineer
        -> model feature matrix
        -> EntityMatcher
        -> match probabilities

This is a small correctness/integration test.
It is NOT a production-scale model-quality evaluation.
"""

import numpy as np
import pandas as pd

from src.features import FeatureEngineer, get_feature_columns
from src.matcher import EntityMatcher


def create_mock_sources():
    """Create small mock source datasets."""

    source1 = pd.DataFrame(
        {
            "entity_id": [
                "S1-001",
                "S1-002",
                "S1-003",
                "S1-004",
            ],
            "business_name": [
                "ABC Electronics",
                "दिल्ली मेडिकल स्टोर",
                "Paris Bakery",
                "Random Traders",
            ],
            "business_address": [
                "12 MG Road Bangalore",
                "45 Main Road Delhi",
                "10 Rue de Paris",
                "Unknown Street",
            ],
            "country": [
                "India",
                "India",
                "France",
                "India",
            ],
        }
    )

    source2 = pd.DataFrame(
        {
            "entity_id": [
                "S2-001",
                "S2-002",
                "S2-003",
                "S2-004",
            ],
            "business_name": [
                "ABC Electronics",
                "दिल्ली मेडिकल स्टोर",
                "Paris Bakery",
                "Completely Different Business",
            ],
            "business_address": [
                "12 MG Road Bangalore",
                None,
                "10 Rue de Paris",
                "999 Unknown Avenue",
            ],
            "country": [
                "India",
                "India",
                "France",
                "India",
            ],
        }
    )

    source3 = pd.DataFrame(
        {
            "entity_id": [
                "S3-001",
                "S3-002",
            ],
            "business_name": [
                "Paris Bakery",
                "Another Business",
            ],
            "business_address": [
                "10 Rue de Paris",
                "Somewhere Else",
            ],
            "country": [
                "France",
                "India",
            ],
        }
    )

    return source1, source2, source3


def create_mock_candidates():
    """Create candidate pairs following the ATLAS candidate contract."""

    return pd.DataFrame(
        {
            "source1_entity_id": [
                "S1-001",
                "S1-002",
                "S1-003",
                "S1-003",
                "S1-004",
            ],
            "candidate_entity_id": [
                "S2-001",
                "S2-002",
                "S2-003",
                "S3-001",
                "S2-004",
            ],
            "candidate_source": [
                "S2",
                "S2",
                "S2",
                "S3",
                "S2",
            ],
            "blocker_sources": [
                ["name_exact", "address_token"],
                ["name_exact"],
                ["name_exact", "country_exact"],
                [
                    "name_exact",
                    "address_token",
                    "country_exact",
                ],
                ["country_exact"],
            ],
            "num_blockers": [
                2,
                1,
                2,
                3,
                1,
            ],
        }
    )


def main():
    print("=" * 70)
    print("ATLAS FEATURE + MATCHER INTEGRATION TEST")
    print("=" * 70)

    # =========================================================
    # 1. Create mock source data
    # =========================================================

    source1, source2, source3 = create_mock_sources()
    candidates = create_mock_candidates()

    print("\nMock data created.")

    print(f"Source 1 records : {len(source1)}")
    print(f"Source 2 records : {len(source2)}")
    print(f"Source 3 records : {len(source3)}")
    print(f"Candidate pairs  : {len(candidates)}")

    # =========================================================
    # 2. Feature engineering
    # =========================================================

    feature_engineer = FeatureEngineer()

    feature_df = feature_engineer.transform(
        source1=source1,
        source2=source2,
        source3=source3,
        candidate_pairs=candidates,
    )

    feature_columns = get_feature_columns()

    X = feature_df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    print("\nFeature matrix:")
    print(f"Rows    : {X.shape[0]}")
    print(f"Columns : {X.shape[1]}")

    # =========================================================
    # 3. Synthetic labels
    # =========================================================
    #
    # These labels are ONLY for integration testing.
    #
    # True matches:
    #
    # S1-001 -> S2-001
    # S1-002 -> S2-002
    # S1-003 -> S2-003
    # S1-003 -> S3-001
    #
    # Non-match:
    #
    # S1-004 -> S2-004

    y = np.array(
        [
            1,
            1,
            1,
            1,
            0,
        ],
        dtype=np.int8,
    )

    # =========================================================
    # 4. Validate feature/label dimensions
    # =========================================================

    assert X.shape[0] == len(candidates)

    assert X.shape[1] == len(feature_columns)

    assert len(X) == len(y)

    print("\n[PASS] Candidate -> feature transformation")
    print("[PASS] Feature matrix shape")
    print("[PASS] Label matrix shape")

    # =========================================================
    # 5. Train matcher
    # =========================================================

    matcher = EntityMatcher(
        max_iter=100,
    )

    matcher.fit(X, y)

    print("[PASS] Matcher training")

    # =========================================================
    # 6. Generate probabilities
    # =========================================================

    scores = matcher.predict_scores(X)

    print("\nMatcher scores:")

    for index, score in enumerate(scores):

        source1_id = feature_df.iloc[index][
            "source1_entity_id"
        ]

        candidate_id = feature_df.iloc[index][
            "candidate_entity_id"
        ]

        print(
            f"{source1_id} -> "
            f"{candidate_id}: "
            f"{score:.4f}"
        )

    # =========================================================
    # 7. Validate probability output
    # =========================================================

    assert len(scores) == len(candidates)

    assert np.all(scores >= 0.0)

    assert np.all(scores <= 1.0)

    print("\n[PASS] Probability generation")
    print("[PASS] Probability range")

    # =========================================================
    # 8. Test binary prediction interface
    # =========================================================

    predictions = matcher.predict(
        X,
        threshold=0.5,
    )

    assert len(predictions) == len(candidates)

    assert np.all(
        np.isin(
            predictions,
            [0, 1],
        )
    )

    print("[PASS] Binary prediction interface")

    # =========================================================
    # 9. Final result
    # =========================================================

    print("\n" + "=" * 70)
    print("ALL FEATURE + MATCHER INTEGRATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()