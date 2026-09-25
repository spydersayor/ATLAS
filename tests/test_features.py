import pandas as pd

from src.features import FeatureEngineer, get_feature_columns


def build_mock_data():
    """
    Small mock dataset used only to verify the feature-engineering
    contract before Satyam's real candidate generator is available.
    """

    # ---------------------------------------------------------
    # SOURCE 1
    # ---------------------------------------------------------

    source1 = pd.DataFrame([
        {
            "entity_id": "S1-001",
            "business_name": "ABC Electronics",
            "business_address": "MG Road Bangalore",
            "country": "India",
        },
        {
            "entity_id": "S1-002",
            "business_name": "राम इलेक्ट्रॉनिक्स",
            "business_address": "Delhi India",
            "country": "India",
        },
        {
            "entity_id": "S1-003",
            "business_name": "Paris Bakery",
            "business_address": "12 Rue Paris",
            "country": "France",
        },
        {
            "entity_id": "S1-004",
            "business_name": "No Match Company",
            "business_address": "Unknown Road",
            "country": "India",
        },
    ])

    # ---------------------------------------------------------
    # SOURCE 2
    # ---------------------------------------------------------

    source2 = pd.DataFrame([
        {
            "entity_id": "S2-001",
            "business_name": "ABC Electronics",
            "business_address": "MG Road Bangalore",
            "country": "India",
        },
        {
            "entity_id": "S2-002",
            "business_name": "ABC Electronic",
            "business_address": "MG Road Bengaluru",
            "country": "India",
        },
        {
            "entity_id": "S2-003",
            "business_name": "राम इलेक्ट्रॉनिक्स",
            "business_address": None,
            "country": "India",
        },
        {
            "entity_id": "S2-004",
            "business_name": "Completely Different",
            "business_address": "Different Road",
            "country": "India",
        },
    ])

    # ---------------------------------------------------------
    # SOURCE 3
    # ---------------------------------------------------------

    source3 = pd.DataFrame([
        {
            "entity_id": "S3-001",
            "business_name": "Paris Bakery",
            "business_address": "12 Rue Paris",
            "country": "France",
        },
        {
            "entity_id": "S3-002",
            "business_name": "Paris Bakerie",
            "business_address": None,
            "country": "France",
        },
    ])

    # ---------------------------------------------------------
    # MOCK CANDIDATE PAIRS
    #
    # This follows the interface expected from Satyam.
    # ---------------------------------------------------------

    candidate_pairs = pd.DataFrame([
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-001",
            "candidate_source": "S2",
            "blocker_sources": [
                "name_exact",
                "address_exact",
            ],
            "num_blockers": 2,
        },
        {
            "source1_entity_id": "S1-001",
            "candidate_entity_id": "S2-002",
            "candidate_source": "S2",
            "blocker_sources": [
                "name_block",
            ],
            "num_blockers": 1,
        },
        {
            "source1_entity_id": "S1-002",
            "candidate_entity_id": "S2-003",
            "candidate_source": "S2",
            "blocker_sources": [
                "name_exact",
            ],
            "num_blockers": 1,
        },
        {
            "source1_entity_id": "S1-003",
            "candidate_entity_id": "S3-001",
            "candidate_source": "S3",
            "blocker_sources": [
                "name_exact",
                "country_exact",
            ],
            "num_blockers": 2,
        },
        {
            "source1_entity_id": "S1-004",
            "candidate_entity_id": "S2-004",
            "candidate_source": "S2",
            "blocker_sources": [
                "country_block",
            ],
            "num_blockers": 1,
        },
    ])

    return (
        source1,
        source2,
        source3,
        candidate_pairs,
    )


def main():

    print("=" * 60)
    print("ATLAS FEATURE ENGINEERING TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # Build mock data
    # ---------------------------------------------------------

    (
        source1,
        source2,
        source3,
        candidate_pairs,
    ) = build_mock_data()

    print("\nMock data created.")
    print(f"Source 1 records : {len(source1)}")
    print(f"Source 2 records : {len(source2)}")
    print(f"Source 3 records : {len(source3)}")
    print(f"Candidate pairs  : {len(candidate_pairs)}")

    # ---------------------------------------------------------
    # Create feature engineer
    # ---------------------------------------------------------

    engineer = FeatureEngineer()

    # ---------------------------------------------------------
    # Generate features
    # ---------------------------------------------------------

    features = engineer.transform(
        source1=source1,
        source2=source2,
        source3=source3,
        candidate_pairs=candidate_pairs,
    )

    # ---------------------------------------------------------
    # Display output
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE OUTPUT")
    print("=" * 60)

    print(features.to_string(index=False))

    # ---------------------------------------------------------
    # Display feature schema
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODEL FEATURE COLUMNS")
    print("=" * 60)

    feature_columns = get_feature_columns()

    for index, column in enumerate(feature_columns, start=1):
        print(f"{index:02d}. {column}")

    # ---------------------------------------------------------
    # Basic shape validation
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SHAPE CHECK")
    print("=" * 60)

    print(f"Rows    : {features.shape[0]}")
    print(f"Columns : {features.shape[1]}")

    assert len(features) == len(candidate_pairs), (
        "Number of feature rows must equal number of candidate pairs."
    )

    # ---------------------------------------------------------
    # Check 1:
    # Exact S1-001 ↔ S2-001 pair
    # ---------------------------------------------------------

    exact = features.iloc[0]

    assert exact["name_exact"] == 1.0
    assert exact["address_exact"] == 1.0
    assert exact["country_exact"] == 1.0

    print("\n[PASS] Exact name/address/country features")

    # ---------------------------------------------------------
    # Check 2:
    # Missing address must NOT destroy name evidence
    # ---------------------------------------------------------

    missing_address = features.iloc[2]

    assert missing_address["address_available_both"] == 0.0
    assert missing_address["address_missing_source2"] == 1.0

    # Name is still an exact match.
    assert missing_address["name_exact"] == 1.0

    print("[PASS] Missing address handling")

    # ---------------------------------------------------------
    # Check 3:
    # France must work without hard-coded country logic
    # ---------------------------------------------------------

    france = features.iloc[3]

    assert france["country_exact"] == 1.0

    print("[PASS] Open-set country handling")

    # ---------------------------------------------------------
    # Check 4:
    # S3 candidate must be accepted
    # ---------------------------------------------------------

    assert france["candidate_source"] == "S3"

    print("[PASS] S3 candidate handling")

    # ---------------------------------------------------------
    # Check 5:
    # Blocker provenance
    # ---------------------------------------------------------

    first = features.iloc[0]

    assert first["was_name_block"] == 1.0
    assert first["was_address_block"] == 1.0
    assert first["multiple_independent_blockers"] == 1.0
    assert first["num_blockers"] == 2.0

    print("[PASS] Blocker provenance features")

    # ---------------------------------------------------------
    # Check 6:
    # Feature schema exists
    # ---------------------------------------------------------

    for column in feature_columns:
        assert column in features.columns, (
            f"Missing feature column: {column}"
        )

    print("[PASS] Feature schema")

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("ALL FEATURE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()