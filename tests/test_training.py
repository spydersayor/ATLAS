import numpy as np
import pandas as pd
import pytest

from src.features import get_feature_columns
from src.training import MatcherTrainer


def _make_source_tables():
    source1 = pd.DataFrame(
        [
            ["S1-1", "Alpha Cafe", "12 Main Street", "IN"],
            ["S1-2", "Beta Store", "24 Park Road", "IN"],
            ["S1-3", "Gamma Foods", "8 Lake Avenue", "IN"],
            ["S1-4", "Delta Market", "91 Hill Road", "IN"],
            ["S1-5", "Epsilon Shop", "17 MG Road", "IN"],
            ["S1-6", "Zeta Mart", "33 Ring Road", "IN"],
            ["S1-7", "Eta Bakery", "44 Church Street", "IN"],
            ["S1-8", "Theta Traders", "55 Main Road", "IN"],
            ["S1-9", "Iota Cafe", "66 Park Street", "IN"],
            ["S1-10", "Kappa Foods", "77 Lake Road", "IN"],
        ],
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    source2 = pd.DataFrame(
        [
            ["S2-1", "Alpha Cafe", "12 Main Street", "IN"],
            ["S2-2", "Beta Store", "24 Park Road", "IN"],
            ["S2-3", "Gamma Foods", "8 Lake Avenue", "IN"],
            ["S2-4", "Delta Market", "91 Hill Road", "IN"],
            ["S2-5", "Epsilon Shop", "17 MG Road", "IN"],
            ["S2-6", "Zeta Mart", "33 Ring Road", "IN"],
            ["S2-7", "Eta Bakery", "44 Church Street", "IN"],
            ["S2-8", "Theta Traders", "55 Main Road", "IN"],
            ["S2-9", "Iota Cafe", "66 Park Street", "IN"],
            ["S2-10", "Kappa Foods", "77 Lake Road", "IN"],
        ],
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    source3 = pd.DataFrame(
        [
            ["S3-1", "Alpha Cafe", "12 Main Street", "IN"],
            ["S3-2", "Beta Store", "24 Park Road", "IN"],
            ["S3-3", "Gamma Foods", "8 Lake Avenue", "IN"],
            ["S3-4", "Delta Market", "91 Hill Road", "IN"],
            ["S3-5", "Epsilon Shop", "17 MG Road", "IN"],
            ["S3-6", "Zeta Mart", "33 Ring Road", "IN"],
            ["S3-7", "Eta Bakery", "44 Church Street", "IN"],
            ["S3-8", "Theta Traders", "55 Main Road", "IN"],
            ["S3-9", "Iota Cafe", "66 Park Street", "IN"],
            ["S3-10", "Kappa Foods", "77 Lake Road", "IN"],
        ],
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    return source1, source2, source3


def _make_candidate_pairs():
    rows = []

    for i in range(1, 11):
        rows.append(
            [
                f"S1-{i}",
                f"S2-{i}",
                "S2",
                ["name_exact", "address_exact"],
                2,
            ]
        )

        wrong_id = (i % 10) + 1

        rows.append(
            [
                f"S1-{i}",
                f"S2-{wrong_id}",
                "S2",
                ["name_token"],
                1,
            ]
        )

        rows.append(
            [
                f"S1-{i}",
                f"S3-{i}",
                "S3",
                ["name_exact", "address_exact"],
                2,
            ]
        )

        wrong_s3_id = ((i + 1) % 10) + 1

        rows.append(
            [
                f"S1-{i}",
                f"S3-{wrong_s3_id}",
                "S3",
                ["name_token"],
                1,
            ]
        )

    return pd.DataFrame(
        rows,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "blocker_sources",
            "num_blockers",
        ],
    )


def _make_ground_truth():
    return {
        f"S1-{i}": {
            f"S2-{i}",
            f"S3-{i}",
        }
        for i in range(1, 11)
    }


def test_training_pipeline_produces_model_and_threshold():
    source1, source2, source3 = _make_source_tables()
    candidate_pairs = _make_candidate_pairs()
    ground_truth = _make_ground_truth()

    trainer = MatcherTrainer(
        validation_fraction=0.3,
        random_state=42,
    )

    result = trainer.train(
        source1=source1,
        source2=source2,
        source3=source3,
        candidate_pairs=candidate_pairs,
        ground_truth=ground_truth,
    )

    assert result.matcher is trainer.matcher

    assert result.train_rows > 0
    assert result.validation_rows > 0

    assert result.train_source1_entities > 0
    assert result.validation_source1_entities > 0

    assert (
        result.train_source1_entities
        + result.validation_source1_entities
        == 10
    )

    assert result.train_positive_rows > 0
    assert result.train_negative_rows > 0

    assert result.validation_positive_rows > 0
    assert result.validation_negative_rows > 0

    assert len(result.validation_scores) == result.validation_rows
    assert len(result.validation_labels) == result.validation_rows

    assert np.all(np.isfinite(result.validation_scores))
    assert np.all(
        (result.validation_scores >= 0)
        & (result.validation_scores <= 1)
    )

    assert set(np.unique(result.validation_labels)) == {0, 1}

    assert 0.0 <= result.threshold_result.threshold <= 1.0
    assert 0.0 <= result.threshold_result.precision <= 1.0
    assert 0.0 <= result.threshold_result.recall <= 1.0
    assert 0.0 <= result.threshold_result.f05 <= 1.0


def test_training_split_is_entity_disjoint():
    source1, source2, source3 = _make_source_tables()
    candidate_pairs = _make_candidate_pairs()
    ground_truth = _make_ground_truth()

    trainer = MatcherTrainer(
        validation_fraction=0.3,
        random_state=42,
    )

    result = trainer.train(
        source1=source1,
        source2=source2,
        source3=source3,
        candidate_pairs=candidate_pairs,
        ground_truth=ground_truth,
    )

    assert (
        result.train_source1_entities
        + result.validation_source1_entities
        == source1["entity_id"].nunique()
    )


def test_training_requires_both_classes_in_training_data():
    source1, source2, source3 = _make_source_tables()

    candidate_pairs = pd.DataFrame(
        [
            [
                "S1-1",
                "S2-1",
                "S2",
                ["name_exact"],
                1,
            ],
            [
                "S1-2",
                "S2-2",
                "S2",
                ["name_exact"],
                1,
            ],
        ],
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "blocker_sources",
            "num_blockers",
        ],
    )

    ground_truth = {
        "S1-1": {"S2-1"},
        "S1-2": {"S2-2"},
    }

    trainer = MatcherTrainer(
        validation_fraction=0.5,
        random_state=42,
    )

    with pytest.raises(ValueError, match="split must contain both"):
        trainer.train(
            source1=source1,
            source2=source2,
            source3=source3,
            candidate_pairs=candidate_pairs,
            ground_truth=ground_truth,
        )


def test_training_rejects_invalid_candidate_contract():
    source1, source2, source3 = _make_source_tables()

    candidate_pairs = pd.DataFrame(
        {
            "source1_entity_id": ["S1-1"],
            "candidate_entity_id": ["S2-1"],
        }
    )

    trainer = MatcherTrainer()

    with pytest.raises(
        ValueError,
        match="Candidate-pair contract violation",
    ):
        trainer.train(
            source1=source1,
            source2=source2,
            source3=source3,
            candidate_pairs=candidate_pairs,
            ground_truth={"S1-1": {"S2-1"}},
        )


def test_training_uses_exact_model_feature_schema():
    assert len(get_feature_columns()) == 25
    assert "source1_entity_id" not in get_feature_columns()
    assert "candidate_entity_id" not in get_feature_columns()
    assert "candidate_source" not in get_feature_columns()