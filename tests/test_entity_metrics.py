from src.evaluator import calculate_entity_metrics


def test_entity_metrics_exact_match():
    true_matches = {"S2-10", "S3-20"}
    predicted_matches = {"S2-10", "S3-20"}

    result = calculate_entity_metrics(true_matches, predicted_matches)

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f0.5"] == 1.0


def test_entity_metrics_partial_multi_match():
    true_matches = {"S2-10", "S3-20"}
    predicted_matches = {"S2-10"}

    result = calculate_entity_metrics(true_matches, predicted_matches)

    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5


def test_entity_metrics_false_positive():
    true_matches = {"S2-10"}
    predicted_matches = {"S2-10", "S2-99"}

    result = calculate_entity_metrics(true_matches, predicted_matches)

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0


def test_entity_metrics_no_match_correctly_predicted():
    true_matches = set()
    predicted_matches = set()

    result = calculate_entity_metrics(true_matches, predicted_matches)

    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f0.5"] == 1.0


def test_entity_metrics_no_match_with_false_prediction():
    true_matches = set()
    predicted_matches = {"S2-99"}

    result = calculate_entity_metrics(true_matches, predicted_matches)

    assert result["tp"] == 0
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f0.5"] == 0.0


def test_entity_metrics_all_true_matches_missed():
    true_matches = {"S2-10", "S3-20"}
    predicted_matches = set()

    result = calculate_entity_metrics(true_matches, predicted_matches)

    assert result["tp"] == 0
    assert result["fp"] == 0
    assert result["fn"] == 2
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f0.5"] == 0.0


def test_entity_metrics_multiple_matches_are_not_forced_to_top_one():
    true_matches = {"S2-10", "S2-11", "S3-20"}
    predicted_matches = {"S2-10", "S2-11", "S3-20"}

    result = calculate_entity_metrics(true_matches, predicted_matches)

    assert result["tp"] == 3
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f0.5"] == 1.0