from pathlib import Path

from src.evaluator import (
    parse_ground_truth,
    get_true_matches,
    is_true_match,
)


def create_sample_ground_truth(tmp_path: Path) -> Path:
    file_path = tmp_path / "ground_truth.tsv"

    file_path.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-A\tS2-1,S3-7\n"
        "S1-B\tS2-2\n"
        "S1-C\t\n"
        "S1-D\tS2-5,S2-5,S3-9\n",
        encoding="utf-8",
    )

    return file_path


def test_parse_ground_truth(tmp_path):
    file_path = create_sample_ground_truth(tmp_path)

    ground_truth = parse_ground_truth(file_path)

    assert ground_truth["S1-A"] == {"S2-1", "S3-7"}
    assert ground_truth["S1-B"] == {"S2-2"}
    assert ground_truth["S1-C"] == set()
    assert ground_truth["S1-D"] == {"S2-5", "S3-9"}


def test_get_true_matches(tmp_path):
    file_path = create_sample_ground_truth(tmp_path)

    ground_truth = parse_ground_truth(file_path)

    assert get_true_matches(ground_truth, "S1-A") == {"S2-1", "S3-7"}
    assert get_true_matches(ground_truth, "S1-C") == set()


def test_unknown_source1_returns_empty_set(tmp_path):
    file_path = create_sample_ground_truth(tmp_path)

    ground_truth = parse_ground_truth(file_path)

    assert get_true_matches(ground_truth, "S1-UNKNOWN") == set()


def test_is_true_match(tmp_path):
    file_path = create_sample_ground_truth(tmp_path)

    ground_truth = parse_ground_truth(file_path)

    assert is_true_match(ground_truth, "S1-A", "S2-1")
    assert is_true_match(ground_truth, "S1-A", "S3-7")

    assert not is_true_match(ground_truth, "S1-A", "S2-999")
    assert not is_true_match(ground_truth, "S1-C", "S2-1")
    
    