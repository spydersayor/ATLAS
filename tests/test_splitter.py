from src.splitter import split_source1_entities


def test_split_is_disjoint():
    source1_ids = [f"S1-{i}" for i in range(100)]

    train_ids, validation_ids = split_source1_entities(
        source1_ids,
        validation_fraction=0.2,
        random_state=42,
    )

    assert train_ids.isdisjoint(validation_ids)


def test_split_contains_all_entities():
    source1_ids = [f"S1-{i}" for i in range(100)]

    train_ids, validation_ids = split_source1_entities(
        source1_ids,
        validation_fraction=0.2,
        random_state=42,
    )

    assert train_ids | validation_ids == set(source1_ids)


def test_split_is_deterministic():
    source1_ids = [f"S1-{i}" for i in range(100)]

    train_1, validation_1 = split_source1_entities(
        source1_ids,
        validation_fraction=0.2,
        random_state=42,
    )

    train_2, validation_2 = split_source1_entities(
        source1_ids,
        validation_fraction=0.2,
        random_state=42,
    )

    assert train_1 == train_2
    assert validation_1 == validation_2


def test_expected_validation_size():
    source1_ids = [f"S1-{i}" for i in range(100)]

    train_ids, validation_ids = split_source1_entities(
        source1_ids,
        validation_fraction=0.2,
        random_state=42,
    )

    assert len(validation_ids) == 20
    assert len(train_ids) == 80
    