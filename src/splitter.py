from __future__ import annotations

from typing import Iterable, Set, Tuple

import numpy as np


def split_source1_entities(
    source1_ids: Iterable[str],
    validation_fraction: float = 0.2,
    random_state: int = 42,
) -> Tuple[Set[str], Set[str]]:
    """
    Split Source-1 entity IDs into training and validation sets.

    The split is:
    - entity-level
    - deterministic for the same random_state
    - disjoint
    - approximately validation_fraction in size
    """

    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be between 0 and 1."
        )

    # Remove duplicates and sort so the split is reproducible
    # across different Python processes.
    unique_ids = np.array(
        sorted(set(source1_ids)),
        dtype=object,
    )

    if len(unique_ids) < 2:
        raise ValueError(
            "At least two unique Source-1 entities are required."
        )

    rng = np.random.default_rng(random_state)
    rng.shuffle(unique_ids)

    validation_size = int(
        round(len(unique_ids) * validation_fraction)
    )

    validation_size = max(1, validation_size)
    validation_size = min(
        len(unique_ids) - 1,
        validation_size,
    )

    validation_ids = set(
        unique_ids[:validation_size]
    )

    train_ids = set(
        unique_ids[validation_size:]
    )

    return train_ids, validation_ids