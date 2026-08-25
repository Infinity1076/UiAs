"""Public data contract and group-disjoint fold generation.

Participant indices, filename mappings, and raw-data parsers are excluded.
Run ``preprocess_data.py`` first, then use a private adapter to load the
processed recordings. The adapter must provide unique ``group_ids`` and two
methods: ``dataset_for_groups(group_ids)`` and
``stage1_dataset_for_groups(group_ids)``. Dataset items must follow
``REQUIRED_BATCH_KEYS`` and ``EXPECTED_SIGNAL_SHAPES``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import torch


REQUIRED_BATCH_KEYS = (
    "input_2d",
    "input_ra",
    "acoustic",
    "label",
    "condition_id",
    "group_id",
)
EXPECTED_SIGNAL_SHAPES = {
    "input_2d": (128, 128),
    "input_ra": (50, 128),
    "acoustic": (401,),
}


def validate_batch(batch: Mapping[str, object]) -> None:
    """Validate the tensor keys and trailing dimensions used by UIAS."""

    missing = set(REQUIRED_BATCH_KEYS) - set(batch)
    if missing:
        raise ValueError(f"dataset adapter omitted batch keys: {sorted(missing)}")

    for key, expected in EXPECTED_SIGNAL_SHAPES.items():
        values = batch[key]
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{key} must be a torch.Tensor")
        if tuple(values.shape[-len(expected) :]) != expected:
            raise ValueError(
                f"{key} trailing shape must be {expected}, got {tuple(values.shape)}"
            )


@dataclass(frozen=True)
class Fold:
    index: int
    train_groups: Sequence[str]
    test_groups: Sequence[str]


def make_group_disjoint_folds(
    group_ids: Sequence[str],
    *,
    folds: int,
    test_groups_per_fold: int,
    seed: int,
) -> list[Fold]:
    """Create deterministic folds with no group overlap."""

    groups = tuple(dict.fromkeys(group_ids))
    if len(groups) != len(group_ids):
        raise ValueError("group_ids must be unique")
    if not 1 <= test_groups_per_fold < len(groups):
        raise ValueError("invalid number of held-out groups")

    candidates = list(combinations(groups, test_groups_per_fold))
    if folds > len(candidates):
        raise ValueError("requested more folds than unique held-out group sets")

    rng = np.random.default_rng(seed)
    selected = rng.permutation(len(candidates))[:folds]
    result: list[Fold] = []
    for index, candidate_index in enumerate(selected):
        test = tuple(candidates[int(candidate_index)])
        train = tuple(group for group in groups if group not in test)
        if set(train) & set(test):
            raise AssertionError("train/test group leakage")
        result.append(Fold(index=index, train_groups=train, test_groups=test))
    return result
