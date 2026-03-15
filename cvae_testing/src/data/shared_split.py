from __future__ import annotations

from typing import Dict, List


def split_groups(group_ids: List[str], split: Dict[str, float], rng) -> Dict[str, List[str]]:
    shuffled = list(group_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * split["train"])
    n_val = int(n * split["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def image_level_split_indices(n_items: int, split: Dict[str, float], rng) -> Dict[str, List[int]]:
    idxs = list(range(n_items))
    rng.shuffle(idxs)
    n_train = int(n_items * split["train"])
    n_val = int(n_items * split["val"])
    return {
        "train": idxs[:n_train],
        "val": idxs[n_train : n_train + n_val],
        "test": idxs[n_train + n_val :],
    }
