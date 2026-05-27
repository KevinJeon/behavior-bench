from dataclasses import dataclass
import numpy as np


@dataclass
class FocalSelection:
    focal_indices: np.ndarray
    other_indices: np.ndarray


def select_ratio_focal_indices(agent_offsets, ratio, seed, min_per_env=1):
    rng = np.random.default_rng(seed)
    selected = []

    for cur, nxt in zip(agent_offsets[:-1], agent_offsets[1:]):
        count = nxt - cur
        if count <= 0:
            continue

        num_focal = max(min_per_env, int(round(count * ratio)))
        num_focal = min(num_focal, count)

        local = rng.choice(count, size=num_focal, replace=False)
        selected.extend(cur + local)

    focal = np.asarray(sorted(selected), dtype=np.int64)
    all_indices = np.arange(agent_offsets[-1], dtype=np.int64)
    other = np.setdiff1d(all_indices, focal, assume_unique=True)
    return FocalSelection(focal, other)