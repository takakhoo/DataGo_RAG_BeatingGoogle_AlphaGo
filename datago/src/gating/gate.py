"""
gating/gate.py

Entropy gating utilities and configuration defaults.
"""
from __future__ import annotations

import math
from typing import Tuple
import numpy as np


DEFAULTS = {
    "H_trigger": 0.7,
    "H_store": 0.9,
}


def entropy_of_policy(p: np.ndarray, base: float = math.e) -> float:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not len(p) or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.any(p > 0):
        raise ValueError("policy must be nonempty, finite, nonnegative, with positive mass")
    if not math.isfinite(base) or base <= 0 or base == 1:
        raise ValueError("entropy base must be positive and not one")
    p = p / p.max()
    p = p / p.sum()
    p_pos = p[p > 0]
    return float(-(p_pos * np.log(p_pos)).sum() / math.log(base))


def normalized_entropy(p: np.ndarray) -> float:
    # normalize by log(num_legal_moves)
    H = entropy_of_policy(p)
    L = float(len(p))
    if L <= 1:
        return 0.0
    H_max = math.log(L)
    return float(np.clip(H / H_max, 0, 1))


def should_trigger(p: np.ndarray, threshold: float = DEFAULTS["H_trigger"]) -> bool:
    return normalized_entropy(p) >= threshold


if __name__ == "__main__":
    import numpy as np
    p = np.ones(361) / 361.0
    print("norm ent", normalized_entropy(p))
