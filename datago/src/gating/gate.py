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
    p = p.astype(float)
    p = p / (p.sum() + 1e-12)
    p_pos = p[p > 0]
    return float(-(p_pos * np.log(p_pos)).sum())


def normalized_entropy(p: np.ndarray) -> float:
    # normalize by log(num_legal_moves)
    L = float(len(p))
    if L <= 1:
        return 0.0
    H = entropy_of_policy(p)
    H_max = math.log(L)
    return float(H / (H_max + 1e-12))


def should_trigger(p: np.ndarray, threshold: float = DEFAULTS["H_trigger"]) -> bool:
    return normalized_entropy(p) >= threshold


if __name__ == "__main__":
    import numpy as np
    p = np.ones(361) / 361.0
    print("norm ent", normalized_entropy(p))
