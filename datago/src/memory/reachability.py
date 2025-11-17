"""
memory/reachability.py

Prototype reachability estimators. Start with cheap heuristics (policy-similarity, parent/child check)
and optionally add short policy-guided rollouts or a learned reachability model.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float)
    b = b.astype(float)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def estimate_reachability(current_policy: np.ndarray, candidate_policy: np.ndarray, method: str = "policy_cosine") -> float:
    """Estimate reachability r in [0,1] between current and candidate using chosen method.

    Methods:
      - policy_cosine: use cosine similarity between policy vectors
      - parent_child: reserved for exact parent/child detection (external check)
      - hybrid: combine heuristics

    Note: this prototype uses only policy cosine and returns a clipped [0,1] value.
    """
    if method == "policy_cosine":
        sim = cosine_similarity(current_policy, candidate_policy)
        # map cosine [-1,1] to [0,1]
        r = max(0.0, (sim + 1.0) / 2.0)
        return r
    elif method == "hybrid":
        sim = cosine_similarity(current_policy, candidate_policy)
        r = max(0.0, (sim + 1.0) / 2.0)
        # small non-linear boost
        return r ** 0.9
    else:
        # fallback
        return 0.0


if __name__ == "__main__":
    import numpy as np
    a = np.random.randn(362)
    b = np.random.randn(362)
    print(estimate_reachability(a, b))
