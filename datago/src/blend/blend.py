"""
blend/blend.py

Reranking, prior construction, and blending utilities.
"""
from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
from src.memory.schema import MemoryEntry
from src.utils import symmetry
from src.memory.reachability import estimate_reachability


def rerank_neighbors(neighbors: List[Tuple[MemoryEntry, float]], current_policy: np.ndarray, alpha: float = 0.7, gamma: float = 0.3) -> List[Tuple[MemoryEntry, float]]:
    """Compute combined weights w = alpha * r + gamma * s for each neighbor, where
    r = reachability estimate (policy similarity by default) and s = structural boost (parent/child).

    This prototype uses a simple heuristic: if neighbor.best_moves contains a move with prob > 0.5,
    we don't treat that as parent/child; structural detection requires full move-history checks.
    """
    weights = []
    for entry, raw_score in neighbors:
        # r: use policy similarity between stored best move distribution and current policy
        # For simplicity, construct a candidate_policy from entry.best_moves mapped to positions.
        # Here we use raw_score as a proxy for similarity when embed->sim provided.
        try:
            # raw_score may be diag: similarity or distance; treat as similarity if in [-1,1] or [0,1]
            r = float(raw_score)
            # clamp
            r = max(0.0, min(1.0, r))
        except Exception:
            r = 0.0
        # placeholder structural boost s
        s = 1.0
        # If metadata contains a parent_child hint, boost
        if entry.metadata.get("relation") in ("parent", "child"):
            s = entry.metadata.get("structural_boost", 2.0)
        w = alpha * r + gamma * s
        weights.append(w)
    # normalize weights
    arr = np.array(weights, dtype=float)
    if arr.sum() <= 0:
        arr = np.ones_like(arr)
    arr = arr / arr.sum()
    reranked = [(neighbors[i][0], float(arr[i])) for i in range(len(neighbors))]
    return reranked


def build_retrieval_prior(reranked_neighbors: List[Tuple[MemoryEntry, float]], current_symmetry: int = 0, board_x: int = 19, board_y: int = 19, top_n: int = 16) -> Dict[str, float]:
    """Construct P_nn(a) as a dict mapping move (string) -> probability mass.

    This prototype expects neighbor.best_moves to be a list of {"move":str, "prob":float} in the canonical orientation.
    A full implementation must align moves back to the current orientation using stored symmetry metadata.
    """
    accum: Dict[str, float] = {}
    for entry, weight in reranked_neighbors[:top_n]:
        for mv in entry.best_moves:
            move = mv.get("move")
            prob = float(mv.get("prob", 0.0))
            # map move from the entry's canonical symmetry into the current symmetry
            mapped_move = move
            try:
                # support integer-index moves (proto format)
                if isinstance(move, str) and move.isdigit():
                    pos = int(move)
                    from_sym = int(entry.metadata.get("symmetry", 0))
                    b_x = int(entry.metadata.get("board_x", board_x))
                    b_y = int(entry.metadata.get("board_y", board_y))
                    mapped_pos = symmetry.map_pos_between_symmetries(pos, from_sym, current_symmetry, board_x=b_x, board_y=b_y)
                    mapped_move = str(int(mapped_pos))
                # else: leave as-is (e.g., PASS or coordinate strings) — extend parsing as needed
            except Exception:
                # if remapping fails, fall back to raw move string
                mapped_move = move
            accum[mapped_move] = accum.get(mapped_move, 0.0) + weight * prob
    # normalize
    total = sum(accum.values())
    if total <= 0:
        return {}
    for k in list(accum.keys()):
        accum[k] = accum[k] / total
    return accum


def blend_priors(p_net: Dict[str, float], p_nn: Dict[str, float], beta: float = 0.4, top_n: int = 16) -> Dict[str, float]:
    """Blend network prior p_net with retrieval prior p_nn.

    p_net and p_nn are dicts move->prob. Return normalized blended dict.
    """
    out = {}
    # take union of moves (optionally limit to top_n by p_net)
    net_items = sorted(p_net.items(), key=lambda x: -x[1])[:top_n]
    keys = set([k for k, _ in net_items]) | set(p_nn.keys())
    for k in keys:
        pn = p_nn.get(k, 0.0)
        pnet = p_net.get(k, 0.0)
        out[k] = (1.0 - beta) * pnet + beta * pn
    # normalize
    s = sum(out.values())
    if s > 0:
        for k in out:
            out[k] = out[k] / s
    return out


if __name__ == "__main__":
    import numpy as np
    # smoke test
    class DummyEntry:
        def __init__(self, id):
            self.id = id
            self.best_moves = [{"move": "D4", "prob": 0.5}, {"move": "Q16", "prob": 0.3}]
            self.metadata = {}
    neighbors = [(DummyEntry(i), 0.9) for i in range(4)]
    rer = rerank_neighbors(neighbors, np.ones(362))
    prior = build_retrieval_prior(rer)
    print(prior)
    print(blend_priors({"D4":0.2, "Q16":0.1}, prior))
