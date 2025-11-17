"""
katago_client.py

Utilities for talking to KataGo's JSON analysis engine and converting analysis results
into canonicalized embeddings suitable for ANN indexing.

This is a lightweight, non-invasive prototype: it expects to receive KataGo JSON analysis
responses (or files) and exposes embedding extraction and board canonicalization helpers.

TODOs:
- Wire to an actual KataGo process (subprocess / socket) if desired.
- Optionally provide a learned projection model to reduce dimensionality.
"""
from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import uuid
from src.utils import symmetry


@dataclass
class AnalysisResult:
    policy: np.ndarray  # shape (board_size,), raw or normalized probabilities
    value: float
    board_fingerprint: str  # canonical board representation (string)
    raw_json: Dict[str, Any]


def canonicalize_board(board_repr: str) -> str:
    """Canonicalize board representation (rotate/reflect) to a canonical orientation.

    For the prototype we accept a canonical string form from the client. In a fuller
    implementation this will rotate/reflect boards and pick a canonical form.
    """
    # Placeholder: assume already canonical
    return board_repr


def canonicalize_policy(policy: np.ndarray, board_x: int = 19, board_y: int = 19) -> Tuple[np.ndarray, int]:
    """Return a canonicalized policy vector and the symmetry used.

    This tries all symmetries (or only 4 if non-square) and picks the lexicographically smallest
    flattened representation as a deterministic canonical form.
    """
    p = policy.astype(float)
    size = board_x * board_y
    if p.size < size:
        # pad if necessary
        p = np.pad(p, (0, max(0, size - p.size)))

    best = None
    best_sym = 0
    sym_upper = 8 if board_x == board_y else 4
    for s in range(sym_upper):
        # build transformed vector
        p_s = np.zeros_like(p)
        for pos in range(size):
            nx, ny = symmetry.apply_symmetry_xy(pos % board_x, pos // board_x, board_x, board_y, s)
            npos = symmetry.xy_to_pos(nx, ny, board_x)
            p_s[npos] = p[pos]
        key = tuple(p_s.tolist())
        if best is None or key < best:
            best = key
            best_sym = s
            best_vec = p_s
    return np.array(best_vec, dtype=float), int(best_sym)


def parse_katago_analysis_json(j: Dict[str, Any]) -> AnalysisResult:
    """Parse a KataGo analysis JSON object into an AnalysisResult.

    Expected minimal JSON fields: 'policy' (list of floats or mapping), 'winrate' or 'value'
    This function should be adapted if your KataGo JSON format differs.
    """
    # Attempt different locations for policy/value depending on the JSON flavor
    policy = None
    value = 0.0

    # Common: a top-level 'moves' or 'policy' array
    if "policy" in j:
        policy = np.array(j["policy"], dtype=float)
    elif "moves" in j:
        # some variants include moves with probs
        moves = j["moves"]
        # create a dense vector later; caller must give board_size or mapping
        # For now, flatten probabilities in order provided
        policy = np.array([float(m.get("prob", 0.0)) for m in moves], dtype=float)
    else:
        # Try common analyze format: j['root']['policy']
        if "root" in j and "policy" in j["root"]:
            policy = np.array(j["root"]["policy"])

    # Value
    if "value" in j:
        value = float(j["value"])
    elif "winrate" in j:
        value = float(j["winrate"])
    elif "root" in j and "value" in j["root"]:
        value = float(j["root"]["value"]) 

    # Board fingerprint (if provided)
    board_fp = j.get("board_fingerprint", j.get("board", "unknown"))

    if policy is None:
        # Fallback: empty small vector
        policy = np.zeros(1, dtype=float)

    return AnalysisResult(policy=policy, value=value, board_fingerprint=board_fp, raw_json=j)


class EmbeddingProjector:
    """Simple embedding projector. Prototype uses a linear random projection or PCA if provided.

    For production, replace with learned MLP or extract internal CNN activations from KataGo.
    """

    def __init__(self, d_out: int = 128, seed: Optional[int] = 1234):
        self.d_out = d_out
        rng = np.random.RandomState(seed)
        # Lazy init: weights will be created when we know input dim
        self.weights: Optional[np.ndarray] = None

    def fit_dummy(self, input_dim: int):
        # For quick prototype, initialize a random projection
        self.weights = np.random.RandomState(0).normal(size=(input_dim, self.d_out)).astype(np.float32)

    def transform(self, policy: np.ndarray, value: Optional[float] = None) -> np.ndarray:
        x = policy.astype(np.float32)
        if value is not None:
            x = np.concatenate([x, np.array([value], dtype=np.float32)])
        if self.weights is None:
            self.fit_dummy(input_dim=x.shape[0])
        emb = x.dot(self.weights)
        # normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb


# Small convenience function
def analysis_to_embedding(j: Dict[str, Any], projector: Optional[EmbeddingProjector] = None) -> Tuple[np.ndarray, AnalysisResult]:
    ar = parse_katago_analysis_json(j)
    if projector is None:
        projector = EmbeddingProjector(d_out=128)
    emb = projector.transform(ar.policy, ar.value)
    return emb, ar


if __name__ == "__main__":
    # Quick smoke test: read a JSON file path from argv and print embedding shape
    import sys
    if len(sys.argv) < 2:
        print("Usage: katago_client.py analysis.json")
        sys.exit(1)
    with open(sys.argv[1], "r") as f:
        j = json.load(f)
    emb, ar = analysis_to_embedding(j)
    print("embedding shape:", emb.shape)
    print("board fp:", ar.board_fingerprint)
