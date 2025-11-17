"""
memory/schema.py

Dataclass definitions for memory entries used by the ANN index.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List
import numpy as np
import time
import uuid


@dataclass
class MemoryEntry:
    id: str
    embed: np.ndarray
    canonical_board: str
    best_moves: List[Dict[str, Any]]  # e.g., [{"move":"D4", "prob":0.45}, ...]
    visits: int = 0
    last_seen: float = field(default_factory=lambda: time.time())
    importance: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, embed: np.ndarray, canonical_board: str, best_moves: List[Dict[str, Any]], **kwargs):
        return cls(id=str(uuid.uuid4()), embed=embed, canonical_board=canonical_board, best_moves=best_moves, **kwargs)


__all__ = ["MemoryEntry"]
