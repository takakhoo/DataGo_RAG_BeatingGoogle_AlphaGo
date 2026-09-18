"""Cosine retrieval with consistent scores and portable, non-executable storage.

FAISS is exact; HNSW is approximate; NumPy is the dependency-free exact reference.
All return cosine similarity (higher is better), including scaled queries.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import numpy as np
from src.memory.schema import MemoryEntry

HAS_FAISS = importlib.util.find_spec("faiss") is not None
HAS_HNSW = importlib.util.find_spec("hnswlib") is not None


class ANNIndex:
    def __init__(self, dim: int, space: str = "cosine", backend: str = "auto"):
        if not isinstance(dim, int) or dim < 1 or space != "cosine":
            raise ValueError("positive integer dim and cosine space required")
        if backend == "auto":
            backend = "faiss" if HAS_FAISS else "hnsw" if HAS_HNSW else "numpy"
        if backend not in ("numpy", "faiss", "hnsw"):
            raise ValueError("backend must be auto, numpy, faiss or hnsw")
        self.dim, self.space, self.backend = dim, space, backend
        self._entries, self._id_to_idx = [], {}
        self._emb_matrix = self._index = None

    def _vector(self, value):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (self.dim,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"expected a finite vector of shape ({self.dim},)")
        scale = np.max(np.abs(vector))
        if scale == 0:
            raise ValueError("zero vectors have no cosine direction")
        vector = vector / scale  # Avoid overflow with large but finite inputs.
        return (vector / np.linalg.norm(vector)).astype(np.float32)

    def add(self, entry: MemoryEntry):
        copied = deepcopy(entry)
        copied.embed = self._vector(entry.embed)
        if not isinstance(entry.id, str) or not entry.id:
            raise ValueError("entry ID must be nonempty text")
        if entry.id in self._id_to_idx:
            self._entries[self._id_to_idx[entry.id]] = copied
        else:
            self._id_to_idx[entry.id] = len(self._entries)
            self._entries.append(copied)
        self._emb_matrix = self._index = None  # Invalidate after insert AND upsert.

    def _build(self):
        self._emb_matrix = np.stack([e.embed for e in self._entries])
        if self.backend == "faiss":
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(self._emb_matrix)
        elif self.backend == "hnsw":
            import hnswlib
            self._index = hnswlib.Index(space="cosine", dim=self.dim)
            self._index.init_index(max_elements=len(self), ef_construction=200, M=16, random_seed=7)
            self._index.set_num_threads(1)
            self._index.add_items(self._emb_matrix, np.arange(len(self)))
            self._index.set_ef(max(50, min(len(self), 200)))

    def retrieve(self, query: np.ndarray, k: int = 10):
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        q = self._vector(query).reshape(1, -1)
        if not self._entries or k == 0:
            return []
        k = min(k, len(self))
        if self._emb_matrix is None:
            self._build()
        if self.backend == "faiss":
            scores, ids = self._index.search(q, k)
            scores, ids = scores[0], ids[0]
        elif self.backend == "hnsw":
            self._index.set_ef(max(50, k))
            ids, distances = self._index.knn_query(q, k=k, num_threads=1)
            scores, ids = 1 - distances[0], ids[0]  # hnsw returns DISTANCE.
        else:
            all_scores = self._emb_matrix @ q[0]
            ids = np.argsort(-all_scores, kind="stable")[:k]
            scores = all_scores[ids]
        results = [(deepcopy(self._entries[int(i)]), float(np.clip(s, -1, 1))) for i, s in zip(ids, scores)]
        return sorted(results, key=lambda item: (-item[1], item[0].id))

    def save(self, path: str):
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        records = []
        for entry in self._entries:
            record = asdict(entry)
            record["embed"] = entry.embed.tolist()
            records.append(record)
        payload = {"version": 1, "dim": self.dim, "space": self.space, "entries": records}
        (directory / "memory.json").write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n")

    def load(self, path: str, *, trusted_legacy=False):
        directory = Path(path)
        if (directory / "memory.json").exists():
            payload = json.loads((directory / "memory.json").read_text())
            if payload.get("version") != 1 or payload.get("dim") != self.dim or payload.get("space") != self.space:
                raise ValueError("incompatible memory schema/dimension/space")
            entries = [MemoryEntry(**record) for record in payload["entries"]]
        elif trusted_legacy:
            import pickle
            # Only migrate files you created yourself; pickle can execute code.
            with (directory / "meta.pkl").open("rb") as stream:
                entries = pickle.load(stream)
        else:
            raise ValueError("memory.json missing; legacy pickle requires explicit trusted_legacy=True")
        fresh = ANNIndex(self.dim, self.space, self.backend)
        for entry in entries:
            fresh.add(entry)
        self._entries, self._id_to_idx = fresh._entries, fresh._id_to_idx
        self._emb_matrix = self._index = None

    def __len__(self):
        return len(self._entries)
