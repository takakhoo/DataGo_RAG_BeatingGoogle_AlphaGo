"""
memory/index.py

A small ANN index wrapper that uses hnswlib if available, otherwise falls back to a brute-force numpy search.
This prototype stores metadata in a simple pickle alongside the index.
"""
from __future__ import annotations

import os
import pickle
from typing import List, Tuple, Optional
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    import hnswlib
    HAS_HNSW = True
except Exception:
    HAS_HNSW = False

from src.memory.schema import MemoryEntry


class ANNIndex:
    def __init__(self, dim: int, space: str = "cosine"):
        self.dim = dim
        self.space = space
        self._entries: List[MemoryEntry] = []
        self._emb_matrix: Optional[np.ndarray] = None
        self._id_to_idx = {}
        # Prefer FAISS (fast) then HNSWLIB, otherwise fallback to brute-force
        self._p = None
        self._inited = False
        if HAS_FAISS:
            # FAISS will be created lazily when saving/loading or explicitly requested
            self._p = None
        elif HAS_HNSW:
            self._p = hnswlib.Index(space=space, dim=dim)
            self._inited = False

    def _rebuild_matrix(self):
        if len(self._entries) == 0:
            self._emb_matrix = None
            return
        self._emb_matrix = np.vstack([e.embed for e in self._entries]).astype(np.float32)

    def add(self, entry: MemoryEntry):
        idx = len(self._entries)
        self._entries.append(entry)
        self._id_to_idx[entry.id] = idx
        # Add to underlying index if available, otherwise rebuild emb matrix
        if HAS_FAISS:
            # FAISS: defer creating the index until save/load or explicit build
            self._rebuild_matrix()
        elif HAS_HNSW:
            if not self._inited:
                # initialize with an estimated capacity (we'll increase on save/load if needed)
                self._p.init_index(max_elements=10000, ef_construction=200, M=16)
                self._inited = True
                self._p.add_items(entry.embed.reshape(1, -1), ids=[idx])
            else:
                self._p.add_items(entry.embed.reshape(1, -1), ids=[idx])
        else:
            # fallback: rebuild emb matrix
            self._rebuild_matrix()

    def retrieve(self, query: np.ndarray, k: int = 10) -> List[Tuple[MemoryEntry, float]]:
        # Prefer FAISS search
        q = query.astype(np.float32).reshape(1, -1)
        if HAS_FAISS:
            try:
                # If we have a faiss index built in memory, use it
                if hasattr(self, "_faiss_index") and self._faiss_index is not None:
                    D, I = self._faiss_index.search(q, k)
                    results = []
                    for i, d in zip(I[0], D[0]):
                        if i < 0 or i >= len(self._entries):
                            continue
                        # faiss returns inner product or distance depending on index; treat d as score
                        results.append((self._entries[int(i)], float(d)))
                    return results
            except Exception:
                pass
        if HAS_HNSW and self._inited:
            labels, distances = self._p.knn_query(q, k=k)
            results = []
            for lbl, dist in zip(labels[0], distances[0]):
                if lbl < 0 or lbl >= len(self._entries):
                    continue
                results.append((self._entries[int(lbl)], float(dist)))
            return results

        # brute-force fallback using cosine similarity
        if self._emb_matrix is None:
            return []
        qv = q.reshape(-1)
        dots = self._emb_matrix.dot(qv)
        norms = np.linalg.norm(self._emb_matrix, axis=1) * (np.linalg.norm(qv) + 1e-12)
        sims = dots / (norms + 1e-12)
        idxs = np.argsort(-sims)[:k]
        return [(self._entries[int(i)], float(sims[int(i)])) for i in idxs]

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        # Save metadata
        meta_path = os.path.join(path, "meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(self._entries, f)
        # For hnsw, save index file
        if HAS_HNSW and self._inited:
            self._p.save_index(os.path.join(path, "hnsw_index.bin"))
        if HAS_FAISS:
            # Build a faiss IndexFlatIP over normalized vectors for inner-product (cosine for normalized)
            try:
                import faiss
                if self._emb_matrix is not None:
                    # ensure float32
                    xb = self._emb_matrix.astype(np.float32)
                    # normalize rows
                    norms = np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
                    xb = xb / norms
                    index = faiss.IndexFlatIP(self.dim)
                    index.add(xb)
                    faiss.write_index(index, os.path.join(path, "faiss_index.bin"))
                    # store in-memory index reference
                    self._faiss_index = index
            except Exception:
                pass
        else:
            # Save embeddings numpy
            if self._emb_matrix is not None:
                np.save(os.path.join(path, "embeddings.npy"), self._emb_matrix)

    def load(self, path: str):
        meta_path = os.path.join(path, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self._entries = pickle.load(f)
            self._id_to_idx = {e.id: i for i, e in enumerate(self._entries)}
            self._rebuild_matrix()
        if HAS_HNSW:
            idx_path = os.path.join(path, "hnsw_index.bin")
            if os.path.exists(idx_path):
                try:
                    self._p.load_index(idx_path)
                    self._inited = True
                except Exception:
                    pass
        if HAS_FAISS:
            try:
                import faiss
                fi = os.path.join(path, "faiss_index.bin")
                if os.path.exists(fi):
                    self._faiss_index = faiss.read_index(fi)
                else:
                    # build index from embeddings.npy if present
                    embf = os.path.join(path, "embeddings.npy")
                    if os.path.exists(embf):
                        xb = np.load(embf).astype(np.float32)
                        norms = np.linalg.norm(xb, axis=1, keepdims=True) + 1e-12
                        xb = xb / norms
                        index = faiss.IndexFlatIP(self.dim)
                        index.add(xb)
                        self._faiss_index = index
            except Exception:
                self._faiss_index = None

    def __len__(self):
        return len(self._entries)


if __name__ == "__main__":
    # smoke test
    import numpy as np
    idx = ANNIndex(dim=16)
    from src.memory.schema import MemoryEntry
    for i in range(10):
        e = MemoryEntry.create(embed=np.random.randn(16).astype(np.float32), canonical_board=f"b{i}", best_moves=[])
        idx.add(e)
    q = np.random.randn(16).astype(np.float32)
    res = idx.retrieve(q, k=5)
    print("retrieved", len(res))
