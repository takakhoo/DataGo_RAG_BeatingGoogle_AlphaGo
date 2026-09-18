"""Offline component diagnostics, NOT Go matches or a KataGo strength benchmark."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent / "datago"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.memory.index import ANNIndex, HAS_FAISS, HAS_HNSW
from src.memory.schema import MemoryEntry
from src.mcts.custom_mcts import CustomMCTS


def toy_evaluator(position):
    return {"root": ({"trap": .9, "safe": .1}, 0),
            "root_trap": ({"punish": .8, "blunder": .2}, 0),
            "root_safe": ({"hold": 1}, 0),
            "root_trap_punish": ({}, -1),
            "root_trap_blunder": ({}, 1),
            "root_safe_hold": ({}, .3)}[position]


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(128, 32))
    query = vectors + rng.normal(scale=.2, size=vectors.shape)
    backends = ["numpy"] + (["faiss"] if HAS_FAISS else []) + (["hnsw"] if HAS_HNSW else [])
    retrieval = []
    for backend in backends:
        index = ANNIndex(32, backend=backend)
        for i, vector in enumerate(vectors):
            index.add(MemoryEntry(str(i), vector, "synthetic-vector", [], last_seen=0))
        baseline = [index.retrieve(q, 1)[0] for q in query]
        with tempfile.TemporaryDirectory() as directory:
            index.save(directory)
            index.load(directory)
            for gain in (.1, 1, 10):
                result = [index.retrieve(q * gain, 1)[0] for q in query]
                retrieval.append({"backend": backend, "query_gain": gain,
                                  "top1_recovery": float(np.mean([r[0].id == str(i) for i, r in enumerate(result)])),
                                  "max_score_change_after_save_and_gain": max(abs(a[1]-b[1]) for a, b in zip(result, baseline))})
    searches = []
    for simulations in (4, 16, 64, 256):
        for label, prior in [("misleading prior", None), ("helpful stored prior", {"trap": .1, "safe": .9})]:
            search = CustomMCTS(toy_evaluator, dirichlet_alpha=0)
            policy, value = search.search("root", simulations, prior)
            searches.append({"prior": label, "simulations": simulations, "safe_probability": policy["safe"], "root_value": value})
    report = {"seed": 7, "scope": "128 noisy synthetic retrieval queries; known six-state adversarial tree, not Go",
              "retrieval": retrieval, "search": searches,
              "caveat": "Helpful prior is hand-specified. Its acquisition cost is not modeled; this is not a fair-compute strength comparison."}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for backend in backends:
        rows = [r for r in retrieval if r["backend"] == backend]
        axes[0].plot([r["query_gain"] for r in rows], [r["top1_recovery"] for r in rows], "o-", label=backend)
    axes[0].set(xscale="log", ylim=(0, 1.05), xlabel="Query gain", ylabel="Correct stored-vector recovery", title="Cosine retrieval survives scaling / reload")
    for label in ("misleading prior", "helpful stored prior"):
        rows = [r for r in searches if r["prior"] == label]
        axes[1].plot([r["simulations"] for r in rows], [r["safe_probability"] for r in rows], "o-", label=label)
    axes[1].set(xscale="log", ylim=(0, 1.05), xlabel="Simulations", ylabel="Visits assigned to safe action", title="Search learns to reject an adversarial trap")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    fig.savefig(output / "diagnostics.png", dpi=170)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/core"))
    run(parser.parse_args().output)
