"""datago

RAG-Augmented MCTS baseline package.

This package contains minimal stubs and entry points for building the
datago workspace described in the top-level README. The real implementations
should live in the subpackages (clients, embeddings, memory, gating, blend, eval).
"""
__version__ = "0.0.0"

from pathlib import Path

def data_dir() -> Path:
    """Return the absolute path to the `data/` directory inside the datago workspace.

    This helper assumes the runtime `sys.path` includes the `src/` directory
    that contains this package (the default layout created here).
    """
    return Path(__file__).resolve().parents[2] / "data"

__all__ = ["__version__", "data_dir"]
