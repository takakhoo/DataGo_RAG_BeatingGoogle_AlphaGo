from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datago"))
import numpy as np
import pytest
from src.memory.index import ANNIndex, HAS_FAISS, HAS_HNSW
from src.memory.schema import MemoryEntry
from src.gating.gate import entropy_of_policy, normalized_entropy
from src.mcts.custom_mcts import CustomMCTS, MCTSNode
from src.mcts.network_evaluator import KataGoNetworkEvaluator
from src.utils.symmetry import apply_symmetry_xy, compose_symmetry, invert_symmetry

BACKENDS = ["numpy"] + (["faiss"] if HAS_FAISS else []) + (["hnsw"] if HAS_HNSW else [])

def entry(id, vector):
    return MemoryEntry(str(id), np.array(vector), "fixture", [], last_seen=0)

@pytest.mark.parametrize("backend", BACKENDS)
def test_scaled_queries_updates_and_roundtrip(tmp_path, backend):
    index = ANNIndex(3, backend=backend)
    index.add(entry("a", [1, 0, 0]))
    index.add(entry("b", [0, 1, 0]))
    assert index.retrieve(np.array([10, 0, 0]), k=30)[0][1] == pytest.approx(1)
    index.save(tmp_path)
    index.add(entry("c", [0, 0, 1]))
    assert index.retrieve(np.array([0, 0, 1]))[0][0].id == "c"
    index.add(entry("a", [0, 0, -1]))
    assert len(index) == 3
    assert index.retrieve(np.array([0, 0, -10]))[0][0].id == "a"
    index.load(tmp_path)
    assert len(index) == 2
    assert index.retrieve(np.array([0, 10, 0]))[0][0].id == "b"
    assert index.retrieve(np.array([1, 0, 0]), k=0) == []

def test_invalid_vectors_and_no_pickle_default(tmp_path):
    index = ANNIndex(2, backend="numpy")
    for vector in ([0, 0], [1, float("nan")], [1], [1, float("inf")]):
        with pytest.raises(ValueError):
            index.add(entry("bad", vector))
    with pytest.raises(ValueError):
        index.load(tmp_path)
    assert len(index) == 0
    index.add(entry("large", [1e300, 1e300]))
    assert index.retrieve(np.array([1e-300, 1e-300]))[0][1] == pytest.approx(1)

def test_entropy_contract():
    assert entropy_of_policy([1, 1], base=2) == pytest.approx(1)
    assert normalized_entropy([1, 1, 1]) == pytest.approx(1)
    assert normalized_entropy([10, 0, 0]) == 0
    for policy in ([], [0], [-1, 2], [float("nan"), 1]):
        with pytest.raises(ValueError):
            normalized_entropy(policy)

def test_all_square_symmetry_compositions():
    for first in range(8):
        for second in range(8):
            for x in range(5):
                for y in range(5):
                    p = apply_symmetry_xy(x, y, 5, 5, first)
                    assert apply_symmetry_xy(*p, 5, 5, second) == apply_symmetry_xy(x, y, 5, 5, compose_symmetry(first, second))
                    assert apply_symmetry_xy(*p, 5, 5, invert_symmetry(first)) == (x, y)

def toy_evaluator(position):
    return {"root": ({"trap": .9, "safe": .1}, 0),
            "root_trap": ({"punish": .8, "blunder": .2}, 0),
            "root_safe": ({"hold": 1}, 0),
            "root_trap_punish": ({}, -1),
            "root_trap_blunder": ({}, 1),
            "root_safe_hold": ({}, .3)}[position]

def test_search_uses_opponent_perspective_and_terminal_states():
    search = CustomMCTS(toy_evaluator, dirichlet_alpha=0)
    policy, value = search.search("root", 512)
    assert policy["safe"] > .9
    assert value > 0
    assert sum(policy.values()) == pytest.approx(1)
    terminal, value = search.search("root_trap_punish", 4)
    assert terminal == {} and value == -1

def test_temperature_ties_and_small_temperature():
    root = MCTSNode("root", children={k: MCTSNode(k, visit_count=5) for k in ("a", "b")})
    for temperature in (0, 1e-5):
        search = CustomMCTS(toy_evaluator, temperature=temperature)
        p = search._get_move_probabilities(root)
        assert sum(p.values()) == pytest.approx(1)
        assert all(np.isfinite(list(p.values())))

def test_missing_katago_state_is_not_fake_pass():
    evaluator = KataGoNetworkEvaluator.__new__(KataGoNetworkEvaluator)
    evaluator.eval_cache = {}
    with pytest.raises(KeyError, match="Go state"):
        evaluator("nonexistent")

def test_illegal_prior_rejected():
    with pytest.raises(ValueError):
        CustomMCTS(toy_evaluator).search("root", 10, {"illegal": 1})

def test_katago_value_conversion_without_subprocess():
    from io import StringIO
    from types import SimpleNamespace
    evaluator = KataGoNetworkEvaluator.__new__(KataGoNetworkEvaluator)
    evaluator.board_size = 9
    evaluator.eval_cache = {}
    evaluator.process = SimpleNamespace(stdin=StringIO(), stdout=StringIO(
        '{"id":"fixture", "rootInfo":{"winrate":0.8}, "moveInfos":[{"move":"D4","prior":1}]}\n'))
    policy, value = evaluator.evaluate("fixture", np.zeros((9, 9)), [], use_cache=False)
    assert policy == {"D4": 1}
    assert value == pytest.approx(.6)
