# tuning/common/game_runner.py

"""
Shared utilities for Phase 2 tuning.

The GameRunner currently supports an offline-json mode that consumes the JSON
artifacts produced by Phase 1 (shallow vs deep comparisons + deep analysis
outputs) and synthesises aggregated metrics that mimic what the live self-play
runner will eventually report.  A live mode skeleton is also provided so that
we can plug in the real KataGo / RAG loop later without rewriting Phase 2.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

DEFAULT_SIMILARITY_WEIGHTS: Dict[str, float] = {
    "policy": 0.40,
    "winrate": 0.25,
    "score_lead": 0.10,
    "visit_distribution": 0.15,
    "stone_count": 0.05,
    "komi": 0.05,
}


@dataclass
class GameMetrics:
    """Aggregated metrics for one synthetic or real game."""

    win: bool
    game_length: int
    avg_move_time_ms: float

    deep_search_count: int
    deep_converged_count: int
    avg_deep_time_ms: float

    avg_value_error: float
    avg_policy_error: float

    rag_hits: int
    rag_misses: int
    new_positions_stored: int

    max_recursion_depth_used: int

    extra: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GameRunner:
    """
    Lightweight facade over either:

    * Offline JSON-driven metrics (`mode="offline_json"`)
    * The future live KataGo + RAG integration (`mode="live"`)
    """

    def __init__(
        self,
        *,
        mode: str = "offline_json",
        phase1_config: Dict[str, Any],
        storage_config: Dict[str, Any],
        similarity_weights: Optional[Dict[str, float]] = None,
        positions_json_dir: Optional[Path] = None,
        positions_per_game: int = 32,
        shallow_visits: int = 800,
        baseline_deep_visits: int = 10_000,
        estimated_visit_time_ms: float = 0.05,
    ) -> None:
        self.mode = mode
        self.phase1_config = phase1_config
        self.storage_config = storage_config
        self.similarity_weights = similarity_weights or DEFAULT_SIMILARITY_WEIGHTS
        self.shallow_visits = shallow_visits
        self.baseline_deep_visits = baseline_deep_visits
        self.estimated_visit_time_ms = estimated_visit_time_ms
        self.positions_per_game = max(1, positions_per_game)
        self._rng = random.Random(0)

        if mode == "offline_json":
            if positions_json_dir is None:
                raise ValueError("positions_json_dir is required in offline_json mode")
            self._offline_records = self._load_offline_positions(positions_json_dir)
        elif mode == "live":
            # Live mode will be wired once the KataGo + RAG harness lands.
            # Keep state placeholders for future use.
            self._offline_records = []
        else:
            raise ValueError(f"Unsupported GameRunner mode: {mode}")

    # ------------------------------------------------------------------ #
    # Offline helpers
    # ------------------------------------------------------------------ #

    def _load_offline_positions(self, directory: Path) -> List[Dict[str, Any]]:
        """
        Load all JSON files under `directory` and normalise them into a flat
        list of position-level dictionaries containing shallow/deep metrics.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Offline positions directory not found: {directory}")

        records: List[Dict[str, Any]] = []
        for path in sorted(directory.glob("**/*.json")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except json.JSONDecodeError as exc:
                # Skip corrupt files but continue loading the rest.
                print(f"[GameRunner] Warning: failed to parse {path}: {exc}")
                continue

            raw_entries: Sequence[Any]
            if isinstance(payload, list):
                raw_entries = payload
            elif isinstance(payload, dict):
                if "positions" in payload and isinstance(payload["positions"], list):
                    raw_entries = payload["positions"]
                elif "entries" in payload and isinstance(payload["entries"], list):
                    raw_entries = payload["entries"]
                else:
                    raw_entries = [payload]
            else:
                continue

            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue
                normalised = self._normalise_offline_entry(entry)
                if normalised:
                    records.append(normalised)

        if not records:
            raise RuntimeError(
                f"No usable offline position records found under {directory}"
            )

        return records

    def _normalise_offline_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Convert a raw JSON record into a standard dictionary of metrics we can
        aggregate. This function is intentionally defensive: if certain fields
        are missing it will fall back to reasonable defaults.
        """

        shallow = entry.get("shallow", {})
        deep = entry.get("deep", {})

        # --- Value error -------------------------------------------------
        value_error = entry.get("value_error")
        if value_error is None:
            shallow_val = _extract_value(shallow)
            deep_val = _extract_value(deep)
            if shallow_val is not None and deep_val is not None:
                value_error = abs(deep_val - shallow_val)
            else:
                value_error = 0.0

        # --- Policy error ------------------------------------------------
        policy_error = entry.get("policy_error")
        if policy_error is None:
            shallow_policy = _extract_policy(shallow) or entry.get("policy_shallow")
            deep_policy = _extract_policy(deep) or entry.get("policy_deep")
            if shallow_policy and deep_policy:
                policy_error = _kl_divergence(deep_policy, shallow_policy)
            else:
                policy_error = 0.0

        # --- Timing ------------------------------------------------------
        deep_visits = (
            deep.get("visits")
            or entry.get("deep_visits")
            or self.baseline_deep_visits
        )
        shallow_visits = (
            shallow.get("visits") or entry.get("shallow_visits") or self.shallow_visits
        )
        deep_time_ms = (
            entry.get("deep_time_ms")
            or deep.get("analysis_time_ms")
            or deep_visits * self.estimated_visit_time_ms
        )
        move_time_ms = entry.get("move_time_ms") or deep_time_ms

        # --- RAG / recursion metadata -----------------------------------
        rag_hit = bool(entry.get("rag_hit", False))
        stored = bool(entry.get("stored", False))
        recursion_depth = int(entry.get("recursion_depth", 0))
        child_nodes = entry.get("child_nodes") or []
        max_possible_recursion = max(recursion_depth, len(child_nodes) // 2)

        state_hash = (
            entry.get("state_hash")
            or entry.get("game_hash")
            or entry.get("hash")
            or entry.get("position_hash")
            or entry.get("positionHash")
        )
        sym_hash = entry.get("sym_hash") or state_hash
        policy = (
            entry.get("policy")
            or _extract_policy(deep)
            or _extract_policy(shallow)
        )
        ownership = entry.get("ownership") or deep.get("ownership")
        winrate = (
            entry.get("winrate")
            or deep.get("winrate")
            or deep.get("value")
            or shallow.get("winrate")
        )
        score_lead = (
            entry.get("score_lead")
            or entry.get("scoreLead")
            or deep.get("scoreLead")
        )
        move_infos = (
            entry.get("move_infos")
            or entry.get("moveInfos")
            or deep.get("moveInfos")
            or shallow.get("moveInfos")
            or []
        )

        normalised = {
            "sym_hash": entry.get("sym_hash"),
            "state_hash": state_hash,
            "policy": policy,
            "ownership": ownership,
            "winrate": winrate,
            "score_lead": score_lead,
            "move_infos": move_infos,
            "value_error": float(value_error),
            "policy_error": float(policy_error),
            "deep_time_ms": float(deep_time_ms),
            "move_time_ms": float(move_time_ms),
            "deep_visits": int(deep_visits),
            "shallow_visits": int(shallow_visits),
            "rag_hit": rag_hit,
            "stored": stored,
            "recursion_depth": recursion_depth,
            "max_possible_recursion": max_possible_recursion,
            "stone_count": entry.get("stone_count"),
            "komi": entry.get("komi"),
            "child_nodes": child_nodes,
            "metadata": {
                "source": entry.get("source"),
                "game_id": entry.get("game_id"),
            },
            "raw_entry": entry,
        }
        return normalised

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run_game(
        self,
        *,
        deep_mcts_max_depth: int,
        policy_delta: float,
        value_delta: float,
        recursion_depth_N: int,
        seed: Optional[int] = None,
        mode_override: Optional[str] = None,
    ) -> GameMetrics:
        """
        Return aggregate metrics for a single (synthetic) game.

        Parameters mirror the Phase 2 knobs we are tuning.  In offline mode
        they adjust how we transform the pre-recorded metrics to simulate what
        would happen under a different deep search budget or convergence rule.
        """

        mode = mode_override or self.mode
        if mode == "offline_json":
            return self._run_game_offline(
                deep_mcts_max_depth=deep_mcts_max_depth,
                policy_delta=policy_delta,
                value_delta=value_delta,
                recursion_depth_N=recursion_depth_N,
                seed=seed,
            )
        if mode == "live":
            # TODO (Phase 2 live integration):
            #   * Call KataGo (shallow visits) to evaluate the current board.
            #   * Compute Phase 1 uncertainty score (w1 E + w2 K) * phase.
            #   * If the score exceeds the storage / lookup thresholds:
            #         - Query the RAG ANN index using sym_hash / embedding.
            #         - If hit: blend policy/value using stored entry.
            #         - Else: run a deep MCTS (deep_mcts_max_depth visits),
            #           with early stop defined by (policy_delta, value_delta).
            #         - Update RAG storage if score_uncertainty >= threshold.
            #   * Track recursion depth by counting how many nested deep / RAG
            #     calls we make (bounded by recursion_depth_N).
            #   * Produce the same GameMetrics structure as offline mode.
            raise NotImplementedError(
                "Live GameRunner mode is not implemented yet. See TODO above."
            )
        raise ValueError(f"Unsupported GameRunner mode '{mode}'")

    # ------------------------------------------------------------------ #
    # Offline implementation
    # ------------------------------------------------------------------ #

    def _run_game_offline(
        self,
        *,
        deep_mcts_max_depth: int,
        policy_delta: float,
        value_delta: float,
        recursion_depth_N: int,
        seed: Optional[int],
    ) -> GameMetrics:
        rng = random.Random(seed if seed is not None else self._rng.randint(0, 2**32))
        population = self._offline_records
        sample_size = min(self.positions_per_game, len(population))
        sample = rng.sample(population, sample_size)

        adjusted_value_errors: List[float] = []
        adjusted_policy_errors: List[float] = []
        adjusted_deep_times: List[float] = []
        move_times: List[float] = []
        rag_hits = 0
        stored_count = 0
        recursion_depths: List[int] = []
        converged = 0
        stored_samples: List[Dict[str, Any]] = []

        depth_ratio = max(1.0, deep_mcts_max_depth) / max(
            1.0, float(self.baseline_deep_visits)
        )
        error_scale = 1.0 / math.sqrt(depth_ratio)

        for record in sample:
            value_err = record["value_error"] * error_scale
            policy_err = record["policy_error"] * error_scale

            deep_visits = max(1, record["deep_visits"])
            time_scale = (deep_mcts_max_depth / deep_visits) if deep_visits else 1.0
            deep_time = record["deep_time_ms"] * max(time_scale, 0.1)

            adjusted_value_errors.append(value_err)
            adjusted_policy_errors.append(policy_err)
            adjusted_deep_times.append(deep_time)
            move_times.append(record["move_time_ms"])

            rag_hits += 1 if record["rag_hit"] else 0
            stored_count += 1 if record["stored"] else 0
            if record["stored"]:
                stored_samples.append(record)

            effective_recursion = min(
                recursion_depth_N,
                max(record["recursion_depth"], record["max_possible_recursion"]),
            )
            recursion_depths.append(effective_recursion)

            if value_err <= value_delta and policy_err <= policy_delta:
                converged += 1

        deep_search_count = len(sample)
        deep_converged_count = converged

        avg_value_error = (
            sum(adjusted_value_errors) / deep_search_count if deep_search_count else 0.0
        )
        avg_policy_error = (
            sum(adjusted_policy_errors) / deep_search_count if deep_search_count else 0.0
        )
        avg_deep_time_ms = (
            sum(adjusted_deep_times) / deep_search_count if deep_search_count else 0.0
        )
        avg_move_time_ms = (
            sum(move_times) / len(move_times) if move_times else avg_deep_time_ms
        )
        max_recursion_depth_used = max(recursion_depths) if recursion_depths else 0

        metrics = GameMetrics(
            win=False,  # Offline mode does not derive true win/loss yet.
            game_length=len(sample),
            avg_move_time_ms=avg_move_time_ms,
            deep_search_count=deep_search_count,
            deep_converged_count=deep_converged_count,
            avg_deep_time_ms=avg_deep_time_ms,
            avg_value_error=avg_value_error,
            avg_policy_error=avg_policy_error,
            rag_hits=rag_hits,
            rag_misses=deep_search_count - rag_hits,
            new_positions_stored=stored_count,
            max_recursion_depth_used=max_recursion_depth_used,
            extra={
                "depth_ratio": depth_ratio,
                "error_scale": error_scale,
                "positions_sampled": sample_size,
                "rag_entries_example": [
                    build_rag_entry_from_deep_eval(
                        state_hash=rec.get("state_hash") or rec.get("sym_hash"),
                        sym_hash=rec.get("sym_hash") or rec.get("state_hash"),
                        deep_eval=_extract_raw_deep_eval(rec),
                        metadata=rec.get("metadata"),
                        similarity_weights=self.similarity_weights,
                    )
                    for rec in stored_samples[:2]
                ],
            },
        )
        return metrics


# ---------------------------------------------------------------------- #
# Helper functions
# ---------------------------------------------------------------------- #


def _extract_value(blob: Dict[str, Any]) -> Optional[float]:
    if blob is None:
        return None
    for key in ("winrate", "value", "scoreLead", "score_lead"):
        if key in blob:
            try:
                return float(blob[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_policy(blob: Dict[str, Any]) -> Optional[List[float]]:
    if blob is None:
        return None
    policy = blob.get("policy")
    if isinstance(policy, list) and policy:
        # Ensure numeric and normalised.
        try:
            floats = [float(x) for x in policy]
        except (TypeError, ValueError):
            return None
        total = sum(floats)
        if total > 0:
            return [x / total for x in floats]
        return floats
    return None


def _kl_divergence(p: Sequence[float], q: Sequence[float], epsilon: float = 1e-12) -> float:
    """
    Compute KL(p || q) with defensive guards against zeros.
    """
    length = min(len(p), len(q))
    divergence = 0.0
    for i in range(length):
        pi = max(float(p[i]), epsilon)
        qi = max(float(q[i]), epsilon)
        divergence += pi * math.log(pi / qi)
    return max(divergence, 0.0)


def _extract_raw_deep_eval(record: Dict[str, Any]) -> Dict[str, Any]:
    raw = record.get("raw_entry") or {}
    deep = raw.get("deep")
    if isinstance(deep, dict):
        return deep
    return raw


def build_rag_entry_from_deep_eval(
    *,
    state_hash: Optional[str],
    sym_hash: Optional[str],
    deep_eval: Dict[str, Any],
    similarity_weights: Optional[Dict[str, float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    max_moves: int = 2,
) -> Dict[str, Any]:
    """
    Construct a RAG entry dictionary using the schema outlined in
    claude_instructions.txt (lines 55-63).
    """

    similarity_weights = similarity_weights or DEFAULT_SIMILARITY_WEIGHTS
    metadata = metadata or {}

    policy = (
        deep_eval.get("policy")
        or _extract_policy(deep_eval)
        or []
    )
    ownership = deep_eval.get("ownership")
    winrate = (
        deep_eval.get("winrate")
        or deep_eval.get("value")
        or 0.0
    )
    score_lead = (
        deep_eval.get("score_lead")
        or deep_eval.get("scoreLead")
    )
    komi = deep_eval.get("komi")
    move_infos = (
        deep_eval.get("move_infos")
        or deep_eval.get("moveInfos")
        or []
    )
    child_nodes = deep_eval.get("child_nodes")

    if not child_nodes:
        child_nodes = []
        for info in move_infos:
            val = info.get("value")
            if val is None:
                val = info.get("winrate")
            child_nodes.append(
                {
                    "hash": info.get("hash")
                    or info.get("node_hash")
                    or info.get("child_hash")
                    or info.get("move"),
                    "move": info.get("move"),
                    "value": val,
                    "pUCT": info.get("pUCT") or info.get("puct"),
                    "policy": info.get("policy") or info.get("prior"),
                }
            )

    def _node_value(node: Dict[str, Any]) -> float:
        val = node.get("value")
        if val is None:
            val = node.get("winrate")
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    sorted_nodes = sorted(child_nodes, key=_node_value, reverse=True)
    top_moves = sorted_nodes[:max_moves]

    return {
        "game_hash": state_hash or sym_hash,
        "sym_hash": sym_hash or state_hash,
        "policy": policy,
        "ownership": ownership,
        "winrate": winrate,
        "score_lead": score_lead,
        "move_infos": move_infos,
        "komi": komi,
        "child_nodes": child_nodes,
        "top_moves": top_moves,
        "similarity_weights": similarity_weights,
        "metadata": metadata,
    }
