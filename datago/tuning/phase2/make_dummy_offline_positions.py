#!/usr/bin/env python3
"""
Utility to generate synthetic offline position JSON files for Phase 2 tuning.

Produces lightweight entries that mimic the schema consumed by
GameRunner._normalise_offline_entry so we can smoke-test the pipeline without
Phase 1 artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List


def _normalise_policy(values: List[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        # fallback to uniform distribution
        return [1.0 / len(values)] * len(values)
    return [v / total for v in values]


def _random_policy(rng: random.Random, length: int = 10) -> List[float]:
    raw = [rng.random() + 1e-6 for _ in range(length)]
    return _normalise_policy(raw)


def _perturb_policy(
    base: List[float], rng: random.Random, noise: float = 0.05
) -> List[float]:
    perturbed = [max(0.0, p + rng.uniform(-noise, noise)) for p in base]
    return _normalise_policy(perturbed)


def _kl_divergence(p: List[float], q: List[float], eps: float = 1e-9) -> float:
    total = 0.0
    for pi, qi in zip(p, q):
        pi = max(pi, eps)
        qi = max(qi, eps)
        total += pi * math.log(pi / qi)
    return max(total, 0.0)


def _make_entry(rng: random.Random, file_idx: int, pos_idx: int) -> dict:
    sym = f"sym_{file_idx}_{pos_idx}"
    state = f"state_{file_idx}_{pos_idx}"
    shallow_policy = _random_policy(rng)
    deep_policy = _perturb_policy(shallow_policy, rng)
    ownership = [rng.uniform(-1.0, 1.0) for _ in range(10)]

    shallow_val = rng.uniform(0.2, 0.8)
    deep_val = min(max(shallow_val + rng.uniform(-0.05, 0.05), 0.0), 1.0)
    deep_time_ms = rng.uniform(1500.0, 3000.0)
    move_time_ms = deep_time_ms * rng.uniform(0.8, 1.2)

    move_infos = [
        {
            "move": f"M{k}",
            "hash": f"child_{file_idx}_{pos_idx}_{k}",
            "value": rng.uniform(0.2, 0.8),
            "pUCT": rng.uniform(0.5, 3.0),
            "policy": rng.random(),
        }
        for k in range(5)
    ]

    entry = {
        "sym_hash": sym,
        "state_hash": state,
        "policy": shallow_policy,
        "ownership": ownership,
        "winrate": shallow_val,
        "score_lead": rng.uniform(-10, 10),
        "move_infos": move_infos,
        "komi": 7.5,
        "query_id": f"dummy_query_{file_idx}_{pos_idx}",
        "stone_count": rng.randint(0, 361),
        "child_nodes": [],
        "shallow": {
            "winrate": shallow_val,
            "policy": shallow_policy,
            "visits": 800,
        },
        "deep": {
            "winrate": deep_val,
            "policy": deep_policy,
            "visits": 10_000,
            "analysis_time_ms": deep_time_ms,
            "moveInfos": move_infos,
        },
        "value_error": abs(deep_val - shallow_val),
        "policy_error": _kl_divergence(deep_policy, shallow_policy),
        "deep_time_ms": deep_time_ms,
        "move_time_ms": move_time_ms,
        "rag_hit": rng.random() < 0.15,
        "stored": rng.random() < 0.3,
        "recursion_depth": rng.randint(0, 3),
    }
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy offline positions.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./tuning_results/offline_positions"),
        help="Directory to write JSON files into.",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=3,
        help="Number of JSON files to create.",
    )
    parser.add_argument(
        "--positions-per-file",
        type=int,
        default=64,
        help="Number of position entries per file.",
    )
    args = parser.parse_args()

    rng = random.Random(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_positions = 0
    for file_idx in range(args.num_files):
        entries = [
            _make_entry(rng, file_idx, pos_idx) for pos_idx in range(args.positions_per_file)
        ]
        out_path = args.output_dir / f"positions_{file_idx}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2)
        total_positions += len(entries)
        print(f"Wrote {len(entries)} entries to {out_path}")

    print(
        f"Generated {total_positions} dummy positions across "
        f"{args.num_files} files in {args.output_dir}"
    )


if __name__ == "__main__":
    main()

