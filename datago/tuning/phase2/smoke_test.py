#!/usr/bin/env python3
"""
Tiny smoke test for GameRunner using offline JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common.game_runner import GameRunner, DEFAULT_SIMILARITY_WEIGHTS


PHASE1_PATH = Path("./tuning_results/phase1/best_config_phase1.json")
PHASE1B_PATH = Path("./tuning_results/phase1b/storage_threshold_results.json")
POSITIONS_DIR = Path("./tuning_results/offline_positions")


def _ensure_placeholder_configs() -> Dict[str, Any]:
    if not PHASE1_PATH.exists():
        PHASE1_PATH.parent.mkdir(parents=True, exist_ok=True)
        placeholder = {
            "w1": 1.0,
            "w2": 1.0,
            "phase_params": {"type": "linear", "a": 1.0, "b": 0.0},
        }
        PHASE1_PATH.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
        print(f"Created placeholder Phase 1 config at {PHASE1_PATH}")
    if not PHASE1B_PATH.exists():
        PHASE1B_PATH.parent.mkdir(parents=True, exist_ok=True)
        placeholder = {
            "threshold_percentile": 90,
            "uncertainty_threshold": 0.9,
        }
        PHASE1B_PATH.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
        print(f"Created placeholder Phase 1b config at {PHASE1B_PATH}")

    with PHASE1_PATH.open("r", encoding="utf-8") as fh:
        phase1_cfg = json.load(fh)
    with PHASE1B_PATH.open("r", encoding="utf-8") as fh:
        storage_cfg = json.load(fh)
    return phase1_cfg, storage_cfg


def main() -> None:
    if not POSITIONS_DIR.exists():
        print(
            f"Offline positions not found at {POSITIONS_DIR}. "
            "Run python phase2/make_dummy_offline_positions.py first."
        )
        raise SystemExit(1)

    phase1_cfg, storage_cfg = _ensure_placeholder_configs()

    runner = GameRunner(
        mode="offline_json",
        phase1_config=phase1_cfg,
        storage_config=storage_cfg,
        similarity_weights=DEFAULT_SIMILARITY_WEIGHTS,
        positions_json_dir=POSITIONS_DIR,
        positions_per_game=16,
        shallow_visits=800,
        baseline_deep_visits=10_000,
    )

    gm = runner.run_game(
        deep_mcts_max_depth=5000,
        policy_delta=0.05,
        value_delta=0.02,
        recursion_depth_N=2,
        seed=0,
    )

    print(json.dumps(gm.to_dict(), indent=2))


if __name__ == "__main__":
    main()

