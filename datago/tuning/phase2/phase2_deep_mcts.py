# tuning/phase2/phase2_deep_mcts.py
"""
Phase 2a: Deep MCTS tuning (offline JSON-driven workflow)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from common.game_runner import GameRunner, GameMetrics, DEFAULT_SIMILARITY_WEIGHTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2a: Deep MCTS tuning")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./tuning_results/phase2/deep_mcts"),
        help="Where to store per-config metrics and summary JSON.",
    )
    parser.add_argument(
        "--positions-json-dir",
        type=Path,
        required=True,
        help="Directory containing offline JSON logs of complex positions.",
    )
    parser.add_argument(
        "--phase1-config",
        type=Path,
        default=Path("./tuning_results/phase1/best_config_phase1.json"),
        help="Phase 1 learned weights JSON file.",
    )
    parser.add_argument(
        "--storage-config",
        type=Path,
        default=Path("./tuning_results/phase1b/storage_threshold_results.json"),
        help="Phase 1b storage threshold JSON file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of synthetic 'games' (samples) to draw per configuration.",
    )
    parser.add_argument(
        "--positions-per-game",
        type=int,
        default=32,
        help="How many JSON entries to treat as one synthetic game.",
    )
    parser.add_argument(
        "--baseline-deep-visits",
        type=int,
        default=10_000,
        help="Visit count used when offline deep analyses were produced.",
    )
    parser.add_argument(
        "--shallow-visits",
        type=int,
        default=800,
        help="Visit count for shallow (baseline) search.",
    )
    parser.add_argument(
        "--target-deep-time-ms",
        type=float,
        default=2_500.0,
        help="Reference target (ms) for deep search time when scoring.",
    )
    parser.add_argument(
        "--alpha-error",
        type=float,
        default=1.0,
        help="Weight for value error penalty in scoring.",
    )
    parser.add_argument(
        "--beta-error",
        type=float,
        default=0.5,
        help="Weight for policy error penalty in scoring.",
    )
    parser.add_argument(
        "--gamma-time",
        type=float,
        default=0.1,
        help="Weight for deep search time penalty in scoring.",
    )
    parser.add_argument(
        "--delta-fail",
        type=float,
        default=0.5,
        help="Weight for non-convergence penalty in scoring.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def aggregate_metrics(metrics: List[GameMetrics]) -> Dict[str, float]:
    if not metrics:
        return {
            "avg_value_error": 0.0,
            "avg_policy_error": 0.0,
            "avg_deep_time_ms": 0.0,
            "deep_search_total": 0,
            "deep_converged_total": 0,
            "avg_move_time_ms": 0.0,
            "rag_hits_total": 0,
            "rag_misses_total": 0,
            "avg_recursion_depth": 0.0,
        }

    deep_search_total = sum(m.deep_search_count for m in metrics)
    deep_converged_total = sum(m.deep_converged_count for m in metrics)
    avg_value_error = mean(m.avg_value_error for m in metrics)
    avg_policy_error = mean(m.avg_policy_error for m in metrics)
    avg_deep_time_ms = mean(m.avg_deep_time_ms for m in metrics)
    avg_move_time_ms = mean(m.avg_move_time_ms for m in metrics)
    rag_hits_total = sum(m.rag_hits for m in metrics)
    rag_misses_total = sum(m.rag_misses for m in metrics)
    avg_recursion_depth = mean(m.max_recursion_depth_used for m in metrics)

    return {
        "avg_value_error": avg_value_error,
        "avg_policy_error": avg_policy_error,
        "avg_deep_time_ms": avg_deep_time_ms,
        "deep_search_total": deep_search_total,
        "deep_converged_total": deep_converged_total,
        "avg_move_time_ms": avg_move_time_ms,
        "rag_hits_total": rag_hits_total,
        "rag_misses_total": rag_misses_total,
        "avg_recursion_depth": avg_recursion_depth,
    }


def compute_score(
    metrics: Dict[str, float],
    *,
    target_deep_time_ms: float,
    alpha_error: float,
    beta_error: float,
    gamma_time: float,
    delta_fail: float,
) -> Tuple[float, Dict[str, float]]:
    deep_search_total = metrics["deep_search_total"]
    deep_converged_total = metrics["deep_converged_total"]
    non_converged_rate = (
        1.0 - (deep_converged_total / deep_search_total)
        if deep_search_total
        else 0.0
    )
    time_ratio = (
        metrics["avg_deep_time_ms"] / target_deep_time_ms if target_deep_time_ms else 0.0
    )
    score = (
        -alpha_error * metrics["avg_value_error"]
        -beta_error * metrics["avg_policy_error"]
        -gamma_time * time_ratio
        -delta_fail * non_converged_rate
    )
    extras = {
        "non_converged_rate": non_converged_rate,
        "time_ratio": time_ratio,
    }
    return score, extras


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.positions_json_dir.exists():
        raise FileNotFoundError(
            f"positions-json-dir does not exist: {args.positions_json_dir}"
        )

    phase1_cfg = load_json(args.phase1_config)
    storage_cfg = load_json(args.storage_config)

    runner = GameRunner(
        mode="offline_json",
        phase1_config=phase1_cfg,
        storage_config=storage_cfg,
        similarity_weights=DEFAULT_SIMILARITY_WEIGHTS,
        positions_json_dir=args.positions_json_dir,
        positions_per_game=args.positions_per_game,
        shallow_visits=args.shallow_visits,
        baseline_deep_visits=args.baseline_deep_visits,
    )

    depth_values = [1_000, 2_000, 5_000, 10_000]
    policy_deltas = [0.02, 0.05, 0.10]
    value_deltas = [0.01, 0.02, 0.05]

    configs: List[Dict[str, float]] = []
    for depth in depth_values:
        for p_delta in policy_deltas:
            for v_delta in value_deltas:
                configs.append(
                    {
                        "deep_mcts_max_depth": depth,
                        "policy_delta": p_delta,
                        "value_delta": v_delta,
                    }
                )

    all_results: List[Dict[str, Any]] = []
    best_entry: Optional[Dict[str, Any]] = None
    best_score = float("-inf")

    for cfg_idx, cfg in enumerate(configs, start=1):
        config_id = (
            f"d{cfg['deep_mcts_max_depth']}_"
            f"p{cfg['policy_delta']}_"
            f"v{cfg['value_delta']}"
        )
        print(f"[Phase2/Deep] ({cfg_idx}/{len(configs)}) Evaluating {config_id}")

        metrics_samples: List[GameMetrics] = []
        for sample_idx in range(args.num_samples):
            gm = runner.run_game(
                deep_mcts_max_depth=cfg["deep_mcts_max_depth"],
                policy_delta=cfg["policy_delta"],
                value_delta=cfg["value_delta"],
                recursion_depth_N=2,  # Phase 2b will sweep this
                seed=sample_idx,
            )
            metrics_samples.append(gm)

        aggregated = aggregate_metrics(metrics_samples)
        score, extras = compute_score(
            aggregated,
            target_deep_time_ms=args.target_deep_time_ms,
            alpha_error=args.alpha_error,
            beta_error=args.beta_error,
            gamma_time=args.gamma_time,
            delta_fail=args.delta_fail,
        )

        result_entry = {
            "config": cfg,
            "config_id": config_id,
            "score": score,
            "metrics": aggregated,
            "extras": extras,
        }
        all_results.append(result_entry)

        output_path = args.output_dir / f"{config_id}.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(result_entry, fh, indent=2)

        if score > best_score:
            best_score = score
            best_entry = result_entry

    summary = {
        "configs": all_results,
        "best": best_entry,
        "scoring": {
            "alpha_error": args.alpha_error,
            "beta_error": args.beta_error,
            "gamma_time": args.gamma_time,
            "delta_fail": args.delta_fail,
            "target_deep_time_ms": args.target_deep_time_ms,
            "num_samples": args.num_samples,
            "positions_per_game": args.positions_per_game,
        },
    }
    summary_path = args.output_dir / "phase2_deep_mcts_results.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if best_entry:
        print(
            "[Phase2/Deep] Best config:",
            best_entry["config"],
            " score=",
            best_entry["score"],
        )
    else:
        print("[Phase2/Deep] No valid configurations evaluated.")


if __name__ == "__main__":
    main()