Phase 2 – Offline Deep Search & Recursion Tuning
================================================

This folder contains the second-stage tuning scripts for the RAG‑MCTS
experiments.  Phase 2 currently runs **entirely on offline JSON logs** that
were generated during Phase 1 (shallow vs deep comparisons and offline deep
analysis).  Once the real self-play harness is ready, these scripts can be
swapped to the live mode without changing their public interfaces.

Prerequisites
-------------

1. Phase 1 tuning completed successfully:
   - `tuning_results/phase1/best_config_phase1.json`
   - `tuning_results/phase1b/storage_threshold_results.json`
2. Offline deep-analysis logs are available under
   `tuning_results/offline_positions/`. Each JSON file should contain the
   fields described in `claude_instructions.txt` (shallow/deep stats, child
   nodes, etc.).
3. Python environment with the packages listed in
   `phase1/requirements_tuning.txt`.

How it works (offline mode)
---------------------------

* `tuning/common/game_runner.py` now exposes a `GameRunner` with two modes.
  We use `mode="offline_json"` to synthesise game-level metrics from the JSON
  directory.  The live mode is stubbed with detailed TODOs for the upcoming
  KataGo/RAG integration.
* `phase2_deep_mcts.py` samples the offline records to estimate
  value/policy error, deep-search time, and convergence rate for each
  `(deep_mcts_max_depth, policy_delta, value_delta)` triple.  It writes
  per-config JSON files plus a consolidated
  `phase2_deep_mcts_results.json`.
* `phase2_recursion.py` loads the best deep-search config and sweeps the
  recursion depth `N`, estimating compute cost and residual error.
* `run_phase2.sh` orchestrates the offline workflow.  It assumes the working
  directory is `datago/tuning/`.

Usage
-----

```bash
# From datago/tuning:
phase2/run_phase2.sh
```

### Dummy Offline Test

To smoke-test Phase 2 without Phase 1 logs:

```bash
cd datago/tuning
python phase2/make_dummy_offline_positions.py --output-dir ./tuning_results/offline_positions
python phase2/smoke_test.py  # optional quick check
phase2/run_phase2.sh
```

The generator creates synthetic offline JSON files so `GameRunner` and the Phase 2
pipelines can run end-to-end without real data.

Key outputs
-----------

* `tuning_results/phase2/deep_mcts/phase2_deep_mcts_results.json`
* `tuning_results/phase2/recursion/phase2_recursion_results.json`

Next steps
----------

Once the live self-play harness is available:

1. Implement `GameRunner`’s `mode="live"` branch (see the TODO comments).
2. Point `phase2_deep_mcts.py` and `phase2_recursion.py` at the live runner by
   removing the offline flag.
3. Validate the offline estimates against real match results.
