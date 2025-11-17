# Phase 1 Tuning Scripts

This directory contains scripts for Phase 1 of the RAG-AlphaGo parameter tuning process.

## Overview

Phase 1 consists of two parts:
1. **Phase 1a**: Tune uncertainty detection parameters (w1, w2, phase function)
2. **Phase 1b**: Tune storage threshold based on Phase 1a results

**Total Time Budget**: 18-20 hours (Phase 1a) + 8-10 hours (Phase 1b) = ~24-30 hours on NVIDIA A100

## Files

### `phase1_uncertainty_tuning.py`
Main script for tuning uncertainty detection parameters.

**Parameters tuned:**
- `w1`: Weight for policy cross-entropy (E)
- `w2`: Weight for value distribution sparseness (K)
- `phase_function_type`: Type of phase function (linear, exponential, piecewise)
- `phase_coefficients`: Coefficients for the phase function based on stones_on_board

**Phase Function:**
The phase function takes stones_on_board as input and outputs a multiplier:
- **Linear**: `phase(s) = a*(s/361) + b` where s = stones_on_board
- **Exponential**: `phase(s) = a*exp(b*s/361) + c`
- **Piecewise**: Different multipliers for early (<120), mid (120-240), late (≥240) game

**Usage:**
```bash
python phase1_uncertainty_tuning.py \
    --output-dir ./tuning_results/phase1 \
    --num-games 150 \
    --parallel-workers 32 \
    --early-stopping-games 100 \
    --early-stopping-threshold 0.40
```

**Arguments:**
- `--output-dir`: Directory to save results (default: `./tuning_results/phase1`)
- `--num-games`: Number of games per configuration (default: 150)
- `--parallel-workers`: Number of parallel game workers (default: 32)
- `--early-stopping-games`: Check early stopping after N games (default: 100)
- `--early-stopping-threshold`: Abort if win rate below threshold (default: 0.40)

**Output:**
- `phase1_results.json`: All configuration results
- `best_config_phase1.json`: Best performing configuration
- Individual config files: `w1_X_w2_Y_early_Z_late_W.json`

### `phase1b_storage_threshold.py`
Script for tuning the storage threshold after Phase 1a completes.

**Parameters tuned:**
- `storage_threshold`: Percentile-based threshold for storing positions in RAG

**Usage:**
```bash
python phase1b_storage_threshold.py \
    --phase1-config ./tuning_results/phase1/best_config_phase1.json \
    --output-dir ./tuning_results/phase1b \
    --num-games 100 \
    --max-db-size 20
```

**Arguments:**
- `--phase1-config`: Path to best config from Phase 1a (required)
- `--output-dir`: Directory to save results (default: `./tuning_results/phase1b`)
- `--num-games`: Number of games per threshold (default: 100)
- `--max-db-size`: Maximum database size in GB (default: 20)

**Output:**
- `storage_threshold_results.json`: All threshold test results
- `percentile_mapping.json`: Mapping from percentiles to absolute thresholds
- `uncertainty_distribution.png`: Distribution plot with threshold lines
- `threshold_comparison.png`: Comparison plots for different thresholds

### `monitor.py`
Real-time monitoring dashboard for tracking tuning progress and system resources.

**Usage (live monitoring):**
```bash
python monitor.py \
    --mode monitor \
    --results-dir ./tuning_results \
    --phase phase1 \
    --refresh 10
```

**Usage (results summary):**
```bash
python monitor.py \
    --mode summary \
    --results-file ./tuning_results/phase1/phase1_results.json
```

**Features:**
- Real-time GPU utilization and temperature
- CPU and RAM monitoring
- Disk usage tracking
- Tuning progress and recent results
- Performance warnings (low GPU utilization, high temperature, etc.)

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy matplotlib psutil
pip install ray[tune] wandb  # Optional but recommended
pip install faiss-gpu  # For Phase 1b ANN search
```

### 2. Run Phase 1a

In one terminal, start the tuning:
```bash
python phase1_uncertainty_tuning.py --parallel-workers 32 --num-games 150
```

In another terminal, monitor progress:
```bash
python monitor.py --mode monitor --phase phase1 --refresh 10
```

### 3. Run Phase 1b

After Phase 1a completes:
```bash
python phase1b_storage_threshold.py \
    --phase1-config ./tuning_results/phase1/best_config_phase1.json \
    --num-games 100
```

### 4. Review Results

Generate summary:
```bash
python monitor.py \
    --mode summary \
    --results-file ./tuning_results/phase1/phase1_results.json
```

## Integration with KataGo

**IMPORTANT**: The current scripts contain placeholder game execution logic. You need to integrate with your actual KataGo client by:

1. **In `phase1_uncertainty_tuning.py`**:
   - Replace `run_single_game()` method with actual KataGo game execution
   - Compute actual uncertainty scores during gameplay
   - Return real win/loss and game statistics

2. **In `phase1b_storage_threshold.py`**:
   - Replace `estimate_percentiles_from_sample()` with real game data collection
   - Replace `evaluate_threshold()` with actual storage monitoring
   - Track real database size and retrieval performance

3. **Integration points**:
   - Import your KataGo client: `from datago.clients.katago_client import KataGoClient`
   - Use the uncertainty config to compute scores at each position
   - Store positions that exceed the threshold in your RAG database
   - Track and report actual metrics

## Expected Timeline

### Phase 1a (18-20 hours)
- 9 configurations to test
- 150 games per configuration (with early stopping)
- ~2 hours per configuration on A100 with 32 parallel workers
- Total: ~18 hours

### Phase 1b (8-10 hours)
- 4 threshold values to test
- 100 games per threshold
- Sample 500 games for percentile estimation (~2 hours)
- Testing thresholds (~6 hours)
- Total: ~8 hours

## Performance Targets (A100)

- **Throughput**: 1000-2000 games/hour with 32-64 parallel workers
- **GPU Utilization**: >90% during game playing
- **Memory**: <60GB VRAM for models + MCTS
- **Retrieval Latency**: <5ms per RAG query

## Troubleshooting

### Low GPU Utilization
- Increase `--parallel-workers`
- Check for CPU bottlenecks in preprocessing
- Verify batch processing is working

### Out of Memory
- Reduce `--parallel-workers`
- Reduce MCTS tree size
- Check for memory leaks

### Slow Progress
- Enable early stopping
- Reduce `--num-games` per config
- Use coarser grid (modify `generate_configs()`)

## Next Steps

After Phase 1 completes:
1. Note the best `w1`, `w2`, `phase_function`, and `storage_threshold`
2. Proceed to Phase 2: Deep MCTS and Recursion Control
3. Use the saved configs for all future phases

## Output Format

### Best Config JSON Format:
```json
{
  "config": {
    "w1": 0.5,
    "w2": 0.5,
    "phase_early_multiplier": 1.0,
    "phase_late_multiplier": 1.0,
    "config_id": "w1_0.50_w2_0.50_early_1.00_late_1.00"
  },
  "results": {
    "total_games": 150,
    "wins": 82,
    "win_rate": 0.547,
    "avg_game_length": 234.5,
    "avg_uncertainty": 0.456,
    "avg_stored_positions": 23.4
  }
}
```

## Questions or Issues?

Refer to the main parameter tuning plan document for detailed explanations of the methodology and parameter relationships.
