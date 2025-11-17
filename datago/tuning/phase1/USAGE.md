# Phase 1 Tuning with Ground Truth Database

## Overview

This tuning approach uses **supervised learning** to optimize uncertainty parameters. Instead of running full games (slow, 18-20 hours), we directly optimize parameters to predict model error using pre-computed ground truth data (4-8 hours).

## Database Format

The tuner expects **two separate JSON files** with the format from `claude_instructions.txt`:

```json
{
  "sym_hash": "lookup_key",
  "state_hash": "unique_id",
  "policy": [362 floats],
  "ownership": [361 floats],
  "winrate": 0.547,
  "score_lead": 2.3,
  "move_infos": [...],
  "komi": 7.5,
  "query_id": "query_123",
  "stone_count": 85,
  "child_nodes": [
    {"hash": "child1", "value": 0.52, "pUCT": 1.23},
    ...
  ]
}
```

You need:
1. **Shallow MCTS database** (800 visits) - fast, potentially inaccurate
2. **Deep MCTS database** (5000+ visits) - slow, ground truth

The tuner will:
- Match positions by `sym_hash`
- Compute uncertainty features from shallow MCTS
- Compute errors by comparing shallow vs deep results
- Optimize parameters to predict errors from features

## Step 1: Generate the Databases

### Option A: Run actual KataGo analysis

```bash
# Play self-play games and collect positions
# For each position, run two analyses:

# 1. Shallow MCTS (800 visits)
katago analysis -model <model> -config analysis.cfg -visits 800 < positions.txt > shallow_results.json

# 2. Deep MCTS (5000 visits) 
katago analysis -model <model> -config analysis.cfg -visits 5000 < positions.txt > deep_results.json
```

### Option B: Use synthetic data generator (for testing)

```bash
# Generate sample databases for testing
python scripts/generate_ground_truth_db.py \
    --num-positions 1000 \
    --output-shallow ./data/shallow_mcts_db.json \
    --output-deep ./data/deep_mcts_db.json
```

## Step 2: Run Supervised Tuning

Once you have both databases, run tuning:

### Option A: Grid Search (Recommended, thorough)

```bash
python tuning/phase1/phase1_uncertainty_tuning.py \
    --shallow-db ./data/shallow_mcts_db.json \
    --deep-db ./data/deep_mcts_db.json \
    --method grid \
    --output-dir ./tuning_results/phase1_supervised \
    --train-split 0.8
```

**What this does:**
- Matches positions by `sym_hash` between databases
- Computes uncertainty features from shallow MCTS
- Computes errors by comparing shallow vs deep
- Tests 35 configurations (7 w1 values × 5 phase functions)
- For each config, computes correlation with ground truth errors
- Uses 80% of data for training, 20% for validation
- Time: ~30 minutes

### Option B: Gradient Optimization (Fast)

```bash
python tuning/phase1/phase1_uncertainty_tuning.py \
    --shallow-db ./data/shallow_mcts_db.json \
    --deep-db ./data/deep_mcts_db.json \
    --method optimize \
    --output-dir ./tuning_results/phase1_fast \
    --train-split 0.8
```

```bash
python tuning/phase1/phase1_uncertainty_tuning.py \
    --database ./data/ground_truth_full.json \
    --method optimize \
    --output-dir ./tuning_results/phase1_supervised \
    --train-split 0.8
```

**What this does:**
- Uses L-BFGS-B to find optimal parameters
- Maximizes Pearson correlation with ground truth errors
- Time: ~5 minutes

## Step 3: Review Results

After tuning completes, check outputs:

### Best Configuration
```bash
cat tuning_results/phase1_supervised/best_uncertainty_config.json
```

Example output:
```json
{
  "config": {
    "w1": 0.65,
    "w2": 0.35,
    "phase_function_type": "linear",
    "phase_coefficients": [0.3, 0.85]
  },
  "validation_metrics": {
    "pearson_correlation": 0.82,
    "spearman_correlation": 0.79,
    "top_k_precision": 0.75,
    "ndcg_score": 0.88
  }
}
```

### Threshold Analysis
```bash
cat tuning_results/phase1_supervised/threshold_analysis.json
```

Example output:
```json
{
  "10": {
    "threshold": 0.85,
    "storage_rate": 0.10,
    "avg_error_stored": 0.089,
    "avg_error_not_stored": 0.032,
    "benefit": 0.057,
    "coverage": 0.82
  },
  "15": {
    "threshold": 0.72,
    "storage_rate": 0.15,
    "avg_error_stored": 0.078,
    "avg_error_not_stored": 0.035,
    "benefit": 0.043,
    "coverage": 0.89
  }
}
```

### Visualization
Check the Pareto frontier plot:
```bash
open tuning_results/phase1_supervised/threshold_pareto_frontier.png
```

This shows:
- **Storage rate vs Error reduction benefit**
- **Storage rate vs Coverage of high-error positions**

**Typical recommendation:** Use 10-15% threshold for good balance.

## Understanding the Metrics

### Correlation Metrics
- **Pearson correlation (0.7-0.9 is good)**: Linear relationship between uncertainty and error
- **Spearman correlation**: Rank correlation (monotonic relationship)
- **NDCG score**: Quality of ranking (0-1, higher is better)
- **Top-K precision**: Among top 10% uncertain, what % are actually high error?

### Threshold Metrics
- **Storage rate**: % of positions stored (lower is more efficient)
- **Benefit**: Difference in error between stored and not-stored positions (higher is better)
- **Coverage**: % of high-error positions captured (higher is better)

## Key Differences from Old Approach

### Old Approach (Win-Rate Based)
- **Time:** 18-20 hours
- **Method:** Run full games, measure win rate
- **Objective:** Indirect (win rate ≠ uncertainty quality)
- **Data needed:** 1350+ games

### New Approach (Supervised Learning)
- **Time:** 4-8 hours total (6 hours DB generation + 0.5 hours tuning)
- **Method:** Direct optimization on labeled data
- **Objective:** Maximize correlation with actual errors
- **Data needed:** 10,000 positions (one-time collection)

## Benefits

1. **Faster:** 4-8 hours vs 18-20 hours
2. **Direct:** Optimizes what you care about (predicting error)
3. **Clear metrics:** Correlation, coverage, benefit
4. **Reusable:** Database can be reused for future tuning
5. **Better insights:** See exactly which positions are problematic

## Next Steps

After Phase 1 tuning:

1. Use the best configuration in your RAG-MCTS system
2. Use the recommended threshold (10-15%) for storage decisions
3. During gameplay:
   - Compute uncertainty for each position
   - If uncertainty > threshold, store in RAG database
   - Retrieve similar positions for uncertain moves

4. Move to Phase 2: Tune similarity threshold and k neighbors
