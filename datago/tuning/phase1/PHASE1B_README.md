# Phase 1b: Relevance Weight Tuning

## Overview

Phase 1b tunes the relevance weights used for RAG similarity scoring. These weights determine how much each feature contributes when comparing retrieved positions to the current query.

**Key Insight**: We test positions with **IDENTICAL sym_hash** (same game state) but **different contexts** (komi, winrate, score_lead, etc.). By measuring how much policy and value distributions change when these contextual features are perturbed, we can determine their optimal weights.

## Methodology

### The Problem

From `claude_instructions.txt`, the relevance scoring uses weighted similarity:

```python
similarity_weights = {
    'policy': 0.40,        # Highest - determines exploration
    'winrate': 0.25,       # High - primary utility component  
    'score_lead': 0.10,    # Moderate - secondary utility
    'visit_distribution': 0.15,  # High - shows what MCTS actually preferred
    'stone_count': 0.05,   # Phase context
    'komi': 0.05,          # Must match exactly (binary)
}
```

**Question**: Are these the optimal weights?

### The Approach

1. **Group positions by sym_hash**: Find positions with identical game states
2. **Measure feature variance**: Within each group, measure how much features vary (komi, winrate, etc.)
3. **Measure policy/value stability**: How much do policy and value change despite feature perturbations?
4. **Optimize weights**: Find weights where high feature similarity predicts low policy/value change

### The Hypothesis

If two positions have:
- **Identical sym_hash** (same board state)
- **Similar features** (according to weights)

Then they should have:
- **Similar policy distributions**
- **Similar value estimates**

We maximize the correlation between feature similarity and policy/value stability.

## Requirements

### Database Format

You need a RAG database with positions that have **duplicate sym_hash** values. This typically happens when:
- Same game state reached via different move orders
- Same position analyzed with different settings (komi variations)
- Same position from different games

Format (from `claude_instructions.txt`):
```json
{
  "sym_hash": "lookup_key",
  "state_hash": "unique_id",
  "policy": [362 floats],
  "winrate": 0.547,
  "score_lead": 2.3,
  "komi": 7.5,
  "stone_count": 85,
  "child_nodes": [
    {"hash": "child1", "value": 0.52, "pUCT": 1.23, "visits": 150},
    ...
  ],
  "move_infos": [...],
  "ownership": [361 floats],
  "query_id": "query_123"
}
```

### Minimum Data Requirements

- At least 50-100 position groups (positions with same sym_hash)
- Each group should have 3+ positions
- Positions within groups should have varied features (different komi, winrate, etc.)

## Usage

### Basic Usage

```bash
cd /scratch2/f004h1v/alphago_project/datago

# Run with grid search (recommended)
./tuning/phase1/run_phase1b.sh \
    ./data/rag_database.json \
    ./tuning_results/phase1b \
    grid \
    3
```

### Arguments

1. **RAG database path** (required): Path to RAG database JSON
2. **Output directory** (default: `./tuning_results/phase1b`)
3. **Method** (default: `grid`):
   - `grid`: Exhaustive grid search (2-3 hours)
   - `optimize`: Gradient-based optimization (30-60 minutes)
4. **Min group size** (default: 3): Minimum positions per sym_hash group

### Python Interface

```bash
python tuning/phase1/phase1b_relevance_weights.py \
    --rag-database ./data/rag_database.json \
    --output-dir ./tuning_results/phase1b \
    --method grid \
    --min-group-size 3
```

## Output

### Results File: `phase1b_results.json`

```json
{
  "best_weights": {
    "policy": 0.423,
    "winrate": 0.267,
    "score_lead": 0.089,
    "visit_distribution": 0.156,
    "stone_count": 0.042,
    "komi": 0.023
  },
  "metrics": {
    "combined_correlation": 0.847,
    "policy_stability_correlation": 0.832,
    "value_stability_correlation": 0.869,
    "feature_sensitivities": {
      "winrate": 0.234,
      "score_lead": 0.089,
      "komi": 0.023,
      "stone_count": 0.042,
      "visit_distribution": 0.156
    },
    "num_groups": 127,
    "num_comparisons": 2341
  },
  "tuning_method": "grid",
  "timestamp": "2025-11-12T22:30:00"
}
```

### Key Metrics

- **combined_correlation**: Overall quality of weights (higher is better, target: >0.8)
- **policy_stability_correlation**: How well weights predict policy stability
- **value_stability_correlation**: How well weights predict value stability
- **feature_sensitivities**: Impact of each feature on predictions

### Visualization

The tuner generates plots showing:
- Feature similarity vs policy distance
- Feature similarity vs value distance
- Weight sensitivity analysis

## Expected Results

### Time Budget

- **Grid search**: 2-3 hours (tests ~375 configurations)
- **Gradient optimization**: 30-60 minutes

### Typical Outcomes

Good weights should show:
- Combined correlation > 0.80
- Policy correlation > 0.75
- Value correlation > 0.75

Common findings:
- **Policy weight** typically highest (0.35-0.45)
- **Winrate weight** second (0.20-0.30)
- **Visit distribution** moderate (0.12-0.18)
- **Score lead, stone count, komi** lower (0.02-0.10 each)

## Integration with Phase 1c

Once Phase 1b completes, use the optimized weights in Phase 1c:

```bash
# Phase 1c will read weights from phase1b_results.json
python tuning/phase1/phase1c_uncertainty_threshold.py \
    --phase1a-config ./tuning_results/phase1a/best_uncertainty_config.json \
    --phase1b-weights ./tuning_results/phase1b/phase1b_results.json \
    --rag-database ./data/rag_database.json \
    --output-dir ./tuning_results/phase1c
```

## Troubleshooting

### Error: "No position groups found"

**Problem**: Database doesn't have duplicate sym_hash values.

**Solution**: 
- Generate more positions from varied games
- Analyze same positions with different komi values
- Use transposition table to find duplicate positions

### Low Correlations (<0.6)

**Problem**: Features don't predict policy/value stability well.

**Possible causes**:
- Not enough position groups
- Positions within groups too similar (no feature variance)
- Database quality issues (inconsistent MCTS results)

**Solution**:
- Collect more diverse position groups
- Increase feature perturbations
- Verify MCTS consistency

### Memory Issues

**Problem**: Large database causes OOM errors.

**Solution**:
- Use `--min-group-size 5` to reduce total comparisons
- Sample subset of position groups
- Run on machine with more RAM

## Theory

### Why This Works

1. **Sym_hash identifies game state**: If sym_hash matches, the board position is identical
2. **Context varies**: Different komi, analysis depth, etc. create feature variations
3. **Policy/value stability**: Good MCTS should produce similar policy/value for same position
4. **Weight optimization**: Features that vary without affecting policy/value get lower weights

### Mathematical Formulation

We optimize weights $w$ to maximize:

$$\text{Correlation}(\text{FeatureSim}_w, \text{PolicyValueStability})$$

Where:
- $\text{FeatureSim}_w = \sum_i w_i \cdot \text{sim}(f_i^{(1)}, f_i^{(2)})$
- $\text{PolicyValueStability} = -\text{KL}(\pi^{(1)} || \pi^{(2)}) - |V^{(1)} - V^{(2)}|$

### Assumptions

1. Same sym_hash implies same optimal policy (under perfect MCTS)
2. Feature perturbations should not drastically change policy if sim is high
3. MCTS results are reasonably consistent (converged search)

## Next Steps

After Phase 1b:
1. ✅ Phase 1a: Uncertainty function optimized
2. ✅ Phase 1b: Relevance weights optimized
3. → **Phase 1c**: Uncertainty threshold tuning with RAG
4. → Phase 2: Gameplay integration and validation
