# Phase 1 Complete Pipeline

## Overview

Phase 1 consists of three sequential tuning steps that must be completed in order:

1. **Phase 1a**: Uncertainty Function Parameters
2. **Phase 1b**: Relevance Comparison Weights  
3. **Phase 1c**: Uncertainty Threshold

Each phase depends on results from previous phases.

## Phase 1a: Uncertainty Function Parameters

**Goal**: Optimize w1, w2, and phase_function to predict model error

**Input**: 
- Shallow MCTS database (800 visits)
- Deep MCTS database (5000+ visits)

**Output**:
- `tuning_results/phase1a/best_uncertainty_config.json`
  - w1, w2 weights
  - phase_function_type and coefficients

**Command**:
```bash
python tuning/phase1/phase1_uncertainty_tuning.py \
    --shallow-db ./data/shallow_mcts_db.json \
    --deep-db ./data/deep_mcts_db.json \
    --method grid \
    --output-dir ./tuning_results/phase1a
```

**Time**: 4-8 hours

## Phase 1b: Relevance Comparison Weights

**Goal**: Optimize similarity weights for RAG retrieval

**Input**:
- RAG database with duplicate sym_hash entries

**Output**:
- `tuning_results/phase1b/phase1b_results.json`
  - Optimized weights for: policy, winrate, score_lead, visit_distribution, stone_count, komi

**Command**:
```bash
python tuning/phase1/phase1b_relevance_weights.py \
    --rag-database ./data/rag_database.json \
    --method grid \
    --output-dir ./tuning_results/phase1b
```

**Time**: 2-4 hours

## Phase 1c: Uncertainty Threshold

**Goal**: Find optimal threshold for when to query RAG

**Input**:
- Phase 1a config (uncertainty parameters)
- Phase 1b weights (relevance weights)
- RAG database (for retrieval)
- Ground truth database (for percentile estimation)

**Output**:
- `tuning_results/phase1c/phase1c_results.json`
  - Best uncertainty threshold percentile
  - Expected win rate and RAG usage statistics

**Command**:
```bash
python tuning/phase1/phase1c_uncertainty_threshold.py \
    --phase1a-config ./tuning_results/phase1a/best_uncertainty_config.json \
    --phase1b-weights ./tuning_results/phase1b/phase1b_results.json \
    --rag-database ./data/rag_database.json \
    --ground-truth-db ./data/shallow_mcts_db.json \
    --output-dir ./tuning_results/phase1c \
    --num-games 100
```

**Time**: 8-10 hours

## Complete Workflow

```bash
# Step 1: Generate databases (if needed)
# TODO: Replace with actual KataGo analysis

# Step 2: Run Phase 1a
python tuning/phase1/phase1_uncertainty_tuning.py \
    --shallow-db ./data/shallow_mcts_db.json \
    --deep-db ./data/deep_mcts_db.json \
    --method grid \
    --output-dir ./tuning_results/phase1a

# Step 3: Run Phase 1b
python tuning/phase1/phase1b_relevance_weights.py \
    --rag-database ./data/rag_database.json \
    --method grid \
    --output-dir ./tuning_results/phase1b

# Step 4: Run Phase 1c
python tuning/phase1/phase1c_uncertainty_threshold.py \
    --phase1a-config ./tuning_results/phase1a/best_uncertainty_config.json \
    --phase1b-weights ./tuning_results/phase1b/phase1b_results.json \
    --rag-database ./data/rag_database.json \
    --ground-truth-db ./data/shallow_mcts_db.json \
    --output-dir ./tuning_results/phase1c
```

**Total Time**: 14-22 hours

## Dependencies Between Phases

```
Phase 1a (Uncertainty Function)
    ↓ produces best_uncertainty_config.json
    ↓
Phase 1b (Relevance Weights)
    ↓ produces phase1b_results.json
    ↓
Phase 1c (Uncertainty Threshold)
    ↓ uses both configs above
    ↓
Final Phase 1 Configuration
```

## Output Files Summary

After completing all phases, you will have:

1. **Phase 1a Output**:
   - `best_uncertainty_config.json` - w1, w2, phase_function

2. **Phase 1b Output**:
   - `phase1b_results.json` - similarity weights

3. **Phase 1c Output**:
   - `phase1c_results.json` - optimal threshold

These three files together define the complete Phase 1 configuration for production use.
