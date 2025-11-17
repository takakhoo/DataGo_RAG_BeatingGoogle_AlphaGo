# RAG-Enhanced MCTS Architecture

## Overview
This document describes the new custom MCTS architecture that enables RAG-based policy modification during tree search.

## Architecture Components

### 1. Network Evaluator (`src/mcts/network_evaluator.py`)
**Purpose:** Extract raw neural network policy and value estimates from KataGo without MCTS.

**Key Method:**
```python
def evaluate(board_state, moves) -> (policy_dict, value):
    # Query KataGo with maxVisits=1 to get raw network output
    # Returns: Dict[str, float] policy + float value
```

**Why:** KataGo's standard analysis mixes network evaluation with MCTS search. To modify the policy *before* MCTS, we need access to the raw network output.

### 2. Custom MCTS (`src/mcts/custom_mcts.py`)
**Purpose:** Implement PUCT-based Monte Carlo Tree Search that accepts modified policy priors.

**Key Method:**
```python
def search(root_state, num_visits, modified_prior=None):
    # modified_prior: Optional Dict[str, float] to override network policy
    # Returns: Dict[str, float] visit-based move probabilities
```

**Why:** KataGo's internal MCTS is a black box. To use RAG-blended policies in the search, we need our own MCTS implementation.

### 3. RAG Integration Pipeline

#### Step-by-Step Flow in `generate_move()`:

**Step 1: Get Raw Network Evaluation**
```python
network_policy, estimated_value = self._evaluate_position(board, moves)
# network_policy: Dict[str, float] from raw network (no MCTS yet)
```

**Step 2: Run Shallow MCTS (No Modifications)**
```python
move_probs, value, stats = self._run_mcts_search(
    board, moves, 
    num_visits=800,
    modified_prior=None  # Use network policy as-is
)
```

**Step 3: Compute Uncertainty**
```python
uncertainty = self.compute_uncertainty(
    network_policy,    # Dict from Step 1
    mcts_stats         # From Step 2
)
```

**Step 4: Query RAG (if uncertain)**
```python
if uncertainty >= threshold:
    rag_hit, neighbor, relevance, use_precomputed = self.query_rag(...)
    
    if rag_hit:
        if use_precomputed:  # relevance >= 0.95
            # Use RAG's stored optimal move directly
            return neighbor.best_moves[0]
        else:
            # **KEY INTEGRATION: Blend network + RAG priors**
            blended_prior = self.blend_with_rag(network_policy, neighbor)
            
            # Re-run MCTS with modified prior!
            move_probs, value, stats = self._run_mcts_search(
                board, moves,
                num_visits=800,
                modified_prior=blended_prior  # RAG-enhanced policy!
            )
```

## Policy Blending Formula

From `src/blend/blend.py`:

```python
P_blended(move) = (1 - β) × P_network(move) + β × P_rag(move)
```

Where:
- `β = 0.4` (from config: `blending.beta`)
- `P_network`: Raw KataGo network policy (Step 1)
- `P_rag`: Retrieval prior from RAG neighbor's move distribution
- `P_blended`: Final prior used in custom MCTS

## Why This Architecture?

### Problem with Original Design:
```
KataGo Analysis (Black Box)
  ├─ Network Evaluation (hidden)
  ├─ MCTS (hidden, uses network policy internally)
  └─ Output: Final move probabilities
```
**Issue:** Cannot inject RAG-modified policy into MCTS search!

### New Design:
```
1. KataGo Network Evaluation (maxVisits=1)
   ↓
2. [Optional] Blend with RAG prior
   ↓
3. Custom MCTS (uses modified prior)
   ↓
4. Output: RAG-enhanced move probabilities
```
**Benefit:** RAG knowledge influences the entire MCTS tree search, not just the final move selection.

## Config Parameters Used

```yaml
blending:
  beta: 0.4                    # Network vs RAG weight
  top_n_neighbors: 1           # 1-NN strategy
  top_n_moves: 5               # Blend top 5 moves
  reranking_alpha: 0.3         # Reachability weight
  reranking_gamma: 0.7         # Structural similarity weight

rag_query:
  uncertainty_threshold: 0.65  # Query RAG when uncertainty >= 0.65
  relevance_threshold: 0.95    # Use precomputed move if relevance >= 0.95

mcts:
  c_puct: 1.5                  # PUCT exploration constant
  temperature: 1.0             # Move selection temperature
```

## Key Files Modified

1. **`src/bot/datago_bot.py`** (Main bot logic)
   - `_evaluate_position()`: Get raw network policy
   - `_run_mcts_search()`: Run custom MCTS with optional modified_prior
   - `blend_with_rag()`: Blend network + RAG policies
   - `generate_move()`: Orchestrate full pipeline
   - `compute_uncertainty()`: Updated for new Dict[str, float] format
   - `deep_mcts_search()`: Returns (move_probs, value, stats) tuple

2. **`src/mcts/network_evaluator.py`** (NEW)
   - `KataGoNetworkEvaluator.evaluate()`: Extract raw network policy/value

3. **`src/mcts/custom_mcts.py`** (NEW)
   - `CustomMCTS.search()`: PUCT-based MCTS with modifiable priors
   - `MCTSNode`: Tree node structure

## Advantages Over Original Architecture

1. **RAG Influence on Search:** RAG priors guide the entire MCTS exploration, not just final selection
2. **Explicit Policy Modification:** Clear separation of network evaluation → blending → search
3. **Configurable MCTS:** Control c_puct, temperature, Dirichlet noise independently
4. **Convergence Control:** Early stopping based on policy/value convergence (deep search)
5. **Debuggability:** Can inspect network policy, blended policy, and MCTS output separately

## Testing Checklist

- [ ] Network evaluator returns correct policy/value format
- [ ] Custom MCTS runs without errors (with/without modified_prior)
- [ ] Blending produces valid probability distribution (sums to 1.0)
- [ ] RAG query works with Dict[str, float] policy
- [ ] High relevance (≥0.95) uses precomputed move
- [ ] Low relevance triggers policy blending + re-search
- [ ] Uncertainty computation handles new data structures
- [ ] Deep MCTS convergence checking works
- [ ] Move selection produces valid GTP coordinates
- [ ] Full game completes without crashes

## Next Steps

1. **Implement Position Hashing:** Replace "hash_placeholder" with actual zobrist/canonical hashing
2. **Test RAG Pipeline:** Verify end-to-end flow with real games
3. **Tune Hyperparameters:** Adjust β, c_puct, thresholds based on performance
4. **Optimize Performance:** Profile MCTS search, consider batch evaluation
5. **Add Logging:** Detailed logs for policy distributions at each stage
