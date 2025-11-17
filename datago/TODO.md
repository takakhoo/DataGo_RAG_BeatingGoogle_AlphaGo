# RAG-MCTS-AlphaGo TODO

**Last Updated:** November 16, 2025  
**Project Status:** Functional but incomplete - System uses real KataGo NN outputs and achieves 80% win rate, but RAG policy blending is NOT implemented

---

## Critical Issues (Priority 1) 🔴

### 1. **PROBABILITY DISTRIBUTION BLENDING NOT IMPLEMENTED**
**Status:** ❌ **CRITICAL GAP**

The system currently retrieves cached analyses from RAG but does NOT blend policies. This means:
- Config has `beta: 0.4` but it's unused
- Retrieved `ctx['policy']` is fetched but never blended with network policy
- Bot wins purely from adaptive visit counts (800 → 2000), not knowledge reuse
- This is NOT a true "RAG-augmented" system - just storage with cache optimization

**What's needed:**
```python
# In generate_move() after line 683 (when cache hit found):
if best_ctx:
    # Extract policies
    network_policy = analysis['policy']  # From current 800-visit search
    rag_policy = best_ctx['policy']      # From cached 2000-visit search
    
    # Blend policies
    beta = self.config['blending']['beta']  # 0.4
    blended_policy = (1 - beta) * network_policy + beta * rag_policy
    blended_policy /= blended_policy.sum()  # Normalize
    
    # Use blended policy to:
    # Option A: Rerank moves by blended probabilities
    # Option B: Force explore top RAG move if different from network move
    # Option C: Modify MCTS priors (requires deeper integration)
```

**Files to modify:**
- `run_datago_recursive_match.py` (lines 678-688 in `generate_move()`)
- Add `blend_policies()` helper function
- Use `reranking_alpha` and `reranking_gamma` for move reranking

**Expected impact:**
- Current: 80% win rate (2000 vs 800 visits = brute force)
- With blending: 85-95% win rate (knowledge reuse + adaptive visits)

---

## High Priority (Priority 2) 🟡

### 2. **Force Exploration of RAG Moves**
**Status:** ⚠️ Partially configured

Config has `force_exploration_top_n: 1` but unclear if implemented.

**What's needed:**
- When RAG returns different move than network, force explore it
- Track whether forced moves improve outcomes
- Adjust exploration rate based on RAG confidence

**Files:** `run_datago_recursive_match.py`

---

### 3. **Move Quality Metrics**
**Status:** ❌ Not implemented

Currently only track win rate, but need:
- Move agreement rate (RAG vs network)
- Move quality score (compare to 10k-visit ground truth)
- Position complexity correlation with RAG effectiveness
- Cache hit rate vs move outcome

**Files:** Add to `run_datago_recursive_match.py` statistics tracking

---

### 4. **Reachability-Based Neighbor Reranking**
**Status:** ⚠️ Configured but not fully implemented

Config has `reranking_alpha: 0.7` and `reranking_gamma: 0.3` but:
- No policy similarity calculation between current and retrieved positions
- No parent/child structural boost
- Simple exact hash matching only

**What's needed:**
```python
def rerank_neighbors(neighbors, current_policy, current_position):
    for neighbor in neighbors:
        # Reachability: policy similarity
        r = cosine_similarity(current_policy, neighbor['policy'])
        
        # Structural boost: parent/child relationship
        s = 2.0 if is_parent_child(current_position, neighbor) else 1.0
        
        # Combined weight
        alpha = 0.7
        gamma = 0.3
        w = alpha * r + gamma * s
        
        neighbor['rerank_score'] = w
    
    return sorted(neighbors, key=lambda x: x['rerank_score'], reverse=True)
```

**Files:** Add to `src/memory/` or `run_datago_recursive_match.py`

---

## Original Design Goals (For Reference)

### High-level Architecture
- Use entropy of policy distribution to gate retrieval at expansion/simulation nodes
- Retrieve K nearest neighbors (ANN), rerank by reachability + structural boost
- **Blend retrieved priors with network priors during expansion** (NOT IMPLEMENTED)
- Store complex positions during simulations
- Prune unreachable states when memory exceeds M_max

---

## Medium Priority (Priority 3) 🟢

### 5. **Uncertainty Threshold Tuning**
**Status:** ✅ Mostly complete (fixed for real NN)

- Current: `0.150` (adjusted from `0.370` for real NN outputs)
- Real NN uncertainty range: 0.05-0.21
- Synthetic data range was: 0.35-0.40

**Potential improvements:**
- Dynamic threshold based on game phase
- Separate thresholds for query vs store
- Track uncertainty distribution over games

**Files:** `src/bot/config.yaml`, `run_datago_recursive_match.py`

---

### 6. **Memory Pruning and Lifecycle Management**
**Status:** ❌ Not implemented

Currently stores indefinitely (1070+ positions). Need:
- Trigger pruning when memory > M_max (e.g., 5000 entries)
- Prune by combined score: importance + reachability
- Track retrieval frequency and outcome correlation
- Remove "unreachable" or low-value positions

**What's needed:**
```python
def prune_memory(self, max_size=5000, prune_fraction=0.15):
    if len(self.position_db) > max_size:
        # Score each entry
        scores = []
        for sym_hash, pos_ctx in self.position_db.items():
            importance = pos_ctx.retrieval_count * 0.5 + pos_ctx.win_rate * 0.5
            reachability = estimate_avg_reachability(pos_ctx)
            score = 0.6 * importance + 0.4 * reachability
            scores.append((sym_hash, score))
        
        # Remove bottom 15%
        scores.sort(key=lambda x: x[1])
        to_remove = int(len(scores) * prune_fraction)
        for sym_hash, _ in scores[:to_remove]:
            del self.position_db[sym_hash]
            # Also remove from ANN index
```

**Files:** Add to `run_datago_recursive_match.py`

---

### 7. **Embedding Improvements**
**Status:** ⚠️ Basic implementation

Current: Uses first 64 elements of policy as embedding
Issues:
- No learned projection
- No incorporation of value/winrate
- No game phase information

**Potential improvements:**
- Add value scalar to embedding
- Include game phase (move number / stones on board)
- Use learned MLP to project to better space
- Experiment with different embedding dimensions

**Files:** `run_datago_recursive_match.py` (store_position method)

---

### 8. **Performance Optimization**
**Status:** ⚠️ Partially optimized

Current optimizations:
- ✅ Cache checking before deep search (saves ~1-2s per hit)
- ✅ Exact hash matching (O(1) lookup)

**Potential improvements:**
- Cache retrieval responses per state fingerprint
- Batch ANN queries when possible
- Limit maximum retrievals per search
- Profile critical paths (genmove_analyze, RAG query, deep search)

**Files:** `run_datago_recursive_match.py`, `src/memory/`

---

## Completed Work ✅

### Already Implemented:
1. ✅ **Real Neural Network Integration**
   - Replaced synthetic `np.random.dirichlet()` with `kata-genmove_analyze`
   - Fixed GTP parser to handle multi-move info lines
   - Extracts real policy, winrate, priors, visits from KataGo

2. ✅ **Uncertainty Calculation**
   - Normalized entropy of policy distribution
   - Visit dispersion and top-move confidence
   - Adjusted threshold from 0.370 → 0.150 for real NN

3. ✅ **RAG Storage and Retrieval**
   - Position database with exact hash matching
   - Stores 1070+ positions with 2000-visit analyses
   - Context selection by game phase and uncertainty

4. ✅ **Recursive Deep Search**
   - Adaptive visit counts (800 standard, 2000 deep)
   - Recursion up to depth 3
   - Child position analysis with caching

5. ✅ **Cache Optimization**
   - Check cache BEFORE expensive deep search
   - "Child cache hit" optimization saves redundant searches
   - 660% cache hit rate (1 query → 6.6 uses)

6. ✅ **Performance Validation**
   - 80% win rate vs pure KataGo (8-2-2 record)
   - 2.3x more efficient than synthetic data approach
   - Average 2446 visits/move (vs 6029 with synthetic)

7. ✅ **Documentation**
   - SYNTHETIC_VS_REAL_NN_COMPARISON.md
   - BOT_IMPLEMENTATION.md
   - IMPLEMENTATION_SUMMARY.md

---

## Actionable Implementation Plan

### Phase 1: Implement Probability Blending (CRITICAL)
**Estimated effort:** 2-4 hours

1. Add `blend_policies()` helper function
2. Modify `generate_move()` to blend when cache hit
3. Implement move reranking by blended probabilities
4. Add logging for blend weight and move changes
5. Test with 10-game match

**Success criteria:**
- Blended policy used in move selection
- Log shows "Using blended policy (beta=0.4)"
- Win rate improves or stays at 80%+

---

### Phase 2: Move Quality Metrics
**Estimated effort:** 2-3 hours

1. Track move agreement (RAG vs network)
2. Compare moves to high-visit ground truth
3. Analyze position complexity vs RAG effectiveness
4. Add metrics to match summary

**Success criteria:**
- Statistics show when RAG improves moves
- Identify position types where RAG helps most

---

### Phase 3: Neighbor Reranking
**Estimated effort:** 3-5 hours

1. Implement policy similarity (cosine)
2. Add parent/child relationship detection
3. Combine reachability + structural boost
4. Rerank neighbors before aggregation

**Success criteria:**
- Top neighbors have high policy similarity
- Parent/child positions prioritized
- Aggregated policy more relevant

---

### Phase 4: Memory Management
**Estimated effort:** 4-6 hours

1. Implement pruning trigger (max_size)
2. Score entries by importance + reachability
3. Remove bottom N%
4. Update ANN index
5. Add periodic snapshots

**Success criteria:**
- Memory stays under max_size
- High-value positions retained
- Low-value positions pruned

---

## Configuration Reference

### Current Settings (src/bot/config.yaml)
```yaml
uncertainty_threshold: 0.150  # Adjusted for real NN
deep_mcts:
  max_visits: 2000
katago:
  visits: 800
blending:
  beta: 0.4                   # ⚠️ NOT USED YET
  reranking_alpha: 0.7        # ⚠️ NOT USED YET
  reranking_gamma: 0.3        # ⚠️ NOT USED YET
  top_n_neighbors: 1
recursion:
  max_recursion_depth: 3
  force_exploration_top_n: 1  # ⚠️ UNCLEAR IF IMPLEMENTED
```

---

## Testing Protocol

### Before Each Change:
1. Run baseline match: `./test_recursive_search.sh`
2. Record win rate, visits/move, RAG activation rate
3. Check logs for errors

### After Each Change:
1. Run same test
2. Compare metrics
3. Verify improvement or equal performance
4. Document findings

### Key Metrics:
- Win rate (target: 80%+)
- Average visits per move (lower = better efficiency)
- RAG activation rate (5% is good for threshold 0.150)
- Cache hit rate (660% is excellent)
- Move agreement rate (new)
- Move quality score (new)

---

## Original Design Goals (Reference)

### High-level Architecture (from initial TODO)
- ✅ Use entropy to gate retrieval
- ✅ Store complex positions in RAG database
- ✅ Retrieve K nearest neighbors with ANN
- ⚠️ Rerank by reachability + structural boost (PARTIAL)
- ❌ **Blend retrieved priors with network priors** (CRITICAL MISSING)
- ❌ Prune unreachable states when memory exceeds M_max

### Implementation Approach

- ✅ Non-invasive Python prototype (current implementation)
- ⚠️ Future: C++ integration for lower latency

### Remaining Original Design Components

**Reachability Estimator Options:**
- A) Parent/child exact check + move-distance heuristic (cheap)
- B) Policy-vector similarity (cosine) - **RECOMMENDED NEXT**
- C) Short policy-guided rollouts (expensive)
- D) Learned model (requires offline training)

**Blending Strategy:**
```python
# Expansion (NOT IMPLEMENTED YET)
if entropy(P) > H_trigger:
    neighbors = memory.retrieve(embed, K)
    reranked = rerank(neighbors, current_state)  # Use alpha, gamma
    P_nn = map_neighbors_to_actions(reranked)
    P_blend = normalize((1 - beta) * P_net + beta * P_nn)
    use P_blend for move selection
```

**Simulation Storing (BASIC VERSION IMPLEMENTED):**
- Current: Store when uncertainty > threshold
- Missing: Store when |value_net - rollout_value| > disagreement threshold
- Missing: Batch buffer flushing

---

## File Structure

### Current Implementation:
- ✅ `src/bot/gtp_controller.py` - GTP protocol, kata-genmove_analyze
- ✅ `src/gating/gate.py` - Entropy calculation and gates
- ✅ `src/memory/rag_memory.py` - ANN index, retrieval
- ✅ `run_datago_recursive_match.py` - Main bot logic
- ✅ `src/bot/config.yaml` - Configuration

### Missing/Incomplete:
- ❌ `src/blend/blend.py` - Policy blending functions
- ⚠️ `src/memory/reachability.py` - Reachability estimation (basic only)
- ⚠️ Move reranking logic (configured but not used)

---

## Known Issues

### 1. **No Policy Blending** (CRITICAL)
- **Impact:** Bot doesn't use RAG knowledge for move improvement
- **Current behavior:** Cache hit → use cached move directly
- **Should be:** Cache hit → blend policies → improve move selection

### 2. **Uncertainty Calculation May Need Refinement**
- Current formula works but could incorporate:
  - Value uncertainty
  - Score lead uncertainty
  - Historical position difficulty

### 3. **Memory Growth Unbounded**
- Currently 1070 positions stored, no pruning
- Will eventually need pruning at ~5000+ entries

### 4. **No Move Outcome Tracking**
- Don't track whether RAG moves win more often
- Can't validate RAG effectiveness empirically

### 5. **GTP Parser Assumes Specific Format**
- Works for KataGo v1.16.4
- May break with different versions
- No fallback for malformed responses

---

## Performance Benchmarks

### Current System (Nov 16, 2025):
- **Win rate:** 80% (8-2-2 vs KataGo)
- **Avg visits/move:** 2446 (vs 6029 synthetic)
- **RAG activation:** 5.1% (23 queries / 454 moves)
- **Cache hit rate:** 660% (152 hits / 23 queries)
- **Avg uncertainty:** 0.080
- **Unique positions:** 1070
- **Efficiency ratio:** 0.149 (vs 0.262 synthetic)

### Target Metrics with Blending:
- **Win rate:** 85-95%
- **Avg visits/move:** 2000-2400 (maintain or improve)
- **RAG activation:** 5-10%
- **Cache hit rate:** 500%+
- **Move quality:** Measurable improvement in complex positions

---

## Development Roadmap

### Immediate (Next Session):
1. Implement policy blending in `generate_move()`
2. Add logging for blend operations
3. Run 10-game test match
4. Validate win rate improvement

### Short Term (1-2 weeks):
1. Implement move quality metrics
2. Add neighbor reranking by policy similarity
3. Track RAG move outcomes
4. Create ablation study framework

### Medium Term (1 month):
1. Implement memory pruning
2. Add embedding improvements
3. Optimize performance bottlenecks
4. Expand test coverage

### Long Term (Future):
1. Learned reachability model
2. C++ integration for production
3. Multi-opponent training
4. Adaptive parameter tuning

---

## Questions to Resolve

1. **When forced to use RAG move, how much to trust it?**
   - Current: Use cached move directly
   - Should we blend confidence with network confidence?

2. **Should we store losing positions?**
   - Pro: Learn what NOT to do
   - Con: Pollutes memory with bad patterns

3. **How to handle symmetries in embedding space?**
   - Current: Uses sym_hash for exact match
   - Could use rotation-invariant embeddings

4. **Should recursion depth affect blending weight?**
   - Deeper = less reliable network analysis?
   - Could increase beta at higher depths

5. **How to evaluate RAG effectiveness offline?**
   - Compare to multi-visit ground truth?
   - Measure knowledge transfer across games?

---

## References

- **SYNTHETIC_VS_REAL_NN_COMPARISON.md** - Performance analysis
- **BOT_IMPLEMENTATION.md** - Architecture details
- **IMPLEMENTATION_SUMMARY.md** - System overview
- **src/bot/config.yaml** - Current configuration

---

**Next Immediate Action:** Implement probability distribution blending in `generate_move()` to actually use the RAG-retrieved policies for move improvement.
