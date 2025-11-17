# Recursive Deep Search Architecture

## Overview

This document describes the recursive deep search system for DataGo, where complex positions trigger deeper, augmented MCTS searches that can recursively analyze child positions.

## Algorithm

### Main Workflow

```
1. During MCTS, generate candidate moves
2. Calculate uncertainty for current position
3. If uncertainty > deep_search_threshold:
   a. Create sym_hash for position
   b. Query RAG database for exact sym_hash match
   c. If exact match found:
      - Retrieve all stored contexts for this position
      - Select best context based on game phase & recency
      - Use cached analysis (skip deep search)
   d. If no match found:
      - Run deep augmented MCTS (more visits)
      - Store results in RAG database
4. During deep search, recursively apply same logic to promising children
5. Respect max_recursion_depth limit
```

### Multi-Context Storage

**Key Innovation:** A single `sym_hash` can store multiple game contexts with different parameters:

```python
PositionContext:
  sym_hash: "hash_123456"
  contexts: [
    {
      move: "D4",
      uncertainty: 0.42,
      deep_visits: 10000,
      policy: [...],
      winrate: 0.52,
      score_lead: 2.5,
      game_phase: 45,  # stones on board
      metadata: {
        recursion_depth: 1,
        move_number: 23,
      },
      timestamp: 1700150000
    },
    {
      move: "D4",
      uncertainty: 0.38,
      deep_visits: 5000,
      policy: [...],
      winrate: 0.48,
      score_lead: -1.2,
      game_phase: 150,  # different game phase
      metadata: {...},
      timestamp: 1700151000
    }
  ]
```

**Context Selection:** When retrieving a cached position, select the most relevant context based on:
- Game phase similarity (50% weight)
- Deep search quality (30% weight)
- Recency (20% weight)

### Recursion Logic

```python
def run_deep_augmented_search(position, uncertainty, depth):
    if depth >= max_recursion_depth:
        return standard_search(position)
    
    # Run deep MCTS on this position
    result = deep_mcts(position, deep_visits=10000)
    
    # Analyze promising children
    for child in result.top_children:
        child_uncertainty = calculate_uncertainty(child)
        
        if child_uncertainty > threshold:
            # Try to retrieve from cache
            cached = query_rag_exact(child.sym_hash)
            
            if cached:
                # Use cached analysis
                child.analysis = cached.get_best_context()
            else:
                # Recursive deep search
                child.analysis = run_deep_augmented_search(
                    child.position,
                    child_uncertainty,
                    depth + 1
                )
                # Store child's analysis
                store_position(child.sym_hash, child.analysis)
    
    return result
```

### Thresholds

From `config.yaml`:

```yaml
rag_query:
  uncertainty_threshold: 0.35  # Trigger RAG lookup

recursion:
  max_recursion_depth: 2  # Prevent infinite recursion
  
deep_mcts:
  max_visits: 10000  # Deep search budget
```

**Deep search threshold:** `uncertainty_threshold + 0.05 = 0.40`
- Positions with uncertainty > 0.40 trigger deep search
- This ensures only truly complex positions incur the overhead

## Implementation Details

### Position Database

```python
# Global position storage
position_db: Dict[str, PositionContext] = {}

# Query by exact sym_hash
def query_rag_exact(sym_hash: str) -> Optional[PositionContext]:
    return position_db.get(sym_hash)

# Store new context for existing or new position
def store_position(sym_hash, move, uncertainty, analysis):
    if sym_hash not in position_db:
        position_db[sym_hash] = PositionContext(sym_hash)
    
    position_db[sym_hash].add_context(
        move=move,
        uncertainty=uncertainty,
        deep_visits=analysis['deep_visits'],
        policy=analysis['policy'],
        winrate=analysis['winrate'],
        score_lead=analysis['score_lead'],
        game_phase=stones_on_board,
        metadata=analysis['metadata']
    )
```

### ANN Index Integration

The position database is integrated with the ANN index for fast retrieval:

1. **Exact lookup:** Check `position_db` dict directly (O(1))
2. **Approximate lookup:** Use ANN index for nearest neighbors
3. **Storage:** Add to both `position_db` and ANN index

```python
# When storing a new unique position
if sym_hash not in position_db:
    # Add to dictionary
    position_db[sym_hash] = PositionContext(sym_hash)
    
    # Add to ANN index for similarity search
    embedding = create_embedding(analysis['policy'])
    entry = MemoryEntry(
        embed=embedding,
        canonical_board=sym_hash,
        best_moves=[{'move': move, 'prob': 1.0}],
        importance=uncertainty
    )
    rag_index.add(entry)
```

### Statistics Tracking

The system tracks comprehensive statistics:

```python
stats = {
    'moves': 0,                    # Total moves generated
    'rag_queries': 0,              # RAG lookups attempted
    'rag_hits': 0,                 # Positions found in cache
    'exact_matches': 0,            # Exact sym_hash matches used
    'deep_searches': 0,            # Deep MCTS runs
    'recursive_searches': 0,       # Searches at depth > 0
    'positions_stored': 0,         # Unique positions
    'contexts_added': 0,           # Total contexts (all positions)
    'total_uncertainty': 0.0,      # For averaging
}
```

**Key metrics:**
- **Cache hit rate:** `rag_hits / rag_queries`
- **Contexts per position:** `contexts_added / positions_stored`
- **Recursion usage:** `recursive_searches / deep_searches`

## Expected Behavior

### Scenario 1: Novel Complex Position

```
Move 15: Calculate uncertainty = 0.45
→ Complex position detected (unc=0.45)
→ Create sym_hash: hash_789123
→ Query RAG: Not found
→ No match found: hash_789123, running deep search
  → Deep search (depth=0, unc=0.45)
  → Analyzing child positions...
    → Child 1: unc=0.38 (below threshold, skip)
    → Child 2: unc=0.43 (above threshold)
      → Child cache miss, running recursive search
        → Deep search (depth=1, unc=0.43)
        → Stored child position in RAG
  → Store main position in RAG
Move 15: DataGo plays D4 [unc=0.45, DEEP, 1250ms]
```

### Scenario 2: Cached Position

```
Move 45: Calculate uncertainty = 0.44
→ Complex position detected (unc=0.44)
→ Create sym_hash: hash_789123
→ Query RAG: Found!
→ Exact match found: hash_789123 (2 contexts)
→ Using cached analysis (deep_visits=10000)
Move 45: DataGo plays D4 [unc=0.44, CACHED(2), 45ms]
```

### Scenario 3: Deep Recursion

```
Move 80: Calculate uncertainty = 0.46
→ Complex position detected (unc=0.46)
→ No match found, running deep search
  → Deep search (depth=0, unc=0.46)
    → Child 1: unc=0.42
      → Recursive search for child
        → Deep search (depth=1, unc=0.42)
          → Grandchild 1: unc=0.40
            → Recursive search for grandchild
              → Deep search (depth=2, unc=0.40)
              → Max recursion depth reached
Move 80: DataGo plays F5 [unc=0.46, DEEP, 3200ms]
```

## Performance Characteristics

### Time Complexity

**Without RAG (standard search):**
- Per move: ~50-150ms (800 visits)

**With RAG (cache hit):**
- Per move: ~45-80ms (cached retrieval)
- **Speedup:** ~1.5-2x faster than running deep search

**With RAG (cache miss, depth 0):**
- Per move: ~800-1500ms (10000 visits)
- **Overhead:** ~10-15x slower than standard (but better quality)

**With RAG (cache miss, depth 1):**
- Per move: ~2000-4000ms (deep search + child searches)
- **Overhead:** ~25-40x slower (but explores search tree deeply)

**With RAG (cache miss, depth 2):**
- Per move: ~5000-10000ms (deep recursion)
- **Overhead:** ~60-100x slower (maximum quality)

### Memory Usage

**Per position context:**
- Policy vector: ~1.4 KB (361 floats)
- Metadata: ~0.5 KB
- **Total:** ~2 KB per context

**Database size estimates:**
- 100 positions × 2 contexts avg = ~400 KB
- 1000 positions × 3 contexts avg = ~6 MB
- 10000 positions × 3 contexts avg = ~60 MB

**Very manageable:** Even with 10k positions and multiple contexts, total memory is <100 MB

### Cache Performance

**Expected hit rates:**
- **Early game (moves 1-50):** 10-20% (few positions seen)
- **Mid game (moves 51-150):** 30-50% (more revisits)
- **Late game (moves 151+):** 40-60% (common endgame patterns)

**Context growth:**
- Same position encountered in different game phases → multiple contexts
- Expected: 1.5-3.0 contexts per unique position

## Configuration

### Tuning Parameters

From `config.yaml`:

```yaml
# Core thresholds
rag_query:
  uncertainty_threshold: 0.35  # When to query RAG
  
# Deep search settings
deep_mcts:
  max_visits: 10000            # Deep search budget
  policy_convergence_threshold: 0.05
  value_convergence_threshold: 0.02
  
# Recursion control
recursion:
  max_recursion_depth: 2       # Prevent runaway recursion
  force_exploration_top_n: 1   # Moves to force explore
  
# Context selection weights
# (implicit in PositionContext.get_best_context)
context_selection:
  phase_weight: 0.5            # Game phase similarity
  quality_weight: 0.3          # Deep search quality
  recency_weight: 0.2          # Temporal relevance
```

### Recommended Settings

**For fast gameplay (minimize latency):**
```yaml
rag_query:
  uncertainty_threshold: 0.45  # Higher threshold = fewer deep searches
deep_mcts:
  max_visits: 5000             # Faster deep search
recursion:
  max_recursion_depth: 1       # Minimal recursion
```

**For high-quality play (maximize strength):**
```yaml
rag_query:
  uncertainty_threshold: 0.35  # Lower threshold = more deep searches
deep_mcts:
  max_visits: 20000            # Thorough deep search
recursion:
  max_recursion_depth: 3       # Deeper recursion
```

**For balanced play (default):**
```yaml
rag_query:
  uncertainty_threshold: 0.35
deep_mcts:
  max_visits: 10000
recursion:
  max_recursion_depth: 2
```

## Testing

### Run Test Match

```bash
./test_recursive_search.sh
```

This runs a 40-move game and shows:
- When deep searches are triggered
- Recursive depth for each search
- Cache hits vs misses
- Database growth statistics

### Expected Output

```
Move 12: DataGo (Black) thinking...
  → Complex position detected (unc=0.42)
  → No match found: hash_456789, running deep search
    → Deep search (depth=0, unc=0.42)
      → Analyzing child positions...
    → Deep search (depth=1, unc=0.39)
  → Stored position in RAG
Move 12: DataGo plays D4 [unc=0.42, DEEP, 1150ms]

Move 24: DataGo (Black) thinking...
  → Complex position detected (unc=0.44)
  → Exact match found: hash_456789 (1 contexts)
  → Using cached analysis (deep_visits=10000)
Move 24: DataGo plays F5 [unc=0.44, CACHED(1), 52ms]

Game 1 Statistics:
  Moves played: 20
  RAG queries: 8
  RAG hits: 3
  Exact matches: 3
  Deep searches: 5
  Recursive searches: 12
  Unique positions: 5
  Total contexts: 5
```

## Future Enhancements

### 1. Adaptive Recursion Depth

Adjust `max_recursion_depth` based on time remaining:
```python
if time_remaining > 60:
    max_depth = 3  # Plenty of time, go deep
elif time_remaining > 30:
    max_depth = 2  # Standard
else:
    max_depth = 1  # Time pressure
```

### 2. Selective Child Analysis

Only recurse on the most promising children:
```python
# Sort children by uncertainty × value
scored_children = sorted(
    children,
    key=lambda c: c.uncertainty * c.expected_value,
    reverse=True
)

# Recurse only on top 3
for child in scored_children[:3]:
    if child.uncertainty > threshold:
        analyze_recursively(child)
```

### 3. Context Pruning

Remove old/low-quality contexts:
```python
def prune_contexts(position_ctx):
    # Keep at most 5 contexts per position
    if len(position_ctx.contexts) > 5:
        # Sort by quality × recency
        scored = sorted(
            position_ctx.contexts,
            key=lambda c: c.deep_visits * recency_factor(c.timestamp),
            reverse=True
        )
        position_ctx.contexts = scored[:5]
```

### 4. Parallel Deep Search

Run multiple deep searches in parallel:
```python
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(run_deep_search, child)
        for child in complex_children
    ]
    results = [f.result() for f in futures]
```

### 5. Transfer Learning

Pre-populate database from professional games:
```python
def load_pro_games(sgf_dir):
    for sgf_file in sgf_dir.glob("*.sgf"):
        game = parse_sgf(sgf_file)
        for position in game.positions:
            if position.uncertainty > threshold:
                store_position(position.sym_hash, position.analysis)
```

## Comparison with Standard RAG

### Standard RAG (Previous Implementation)

```
1. Generate move with KataGo
2. Calculate uncertainty
3. If high: Query RAG and blend with policy
4. Store position if very high uncertainty
```

**Limitations:**
- Only stores/retrieves at move generation time
- No deep search on complex positions
- Single context per position
- No exploration of search tree

### Recursive Deep Search (New Implementation)

```
1. Generate candidate moves
2. Calculate uncertainty
3. If high:
   a. Query RAG for cached analysis
   b. If not found: Run deep MCTS
   c. Recursively analyze promising children
   d. Store multi-context results
4. Use best analysis to guide move selection
```

**Advantages:**
- Deeper analysis of complex positions
- Recursive exploration of search tree
- Multiple contexts per position (different game phases)
- Cache reuse across games
- Builds knowledge base automatically

**Trade-offs:**
- Higher computational cost (10-100x for cache misses)
- More memory usage (multiple contexts)
- Requires careful recursion depth tuning
- More complex implementation

## Conclusion

The recursive deep search architecture provides a sophisticated approach to RAG-enhanced Go playing:

1. **Adaptive:** Only searches deeply when positions are truly complex
2. **Cached:** Reuses previous deep analyses for identical positions
3. **Recursive:** Explores the search tree deeply to find best lines
4. **Multi-context:** Stores different game contexts for same position
5. **Scalable:** Database grows organically during play

This approach combines the best of both worlds:
- **Speed:** Standard search for simple positions (~100ms/move)
- **Quality:** Deep search for complex positions (~1-3s/move)
- **Learning:** Automatic knowledge base growth over time
