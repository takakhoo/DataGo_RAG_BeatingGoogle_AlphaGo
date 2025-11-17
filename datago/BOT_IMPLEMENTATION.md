# DataGo Bot Implementation Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Configuration](#configuration)
5. [Implementation Details](#implementation-details)
6. [Usage](#usage)
7. [Parameter Tuning](#parameter-tuning)
8. [Development Roadmap](#development-roadmap)

---

## Overview

The DataGo bot is a **Retrieval-Augmented Generation (RAG) enhanced Go playing engine** that combines KataGo's powerful MCTS search with a database of previously analyzed complex positions. The bot detects uncertain positions during gameplay, queries a RAG database for similar situations, and blends retrieved knowledge with neural network priors to make stronger moves.

### Key Features

- **Uncertainty Detection**: Identifies complex positions using policy entropy and value variance
- **RAG Integration**: Queries database of deeply analyzed positions for similar patterns
- **Adaptive Search**: Performs deep MCTS (10,000+ visits) on novel complex positions
- **Online Learning**: Stores newly discovered complex positions during gameplay
- **Gomill Integration**: Plays games against KataGo and other engines through the gomill library
- **Comprehensive Logging**: Tracks RAG queries, deep searches, and performance metrics

### Design Philosophy

The bot follows the principle that **deeper search always yields better results** for complex positions. Rather than relying solely on the neural network's policy/value estimates, the bot:

1. Detects when the position is uncertain (high entropy, value variance)
2. Checks if similar positions have been deeply analyzed before
3. Blends stored deep analysis with current network estimates
4. Performs new deep analysis for novel complex positions
5. Stores the results for future retrieval

---

## Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      DataGo Bot                              │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐     ┌────────────┐ │
│  │   Gomill     │      │  DataGoBot   │     │   KataGo   │ │
│  │   Player     │─────>│    Core      │────>│   Client   │ │
│  │  (Interface) │      │              │     │            │ │
│  └──────────────┘      └──────────────┘     └────────────┘ │
│         │                      │                    │       │
│         │                      ▼                    │       │
│         │              ┌──────────────┐             │       │
│         │              │ Uncertainty  │             │       │
│         │              │  Detection   │             │       │
│         │              └──────────────┘             │       │
│         │                      │                    │       │
│         │                      ▼                    │       │
│         │              ┌──────────────┐             │       │
│         │              │  RAG Query   │             │       │
│         │              │   Engine     │             │       │
│         │              └──────────────┘             │       │
│         │                      │                    │       │
│         │                      ▼                    │       │
│         │              ┌──────────────┐             │       │
│         │              │   Blending   │             │       │
│         │              │   & MCTS     │─────────────┘       │
│         │              └──────────────┘                     │
│         │                      │                            │
│         └──────────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   RAG Database      │
                    │  (ANN Index + JSON) │
                    └─────────────────────┘
```

### Pipeline Flow

```
┌─────────────┐
│ New Move    │
│ Request     │
└──────┬──────┘
       │
       ▼
┌────────────────────────────────────────┐
│ 1. Shallow MCTS (800 visits)          │
│    - Get policy distribution          │
│    - Get value estimate               │
│    - Get move information             │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│ 2. Compute Uncertainty                │
│    uncertainty = (w1*E + w2*K) * phase│
│    - E: Policy entropy                │
│    - K: Value variance                │
│    - phase: Game phase multiplier     │
└──────┬─────────────────────────────────┘
       │
       ▼
   ┌───────────┐
   │uncertainty│
   │>threshold?│
   └─────┬─────┘
         │ Yes
         ▼
┌────────────────────────────────────────┐
│ 3. Query RAG Database                 │
│    - Find k nearest neighbors         │
│    - Compute relevance scores         │
└──────┬─────────────────────────────────┘
       │
       ▼
  ┌──────────┐
  │ RAG Hit? │
  └────┬─────┘
       │
       ├─Yes─>┌────────────────────────────┐
       │      │ High relevance (≥90%)      │
       │      │ - Blend stored policy      │
       │      │ - Blend stored value       │
       │      └────────────────────────────┘
       │
       ├─Partial>┌────────────────────────────┐
       │         │ Low relevance (<90%)       │
       │         │ - Force exploration of     │
       │         │   best RAG moves           │
       │         └────────────────────────────┘
       │
       └─No──>┌────────────────────────────────┐
              │ No RAG match found             │
              │ - Perform deep MCTS (10k vis)  │
              │ - Store in RAG database        │
              └────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Select Move    │
                    │ from Final     │
                    │ Policy         │
                    └────────────────┘
```

---

## Components

### 1. DataGoBot (`datago_bot.py`)

**Core bot implementation** that orchestrates the entire RAG-MCTS pipeline.

#### Key Classes

##### `GameState`
Represents the current state of the Go game.

**Fields:**
- `board`: 19×19 numpy array (0=empty, 1=black, -1=white)
- `current_player`: 1 for black, -1 for white
- `move_number`: Current move count
- `komi`: Komi value
- `history`: List of (move, player) tuples
- `captures`: Dictionary tracking captures for each player
- `ko_point`: Current ko point if any

**Methods:**
- `stones_on_board()`: Count total stones on board (for phase detection)
- `to_dict()`: Serialize for logging

##### `MoveDecision`
Encapsulates a move decision with all associated metadata.

**Fields:**
- `move`: Move in GTP format (e.g., "D4", "Q16", "pass")
- `policy`: Policy distribution over all moves
- `value`: Estimated position value
- `winrate`: Estimated win rate
- `score_lead`: Estimated score lead
- `uncertainty`: Computed uncertainty score
- `rag_queried`: Whether RAG was queried
- `rag_hit`: Whether RAG found relevant position
- `rag_relevance`: Relevance score (0-1)
- `used_deep_search`: Whether deep MCTS was performed
- `time_taken_ms`: Time to generate move
- `visits`: Number of MCTS visits used

##### `DataGoBot`
Main bot class implementing the RAG-MCTS pipeline.

**Key Methods:**

###### `__init__(config_path)`
Initialize bot with YAML configuration.
- Loads configuration
- Sets up logging
- Starts KataGo subprocess
- Initializes RAG database index

###### `generate_move() -> MoveDecision`
**Main entry point** for move generation. Implements the full pipeline:
1. Shallow MCTS search (800 visits)
2. Uncertainty computation
3. RAG query (if uncertain)
4. Knowledge blending or forced exploration
5. Deep MCTS (if novel complex position)
6. Move selection

###### `compute_uncertainty(policy, move_infos, stones_on_board) -> float`
Computes uncertainty score using:
```
uncertainty = (w1 * E + w2 * K) * phase(stones_on_board)
```
- **E**: Normalized policy cross-entropy
- **K**: Value distribution variance (normalized)
- **phase**: Game phase multiplier

###### `query_rag(position_hash, policy, k) -> Tuple[bool, List, float]`
Queries RAG database:
1. Performs k-NN search in ANN index
2. Computes relevance scores
3. Returns (hit, neighbors, max_relevance)

###### `_compute_relevance(rag_entry, current_policy) -> float`
Computes weighted relevance score:
```python
relevance = (
    0.40 * policy_similarity +
    0.25 * winrate_similarity +
    0.10 * score_lead_similarity +
    0.15 * visit_distribution_similarity +
    0.05 * stone_count_similarity +
    0.05 * komi_similarity
)
```

###### `blend_with_rag(network_policy, rag_neighbors) -> Dict`
Blends network policy with RAG retrieval prior:
1. Rerank neighbors by reachability and structural hints
2. Build retrieval prior from top-k neighbors
3. Blend: `P = (1-β)*P_network + β*P_RAG`

###### `deep_mcts_search(board_state, moves) -> Dict`
Performs deep MCTS search:
- Uses 10,000+ visits (configurable)
- Checks for convergence
- Returns refined policy/value estimates

###### `store_position(position_hash, analysis, uncertainty)`
Stores analyzed position in RAG:
- Extracts policy, value, winrate, score_lead
- Creates MemoryEntry with metadata
- Adds to ANN index

### 2. GomillPlayer (`gomill_player.py`)

**Gomill integration** allowing DataGo to play through the gomill library.

#### Key Methods

##### `setup_game(board_size, komi)`
Initialize a new game:
- Creates gomill Board
- Initializes bot's GameState
- Resets move history

##### `genmove() -> str`
Generate move for this player:
1. Calls `bot.generate_move()`
2. Handles move on gomill board
3. Updates bot's game state
4. Returns move in GTP format

##### `handle_move(color, move)`
Process a move (by either player):
- Updates gomill board
- Records in move history
- Syncs bot's internal state

##### `play_game_vs_katago(...)` (Function)
Orchestrates a full game:
1. Initialize DataGo bot
2. Initialize KataGo through GTP
3. Alternately generate moves
4. Handle passes and resignation
5. Score final position
6. Return results and statistics

### 3. Configuration (`config.yaml`)

**Centralized configuration** for all bot parameters, organized by tuning phase.

#### Major Sections

##### Uncertainty Detection (Phase 1a)
- `w1`, `w2`: Weights for entropy and variance
- `phase_function_type`: Linear, exponential, or piecewise
- `phase_function_coefficients`: Phase function parameters

##### Relevance Weights (Phase 1b)
- `policy_weight`: 0.40
- `winrate_weight`: 0.25
- `score_lead_weight`: 0.10
- `visit_distribution_weight`: 0.15
- `stone_count_weight`: 0.05
- `komi_weight`: 0.05
- `relevance_threshold`: 0.90 (for blending vs. forced exploration)

##### RAG Query (Phase 1c)
- `uncertainty_threshold`: When to query RAG
- `max_queries_per_game`: Rate limiting
- `max_query_time_ms`: Performance constraint

##### Deep MCTS (Phase 2)
- `max_visits`: 10,000 (configurable)
- `policy_convergence_threshold`: 0.05
- `value_convergence_threshold`: 0.02
- `convergence_check_interval`: 500

##### Blending (Phase 3)
- `beta`: 0.4 (RAG prior weight)
- `reranking_alpha`: 0.7 (reachability weight)
- `reranking_gamma`: 0.3 (structural weight)
- `top_n_neighbors`: 16
- `top_n_moves`: 16

##### Recursion (Phase 3)
- `max_recursion_depth`: 2
- `force_exploration_top_n`: 2

##### KataGo Configuration
- `executable_path`, `model_path`, `config_path`
- `visits`: 800 (standard search)
- `max_time_per_move`: 30.0
- Analysis settings (ownership, policy, move info)

##### RAG Database
- `database_path`: Path to JSON database
- ANN settings (FAISS or HNSW)
- Storage management (pruning, refresh)

##### Game Settings
- `mode`: "vs_katago", "self_play", etc.
- Time settings, resignation threshold

##### Online Learning
- Background analysis, query frequency tracking

##### Logging
- Log levels, file paths, profiling

---

## Configuration

### Config File Structure

The `config.yaml` file is organized into logical sections corresponding to the parameter tuning phases described in `parameter_tuning_plan.txt`.

### Key Parameters

#### Must Be Tuned (Phase 1-3)

| Parameter | Default | Range | Tuning Phase | Description |
|-----------|---------|-------|--------------|-------------|
| `w1` | 0.5 | [0.0, 1.0] | 1a | Weight for policy entropy |
| `w2` | 0.5 | [0.0, 1.0] | 1a | Weight for value variance |
| `phase_function_coefficients` | [0.5, 0.75] | Various | 1a | Game phase multiplier params |
| `policy_weight` | 0.40 | [0.0, 1.0] | 1b | Relevance: policy similarity |
| `winrate_weight` | 0.25 | [0.0, 1.0] | 1b | Relevance: winrate similarity |
| `uncertainty_threshold` | 0.75 | [0.0, 1.0] | 1c | When to query RAG |
| `max_visits` (deep) | 10000 | [1000, 50000] | 2 | Deep MCTS visits |
| `beta` | 0.4 | [0.0, 1.0] | 3 | RAG prior blending weight |
| `max_recursion_depth` | 2 | [1, 5] | 3 | RAG recursion limit |

#### Can Use Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `visits` (shallow) | 800 | Standard MCTS visits |
| `board_size` | 19 | Board size |
| `komi` | 7.5 | Komi value |
| `top_n_neighbors` | 16 | Neighbors for blending |
| `relevance_threshold` | 0.90 | High-confidence threshold |

### Example Configuration

```yaml
uncertainty_detection:
  w1: 0.5
  w2: 0.5
  phase_function_type: "linear"
  phase_function_coefficients: [0.5, 0.75]

relevance_weights:
  policy_weight: 0.40
  winrate_weight: 0.25
  score_lead_weight: 0.10
  visit_distribution_weight: 0.15
  stone_count_weight: 0.05
  komi_weight: 0.05
  relevance_threshold: 0.90

rag_query:
  uncertainty_threshold: 0.75
  max_queries_per_game: 50
  max_query_time_ms: 5.0

deep_mcts:
  max_visits: 10000
  policy_convergence_threshold: 0.05
  value_convergence_threshold: 0.02

blending:
  beta: 0.4
  reranking_alpha: 0.7
  reranking_gamma: 0.3
  top_n_neighbors: 16

recursion:
  max_recursion_depth: 2
  force_exploration_top_n: 2
```

---

## Implementation Details

### Uncertainty Detection

The bot computes uncertainty using a **two-component formula**:

```
uncertainty = (w1 * E + w2 * K) * phase(stones_on_board)
```

#### Component E: Policy Cross-Entropy

Measures how uncertain the policy distribution is:

```python
def normalized_entropy(policy: np.ndarray) -> float:
    """Compute normalized Shannon entropy of policy."""
    L = len(policy)
    if L <= 1:
        return 0.0
    
    # Compute entropy
    p = policy / (policy.sum() + 1e-12)
    p_pos = p[p > 0]
    H = -(p_pos * np.log(p_pos)).sum()
    
    # Normalize by max possible entropy
    H_max = np.log(L)
    return H / (H_max + 1e-12)
```

**Intuition**: High entropy = many equally good moves = uncertain position.

#### Component K: Value Distribution Sparseness

Measures variance in value estimates across candidate moves:

```python
values = [move['winrate'] for move in move_infos]
value_variance = np.var(values)
K = min(1.0, value_variance / 0.25)  # Normalize by max variance
```

**Intuition**: High variance = moves lead to very different outcomes = uncertain position.

#### Phase Function

Adjusts uncertainty based on game phase (stones on board):

**Linear** (default):
```
phase(s) = a * (s/361) + b
```
Example: `[0.5, 0.75]` → phase increases from 0.75 (opening) to 1.25 (endgame)

**Exponential**:
```
phase(s) = a * exp(b * s/361) + c
```
Example: `[0.5, 0.5, 0.5]` → emphasizes late-game uncertainty

**Piecewise**:
```
phase(s) = { a if s < 120 (early)
           { b if 120 ≤ s < 240 (mid)
           { c if s ≥ 240 (late)
```
Example: `[0.8, 1.0, 1.2]` → increasing emphasis through game phases

### RAG Query Process

#### 1. Embedding Construction

Current implementation uses **simplified embedding** (policy vector). Full implementation should include:

```python
embedding = np.concatenate([
    policy,                    # 362-dim (19×19 + pass)
    [winrate],                 # 1-dim
    [score_lead],              # 1-dim
    ownership,                 # 361-dim
    visit_distribution,        # Top-k moves' visit counts
])
```

#### 2. ANN Search

Uses FAISS or HNSW for fast approximate nearest neighbor search:

```python
neighbors = rag_index.query(
    query_embedding=embedding,
    k=32,  # Retrieve 32 neighbors
)
# Returns: List[(MemoryEntry, distance)]
```

#### 3. Relevance Scoring

Computes **weighted similarity** across multiple dimensions:

```python
relevance = (
    0.40 * cosine_similarity(policy_current, policy_stored) +
    0.25 * (1 - |winrate_current - winrate_stored|) +
    0.10 * score_lead_similarity +
    0.15 * visit_distribution_overlap +
    0.05 * stone_count_similarity +
    0.05 * komi_match
)
```

#### 4. Hit Detection

```python
if max_relevance >= 0.90:
    # High confidence: blend policies
    return (True, neighbors, max_relevance)
else:
    # Low confidence: force exploration or deep search
    return (False, neighbors, max_relevance)
```

### Knowledge Blending

When RAG finds a high-relevance match, blend stored and network policies:

#### 1. Rerank Neighbors

```python
def rerank_neighbors(neighbors, current_policy, alpha, gamma):
    """
    Combine similarity and structural scores:
    
    weight = alpha * reachability + gamma * structural_boost
    
    - reachability: Policy similarity (how easy to reach stored position)
    - structural_boost: Parent/child relationship hints
    """
    weights = []
    for entry, raw_score in neighbors:
        r = raw_score  # Reachability from ANN similarity
        s = 1.0  # Default structural boost
        
        # Boost parent/child positions
        if entry.metadata.get("relation") in ("parent", "child"):
            s = 2.0
        
        w = alpha * r + gamma * s
        weights.append(w)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    return [(neighbors[i][0], weights[i]) for i in range(len(neighbors))]
```

#### 2. Build Retrieval Prior

```python
def build_retrieval_prior(reranked_neighbors, top_n=16):
    """
    Aggregate best moves from top-k neighbors:
    
    P_RAG(move) = Σ weight_i * prob_i(move)
    """
    accum = {}
    
    for entry, weight in reranked_neighbors[:top_n]:
        for move_info in entry.best_moves:
            move = move_info['move']
            prob = move_info['prob']
            accum[move] = accum.get(move, 0.0) + weight * prob
    
    # Normalize
    total = sum(accum.values())
    return {k: v/total for k, v in accum.items()}
```

#### 3. Blend Priors

```python
def blend_priors(P_network, P_RAG, beta=0.4):
    """
    Blend network and retrieval priors:
    
    P_final = (1-β)*P_network + β*P_RAG
    """
    blended = {}
    all_moves = set(P_network.keys()) | set(P_RAG.keys())
    
    for move in all_moves:
        p_net = P_network.get(move, 0.0)
        p_rag = P_RAG.get(move, 0.0)
        blended[move] = (1 - beta) * p_net + beta * p_rag
    
    # Normalize
    total = sum(blended.values())
    return {k: v/total for k, v in blended.items()}
```

### Deep MCTS

For novel complex positions (no RAG hit), perform deep search:

#### 1. Query KataGo with High Visit Count

```python
deep_analysis = katago.query(
    board_state=current_board,
    moves=move_history,
    visits=10000,  # Much higher than standard 800
)
```

#### 2. Check for Convergence (Optional Early Stopping)

```python
def check_convergence(analysis_history, threshold):
    """
    Stop early if policy/value have converged:
    
    - Policy change < 0.05
    - Value change < 0.02
    """
    if len(analysis_history) < 2:
        return False
    
    prev = analysis_history[-2]
    curr = analysis_history[-1]
    
    policy_diff = kl_divergence(curr['policy'], prev['policy'])
    value_diff = abs(curr['winrate'] - prev['winrate'])
    
    return (policy_diff < 0.05 and value_diff < 0.02)
```

#### 3. Store in RAG Database

```python
entry = MemoryEntry(
    position_hash=compute_hash(board_state),
    policy=deep_analysis['policy'],
    best_moves=deep_analysis['moveInfos'][:2],
    metadata={
        'winrate': deep_analysis['winrate'],
        'score_lead': deep_analysis['scoreLead'],
        'uncertainty': uncertainty_score,
        'stone_count': stones_on_board,
        'visits': 10000,
        'timestamp': time.time(),
    }
)

rag_index.add(entry)
```

### Online Learning

During gameplay, the bot continuously learns:

#### Store New Positions

Whenever a deep search is performed:
1. Extract refined policy/value
2. Store in RAG database
3. Add to ANN index

#### Background Analysis (Optional)

For positions that exceed uncertainty threshold but aren't immediately deep-searched:

```python
if online_learning['enable_background_analysis']:
    background_queue.append({
        'position_hash': hash,
        'board_state': board,
        'move_history': moves,
        'priority': uncertainty,
    })
    
    # Process queue asynchronously
    if len(background_queue) > 0:
        spawn_background_worker()
```

#### Pruning

When database reaches size limit:

```python
def prune_database(rag_index, max_size_gb):
    """
    Remove least-used entries:
    
    - Track query frequency per entry
    - Remove entries with frequency < threshold
    - Keep high-uncertainty positions
    """
    entries_by_frequency = sorted(
        rag_index.entries,
        key=lambda e: e.metadata.get('query_count', 0)
    )
    
    # Remove bottom 10%
    num_to_remove = len(entries_by_frequency) // 10
    for entry in entries_by_frequency[:num_to_remove]:
        rag_index.remove(entry.position_hash)
```

---

## Usage

### Installation

#### 1. Install Dependencies

```bash
cd DataGO
pip install -r requirements.txt
pip install gomill pyyaml
```

#### 2. Install KataGo

Download from: https://github.com/lightvector/KataGo/releases

```bash
# Example for macOS ARM64
wget https://github.com/lightvector/KataGo/releases/download/v1.14.1/katago-v1.14.1-macos-arm64.zip
unzip katago-v1.14.1-macos-arm64.zip
chmod +x katago

# Download a model
wget https://github.com/lightvector/KataGo/releases/download/v1.14.1/kata1-b18c384nbt-s7528563712-d4434368680.bin.gz
```

#### 3. Configure KataGo

Create a GTP config file (`katago_gtp.cfg`):

```ini
# KataGo GTP Configuration
numSearchThreads = 4
nnCacheSizePowerOfTwo = 23
nnMutexPoolSizePowerOfTwo = 17
```

#### 4. Initialize RAG Database

```bash
# Create empty database or use existing
mkdir -p rag_store/rag_output
echo '{"entries": []}' > rag_store/rag_output/rag_database.json
```

### Running the Bot

#### Interactive Testing

```python
from src.bot.datago_bot import DataGoBot

# Initialize bot
bot = DataGoBot("src/bot/config.yaml")

# Start new game
bot.new_game(board_size=19, komi=7.5)

# Generate move
decision = bot.generate_move()
print(f"Move: {decision.move}")
print(f"Uncertainty: {decision.uncertainty:.3f}")
print(f"RAG queried: {decision.rag_queried}")
print(f"Time: {decision.time_taken_ms:.1f}ms")

# Get statistics
stats = bot.get_statistics()
print(stats)

# Cleanup
bot.shutdown()
```

#### Play Against KataGo

```bash
python src/bot/gomill_player.py \
    --config src/bot/config.yaml \
    --katago-executable /path/to/katago \
    --katago-model /path/to/model.bin.gz \
    --katago-config katago_gtp.cfg \
    --board-size 19 \
    --komi 7.5 \
    --color black \
    --save-sgf games/game_001.sgf
```

#### Play Multiple Games

```bash
python src/bot/gomill_player.py \
    --config src/bot/config.yaml \
    --katago-executable /path/to/katago \
    --katago-model /path/to/model.bin.gz \
    --katago-config katago_gtp.cfg \
    --num-games 10 \
    --output-json results/games_vs_katago.json
```

### Monitoring

#### Check Logs

```bash
tail -f logs/datago_bot.log
```

#### View Statistics

```python
import json

with open('results/games_vs_katago.json') as f:
    results = json.load(f)

for i, game in enumerate(results):
    print(f"Game {i+1}:")
    print(f"  Winner: {game['winner']}")
    print(f"  Moves: {game['total_moves']}")
    print(f"  RAG queries: {game['statistics']['rag_queries']}")
    print(f"  RAG hit rate: {game['statistics'].get('rag_hit_rate', 0):.1%}")
    print(f"  Deep searches: {game['statistics']['deep_searches']}")
```

---

## Parameter Tuning

The bot's parameters must be tuned through a **systematic 3-phase process** as described in `tuning/parameter_tuning_plan.txt`.

### Phase 1a: Uncertainty Detection (18-20 hours)

**Goal**: Find optimal `(w1, w2, phase_function)` that best identifies uncertain positions.

**Method**:
1. Grid search over weight combinations
2. Evaluate on ground truth database
3. Measure win rate vs. baseline KataGo
4. Use A100 GPU for parallel game execution (32-64 games simultaneously)

**Search Space**:
- `w1`: [0.3, 0.5, 0.7] (constraint: w1 + w2 = 1.0)
- `w2`: [0.3, 0.5, 0.7]
- Phase function: Linear (test others if time permits)
- Phase coefficients: 3 combinations

**Total configurations**: 3 × 3 = 9

**Script**: `tuning/phase1/phase1_uncertainty_tuning.py`

```bash
cd tuning
python phase1/phase1_uncertainty_tuning.py \
    --output-dir ./tuning_results/phase1 \
    --num-games 150 \
    --parallel-workers 32 \
    --early-stopping-threshold 0.40
```

### Phase 1b: Relevance Weights (2-4 hours)

**Goal**: Find optimal relevance comparison weights for RAG retrieval.

**Method**:
1. Supervised learning on labeled position pairs
2. Test variations around baseline weights
3. Evaluate using ROC-AUC and calibration metrics

**Search Space**:
- Focus on `policy_weight` and `winrate_weight`
- Test ±0.05, ±0.10 variations
- ~20-30 combinations total

**Script**: `tuning/phase1/phase1b_relevance_weights.py`

```bash
python phase1/phase1b_relevance_weights.py \
    --ground-truth-db data/ground_truth_sample.json \
    --output-dir ./tuning_results/phase1b
```

### Phase 1c: Uncertainty Threshold (8-10 hours)

**Goal**: Find optimal `uncertainty_threshold` for RAG queries.

**Method**:
1. Use fixed uncertainty detection and relevance weights from Phase 1a/1b
2. Test percentile-based thresholds (top 5%, 10%, 15%, 20%, 25%)
3. Evaluate win rate and RAG efficiency

**Script**: `tuning/phase1/phase1c_uncertainty_threshold.py`

```bash
python phase1/phase1c_uncertainty_threshold.py \
    --phase1-config ./tuning_results/phase1/best_config_phase1.json \
    --phase1b-config ./tuning_results/phase1b/best_relevance_weights.json \
    --output-dir ./tuning_results/phase1c \
    --num-games 100
```

### Phase 2: Deep MCTS Parameters (8-10 hours)

**Goal**: Find optimal deep MCTS visit count and convergence thresholds.

**Method**:
1. Test different visit counts: [1000, 2000, 5000, 10000]
2. Test convergence thresholds
3. Measure quality vs. computation trade-off

**Script**: `tuning/phase2/phase2_deep_mcts.py`

```bash
cd tuning
phase2/run_phase2.sh
```

### Phase 3: Blending & Recursion (8-10 hours)

**Goal**: Find optimal blending parameters and recursion depth.

**Parameters to tune**:
- `beta`: RAG prior weight [0.2, 0.4, 0.6]
- `reranking_alpha`: Reachability weight [0.5, 0.7, 0.9]
- `max_recursion_depth`: [1, 2, 3]

**Method**: Game-based evaluation with full pipeline

### Applying Tuned Parameters

After tuning, update `config.yaml`:

```yaml
# From Phase 1a
uncertainty_detection:
  w1: 0.52  # Tuned value
  w2: 0.48  # Tuned value
  phase_function_coefficients: [0.6, 0.7]  # Tuned

# From Phase 1b
relevance_weights:
  policy_weight: 0.45  # Tuned (was 0.40)
  winrate_weight: 0.28  # Tuned (was 0.25)
  # ... other weights

# From Phase 1c
rag_query:
  uncertainty_threshold: 0.82  # Tuned (top 12% uncertain positions)

# From Phase 2
deep_mcts:
  max_visits: 8000  # Tuned (down from 10000 for efficiency)
  policy_convergence_threshold: 0.04  # Tuned

# From Phase 3
blending:
  beta: 0.45  # Tuned (was 0.40)
  reranking_alpha: 0.75  # Tuned

recursion:
  max_recursion_depth: 2  # Tuned (confirmed optimal)
```

---

## Development Roadmap

### Phase 1: Core Implementation ✅

- [x] DataGoBot class with RAG-MCTS pipeline
- [x] Uncertainty detection (entropy + variance)
- [x] RAG query interface
- [x] Knowledge blending
- [x] Gomill integration
- [x] Configuration system
- [x] Documentation

### Phase 2: Integration & Testing 🚧

- [ ] Fix import paths in existing codebase
- [ ] Integrate with existing `memory/index.py` and `memory/schema.py`
- [ ] Implement proper board state hashing
- [ ] Test shallow MCTS with KataGo
- [ ] Test RAG query with sample database
- [ ] Test full pipeline on simple positions

### Phase 3: Advanced Features

- [ ] Implement child node comparison for relevance scoring
- [ ] Add structural relationship detection (parent/child positions)
- [ ] Implement forced exploration logic
- [ ] Add background analysis queue
- [ ] Implement database pruning
- [ ] Add SGF game saving

### Phase 4: Optimization

- [ ] GPU batch processing for MCTS
- [ ] Parallel MCTS with multiple threads
- [ ] Optimize ANN index (tune FAISS/HNSW parameters)
- [ ] Profile and optimize hot paths
- [ ] Memory usage optimization

### Phase 5: Parameter Tuning

- [ ] Phase 1a: Uncertainty detection weights
- [ ] Phase 1b: Relevance comparison weights
- [ ] Phase 1c: Uncertainty threshold
- [ ] Phase 2: Deep MCTS parameters
- [ ] Phase 3: Blending & recursion

### Phase 6: Evaluation

- [ ] Play 100+ games vs. baseline KataGo
- [ ] Measure win rate improvement
- [ ] Analyze RAG hit rate and effectiveness
- [ ] Profile computational overhead
- [ ] Compare different board sizes (9×9, 13×13, 19×19)

### Future Enhancements

- [ ] Support for different rulesets (Japanese, Korean, AGA)
- [ ] Time control strategies
- [ ] Opening book integration
- [ ] Tournament mode with multiple opponents
- [ ] Web interface for visualization
- [ ] Real-time analysis dashboard

---

## Known Issues & TODOs

### Critical

1. **Import path issues**: Need to fix imports to use installed package or add parent to sys.path
2. **Board hashing**: Implement canonical position hashing (currently placeholder)
3. **Move execution**: Add proper board state updates after moves
4. **KataGo integration**: Test subprocess communication with real KataGo

### Important

1. **Child node comparison**: Implement proper visit distribution similarity
2. **Forced exploration**: Add logic to force RAG moves when relevance < threshold
3. **Convergence checking**: Implement incremental convergence checks during deep MCTS
4. **Database persistence**: Add proper save/load for ANN index

### Nice to Have

1. **SGF export**: Implement full SGF game saving
2. **Analysis mode**: Add position analysis without playing
3. **Undo/redo**: Support move takeback
4. **Debug visualization**: Board rendering, policy heatmaps
5. **Config validation**: Add schema validation for config.yaml

---

## Conclusion

The DataGo bot implements a sophisticated RAG-enhanced MCTS system that:

1. **Detects uncertainty** using entropy and value variance
2. **Queries a knowledge base** of deeply analyzed positions
3. **Blends retrieved knowledge** with neural network priors
4. **Performs deep search** on novel complex positions
5. **Learns continuously** by storing new analysis

The modular design allows for:
- Easy parameter tuning through configuration
- Clear separation of concerns (detection, retrieval, blending)
- Integration with existing tools (KataGo, gomill)
- Comprehensive logging and monitoring

The implementation is designed to be **tuned systematically** through the 3-phase process described in the parameter tuning plan, with all parameters exposed in the configuration file for future adjustments.

**Next steps**: 
1. Fix integration issues with existing codebase
2. Test with real KataGo and sample RAG database  
3. Begin Phase 1 parameter tuning
4. Iterate based on win rate results

---

## References

- **Parameter Tuning Plan**: `tuning/parameter_tuning_plan.txt`
- **Phase 1 Tuning**: `tuning/phase1/README.md`
- **Phase 2 Tuning**: `tuning/phase2/README.md`
- **KataGo**: https://github.com/lightvector/KataGo
- **Gomill**: https://github.com/mattheww/gomill
- **FAISS**: https://github.com/facebookresearch/faiss
- **HNSW**: https://github.com/nmslib/hnswlib
