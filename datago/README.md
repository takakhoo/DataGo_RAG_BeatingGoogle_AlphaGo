datago – RAG-Augmented MCTS Workspace

Purpose
- Implement the selective Retrieval-Augmented (RAG) assistance around KataGo’s search, as outlined in the plan: entropy-gated root prior blending, optional leaf prior blending, and a compact curated memory with ANN.

Initial structure (proposed)
- notebooks/                  # Exploration/visual sanity checks
- src/
  - embeddings/               # State encoders / feature extraction wrappers
  - memory/                   # ANN index (FAISS/HNSW), add/prune, IO
  - gating/                   # Entropy and similarity gates
  - blend/                    # Prior/optional value blending utilities
  - clients/                  # JSON-analysis-engine clients and helpers
  - eval/                     # Match runners, metrics, ablations
- data/
  - memory/                   # Serialized memory shards/checkpoints
  - sgf/                      # Curated SGF positions for testing
  - tmp/                      # Scratch outputs, plots

Dependencies
- Python 3.12+ (use the project venv):
  source /scratch2/f004h1v/alphago_project/venv/bin/activate
- Add when implementing:
  pip install faiss-cpu or faiss-gpu  # per host setup
  pip install numpy pandas tqdm pyyaml rich

Interfaces to KataGo
- Preferred interface: JSON analysis engine.
  - Start from run/: ../KataGo/cpp/build-opencl/katago analysis -model default_model.bin.gz -config analysis.cfg
  - Use src/clients to: send-analyze-requests, parse JSON, and retrieve policy/value/ownership and (if available) feature vectors.
- Alternative: GTP with kata-analyze (less convenient for batch).

Key algorithms
- Entropy gating: compute H(policy) at root and compare to H_max.
- Retrieval: encode state -> ANN query -> top-K neighbors -> alignment (rot/ref) -> build retrieval prior.
- Blending: P'(a) = (1-β) P_nn(a) + β P_ret(a) at root; tiny value nudge optional.
- Maintenance: importance-scored add/prune; keep memory size M small (≈5k entries).

### Algorithm detail — two-step RAG augmentation (recommended)

- Entropy gating (both expansion and simulation):
  - At each decision point (root expansion and at selective intermediate nodes during simulation), compute the policy entropy H(P) = -∑_a P(a) log P(a). If H(P) exceeds a configured threshold H_trigger, trigger retrieval and augmentation for that step. This allows more aggressive augmentation only on uncertain/complex positions.
- Retrieval and reranking:
  - Query ANN for the top-K nearest stored embeddings for the current state.
  - Rerank these K candidates by a combination of:
    - Reachability score: estimate how reachable the candidate position is from the current node (e.g., short move-distance, high probability under the current policy, or a learned reachability model). Use a normalized score r ∈ [0,1].
    - Structural relation boost: boost parent or direct-child positions when a stored state is recognized as a direct ancestor/descendant (exact move match after un-rotation) — apply a multiplicative factor s > 1 when detected.
    - Combined ranking weight w = α * r + γ * s where α, γ are tunable.
  - The reranked neighbors produce an action prior P_nn(a) by aligning neighbor move(s) to the current board orientation and accumulating probability mass for their suggested moves.
- Expansion augmentation:
  - During node expansion when H(P) > H_trigger, construct a blended prior for the new child actions:
    - P_blend(a) = (1 - β_exp) * P_net(a) + β_exp * P_nn(a), where β_exp is the expansion blending weight.
  - If retrieved neighbors include parent/child states mapping to a single move, give that move an extra nudge proportional to the parent/child boost to reflect proven reachability.
- Simulation augmentation:
  - During simulation (rollouts or value/policy-guided playouts), if a simulation reaches a node whose entropy crosses H_trigger and ANN retrieval finds similar states, mix retrieved move distributions into the simulation policy at that node by:
    - P_sim(a) = (1 - β_sim) * P_current(a) + β_sim * P_nn(a)
  - This biases simulations toward moves seen in similar complex positions without fully overriding network guidance. β_sim may be smaller than β_exp.
- Storing positions during simulations:
  - If a simulation visits a node with high complexity (e.g., H(P) ≥ H_store and/or large disagreement between value estimate and rollout), capture a compact entry (embedding, canonicalized board, best-move candidates, metadata like frequency and timestamp) to a temporary buffer.
  - Buffer entries are periodically merged into the persistent ANN memory on low-load intervals or per configured batch size.
- Maintenance, pruning and lifecycle:
  - When the persistent memory size exceeds a configured population threshold M_max, trigger a pruning-and-add cycle:
    - Compute an importance score for each entry (recentness, frequency of retrieval, retrieval relevance, outcome-based utility) and evaluate its reachability-based ranking using the same rerank metric w = α * r + γ * s (reachability r and structural boost s).
    - Prune entries with low combined score (effectively those judged most "unreachable" under the rerank metric) — i.e., use the same reachability/unreachability metric you use to rerank retrieval results. After pruning, add buffered new entries or reweighted candidates.
  - Periodic re-indexing may be necessary for ANN structures (HNSW/FAISS) to keep performance predictable.

### Implementation notes / Contracts
- Inputs/outputs:
  - Embedding extractor signature: embed(state) -> float[d].
  - Retrieval signature: retrieve(embed, K) -> list[(id, score, canonical_state, action_hint, metadata)].
  - Rerank function: rerank(retrieved, state) -> weighted list used to build P_nn(a).
- Failure modes:
  - If ANN retrieval fails or returns low-quality matches (low cosine similarity), the system should gracefully fall back to network-only policy.
- Edge cases:
  - Transpositions and symmetries: canonicalize states consistently (rot/ref) to increase matches; align moves back to current orientation on blending.
  - Large branching factors: restrict blending to top-N moves per node to limit effect on search distribution.
  - Concurrency: only commit new entries after consensus or periodic batching to avoid high write contention on indices.

Milestones (from the 4-week plan)
1) Week 1: Baseline + tooling
   - Run baseline fixed-visit matches from analysis clients; export CSV of entropies/top-moves.
2) Week 2: Embeddings + ANN
   - Implement embedding extraction and HNSW index; build a small curated memory.
3) Week 3: Root-only RAG
   - Wire entropy gating + prior blending; online add/prune; run matches/ablations.
4) Week 4: Optional leaf blend + light tune
   - Small leaf blending; light encoder tune; finalize report.

How to start development here
1) Set up dependencies in venv.
2) Create src/clients to talk to analysis engine; write a minimal request/response loop.
3) Implement entropy estimation + logging for a test set of positions.
4) Stand up FAISS index with a tiny memory; build prior from neighbors.
5) Add a command-line tool to run a root-only RAG evaluation on SGF positions.

References
- KataGo GitHub: https://github.com/lightvector/KataGo
- Networks (models): https://katagotraining.org/networks/


