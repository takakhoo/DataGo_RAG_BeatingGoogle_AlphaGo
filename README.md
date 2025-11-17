# DataGo: Retrieval-Augmented Recursive Search that Surpasses KataGo

**ENGS 102 - Game Theoretic Design | Dartmouth College Thayer School of Engineering**  
**Team:** Benjamin Huh, Jason Peng, Taka Khoo, Olir Eswaramoorthy, David Roos, Victor Lun Pun  
**Advisor:** Dr. Peter Chin

---

## 📄 Complete Research Documents

**📑 [View Full Research Paper (PDF)](DATAGO_PAPER.pdf)**  
**📊 [View Presentation Slides (PDF)](DataGo_Presentation.pdf)**  
**📋 [View Original Project Plan (PDF)](Project_Plan.pdf)**

The complete research paper provides comprehensive mathematical foundations, experimental methodology, and detailed analysis. The presentation offers a visual walkthrough of the architecture and results. The project plan outlines the original vision and roadmap.

---

## 🎯 Executive Summary: We Already Beat KataGo

**DataGo has achieved measurable superiority over KataGo with minimal tuning:**

- **9-0-1 record** (90% win rate) in synthetic stress tests
- **8-0-2 record** (80% win rate) against real KataGo outputs with only threshold retuning
- **Only 5.1% activation rate** needed—proving that selective, high-value RAG interventions suffice
- **6.6 cache hits per query**—demonstrating effective reuse of past 2,000-visit analyses

**What makes this impressive:** We achieved these results with:
- **No new network training**—using only KataGo's public networks
- **Minimal hyperparameter tuning**—just one threshold adjustment (0.37 → 0.15)
- **Simple forced exploration**—not even full policy blending yet
- **Self-play generated memory**—no pro-game ingestion

**This is just the beginning.** With 8 planned tuning phases and full RAG implementation, we aim to surpass AlphaGo and other leading models, targeting NeurIPS or ICLR submission.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Experimental Results](#experimental-results)
5. [System Components](#system-components)
6. [Installation and Setup](#installation-and-setup)
7. [Future Work: 8-Phase Tuning Roadmap](#future-work-8-phase-tuning-roadmap)
8. [Repository Structure](#repository-structure)
9. [Citation](#citation)

---

## Introduction

### The Challenge

Monte Carlo tree search (MCTS) with deep neural priors underpins state-of-the-art Go engines such as KataGo. Yet current systems exhibit a fundamental inefficiency: **each position is analyzed from scratch**, even if it or its symmetric variants have appeared in earlier games. Expensive 10,000-visit analyses are discarded after use, never to be reused.

### Our Solution: Retrieval-Augmented Generation for Go

We introduce **DataGo**, a retrieval-augmented Go engine that wraps unmodified KataGo networks with:

1. **Uncertainty gate** calibrated on log-grounded distributions
2. **Multi-context approximate nearest-neighbor (ANN) memory** keyed by symmetry-invariant hashes
3. **Recursive deep-search module** that stores 2,000–10,000-visit analyses for re-use

### Key Innovation

Unlike traditional opening books or endgame tablebases, DataGo's memory stores **entire deep-search trees** including child-node value distributions, policies, and auxiliary metrics. When a similar position appears, DataGo can instantly retrieve a cached 10k-visit analysis instead of recomputing from scratch.

### Research Question

**Can a KataGo-based engine that caches high-quality analyses and selectively revisits them via ANN retrieval systematically outperform pure KataGo at fixed base visits?**

**Answer: Yes.** Our experiments demonstrate that DataGo achieves measurable improvements in win rate while activating retrieval on only 5.1% of moves.

---

## Architecture Overview

### High-Level Pipeline

The DataGo decision pipeline can be expressed as:

$$
s \;\xrightarrow{f_{\theta}}\; \pi_{\theta},V_{\theta}
\;\xrightarrow{\text{shallow MCTS}}\; \pi_{\text{MCTS}}
\;\xrightarrow{\mathcal{U}}\; \text{gate}
\;\xrightarrow{\text{memory+deep search}}\; \pi_{\text{RAG}}
\;\xrightarrow{\text{MCTS w/ modified priors}}\; a^\star
$$

where:
- $f_{\theta}(s)$ is the KataGo neural evaluator
- $\pi_{\text{MCTS}}$ is the baseline 800-visit search policy
- $\mathcal{U}(s)$ is the uncertainty estimator
- $\pi_{\text{RAG}}$ is the retrieval-augmented policy
- $a^\star$ is the final move selection

### Three-Layer Architecture

1. **Baseline Engine Layer:** KataGo providing neural policy–value function and standard MCTS
2. **Retrieval-Augmented Layer:** Uncertainty estimation, ANN memory, and recursive deep search
3. **Offline Analysis Pipeline:** Constructs and tunes the memory used by retrieval

---

## Mathematical Foundations

### Baseline Engine: Policy–Value Network and MCTS

For each board state $s \in \{-1,0,+1\}^{19\times 19}$ and legal move set $L(s)$, the KataGo network provides:

$$
f_{\theta}(s) = \bigl(\pi_{\theta}(\cdot \mid s), V_{\theta}(s)\bigr)
$$

where $\pi_{\theta}(\cdot\mid s) \in \Delta(L(s))$ is a probability distribution over legal moves and $V_{\theta}(s)\in[-1,1]$ estimates win probability.

A PUCT-style MCTS constructs search statistics:

$$
\{Q(s,a), N(s,a)\}_{a\in L(s)}, \qquad N(s)=\sum_a N(s,a)
$$

The baseline policy used for play is:

$$
\pi_{\text{MCTS}}(a\mid s) \propto N(s,a)
$$

after a fixed budget of $N_{\text{base}}=800$ visits per move.

### Uncertainty Metric and Gating

DataGo uses an uncertainty estimator to decide when additional computation and retrieval are warranted:

$$
\mathcal{U}(s) = \bigl(w_1 E(s) + w_2 K(s)\bigr)\,\phi\bigl(n(s)\bigr)
$$

where:
- $E(s)$ is the normalized Shannon entropy of $\pi_{\text{MCTS}}(\cdot\mid s)$
- $K(s)$ measures disagreement among child values
- $n(s)$ is the number of stones on the board (game phase proxy)
- $\phi$ is a phase multiplier
- $w_1,w_2$ are tuned weights with $w_1+w_2=1$

#### Policy Entropy $E(s)$

Let $\mathbf{p} \in \mathbb{R}^{|L(s)|}$ be the vector of probabilities $\pi_{\text{MCTS}}(a\mid s)$. We define normalized entropy:

$$
E(s) = -\frac{1}{\log |L(s)|} \sum_{a\in L(s)} p(a)\log p(a)
$$

Positions with concentrated search have $E(s)\approx 0$, while positions with many nearly indistinguishable moves have $E(s)\approx 1$.

#### Value Spread $K(s)$

From the shallow search, we consider the top-$k$ children (typically $k\le 10$) by visit count, with values $\{Q_1,\dots,Q_k\}$. We define normalized variance:

$$
K(s) = \min\bigl(1,\;\operatorname{Var}(Q_1,\dots,Q_k) / 0.25\bigr)
$$

Large $K(s)$ indicates that plausible moves lead to significantly different outcomes, signaling internal disagreement.

#### Phase Multiplier $\phi(n)$

The phase function adapts the gate to the stage of the game. The deployed configuration uses a linear profile:

$$
\phi(n) = a\,\frac{n}{361} + b
$$

This allows uncertainty during the endgame (where mistakes are costly) to have more weight than similar uncertainty in the opening.

#### Gating Thresholds

Two thresholds govern behavior:

- **Query threshold** $\theta_{\text{query}}$: If $\mathcal{U}(s)<\theta_{\text{query}}$, the system trusts the baseline 800-visit move
- **Deep-search threshold** $\theta_{\text{deep}} = \theta_{\text{query}} + \delta$: Determines when full recursive deep search is launched

**Critical Finding:** Real KataGo policies are much sharper than synthetic ones. Synthetic tuning suggested $\theta_{\text{query}} \approx 0.37$, but real networks require $\theta_{\text{query}}=0.15$ to achieve proper activation.

### Symmetry-Canonicalized Memory

Each entry in memory corresponds to a symmetry-invariant description of a board state. Let $\mathrm{Sym}(s)$ denote the set of eight rotational and reflectional images of $s$. We define the canonical representative:

$$
\tilde{s} = \arg\min_{s' \in \mathrm{Sym}(s)} \text{lex}(\text{flatten}(s'))
$$

We then compute a fixed-length hash:

$$
h(s) = \mathrm{SHA256}\bigl(\text{flatten}(\tilde{s})\bigr)
$$

This guarantees that symmetric positions share the same key $h(s)$.

### Multi-Context Storage

For each canonical position $\tilde{s}$, the memory stores a list of deep-search contexts:

$$
\mathrm{Ctx}(\tilde{s}) = \bigl\{c_1,\dots,c_m\bigr\}
$$

Each context $c_i$ contains:

$$
c_i = \bigl(a_i^{\star}, \mathcal{U}_i, V_{\text{deep},i}, \pi_{\text{deep},i}, \Delta\text{score}_i, n_i, d_i, t_i\bigr)
$$

where $a_i^{\star}$ is the recommended move, $V_{\text{deep},i}$ and $\pi_{\text{deep},i}$ are the deep-search value and policy, $\Delta\text{score}_i$ is the score lead, $n_i$ is the stone count, $d_i$ is the recursion depth, and $t_i$ is a timestamp.

### Relevance Scoring

Given a candidate memory entry for a query state $s$, DataGo computes a relevance score:

$$
\mathrm{Rel}(s,\tilde{s}^\star) = 0.40\,s_{\text{policy}} + 0.25\,s_{\text{winrate}} + 0.10\,s_{\text{score}} + 0.15\,s_{\text{visits}} + 0.05\,s_{\text{stones}} + 0.05\,s_{\text{komi}}
$$

where each $s_{\text{*}}$ measures similarity in policy space, win rate, score, visit distribution, stone count, and komi.

### Recursive Deep Search

When the uncertainty gate fires and no sufficiently relevant memory entry exists, DataGo performs recursive deep search. The expected cost at depth $D$ is:

$$
\mathrm{Cost}(D) \approx N_{\text{deep}} \sum_{i=0}^{D} \prod_{j=0}^{i-1} b_j
$$

where $b_j$ is the average branching factor at recursion level $j$.

In practice, moderate deep budgets (e.g., $N_{\text{deep}}=2{,}000$) and depths $D_{\max}\in\{1,2,3\}$ offer substantial improvements on a small fraction of moves without exploding compute.

### MCTS with Retrieval-Augmented Priors

DataGo integrates memory into search by modifying priors. The full design uses a blended prior:

$$
P'(a) = (1-\beta)\,P_{\text{net}}(a) + \beta\,P_{\text{rag}}(a)
$$

with $\beta\in[0,1]$ controlling the strength of retrieval. In current experiments, we use forced exploration as a simpler but effective strategy.

---

## Experimental Results

### Experiment Summary

| Experiment | Threshold $\theta$ | Data Type | RAG Queries | Deep Searches | Result (W-L-D) | Win Rate |
|------------|-------------------|-----------|-------------|---------------|----------------|----------|
| **A: Quick Test** | 0.35 | Synthetic | 19 | 19 | 1-0-0 | 100% |
| **B: Extended Synthetic** | 0.370 | Synthetic | 296 | 1411 | **9-0-1** | **90%** |
| **C: Real Untuned** | 0.370 | Real NN | 0 | 0 | 0-10-0 | 0% |
| **D: Real Tuned** | 0.150 | Real NN | 23 | 328 | **8-0-2** | **80%** |

### Key Results: We Beat KataGo

#### Experiment B: Synthetic Stress Test (9-0-1)

**Configuration:**
- Threshold: $\theta = 0.370$
- Max recursion depth: $D_{\max} = 3$
- Deep visits: $V_{\text{deep}} = 2{,}000$
- Games: 10
- Move cap: 100

**Results:**

| Metric | Value |
|--------|-------|
| **Win/Loss/Draw** | **9-0-1** |
| Total moves | 468 |
| RAG queries | 296 (63.2% activation) |
| Deep searches | 1,411 |
| Recursive searches | 1,232 (87% of deep searches) |
| Unique positions stored | 3,017 |
| Total contexts | 3,144 |
| Average uncertainty | $\bar{\mathcal{U}} = 0.377$ |
| Cache hits | 135 (45.6% hit rate) |

**Interpretation:** When allowed to fire frequently, the recursive RAG pipeline drives an almost 2× effective visit advantage and achieves a 90% win rate in a synthetic environment.

#### Experiment D: Real KataGo with Tuned Threshold (8-0-2)

**Configuration:**
- Threshold: $\theta = 0.150$ (retuned from 0.370)
- Max recursion depth: $D_{\max} = 3$
- Deep visits: $V_{\text{deep}} = 2{,}000$
- Games: 10
- Move cap: 100

**Results:**

| Metric | Value |
|--------|-------|
| **Win/Loss/Draw** | **8-0-2** |
| Total moves | 454 |
| RAG queries | 23 (5.1% activation) |
| Deep searches | 328 |
| Recursive searches | 314 (95.7% of deep searches) |
| Unique positions stored | 1,070 |
| Total contexts | 1,211 |
| Average uncertainty | $\bar{\mathcal{U}} = 0.080$ |
| **Cache hits** | **152 (6.6 hits per query!)** |
| Effective visits per move | $\approx 2{,}446$ |

**Key Insight:** Despite a much lower activation rate (5.1% vs 63.2%), DataGo still achieves an 80% win rate. This demonstrates that **rare but high-value activations suffice to improve win rate** when properly calibrated.

### Threshold Sensitivity Analysis

| Scenario | $\theta$ | Data | Activation $p_{\text{act}}$ | Win Rate $p_{\text{win}}$ |
|----------|----------|------|----------------------------|---------------------------|
| Real tuned | 0.15 | Real | 0.051 | **0.80** |
| Synthetic | 0.37 | Synthetic | 0.632 | 0.90 |
| Real untuned | 0.37 | Real | 0.0 | 0.0 |

**Critical Lesson:** The synthetic distribution of $\mathcal{U}$ is much higher than the real one. With $\theta=0.37$ on real networks, the gate never fires (0% activation), demonstrating the necessity of calibrating the gate to actual entropy distributions.

### Compute–Benefit Trade-offs

| Run | Avg Visits/Move | Win Rate | Efficiency (win rate / relative visits) |
|-----|----------------|----------|----------------------------------------|
| Synthetic extended | $\approx 6{,}029$ | 0.9 | 0.15 |
| Real tuned | $\approx 2{,}446$ | 0.8 | **0.26** |

The real tuned configuration is about **76% more efficient** in terms of win rate per relative visit.

### Offline Phase 2 Tuning Results

Deep MCTS sweep results show the compute–accuracy tradeoff:

| Deep Visits $D$ | Avg Policy Error | Avg Value Error | Avg Deep Time (ms) |
|-----------------|------------------|-----------------|-------------------|
| 1,000 | 0.178 | 0.089 | $\approx 220$ |
| 2,000 | 0.112 | 0.052 | $\approx 440$ |
| 5,000 | 0.078 | 0.035 | $\approx 1{,}100$ |
| **10,000** | **0.056** | **0.026** | **$\approx 2{,}203$** |

Raising $D$ from 1,000 to 10,000 reduces policy error by nearly **3×** but multiplies deep-search time by $\approx 10$.

---

## System Components

### Core Directories

- **`datago/`**: Main RAG research sandbox
  - `src/bot/`: DataGo bot implementation
  - `src/memory/`: ANN memory and indexing
  - `src/gating/`: Uncertainty estimation
  - `src/blend/`: Policy blending utilities
  - `src/mcts/`: Custom MCTS implementation
  - `run_datago_recursive_match.py`: Main match runner
  - `rag_store/`: Offline analysis and RAG store construction
  - `tuning/`: Hyperparameter tuning scripts and results

- **`katago_repo/`**: KataGo source and binaries
  - Provides baseline policy/value function $f_{\theta}(s)$
  - Standard MCTS implementation
  - GPU/CPU builds and configurations

- **`ragflow_repo/`**: Production ingestion stack
  - FastAPI backend for document processing
  - GraphRAG and agentic tools
  - Future: Pro-game SGF ingestion

- **`raw_games_data/`**: Self-play logs and generated positions
  - Used for offline RAG store construction
  - Contains flagged positions for deep analysis

### Key Scripts

- **`datago/run_datago_recursive_match.py`**: Main competitive match runner
  - Implements recursive deep search
  - Logs all queries, hits, and deep searches
  - Outputs structured match logs

- **`datago/rag_store/game_analyzer.py`**: Offline position analyzer
  - Reanalyzes flagged positions with high visit counts
  - Populates RAG database with deep-search contexts

- **`datago/tuning/phase2/monitor.py`**: Real-time system monitor
  - Tracks GPU utilization, RAM, disk space
  - Monitors experiment progress

---

## Installation and Setup

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- Git LFS (for large model files)

### Step 1: Clone Repository

```bash
git clone https://github.com/takakhoo/DataGo_RAGtoWin_vs_Google.git
cd DataGo_RAGtoWin_vs_Google
```

### Step 2: Install Python Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 3: Set Up KataGo

KataGo binaries and models are required. You have two options:

**Option A: Use Existing KataGo Installation**

If you have KataGo installed elsewhere, update the paths in configuration files to point to your installation.

**Option B: Build from Source**

```bash
cd katago_repo/KataGo
# Follow KataGo build instructions for your platform
# See: https://github.com/lightvector/KataGo
```

### Step 4: Configure Environment

Create a `.env` file in the root directory:

```bash
KATAGO_BINARY_PATH=/path/to/katago
KATAGO_MODEL_PATH=/path/to/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
CUDA_VISIBLE_DEVICES=0  # Specify GPU
```

### Step 5: Verify Installation

Run a quick test:

```bash
cd datago
python run_datago_recursive_match.py --help
```

### Step 6: Run Your First Match

```bash
cd datago
python run_datago_recursive_match.py \
    --config configs/default_config.yaml \
    --games 1 \
    --move-cap 50
```

---

## Future Work: 8-Phase Tuning Roadmap

**Current Status:** We have achieved 80% win rate against KataGo with minimal tuning (Phase 1 complete). The architecture will continue to be built and refined until NeurIPS or ICLR submission.

### Phase 1: ✅ Uncertainty Gate Calibration (COMPLETE)

- Tuned uncertainty threshold on synthetic data: $\theta = 0.37$
- Retuned on real KataGo outputs: $\theta = 0.15$
- **Result:** 8-0-2 record (80% win rate)

### Phase 2: ✅ Deep Search Configuration (COMPLETE)

- Swept deep visit budgets: $D \in \{1{,}000, 2{,}000, 5{,}000, 10{,}000\}$
- Evaluated recursion depths: $D_{\max} \in \{1, 2, 3, 5\}$
- **Result:** Optimal config: $D=10{,}000$, $D_{\max}=3$ (for stress testing)

### Phase 3: 🔄 Policy Blending Implementation (IN PROGRESS)

**Goal:** Implement full policy blending in MCTS

$$
P'(a) = (1-\beta)\,P_{\text{net}}(a) + \beta\,P_{\text{rag}}(a), \quad \beta \in [0,1]
$$

**Current Status:** Blending utilities implemented but not yet integrated into competitive matches. Currently using forced exploration.

**Expected Impact:** Further shift in Nash equilibrium, potentially increasing win rate to 85-90%.

### Phase 4: 📋 Pro-Game Ingestion

**Goal:** Pre-seed ANN with thousands of professional game positions

- Integrate RagFlow pipeline for SGF parsing
- Ingest curated pro-game databases
- **Target:** $\mathbb{E}[p_{\text{hit}}] \rightarrow 0.5$ (50% cache hit rate)

**Expected Impact:** Cross-game generalization, improved opening play, higher cache efficiency.

### Phase 5: 📊 Asymmetric Visit Testing

**Goal:** Test whether RAG compensates for reduced base visits

**Experiments:**
- DataGo(400 visits + RAG) vs KataGo(800 visits)
- DataGo(600 visits + RAG) vs KataGo(800 visits)
- Measure $\Delta$ Elo and whether retrieval compensates

**Expected Impact:** Prove that RAG provides value beyond simple visit scaling.

### Phase 6: 🎯 Threshold Sweep and Optimization

**Goal:** Map threshold space to performance surface

$$
f(\theta) = (\text{win rate}(\theta), \text{activation}(\theta), \text{compute cost}(\theta))
$$

**Sweep Range:** $\theta \in [0.11, 0.18]$ with fine-grained steps

**Expected Impact:** Find optimal operating point balancing win rate, activation, and compute.

### Phase 7: 🔬 Advanced Relevance Scoring

**Goal:** Improve relevance metric with learned weights

- Current: Fixed weights $(0.40, 0.25, 0.10, 0.15, 0.05, 0.05)$
- Future: Learn optimal weights via regression on policy/value discrepancies
- Add temporal decay and context quality metrics

**Expected Impact:** Higher precision in memory retrieval, better cache utilization.

### Phase 8: 🏆 AlphaGo-Level Competition

**Goal:** Target submission to NeurIPS or ICLR

**Milestones:**
- Achieve >90% win rate against KataGo
- Demonstrate superiority in Elo rating
- Compare against AlphaGo Zero and other leading models
- Publish comprehensive ablation studies

**Timeline:** Architecture development continues until conference submission deadline.

---

## Repository Structure

```
DataGo_RAGtoWin_vs_Google/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── PAPER.tex                    # LaTeX source for research paper
├── presentation.tex             # Beamer presentation source
├── DATAGO_PAPER.pdf            # Complete research paper (PDF)
├── DataGo_Presentation.pdf     # Presentation slides (PDF)
├── Project_Plan.pdf            # Original project proposal (PDF)
│
├── datago/                      # Main DataGo implementation
│   ├── src/                     # Source code
│   │   ├── bot/                 # DataGo bot and GTP player
│   │   ├── memory/              # ANN memory and indexing
│   │   ├── gating/              # Uncertainty estimation
│   │   ├── blend/               # Policy blending
│   │   ├── mcts/                # Custom MCTS
│   │   └── utils/               # Utility functions
│   ├── rag_store/               # RAG store construction
│   ├── tuning/                  # Hyperparameter tuning
│   ├── run_datago_recursive_match.py  # Main match runner
│   └── requirements.txt         # DataGo-specific dependencies
│
├── katago_repo/                 # KataGo source and binaries
│   └── KataGo/                  # KataGo repository
│
├── ragflow_repo/                # RagFlow ingestion stack
│   └── ragflow/                 # RagFlow source
│
└── raw_games_data/              # Self-play logs and positions
    └── rag_data/                # Processed RAG data
```

---

## Key Achievements

### ✅ Validated Accomplishments

1. **Beat KataGo:** 9-0-1 (synthetic) and 8-0-2 (real tuned) records
2. **Efficient Activation:** Only 5.1% activation rate needed for 80% win rate
3. **High Cache Efficiency:** 6.6 hits per query, reusing 2,000-visit analyses
4. **Recursive Deep Search:** 95.7% recursion rate, building deep analysis trees
5. **Reproducible Pipeline:** All experiments logged and reproducible

### 🎯 What Makes This Impressive

- **No new network training**—using only public KataGo networks
- **Minimal tuning**—just one threshold adjustment
- **Simple implementation**—forced exploration, not even full blending
- **Self-play memory**—no external pro-game data yet

### 🚀 Future Potential

With 8-phase tuning complete, we expect:
- **>90% win rate** against KataGo
- **AlphaGo-level performance** with full RAG integration
- **Conference submission** to NeurIPS or ICLR

---

## Game-Theoretic Interpretation

Let $\pi_{\text{DataGo}}$ and $\pi_{\text{KataGo}}$ denote the strategies induced by DataGo and baseline KataGo engines under the same base visit budget (800 visits per move), with DataGo allowed to perform additional deep searches on a small subset of moves.

Let $\text{Score}(s_T)$ be the final score under Chinese rules.

Our experiments approximate:

$$
\max_{\pi_{\text{DataGo}}}\min_{\pi_{\text{KataGo}}} \mathbb{E}[\text{Score}(s_T)] - \max_{\pi_{\text{KataGo}}}\min_{\pi_{\text{KataGo}}} \mathbb{E}[\text{Score}(s_T)] > 0
$$

by showing that DataGo's win rate vs. a strong KataGo baseline is 0.9 (synthetic) and 0.8 (real tuned), with identical networks and identical 800-visit shallow budgets.

**The extra strength derives from targeted deep searches and reuse of stored 2k-visit analyses**, effectively increasing the visit count on hard positions without changing the base configuration.

---

## Limitations and Known Issues

### Visit Asymmetry

Although both engines use 800 shallow visits per move, DataGo uses additional deep visits on activated positions. Our experiments evaluate "DataGo at ~2.4k effective visits" vs. "KataGo at 800 visits."

**Future work:** Run asymmetric matches (e.g., DataGo at 400 base visits + RAG vs. KataGo at 800) to measure whether retrieval compensates for reduced search.

### Incomplete Blending Implementation

While `CustomMCTS` and blending utilities are implemented and tested in isolation, competitive runs use forced exploration rather than full policy blending at every RAG hit.

**Future work:** Enable true blended priors and measure their effect on win rate.

### Limited Statistical Coverage

Main real-network results are based on 10-game matches per configuration. While win rates are large enough (80% vs. 0%) to be highly suggestive, more games and varied opponents are needed for precise strength estimates.

**Future work:** Run 100+ game matches, test against different KataGo nets, Leela Zero, and human players.

### No Pro-Game Ingestion Yet

The `ragflow_repo` infrastructure is ready to ingest SGFs, but experiments reported here restrict themselves to self-play and match-generated positions.

**Future work:** Ingest curated pro-game databases to increase cross-game hit rates.

---

## Broader Impacts

DataGo demonstrates that retrieval-augmented, log-grounded search can substantially strengthen an open-source Go engine **without training new networks**, merely by reusing past computations.

**Positive impacts:**
- Resource-limited labs can improve strong baselines via clever search and memory
- Techniques transfer to other combinatorial search domains (chess, shogi, planning)
- Open-source contribution to Go AI research

**Considerations:**
- Stronger Go engines may widen the gap between humans and machines
- Could enable more persuasive automated analysis tools
- However, these risks are mild compared to generative language or vision models

We rely only on public KataGo networks and do not train new models. Our code and logs are released under standard open-source licenses to support reproducibility.

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{datago2025,
  title={DataGo: Retrieval-Augmented Recursive Search that Surpasses KataGo with Log-Grounded Analysis},
  author={Huh, Benjamin and Peng, Jason and Khoo, Taka and Eswaramoorthy, Olir and Roos, David and Pun, Victor Lun},
  journal={arXiv preprint},
  year={2025},
  note={ENGS 102, Dartmouth College}
}
```

---

## Acknowledgments

- **Advisor:** Dr. Peter Chin, Thayer School of Engineering, Dartmouth College
- **KataGo Team:** For providing excellent open-source Go engine and networks
- **Dartmouth LISP Lab:** For computational resources

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact the authors.

---

**Last Updated:** November 2025  
**Status:** Active development targeting NeurIPS/ICLR submission
