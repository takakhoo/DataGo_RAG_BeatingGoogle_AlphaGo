"""
Phase 1c: Uncertainty Threshold Tuning via RAG-Augmented Gameplay
Uses a STATIC pre-populated RAG database to augment MCTS search during gameplay.
Tests different uncertainty thresholds for when to query the RAG database.

Does NOT add new positions - only retrieves from existing database.
Blends retrieved policy/value into MCTS based on similarity.

Prerequisites: 
- Phase 1a: Uncertainty function parameters (w1, w2, phase_function)
- Phase 1b: Relevance comparison weights

Hardware: NVIDIA A100
Time Budget: 8-10 hours
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.memory.index import ANNIndex
    from src.memory.schema import MemoryEntry
    HAS_MEMORY = True
except ImportError:
    HAS_MEMORY = False
    print("Warning: Memory modules not available")


@dataclass
class UncertaintyThresholdConfig:
    """Configuration for uncertainty threshold testing"""
    threshold_percentile: float  # Query RAG for top X% uncertain positions
    absolute_threshold: float  # Computed absolute uncertainty threshold
    relevance_threshold: float = 0.90  # Minimum similarity for blending (from claude_instructions)
    k_neighbors: int = 1  # Number of nearest neighbors to retrieve
    config_id: str = ""
    
    def __post_init__(self):
        if not self.config_id:
            self.config_id = f"threshold_{self.threshold_percentile:.1f}pct_rel_{self.relevance_threshold:.2f}"


@dataclass
class RAGAugmentedMetrics:
    """Metrics for RAG-augmented gameplay"""
    config_id: str
    win_rate: float
    total_games: int
    
    # RAG usage statistics
    total_positions: int
    rag_queries: int  # How many times we queried RAG
    rag_query_rate: float  # % of positions that triggered RAG query
    
    # RAG effectiveness
    high_relevance_hits: int  # Found match with relevance >= threshold (blended)
    low_relevance_hits: int   # Found match with relevance < threshold (forced moves)
    no_hits: int  # No match found in RAG
    
    avg_relevance_score: float  # Average similarity of retrieved entries
    avg_retrieval_time_ms: float
    
    # Game outcomes
    avg_game_length: float
    avg_uncertainty_per_game: float


@dataclass
class RAGEntry:
    """Entry in the RAG database (from offline deep analysis)"""
    sym_hash: str  # Lookup key
    state_hash: str  # Unique ID
    embedding: np.ndarray  # Vector for ANN search
    
    # MCTS results from deep search (5000+ visits)
    policy: np.ndarray  # Policy distribution (362 floats: 361 moves + pass)
    winrate: float
    score_lead: float
    ownership: Optional[np.ndarray]  # Ownership prediction (361 floats)
    
    # Child node information
    child_nodes: List[Dict]  # [{"hash": str, "value": float, "pUCT": float, "move": int}, ...]
    best_2_moves: List[int]  # Top 2 moves by value
    
    # Context information
    stone_count: int
    komi: float
    move_infos: List[Dict]


def compute_relevance_score(query_entry: Dict, rag_entry: RAGEntry,
                            weights: Optional[Dict] = None) -> float:
    """
    Compute relevance score between query position and RAG entry.
    
    From claude_instructions.txt:
    similarity_weights = {
        'policy': 0.40,
        'winrate': 0.25,  
        'score_lead': 0.10,
        'visit_distribution': 0.15,
        'stone_count': 0.05,
        'komi': 0.05
    }
    """
    if weights is None:
        weights = {
            'policy': 0.40,
            'winrate': 0.25,
            'score_lead': 0.10,
            'visit_distribution': 0.15,
            'stone_count': 0.05,
            'komi': 0.05
        }
    
    score = 0.0
    
    # Policy similarity (cosine similarity)
    if 'policy' in query_entry and rag_entry.policy is not None:
        q_policy = np.array(query_entry['policy'])
        r_policy = rag_entry.policy
        policy_sim = np.dot(q_policy, r_policy) / (np.linalg.norm(q_policy) * np.linalg.norm(r_policy) + 1e-10)
        score += weights['policy'] * policy_sim
    
    # Winrate similarity (1 - absolute difference)
    if 'winrate' in query_entry:
        winrate_sim = 1.0 - abs(query_entry['winrate'] - rag_entry.winrate)
        score += weights['winrate'] * winrate_sim
    
    # Score lead similarity (1 - normalized difference)
    if 'score_lead' in query_entry:
        score_diff = abs(query_entry['score_lead'] - rag_entry.score_lead)
        score_lead_sim = 1.0 / (1.0 + score_diff / 10.0)  # Normalize by typical range
        score += weights['score_lead'] * score_lead_sim
    
    # Visit distribution similarity (compare child node overlap)
    if 'child_nodes' in query_entry and rag_entry.child_nodes:
        query_children = {child['hash'] for child in query_entry['child_nodes']}
        rag_children = {child['hash'] for child in rag_entry.child_nodes}
        overlap = len(query_children & rag_children)
        total = len(query_children | rag_children)
        visit_sim = overlap / max(1, total)
        score += weights['visit_distribution'] * visit_sim
    
    # Stone count similarity (phase matching)
    if 'stone_count' in query_entry:
        stone_diff = abs(query_entry['stone_count'] - rag_entry.stone_count)
        stone_sim = 1.0 / (1.0 + stone_diff / 50.0)  # Normalize by typical variation
        score += weights['stone_count'] * stone_sim
    
    # Komi must match exactly (binary)
    if 'komi' in query_entry:
        komi_match = 1.0 if abs(query_entry['komi'] - rag_entry.komi) < 0.1 else 0.0
        score += weights['komi'] * komi_match
    
    return score


def blend_policy_value(current_policy: np.ndarray, current_value: float,
                       rag_policy: np.ndarray, rag_value: float,
                       relevance: float, blend_factor: float = 0.5) -> Tuple[np.ndarray, float]:
    """
    Blend current MCTS policy/value with retrieved RAG entry.
    
    Args:
        current_policy: Current policy from shallow MCTS
        current_value: Current value estimate
        rag_policy: Retrieved policy from deep MCTS
        rag_value: Retrieved value estimate
        relevance: Relevance score [0, 1]
        blend_factor: How much to blend (0=current only, 1=RAG only)
        
    Returns:
        (blended_policy, blended_value)
    """
    # Adjust blend factor by relevance (higher relevance = more RAG influence)
    effective_blend = blend_factor * relevance
    
    # Blend policy (weighted average, then renormalize)
    blended_policy = (1 - effective_blend) * current_policy + effective_blend * rag_policy
    blended_policy = blended_policy / (np.sum(blended_policy) + 1e-10)
    
    # Blend value (weighted average)
    blended_value = (1 - effective_blend) * current_value + effective_blend * rag_value
    
    return blended_policy, blended_value


class RAGAugmentedGameExecutor:
    """
    Executes games with RAG augmentation but NO storage of new positions.
    
    Uses static RAG database for retrieval only.
    """
    
    def __init__(self, 
                 rag_database_path: str,
                 uncertainty_config: Dict,
                 threshold_config: UncertaintyThresholdConfig,
                 relevance_weights: Dict):
        """
        Args:
            rag_database_path: Path to pre-populated RAG database
            uncertainty_config: Phase 1a config (w1, w2, phase_function)
            threshold_config: Threshold for when to query RAG
            relevance_weights: Phase 1b weights for similarity scoring
        """
        self.uncertainty_config = uncertainty_config
        self.threshold_config = threshold_config
        self.relevance_weights = relevance_weights
        
        # Load RAG database
        print(f"Loading RAG database from {rag_database_path}...")
        self.rag_entries = self.load_rag_database(rag_database_path)
        print(f"  Loaded {len(self.rag_entries)} RAG entries")
        
        # Initialize ANN index if available
        if HAS_MEMORY:
            self.ann_index = self.build_ann_index()
        else:
            self.ann_index = None
            print("  Warning: ANN index not available, will use linear search")
        
        # Statistics
        self.stats = {
            'total_positions': 0,
            'rag_queries': 0,
            'high_relevance_hits': 0,
            'low_relevance_hits': 0,
            'no_hits': 0,
            'relevance_scores': [],
            'retrieval_times': []
        }
    
    def load_rag_database(self, db_path: str) -> List[RAGEntry]:
        """Load RAG database from JSON"""
        with open(db_path, 'r') as f:
            data = json.load(f)
        
        entries = []
        for entry_dict in data.get('entries', []):
            entry = RAGEntry(
                sym_hash=entry_dict['sym_hash'],
                state_hash=entry_dict['state_hash'],
                embedding=np.array(entry_dict['embedding']),
                policy=np.array(entry_dict['policy']),
                winrate=entry_dict['winrate'],
                score_lead=entry_dict['score_lead'],
                ownership=np.array(entry_dict['ownership']) if 'ownership' in entry_dict else None,
                child_nodes=entry_dict.get('child_nodes', []),
                best_2_moves=entry_dict.get('best_2_moves', []),
                stone_count=entry_dict['stone_count'],
                komi=entry_dict['komi'],
                move_infos=entry_dict.get('move_infos', [])
            )
            entries.append(entry)
        
        return entries
    
    def build_ann_index(self) -> Optional[ANNIndex]:
        """Build ANN index from RAG entries"""
        if not self.rag_entries:
            return None
        
        # Create memory entries from RAG entries
        memory_entries = []
        for rag_entry in self.rag_entries:
            mem_entry = MemoryEntry(
                id=rag_entry.state_hash,
                embed=rag_entry.embedding.tolist(),
                canonical_board=rag_entry.sym_hash,
                best_moves=rag_entry.best_2_moves,
                visits=5000,  # Deep MCTS visits
                importance=1.0,
                metadata={'winrate': rag_entry.winrate, 'stone_count': rag_entry.stone_count}
            )
            memory_entries.append(mem_entry)
        
        # Build index
        index = ANNIndex(
            dim=len(self.rag_entries[0].embedding),
            backend='faiss',
            metric='cosine'
        )
        
        for entry in memory_entries:
            index.add(entry)
        
        return index
    
    def compute_uncertainty(self, position_data: Dict) -> float:
        """
        Compute uncertainty score for a position.
        
        Uses Phase 1a learned parameters: (w1*E + w2*K) * phase(stones)
        """
        w1 = self.uncertainty_config['w1']
        w2 = self.uncertainty_config['w2']
        phase_type = self.uncertainty_config['phase_function_type']
        phase_coef = self.uncertainty_config['phase_coefficients']
        
        # Extract features
        policy_entropy = position_data.get('policy_entropy', 0.0)
        value_sparseness = position_data.get('value_sparseness', 0.0)
        stones_on_board = position_data.get('stone_count', 0)
        
        # Compute phase multiplier
        s = stones_on_board / 361.0
        if phase_type == 'linear':
            a, b = phase_coef[0], phase_coef[1]
            phase = a * s + b
        elif phase_type == 'exponential':
            a, b, c = phase_coef
            phase = a * np.exp(b * s) + c
        elif phase_type == 'piecewise':
            early_mult, mid_mult, late_mult = phase_coef
            if stones_on_board < 120:
                phase = early_mult
            elif stones_on_board < 240:
                phase = mid_mult
            else:
                phase = late_mult
        else:
            phase = 1.0
        
        # Compute uncertainty
        uncertainty = (w1 * policy_entropy + w2 * value_sparseness) * phase
        return uncertainty
    
    def query_rag(self, position_data: Dict) -> Optional[Tuple[RAGEntry, float]]:
        """
        Query RAG database for similar position.
        
        Returns:
            (rag_entry, relevance_score) if found, None otherwise
        """
        start_time = time.time()
        
        # Use ANN index if available
        if self.ann_index is not None and 'embedding' in position_data:
            query_embedding = np.array(position_data['embedding'])
            
            # Retrieve k nearest neighbors
            results = self.ann_index.retrieve(
                query_embedding,
                k=self.threshold_config.k_neighbors
            )
            
            if results:
                # Get best match
                best_result = results[0]
                rag_entry = next((e for e in self.rag_entries if e.state_hash == best_result.id), None)
                
                if rag_entry:
                    # Compute relevance score using Phase 1b weights
                    relevance = compute_relevance_score(position_data, rag_entry, self.relevance_weights)
                    
                    retrieval_time = (time.time() - start_time) * 1000  # ms
                    self.stats['retrieval_times'].append(retrieval_time)
                    self.stats['relevance_scores'].append(relevance)
                    
                    return (rag_entry, relevance)
        else:
            # Fallback: Linear search by sym_hash
            sym_hash = position_data.get('sym_hash', '')
            for rag_entry in self.rag_entries:
                if rag_entry.sym_hash == sym_hash:
                    relevance = compute_relevance_score(position_data, rag_entry, self.relevance_weights)
                    
                    retrieval_time = (time.time() - start_time) * 1000
                    self.stats['retrieval_times'].append(retrieval_time)
                    self.stats['relevance_scores'].append(relevance)
                    
                    return (rag_entry, relevance)
        
        return None
    
    def process_position(self, position_data: Dict) -> Dict:
        """
        Process a single position during MCTS search.
        
        Logic from claude_instructions.txt:
        1. Compute uncertainty
        2. If uncertainty > threshold: Query RAG
        3. If found with high relevance (>90%): Blend policy/value
        4. If found with low relevance: Force best 2 moves as priority
        5. If not found: Use current MCTS results only
        
        Returns:
            Augmented position data with RAG-enhanced policy/value
        """
        self.stats['total_positions'] += 1
        
        # Compute uncertainty
        uncertainty = self.compute_uncertainty(position_data)
        
        # Check if we should query RAG
        if uncertainty < self.threshold_config.absolute_threshold:
            # Low uncertainty, use current MCTS results
            return position_data
        
        # High uncertainty, query RAG
        self.stats['rag_queries'] += 1
        result = self.query_rag(position_data)
        
        if result is None:
            # No match found in RAG
            self.stats['no_hits'] += 1
            return position_data
        
        rag_entry, relevance = result
        
        # Check relevance threshold
        if relevance >= self.threshold_config.relevance_threshold:
            # High relevance: Blend policy and value
            self.stats['high_relevance_hits'] += 1
            
            current_policy = np.array(position_data.get('policy', []))
            current_value = position_data.get('winrate', 0.5)
            
            blended_policy, blended_value = blend_policy_value(
                current_policy=current_policy,
                current_value=current_value,
                rag_policy=rag_entry.policy,
                rag_value=rag_entry.winrate,
                relevance=relevance,
                blend_factor=0.5  # Can be tuned
            )
            
            # Update position data with blended results
            augmented_data = position_data.copy()
            augmented_data['policy'] = blended_policy.tolist()
            augmented_data['winrate'] = blended_value
            augmented_data['rag_augmented'] = True
            augmented_data['rag_method'] = 'blend'
            augmented_data['relevance'] = relevance
            
            return augmented_data
        else:
            # Low relevance: Force best 2 moves as exploration priority
            self.stats['low_relevance_hits'] += 1
            
            augmented_data = position_data.copy()
            augmented_data['forced_exploration_moves'] = rag_entry.best_2_moves
            augmented_data['rag_augmented'] = True
            augmented_data['rag_method'] = 'forced_exploration'
            augmented_data['relevance'] = relevance
            
            return augmented_data
    
    def run_game(self, game_id: int) -> Dict:
        """
        Run a single game with RAG augmentation.
        
        PLACEHOLDER: Replace with actual KataGo integration.
        This should play a full game, calling process_position() for each MCTS search.
        
        Returns:
            Game results including win/loss and RAG usage statistics
        """
        # PLACEHOLDER: Simulated game
        import random
        
        # Simulate ~250 positions per game
        num_positions = random.randint(200, 300)
        
        for _ in range(num_positions):
            # Simulate position data
            position_data = {
                'sym_hash': f'hash_{random.randint(0, 10000)}',
                'embedding': np.random.randn(128),
                'policy': np.random.dirichlet(np.ones(362)),
                'winrate': random.uniform(0.3, 0.7),
                'score_lead': random.uniform(-5, 5),
                'policy_entropy': random.uniform(2.0, 5.0),
                'value_sparseness': random.uniform(0.2, 0.8),
                'stone_count': random.randint(20, 300),
                'komi': 7.5,
                'child_nodes': []
            }
            
            # Process position (may augment with RAG)
            augmented = self.process_position(position_data)
        
        # Simulate game result
        game_result = {
            'game_id': game_id,
            'win': random.random() > 0.5,  # Replace with actual result
            'game_length': num_positions,
            'positions_processed': self.stats['total_positions'],
            'rag_queries': self.stats['rag_queries'],
            'high_relevance_hits': self.stats['high_relevance_hits'],
            'low_relevance_hits': self.stats['low_relevance_hits'],
            'no_hits': self.stats['no_hits']
        }
        
        return game_result


class Phase1cTuner:
    """Tunes uncertainty threshold for RAG queries using static database"""
    
    def __init__(self,
                 phase1a_config_path: str,
                 phase1b_weights_path: str,
                 rag_database_path: str,
                 output_dir: str = "./tuning_results/phase1c",
                 num_games_per_threshold: int = 100):
        """
        Args:
            phase1a_config_path: Path to best config from Phase 1a
            phase1b_weights_path: Path to relevance weights from Phase 1b
            rag_database_path: Path to pre-populated RAG database (static)
            output_dir: Directory to save results
            num_games_per_threshold: Number of games to test per threshold
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Phase 1a best config (uncertainty function parameters)
        with open(phase1a_config_path, 'r') as f:
            phase1a_data = json.load(f)
            self.phase1a_config = phase1a_data['config']
        
        print(f"Loaded Phase 1a config: w1={self.phase1a_config['w1']:.3f}, w2={self.phase1a_config['w2']:.3f}")
        
        # Load Phase 1b relevance weights
        with open(phase1b_weights_path, 'r') as f:
            phase1b_data = json.load(f)
            self.relevance_weights = phase1b_data['best_weights']
        
        print(f"Loaded Phase 1b relevance weights:")
        print(f"  policy: {self.relevance_weights['policy']:.3f}")
        print(f"  winrate: {self.relevance_weights['winrate']:.3f}")
        print(f"  score_lead: {self.relevance_weights['score_lead']:.3f}")
        print(f"  visit_distribution: {self.relevance_weights['visit_distribution']:.3f}")
        print(f"  stone_count: {self.relevance_weights['stone_count']:.3f}")
        print(f"  komi: {self.relevance_weights['komi']:.3f}")
        
        self.rag_database_path = rag_database_path
        self.num_games_per_threshold = num_games_per_threshold
        self.results: List[Dict] = []
    
    def estimate_percentiles_from_database(self, ground_truth_db_path: str) -> Dict[float, float]:
        """
        Estimate uncertainty percentiles from the ground truth database.
        
        This maps percentiles to absolute threshold values without running games.
        """
        print(f"\nEstimating percentiles from ground truth database...")
        
        with open(ground_truth_db_path, 'r') as f:
            data = json.load(f)
        
        # Compute uncertainty for all positions
        uncertainties = []
        for pos in data['positions']:
            # Compute uncertainty using Phase 1a parameters
            w1 = self.phase1a_config['w1']
            w2 = self.phase1a_config['w2']
            phase_type = self.phase1a_config['phase_function_type']
            phase_coef = self.phase1a_config['phase_coefficients']
            
            E = pos['features']['policy_cross_entropy']
            K = pos['features']['value_sparseness']
            stones = pos['stones_on_board']
            
            # Compute phase
            s = stones / 361.0
            if phase_type == 'linear':
                a, b = phase_coef[0], phase_coef[1]
                phase = a * s + b
            else:
                phase = 1.0
            
            uncertainty = (w1 * E + w2 * K) * phase
            uncertainties.append(uncertainty)
        
        uncertainties = np.array(uncertainties)
        
        # Calculate percentile thresholds
        percentiles_to_test = [5, 10, 15, 20, 25]
        percentile_map = {}
        
        for percentile in percentiles_to_test:
            threshold = np.percentile(uncertainties, 100 - percentile)
            percentile_map[percentile] = threshold
            print(f"  Top {percentile}% → threshold {threshold:.4f}")
        
        # Save mapping
        mapping_file = self.output_dir / "percentile_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(percentile_map, f, indent=2)
        
        # Plot distribution
        self.plot_uncertainty_distribution(uncertainties, percentile_map)
        
        return percentile_map
    
    def plot_uncertainty_distribution(self, scores: np.ndarray, percentile_map: Dict[float, float]):
        """Plot uncertainty score distribution with threshold lines"""
        plt.figure(figsize=(12, 6))
        
        plt.hist(scores, bins=100, alpha=0.7, edgecolor='black')
        plt.xlabel('Uncertainty Score')
        plt.ylabel('Frequency')
        plt.title('Uncertainty Score Distribution (Phase 1a Parameters)')
        
        # Add vertical lines for thresholds
        colors = ['red', 'orange', 'yellow', 'green', 'blue']
        for (percentile, threshold), color in zip(sorted(percentile_map.items()), colors):
            plt.axvline(threshold, color=color, linestyle='--', linewidth=2,
                       label=f'Top {percentile}% (threshold={threshold:.3f})')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = self.output_dir / "uncertainty_distribution.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved distribution plot to {plot_file}")
        plt.close()
    
    def evaluate_threshold(self, config: UncertaintyThresholdConfig) -> RAGAugmentedMetrics:
        """
        Evaluate a specific uncertainty threshold.
        
        Plays games with RAG augmentation at positions exceeding the threshold.
        """
        print(f"\nEvaluating threshold: Top {config.threshold_percentile}% (abs={config.absolute_threshold:.4f})")
        
        # Create game executor with Phase 1a config and Phase 1b weights
        executor = RAGAugmentedGameExecutor(
            rag_database_path=self.rag_database_path,
            uncertainty_config=self.phase1a_config,
            threshold_config=config,
            relevance_weights=self.relevance_weights
        )
        
        # Run games
        game_results = []
        wins = 0
        
        for game_id in range(self.num_games_per_threshold):
            if game_id % 20 == 0:
                print(f"  Game {game_id}/{self.num_games_per_threshold}...")
            
            result = executor.run_game(game_id)
            game_results.append(result)
            
            if result['win']:
                wins += 1
        
        # Aggregate metrics
        win_rate = wins / self.num_games_per_threshold
        
        total_positions = sum(r['positions_processed'] for r in game_results)
        total_queries = sum(r['rag_queries'] for r in game_results)
        
        metrics = RAGAugmentedMetrics(
            config_id=config.config_id,
            win_rate=win_rate,
            total_games=self.num_games_per_threshold,
            total_positions=total_positions,
            rag_queries=total_queries,
            rag_query_rate=total_queries / max(1, total_positions),
            high_relevance_hits=sum(r['high_relevance_hits'] for r in game_results),
            low_relevance_hits=sum(r['low_relevance_hits'] for r in game_results),
            no_hits=sum(r['no_hits'] for r in game_results),
            avg_relevance_score=float(np.mean(executor.stats['relevance_scores'])) if executor.stats['relevance_scores'] else 0.0,
            avg_retrieval_time_ms=float(np.mean(executor.stats['retrieval_times'])) if executor.stats['retrieval_times'] else 0.0,
            avg_game_length=float(np.mean([r['game_length'] for r in game_results])),
            avg_uncertainty_per_game=0.0  # Can be computed if needed
        )
        
        print(f"  Win rate: {metrics.win_rate:.3f} ({wins}/{self.num_games_per_threshold})")
        print(f"  RAG query rate: {metrics.rag_query_rate:.3f} ({metrics.rag_queries}/{metrics.total_positions})")
        print(f"  High relevance hits: {metrics.high_relevance_hits}")
        print(f"  Low relevance hits: {metrics.low_relevance_hits}")
        print(f"  No hits: {metrics.no_hits}")
        print(f"  Avg relevance: {metrics.avg_relevance_score:.3f}")
        print(f"  Avg retrieval time: {metrics.avg_retrieval_time_ms:.2f} ms")
        
        return metrics
    
    def run_tuning(self, ground_truth_db_path: str) -> Tuple[UncertaintyThresholdConfig, RAGAugmentedMetrics]:
        """Run Phase 1c tuning"""
        print("="*80)
        print("PHASE 1C: UNCERTAINTY THRESHOLD TUNING (RAG-AUGMENTED)")
        print("="*80)
        print(f"Output directory: {self.output_dir}")
        print(f"RAG database: {self.rag_database_path}")
        print(f"Games per threshold: {self.num_games_per_threshold}")
        print(f"Mode: READ-ONLY (no new positions stored)")
        print("="*80)
        print(f"Games per threshold: {self.num_games_per_threshold}")
        print(f"Max database size: {self.max_database_size_gb} GB")
        print("="*80)
        
        # Step 1: Estimate percentile thresholds
        percentile_map = self.estimate_percentiles_from_sample()
        
        # Step 2: Generate configs to test
        configs = []
        for percentile, threshold in percentile_map.items():
            config = StorageThresholdConfig(
                percentile=percentile,
                absolute_threshold=threshold
            )
            configs.append(config)
        
        # Step 3: Evaluate each threshold
        results = []
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Testing threshold...")
            metrics = self.evaluate_threshold(config)
            
            result = {
                "config": asdict(config),
                "metrics": asdict(metrics)
            }
            results.append(result)
            
            # Save intermediate results
            result_file = self.output_dir / f"{config.config_id}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # Step 4: Analyze results and find optimal threshold
        best_config, best_metrics = self.find_optimal_threshold(configs, results)
        
        # Step 5: Save all results
        summary_file = self.output_dir / "storage_threshold_results.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "phase1_config": self.phase1_config,
                "percentile_mapping": percentile_map,
                "all_results": results,
                "best_config": asdict(best_config),
                "best_metrics": asdict(best_metrics)
            }, f, indent=2)
        
        # Step 6: Create visualization
        self.plot_threshold_comparison(configs, results)
        
        print("\n" + "="*80)
        print("OPTIMAL STORAGE THRESHOLD")
        print("="*80)
        print(f"Percentile: Top {best_config.percentile}%")
        print(f"Absolute threshold: {best_config.absolute_threshold:.4f}")
        print(f"Win rate: {best_metrics.win_rate:.3f}")
        print(f"Database size: {best_metrics.database_size_mb:.1f} MB")
        print(f"Positions per game: {best_metrics.avg_positions_per_game:.1f}")
        print("="*80)
        
        return best_config, best_metrics
    
    def plot_threshold_comparison(self, configs: List[UncertaintyThresholdConfig], results: List[Dict]):
        """Create comparison plots for different thresholds"""
        metrics_list = [RAGAugmentedMetrics(**r['metrics']) for r in results]
        percentiles = [c.threshold_percentile for c in configs]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Win rate vs percentile
        axes[0, 0].plot(percentiles, [m.win_rate for m in metrics_list], 'o-', linewidth=2)
        axes[0, 0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        axes[0, 0].set_xlabel('Uncertainty Threshold (Top X%)')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_title('Win Rate vs Uncertainty Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: RAG query rate vs percentile
        axes[0, 1].plot(percentiles, [m.rag_query_rate * 100 for m in metrics_list], 
                       's-', linewidth=2, color='orange')
        axes[0, 1].set_xlabel('Uncertainty Threshold (Top X%)')
        axes[0, 1].set_ylabel('RAG Query Rate (%)')
        axes[0, 1].set_title('RAG Usage vs Uncertainty Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Hit distribution
        x = np.arange(len(percentiles))
        width = 0.25
        axes[1, 0].bar(x - width, [m.high_relevance_hits for m in metrics_list], 
                      width, label='High Relevance', color='green', alpha=0.7)
        axes[1, 0].bar(x, [m.low_relevance_hits for m in metrics_list], 
                      width, label='Low Relevance', color='yellow', alpha=0.7)
        axes[1, 0].bar(x + width, [m.no_hits for m in metrics_list], 
                      width, label='No Match', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Uncertainty Threshold (Top X%)')
        axes[1, 0].set_ylabel('Number of Hits')
        axes[1, 0].set_title('RAG Retrieval Results')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f'{p}%' for p in percentiles])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Average relevance score
        axes[1, 1].plot(percentiles, [m.avg_relevance_score for m in metrics_list], 
                       'd-', linewidth=2, color='purple')
        axes[1, 1].axhline(0.90, color='red', linestyle='--', label='Relevance Threshold')
        axes[1, 1].set_xlabel('Uncertainty Threshold (Top X%)')
        axes[1, 1].set_ylabel('Average Relevance Score')
        axes[1, 1].set_title('Retrieval Quality vs Uncertainty Threshold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_file = self.output_dir / "threshold_comparison.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\nSaved comparison plots to {plot_file}")
        plt.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1c: Uncertainty Threshold Tuning with RAG")
    parser.add_argument("--phase1a-config", type=str, required=True,
                       help="Path to best config from Phase 1a (uncertainty function)")
    parser.add_argument("--phase1b-weights", type=str, required=True,
                       help="Path to relevance weights from Phase 1b")
    parser.add_argument("--rag-database", type=str, required=True,
                       help="Path to pre-populated RAG database (static, read-only)")
    parser.add_argument("--ground-truth-db", type=str, required=True,
                       help="Path to ground truth database for percentile estimation")
    parser.add_argument("--output-dir", type=str, default="./tuning_results/phase1c",
                       help="Output directory for results")
    parser.add_argument("--num-games", type=int, default=100,
                       help="Number of games per threshold")
    
    args = parser.parse_args()
    
    tuner = Phase1cTuner(
        phase1a_config_path=args.phase1a_config,
        phase1b_weights_path=args.phase1b_weights,
        rag_database_path=args.rag_database,
        output_dir=args.output_dir,
        num_games_per_threshold=args.num_games
    )
    
    best_config, best_metrics = tuner.run_tuning(
        ground_truth_db_path=args.ground_truth_db
    )
    
    print("\n✓ Phase 1c tuning completed successfully!")
    print(f"  Best threshold: Top {best_config.threshold_percentile}%")
    print(f"  Win rate: {best_metrics.win_rate:.3f}")
    print(f"  RAG effectiveness: {best_metrics.high_relevance_hits} high-relevance hits")


if __name__ == "__main__":
    main()
