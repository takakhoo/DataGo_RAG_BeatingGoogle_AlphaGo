"""
datago_bot.py

Main DataGo bot implementation that integrates KataGo's MCTS with a 
retrieval-augmented generation (RAG) system for improved play on 
uncertain/complex positions.

This bot implements the full RAG-MCTS pipeline:
1. Use KataGo for standard MCTS search (800 visits)
2. Detect uncertain positions using entropy-based metrics
3. Query RAG database for similar complex positions
4. Blend retrieved knowledge with network priors
5. Perform deep MCTS on novel complex positions
6. Store new positions in RAG database
"""

from __future__ import annotations

import json
import math
import time
import logging
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import yaml

# Import DataGo components
from src.clients.katago_client import AnalysisResult
from src.memory.index import ANNIndex
from src.memory.schema import MemoryEntry
from src.blend.blend import (
    rerank_neighbors,
    build_retrieval_prior,
    blend_priors,
)
from src.gating.gate import entropy_of_policy, normalized_entropy
from src.mcts.custom_mcts import CustomMCTS
from src.mcts.network_evaluator import KataGoNetworkEvaluator


logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Represents the current state of the Go game."""
    
    board: np.ndarray  # 19x19 array, 0=empty, 1=black, -1=white
    current_player: int  # 1=black, -1=white
    move_number: int
    komi: float
    history: List[Tuple[str, int]]  # List of (move, player) tuples
    captures: Dict[int, int]  # {player: num_captures}
    ko_point: Optional[Tuple[int, int]]  # Current ko point if any
    
    def stones_on_board(self) -> int:
        """Count total stones on board."""
        return int(np.abs(self.board).sum())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "move_number": self.move_number,
            "current_player": self.current_player,
            "stones_on_board": self.stones_on_board(),
            "komi": self.komi,
            "history": self.history,
            "captures": self.captures,
        }


@dataclass
class MoveDecision:
    """Represents a move decision with associated metadata."""
    
    move: str  # Move in GTP format (e.g., "D4", "Q16", "pass", "resign")
    policy: np.ndarray  # Policy distribution over all moves
    value: float  # Estimated value of position
    winrate: float  # Estimated win rate
    score_lead: float  # Estimated score lead
    
    # RAG-related metadata
    uncertainty: float  # Computed uncertainty score
    rag_queried: bool  # Whether RAG was queried
    rag_hit: bool  # Whether RAG found relevant position
    rag_relevance: float  # Relevance score if RAG hit
    used_deep_search: bool  # Whether deep MCTS was performed
    
    # Timing information
    time_taken_ms: float
    visits: int  # Number of MCTS visits
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "move": self.move,
            "value": float(self.value),
            "winrate": float(self.winrate),
            "score_lead": float(self.score_lead),
            "uncertainty": float(self.uncertainty),
            "rag_queried": self.rag_queried,
            "rag_hit": self.rag_hit,
            "rag_relevance": float(self.rag_relevance) if self.rag_hit else None,
            "used_deep_search": self.used_deep_search,
            "time_taken_ms": float(self.time_taken_ms),
            "visits": self.visits,
        }


class DataGoBot:
    """
    RAG-enhanced Go bot that combines KataGo with retrieval-augmented generation.
    
    The bot operates in the following stages:
    1. Shallow MCTS search using KataGo (default: 800 visits)
    2. Uncertainty detection using policy entropy and value variance
    3. RAG query for high-uncertainty positions
    4. Knowledge blending or forced exploration based on relevance
    5. Deep MCTS for novel complex positions (optional)
    6. Online learning: store new complex positions in RAG
    """
    
    def __init__(self, config_path: str):
        """
        Initialize DataGo bot with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize logging
        self._setup_logging()
        
        # Log experimental features if enabled
        self._log_experimental_features()
        
        # Initialize network evaluator (KataGo for raw network evals only)
        self.network_evaluator = KataGoNetworkEvaluator(
            katago_executable=self.config['katago']['executable_path'],
            model_path=self.config['katago']['model_path'],
            config_path=self.config['katago']['config_path'],
            board_size=self.config['katago']['board_size'],
        )
        
        # Initialize custom MCTS (for search with modified priors)
        self.mcts = CustomMCTS(
            network_evaluator=self.network_evaluator,
            c_puct=self.config.get('mcts', {}).get('c_puct', 1.5),
            temperature=self.config.get('mcts', {}).get('temperature', 1.0),
        )
        
        # Keep old KataGo process reference for compatibility
        self.katago_process = self.network_evaluator.process
        
        # Initialize RAG database
        logger.info("Initializing RAG database...")
        rag_config = self.config['rag_database']
        
        # Use a default embedding dimension (sym_hash based, can be adjusted)
        embedding_dim = rag_config.get('embedding_dim', 64)  # Default 64 for sym_hash
        
        self.rag_index = ANNIndex(
            dim=embedding_dim,
            space=rag_config['ann'].get('distance_metric', 'cosine'),
        )
        
        # Game state
        self.game_state: Optional[GameState] = None
        
        # Statistics tracking
        self.stats = {
            "moves_played": 0,
            "rag_queries": 0,
            "rag_hits": 0,
            "deep_searches": 0,
            "positions_stored": 0,
            "total_time_ms": 0.0,
            "forced_explorations": 0,
        }
        
        # Recursion tracking for RAG queries
        self.current_recursion_depth = 0
        self.max_recursion_depth = self.config['recursion']['max_recursion_depth']
        
        # Background analysis queue (for online learning)
        self.background_queue: List[Dict[str, Any]] = []
        
        logger.info("DataGo bot initialized successfully")
    
    def _setup_logging(self):
        """Configure logging based on config."""
        log_level = getattr(logging, self.config['logging']['level'])
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # File handler (if enabled)
        handlers = [console_handler]
        if self.config['logging'].get('log_file'):
            log_file = Path(self.config['logging']['log_file'])
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
        )
    
    def _log_experimental_features(self):
        """Log information about enabled experimental features."""
        exp_config = self.config.get('experimental', {})
        
        enabled_features = []
        
        if exp_config.get('use_gpu_batch', False):
            enabled_features.append(
                f"GPU batch processing (device={exp_config.get('gpu_device', 0)}, "
                f"batch_size={exp_config.get('gpu_batch_size', 64)})"
            )
        
        if exp_config.get('adaptive_threshold', False):
            enabled_features.append("Adaptive uncertainty thresholds")
        
        if exp_config.get('learn_from_opponent', False):
            enabled_features.append("Learning from opponent moves")
        
        if exp_config.get('parallel_mcts', False):
            enabled_features.append(
                f"Parallel MCTS ({exp_config.get('num_mcts_threads', 4)} threads)"
            )
        
        if enabled_features:
            logger.warning(
                "EXPERIMENTAL FEATURES ENABLED (may not be fully tuned):\n  - " +
                "\n  - ".join(enabled_features)
            )
        else:
            logger.info("No experimental features enabled")
    
    def is_experimental_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if an experimental feature is enabled.
        
        Args:
            feature_name: Name of the experimental feature
                         (use_gpu_batch, adaptive_threshold, 
                          learn_from_opponent, parallel_mcts)
        
        Returns:
            True if feature is enabled
        """
        return self.config.get('experimental', {}).get(feature_name, False)
    
    def get_experimental_config(self, key: str, default: Any = None) -> Any:
        """
        Get experimental configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        return self.config.get('experimental', {}).get(key, default)
    
    def _evaluate_position(
        self,
        board_state: np.ndarray,
        moves: List[str],
    ) -> Tuple[Dict[str, float], float]:
        """
        Get raw network evaluation (policy + value) without MCTS.
        
        Args:
            board_state: Current board state
            moves: Move history in GTP format
            
        Returns:
            Tuple of (policy_dict, value)
        """
        position_hash = f"pos_{len(moves)}"  # TODO: Better hashing
        
        return self.network_evaluator.evaluate(
            position_hash=position_hash,
            board_state=board_state,
            moves=moves,
            komi=self.game_state.komi,
            use_cache=True,
        )
    
    def _stop_katago(self):
        """Stop KataGo subprocess."""
        if hasattr(self, 'network_evaluator') and self.network_evaluator:
            self.network_evaluator.stop()
            logger.info("KataGo network evaluator stopped")
    
    def _run_mcts_search(
        self,
        board_state: np.ndarray,
        moves: List[str],
        num_visits: int,
        modified_prior: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """
        Run custom MCTS search with optionally modified priors.
        
        This is the core of the RAG-MCTS architecture:
        1. Get raw network policy and value
        2. Optionally blend with RAG knowledge
        3. Run MCTS with the (potentially modified) prior
        
        Args:
            board_state: Current board state
            moves: Move history
            num_visits: Number of MCTS simulations
            modified_prior: Optional modified policy (e.g., blended with RAG)
            
        Returns:
            Tuple of (move_probabilities, estimated_value, mcts_stats)
        """
        position_hash = f"pos_{len(moves)}"  # TODO: Better hashing
        
        # Run MCTS with (optionally modified) priors
        move_probs, value = self.mcts.search(
            position_hash=position_hash,
            num_simulations=num_visits,
            modified_prior=modified_prior,
        )
        
        # Get statistics
        stats = self.mcts.get_statistics()
        
        return move_probs, value, stats
    
    def compute_uncertainty(
        self,
        policy: Dict[str, float],
        mcts_stats: Dict[str, Any],
        stones_on_board: int,
    ) -> float:
        """
        Compute uncertainty score for current position.
        
        Uncertainty = (w1*E + w2*K) * phase(stones_on_board)
        
        Where:
        - E: Policy cross-entropy (normalized)
        - K: Value distribution sparseness (from MCTS Q-values)
        - phase: Game phase multiplier based on stones_on_board
        
        Args:
            policy: Policy distribution (dict: move -> prob)
            mcts_stats: MCTS statistics including top moves with Q-values
            stones_on_board: Number of stones on board
            
        Returns:
            Uncertainty score (higher = more uncertain)
        """
        config = self.config['uncertainty_detection']
        
        # Convert policy dict to array for entropy computation
        policy_array = np.array(list(policy.values()))
        
        # Compute E: Policy cross-entropy
        E = normalized_entropy(policy_array)
        
        # Compute K: Value distribution sparseness
        # Extract Q-values from top moves and compute variance
        top_moves = mcts_stats.get('top_moves', [])
        if top_moves:
            q_values = np.array([m.get('q_value', 0.0) for m in top_moves])
            value_variance = float(np.var(q_values))
            # Normalize by theoretical max variance (0.25 for values in [0,1])
            K = min(1.0, value_variance / 0.25)
        else:
            K = 0.0
        
        # Compute phase multiplier
        phase_mult = self._compute_phase_multiplier(stones_on_board)
        
        # Combined uncertainty
        w1 = config['w1']
        w2 = config['w2']
        uncertainty = (w1 * E + w2 * K) * phase_mult
        
        return float(uncertainty)
    
    def _compute_phase_multiplier(self, stones_on_board: int) -> float:
        """
        Compute game phase multiplier based on stones on board.
        
        Args:
            stones_on_board: Number of stones on board
            
        Returns:
            Phase multiplier
        """
        config = self.config['uncertainty_detection']
        phase_type = config['phase_function_type']
        coeffs = config['phase_function_coefficients']
        
        # Normalize stone count to [0, 1]
        s = stones_on_board / 361.0
        
        if phase_type == "linear":
            # phase(s) = a*s + b
            a, b = coeffs
            return a * s + b
        
        elif phase_type == "exponential":
            # phase(s) = a*exp(b*s) + c
            a, b, c = coeffs
            return a * math.exp(b * s) + c
        
        elif phase_type == "piecewise":
            # Different multipliers for early/mid/late game
            early_mult, mid_mult, late_mult = coeffs
            if stones_on_board < 120:
                return early_mult
            elif stones_on_board < 240:
                return mid_mult
            else:
                return late_mult
        
        else:
            logger.warning(f"Unknown phase function type: {phase_type}, using 1.0")
            return 1.0
    
    def query_rag(
        self,
        position_hash: str,
        sym_hash: str,
        policy: np.ndarray,
        winrate: float = None,
        score_lead: float = None,
        visit_distribution: np.ndarray = None,
    ) -> Tuple[bool, Optional[MemoryEntry], float, bool]:
        """
        Query RAG database for similar positions using 1-NN exact matching strategy.
        
        ALGORITHM (from config rag_strategy):
        1. ANN lookup: Use cosine similarity to find 1-nearest neighbor based on sym_hash
           - Retrieves the closest sym_hash in the database
           - Does NOT guarantee exact match, just closest match
        2. Check if retrieved sym_hash is IDENTICAL to query sym_hash:
           a. If NOT identical (different board position):
              → Return neighbor with use_precomputed=False (force exploration)
           b. If IDENTICAL (exact same board position):
              → Compute similarity score using relevance_weights on other parameters
              → If similarity >= relevance_threshold: Return with use_precomputed=True
              → If similarity < relevance_threshold: Return with use_precomputed=False
        
        Args:
            position_hash: Hash of current position
            sym_hash: Symmetry-invariant hash for ANN lookup
            policy: Policy distribution for current position
            winrate: Current winrate (optional)
            score_lead: Current score lead (optional)
            visit_distribution: Current visit distribution (optional)
            
        Returns:
            Tuple of (hit, neighbor_entry, relevance, use_precomputed)
            - hit: Whether any neighbor was found
            - neighbor_entry: The retrieved MemoryEntry (or None)
            - relevance: Relevance score (0.0 if no exact match)
            - use_precomputed: Whether to use precomputed optimal move directly
        """
        start_time = time.time()
        
        rag_config = self.config['rag_strategy']
        
        # Step 1: Query ANN index for 1-nearest neighbor using sym_hash
        # Note: The actual ANN lookup would use sym_hash embedding
        k = self.config['rag_database']['ann']['num_neighbors']  # Should be 1
        neighbors = self.rag_index.query(
            query_embedding=policy,  # In production, would query by sym_hash
            k=k,
        )
        
        if not neighbors:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"RAG query: no neighbors found ({elapsed_ms:.2f}ms)")
            return False, None, 0.0, False
        
        # Get the 1-nearest neighbor
        neighbor_entry, ann_distance = neighbors[0]
        
        # Step 2: Check if sym_hash is IDENTICAL to query
        stored_sym_hash = neighbor_entry.metadata.get('sym_hash', '')
        
        if rag_config.get('check_exact_sym_hash_match', True):
            if stored_sym_hash != sym_hash:
                # Different board position: force exploration of RAG's best move
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(
                    f"RAG query: sym_hash mismatch, force exploration "
                    f"({elapsed_ms:.2f}ms)"
                )
                return True, neighbor_entry, 0.0, False  # use_precomputed=False
        
        # Step 3: Exact sym_hash match - compute detailed similarity
        relevance = self._compute_relevance(
            neighbor_entry,
            policy,
            current_winrate=winrate,
            current_score_lead=score_lead,
            current_visit_distribution=visit_distribution,
        )
        
        # Step 4: Compare relevance to threshold
        threshold = self.config['relevance_weights']['relevance_threshold']
        use_precomputed = relevance >= threshold
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if use_precomputed:
            logger.debug(
                f"RAG query: exact match with high relevance={relevance:.3f} >= {threshold}, "
                f"use precomputed move ({elapsed_ms:.2f}ms)"
            )
        else:
            logger.debug(
                f"RAG query: exact match but low relevance={relevance:.3f} < {threshold}, "
                f"force exploration ({elapsed_ms:.2f}ms)"
            )
        
        return True, neighbor_entry, relevance, use_precomputed
    
    def _compute_similarity(
        self,
        value1: Any,
        value2: Any,
        similarity_config: Dict[str, Any],
    ) -> float:
        """
        Generic similarity computation based on config method.
        
        Args:
            value1: First value (current)
            value2: Second value (stored)
            similarity_config: Configuration dict with 'method' and other params
            
        Returns:
            Similarity score in [0, 1]
        """
        method = similarity_config.get('method', 'cosine')
        
        if method == 'cosine':
            # Cosine similarity for vectors
            if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                norm1 = np.linalg.norm(value1)
                norm2 = np.linalg.norm(value2)
                if norm1 > 0 and norm2 > 0:
                    return float(np.dot(value1, value2) / (norm1 * norm2))
            return 0.0
        
        elif method == 'inverse_absolute_diff':
            # 1 - |v1 - v2| / max_diff
            max_diff = similarity_config.get('max_diff', 1.0)
            diff = abs(float(value1) - float(value2))
            return float(max(0.0, 1.0 - min(diff / max_diff, 1.0)))
        
        elif method == 'exact_match':
            # Binary exact match with tolerance
            tolerance = similarity_config.get('tolerance', 0.01)
            return 1.0 if abs(float(value1) - float(value2)) < tolerance else 0.0
        
        elif method == 'euclidean':
            # Euclidean distance-based similarity
            if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                dist = np.linalg.norm(value1 - value2)
                return float(1.0 / (1.0 + dist))
            return 0.0
        
        elif method == 'kl_divergence':
            # KL divergence-based similarity
            if isinstance(value1, np.ndarray) and isinstance(value2, np.ndarray):
                # Add small epsilon to avoid log(0)
                eps = 1e-10
                p1 = value1 + eps
                p2 = value2 + eps
                p1 = p1 / p1.sum()
                p2 = p2 / p2.sum()
                kl = np.sum(p1 * np.log(p1 / p2))
                return float(np.exp(-kl))  # Convert to similarity
            return 0.0
        
        else:
            logger.warning(f"Unknown similarity method: {method}, returning 0.0")
            return 0.0
    
    def _compute_relevance(
        self,
        rag_entry: MemoryEntry,
        current_policy: np.ndarray,
        current_winrate: float = None,
        current_score_lead: float = None,
        current_visit_distribution: np.ndarray = None,
    ) -> float:
        """
        Compute relevance score between RAG entry and current position.
        
        Uses weighted combination of multiple similarity metrics as defined
        in the config's similarity_functions section.
        
        Args:
            rag_entry: Retrieved memory entry
            current_policy: Current position's policy
            current_winrate: Current winrate (optional)
            current_score_lead: Current score lead (optional)
            current_visit_distribution: Current visit distribution (optional)
            
        Returns:
            Relevance score in [0, 1]
        """
        weights = self.config['relevance_weights']
        sim_functions = self.config['similarity_functions']
        
        # Policy similarity
        stored_policy = rag_entry.policy
        if sim_functions['policy_similarity'].get('normalize', True):
            # Normalize policies before comparison
            current_policy_norm = current_policy / (current_policy.sum() + 1e-9)
            stored_policy_norm = stored_policy / (stored_policy.sum() + 1e-9)
            policy_sim = self._compute_similarity(
                current_policy_norm,
                stored_policy_norm,
                sim_functions['policy_similarity']
            )
        else:
            policy_sim = self._compute_similarity(
                current_policy,
                stored_policy,
                sim_functions['policy_similarity']
            )
        
        # Winrate similarity
        if current_winrate is None:
            current_winrate = 0.5
        stored_winrate = rag_entry.metadata.get('winrate', 0.5)
        winrate_sim = self._compute_similarity(
            current_winrate,
            stored_winrate,
            sim_functions['winrate_similarity']
        )
        
        # Score lead similarity
        if current_score_lead is None:
            current_score_lead = 0.0
        stored_score = rag_entry.metadata.get('score_lead', 0.0)
        score_sim = self._compute_similarity(
            current_score_lead,
            stored_score,
            sim_functions['score_lead_similarity']
        )
        
        # Visit distribution similarity
        if current_visit_distribution is not None:
            stored_visit_dist = rag_entry.metadata.get('visit_distribution', None)
            if stored_visit_dist is not None:
                stored_visit_dist = np.array(stored_visit_dist)
                if sim_functions['visit_distribution_similarity'].get('normalize', True):
                    current_visit_norm = current_visit_distribution / (current_visit_distribution.sum() + 1e-9)
                    stored_visit_norm = stored_visit_dist / (stored_visit_dist.sum() + 1e-9)
                    visit_sim = self._compute_similarity(
                        current_visit_norm,
                        stored_visit_norm,
                        sim_functions['visit_distribution_similarity']
                    )
                else:
                    visit_sim = self._compute_similarity(
                        current_visit_distribution,
                        stored_visit_dist,
                        sim_functions['visit_distribution_similarity']
                    )
            else:
                visit_sim = 0.5  # Default if no stored visit distribution
        else:
            visit_sim = 0.5  # Default if no current visit distribution
        
        # Stone count similarity (phase matching)
        current_stones = self.game_state.stones_on_board()
        stored_stones = rag_entry.metadata.get('stone_count', 180)
        stone_sim = self._compute_similarity(
            current_stones,
            stored_stones,
            sim_functions['stone_count_similarity']
        )
        
        # Komi similarity (exact match)
        current_komi = self.game_state.komi
        stored_komi = rag_entry.metadata.get('komi', 7.5)
        komi_sim = self._compute_similarity(
            current_komi,
            stored_komi,
            sim_functions['komi_similarity']
        )
        
        # Weighted combination
        relevance = (
            weights['policy_weight'] * policy_sim +
            weights['winrate_weight'] * winrate_sim +
            weights['score_lead_weight'] * score_sim +
            weights['visit_distribution_weight'] * visit_sim +
            weights['stone_count_weight'] * stone_sim +
            weights['komi_weight'] * komi_sim
        )
        
        return float(relevance)
    
    def force_exploration(
        self,
        network_policy: Dict[str, float],
        rag_entry: MemoryEntry,
    ) -> Dict[str, float]:
        """
        Force exploration of top RAG moves as first priority.
        
        From config recursion.force_exploration_top_n:
        When relevance < threshold, force explore the top N moves from RAG
        as first priority in tree search.
        
        Args:
            network_policy: Original policy from network
            rag_entry: RAG entry with moves to force explore
            
        Returns:
            Modified policy with forced exploration moves prioritized
        """
        force_top_n = self.config['recursion']['force_exploration_top_n']
        
        # Get top moves from RAG entry
        rag_best_moves = rag_entry.best_moves[:force_top_n]
        
        if not rag_best_moves:
            logger.warning("RAG entry has no best moves for forced exploration")
            return network_policy
        
        # Create modified policy with forced exploration
        modified_policy = network_policy.copy()
        
        # Boost the RAG-recommended moves significantly
        boost_factor = 2.0  # Make them twice as likely to be explored first
        for move_info in rag_best_moves:
            move_str = move_info.get('move', '')
            if move_str in modified_policy:
                modified_policy[move_str] *= boost_factor
        
        # Renormalize
        total = sum(modified_policy.values())
        if total > 0:
            modified_policy = {k: v / total for k, v in modified_policy.items()}
        
        logger.info(
            f"Forced exploration of top {force_top_n} RAG moves: "
            f"{[m.get('move') for m in rag_best_moves]}"
        )
        self.stats['forced_explorations'] += 1
        
        return modified_policy
    
    def blend_with_rag(
        self,
        network_policy: Dict[str, float],
        rag_entry: MemoryEntry,
    ) -> Dict[str, float]:
        """
        Blend network policy with RAG retrieval prior.
        
        Note: With 1-NN strategy, we use only the single retrieved neighbor
        as specified in config blending.top_n_neighbors = 1
        
        Args:
            network_policy: Policy from KataGo (dict: move -> prob)
            rag_entry: Retrieved RAG entry (single neighbor from 1-NN)
            
        Returns:
            Blended policy distribution
        """
        # Convert single entry to list format expected by rerank_neighbors
        rag_neighbors = [(rag_entry, 0.0)]  # Distance not used in 1-NN case
        
        # Rerank neighbors based on reachability and structure
        reranked = rerank_neighbors(
            rag_neighbors,
            current_policy=np.array(list(network_policy.values())),
            alpha=self.config['blending']['reranking_alpha'],
            gamma=self.config['blending']['reranking_gamma'],
        )
        
        # Build retrieval prior from reranked neighbors
        retrieval_prior = build_retrieval_prior(
            reranked,
            current_symmetry=0,  # Placeholder
            board_x=self.config['katago']['board_size'],
            board_y=self.config['katago']['board_size'],
            top_n=self.config['blending']['top_n_neighbors'],
        )
        
        # Blend network prior with retrieval prior
        blended = blend_priors(
            network_policy,
            retrieval_prior,
            beta=self.config['blending']['beta'],
            top_n=self.config['blending']['top_n_moves'],
        )
        
        return blended
    
    def deep_mcts_search(
        self,
        board_state: np.ndarray,
        moves: List[str],
        modified_prior: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
        """
        Perform deep MCTS search on complex position with early stopping.
        
        Implements convergence checking based on config:
        - Check every convergence_check_interval visits
        - Stop if policy changes < policy_convergence_threshold
        - Stop if value changes < value_convergence_threshold
        - Minimum visits: min_visits_before_convergence
        
        Args:
            board_state: Current board state
            moves: Move history leading to this position
            modified_prior: Optional modified policy prior (e.g., from RAG)
            
        Returns:
            Tuple of (move_probabilities, value, stats)
        """
        deep_config = self.config['deep_mcts']
        max_visits = deep_config['max_visits']
        policy_threshold = deep_config['policy_convergence_threshold']
        value_threshold = deep_config['value_convergence_threshold']
        check_interval = deep_config['convergence_check_interval']
        min_visits = deep_config['min_visits_before_convergence']
        
        logger.info(f"Starting deep MCTS search with max {max_visits} visits")
        
        # Track previous results for convergence checking
        prev_policy_array = None
        prev_value = None
        current_visits = min_visits  # Start with minimum visits
        
        move_probs = {}
        value = 0.5
        stats = {}
        
        while current_visits <= max_visits:
            # Run MCTS with current visit count
            move_probs, value, stats = self._run_mcts_search(
                board_state,
                moves,
                num_visits=current_visits,
                modified_prior=modified_prior,
            )
            
            # Check convergence if past minimum visits
            if current_visits >= min_visits and prev_policy_array is not None:
                # Convert policy dict to array for comparison
                current_policy_array = np.array(list(move_probs.values()))
                
                # Compute policy change (L1 distance)
                # Align arrays if sizes differ
                min_len = min(len(current_policy_array), len(prev_policy_array))
                policy_change = np.abs(
                    current_policy_array[:min_len] - prev_policy_array[:min_len]
                ).sum()
                
                # Compute value change
                value_change = abs(value - prev_value)
                
                logger.debug(
                    f"Deep MCTS @ {current_visits} visits: "
                    f"policy_change={policy_change:.4f}, value_change={value_change:.4f}"
                )
                
                # Check if converged
                if policy_change < policy_threshold and value_change < value_threshold:
                    logger.info(
                        f"Deep MCTS converged at {current_visits} visits "
                        f"(policy_change={policy_change:.4f}, value_change={value_change:.4f})"
                    )
                    self.stats['deep_searches'] += 1
                    return move_probs, value, stats
            
            # Update previous results
            prev_policy_array = np.array(list(move_probs.values()))
            prev_value = value
            
            # Increment visits for next iteration
            current_visits = min(current_visits + check_interval, max_visits)
        
        # Reached max visits without convergence
        logger.info(f"Deep MCTS completed {max_visits} visits (max reached)")
        self.stats['deep_searches'] += 1
        
        return move_probs, value, stats
    
    def store_position(
        self,
        position_hash: str,
        sym_hash: str,
        analysis: Dict[str, Any],
        uncertainty: float,
    ):
        """
        Store analyzed position in RAG database.
        
        Args:
            position_hash: Hash of position
            sym_hash: Symmetry-invariant hash
            analysis: KataGo analysis results
            uncertainty: Computed uncertainty score
        """
        # Extract relevant information
        policy = np.array(analysis.get('policy', []))
        
        # Create memory entry
        entry = MemoryEntry(
            position_hash=position_hash,
            policy=policy,
            best_moves=analysis.get('moveInfos', [])[:2],  # Top 2 moves
            metadata={
                'sym_hash': sym_hash,
                'winrate': analysis.get('winrate', 0.5),
                'score_lead': analysis.get('scoreLead', 0.0),
                'uncertainty': uncertainty,
                'stone_count': self.game_state.stones_on_board(),
                'komi': self.game_state.komi,
                'move_number': self.game_state.move_number,
                'visits': analysis.get('visits', 0),
                'query_count': 0,  # Initialize query frequency tracking
                'last_accessed': time.time(),
            }
        )
        
        # Add to RAG database
        self.rag_index.add(entry)
        self.stats['positions_stored'] += 1
        
        logger.info(f"Stored position {position_hash} in RAG database")
    
    def queue_background_analysis(
        self,
        position_hash: str,
        board_state: np.ndarray,
        moves: List[str],
        uncertainty: float,
    ):
        """
        Queue position for background deep analysis.
        
        From config online_learning:
        - Only queue if enable_background_analysis is True
        - Check max_background_jobs limit
        - Only analyze if uncertainty >= background_analysis_threshold
        
        Args:
            position_hash: Hash of position
            board_state: Board state
            moves: Move history
            uncertainty: Uncertainty score
        """
        online_config = self.config['online_learning']
        
        if not online_config.get('enable_background_analysis', True):
            return
        
        # Check threshold
        bg_threshold = online_config.get('background_analysis_threshold', 0.80)
        if uncertainty < bg_threshold:
            logger.debug(
                f"Position uncertainty {uncertainty:.3f} < threshold {bg_threshold}, "
                "skipping background analysis"
            )
            return
        
        # Check queue size limit
        max_jobs = online_config.get('max_background_jobs', 10)
        if len(self.background_queue) >= max_jobs:
            logger.warning(
                f"Background queue full ({len(self.background_queue)}/{max_jobs}), "
                "skipping new analysis"
            )
            return
        
        # Add to queue
        job = {
            'position_hash': position_hash,
            'board_state': board_state.copy(),
            'moves': moves.copy(),
            'uncertainty': uncertainty,
            'queued_at': time.time(),
        }
        self.background_queue.append(job)
        
        logger.debug(
            f"Queued background analysis for position {position_hash} "
            f"(uncertainty={uncertainty:.3f})"
        )
    
    def process_background_queue(self, max_jobs: int = 1):
        """
        Process background analysis jobs from queue.
        
        Args:
            max_jobs: Maximum number of jobs to process in this call
        """
        if not self.config['online_learning'].get('enabled', True):
            return
        
        processed = 0
        while self.background_queue and processed < max_jobs:
            job = self.background_queue.pop(0)
            
            logger.info(
                f"Processing background analysis for position {job['position_hash']}"
            )
            
            # Perform deep MCTS
            analysis = self.deep_mcts_search(
                job['board_state'],
                job['moves'],
            )
            
            # Store in RAG database
            sym_hash = "sym_hash_placeholder"  # TODO: Compute actual sym_hash
            self.store_position(
                job['position_hash'],
                sym_hash,
                analysis,
                job['uncertainty'],
            )
            
            processed += 1
        
        if processed > 0:
            logger.info(f"Processed {processed} background analysis jobs")
    
    def should_resign(self, winrate: float) -> bool:
        """
        Determine if bot should resign based on winrate.
        
        From config game:
        - Resign if winrate < resignation_threshold
        - Only after min_moves_before_resignation moves
        - Returns False if resignation is disabled (threshold = 0.0)
        
        Args:
            winrate: Current estimated winrate
            
        Returns:
            True if should resign
        """
        game_config = self.config['game']
        resignation_threshold = game_config.get('resignation_threshold', 0.05)
        min_moves = game_config.get('min_moves_before_resignation', 100)
        
        # Check if resignation is disabled
        if resignation_threshold <= 0.0:
            return False
        
        # Check minimum moves
        if self.game_state.move_number < min_moves:
            return False
        
        # Check winrate threshold
        should_resign = winrate < resignation_threshold
        
        if should_resign:
            logger.info(
                f"Resignation condition met: winrate {winrate:.4f} < "
                f"threshold {resignation_threshold} at move {self.game_state.move_number}"
            )
        
        return should_resign
    
    def check_time_limit(self, elapsed_ms: float) -> bool:
        """
        Check if time limit for move has been exceeded.
        
        From config katago.max_time_per_move:
        - Returns True if time limit exceeded
        - Returns False if no time limit (max_time_per_move = 0)
        
        Args:
            elapsed_ms: Time elapsed in milliseconds
            
        Returns:
            True if time limit exceeded
        """
        max_time_sec = self.config['katago'].get('max_time_per_move', 30.0)
        
        if max_time_sec <= 0:
            return False  # No time limit
        
        max_time_ms = max_time_sec * 1000
        return elapsed_ms >= max_time_ms
    
    def generate_move(self) -> MoveDecision:
        """
        Generate next move for current game state.
        
        NEW ARCHITECTURE: RAG-MCTS with Custom Search
        1. Get raw network policy + value (no MCTS yet)
        2. Compute uncertainty
        3. If uncertain: Query RAG and blend priors
        4. Run custom MCTS with (potentially blended) priors
        5. Select move from MCTS results
        
        Returns:
            MoveDecision with chosen move and metadata
        """
        start_time = time.time()
        
        moves_history = [m[0] for m in self.game_state.history]
        
        # Step 1: Get raw network evaluation (policy + value, NO MCTS yet)
        logger.debug("Step 1: Getting raw network evaluation")
        network_policy, network_value = self._evaluate_position(
            self.game_state.board,
            moves_history,
        )
        
        # Step 2: Run shallow MCTS to assess uncertainty
        logger.debug(f"Step 2: Running shallow MCTS ({self.config['katago']['visits']} visits)")
        move_probs, estimated_value, mcts_stats = self._run_mcts_search(
            self.game_state.board,
            moves_history,
            num_visits=self.config['katago']['visits'],
            modified_prior=None,  # Use raw network policy
        )
        
        # Step 3: Compute uncertainty
        logger.debug("Step 3: Computing uncertainty")
        uncertainty = self.compute_uncertainty(
            move_probs,
            mcts_stats,
            self.game_state.stones_on_board(),
        )
        
        # Initialize decision metadata
        rag_queried = False
        rag_hit = False
        rag_relevance = 0.0
        used_deep_search = False
        used_blended_prior = False
        final_move_probs = move_probs
        final_value = estimated_value
        
        # Step 4: Check if position is uncertain enough to query RAG
        logger.debug("Step 4: Checking uncertainty threshold")
        threshold = self.config['rag_query']['uncertainty_threshold']
        max_queries = self.config['rag_query']['max_queries_per_game']
        
        if uncertainty >= threshold and self.stats['rag_queries'] < max_queries:
            rag_queried = True
            self.stats['rag_queries'] += 1
            logger.info(f"Position is uncertain ({uncertainty:.3f}), querying RAG")
            
            # Query RAG database using 1-NN exact matching strategy
            position_hash = "hash_placeholder"  # TODO: Implement proper hashing
            sym_hash = "sym_hash_placeholder"  # TODO: Compute actual sym_hash
            
            # Convert network policy for RAG query
            network_policy_array = np.array(list(network_policy.values()))
            
            rag_hit, neighbor_entry, rag_relevance, use_precomputed = self.query_rag(
                position_hash,
                sym_hash,
                network_policy_array,
                winrate=estimated_value,
                score_lead=0.0,  # TODO: Extract from MCTS stats
            )
            
            if rag_hit and neighbor_entry:
                self.stats['rag_hits'] += 1
                self.update_query_statistics(position_hash)
                
                if use_precomputed:
                    # High relevance (≥0.95): Use RAG's optimal move directly
                    logger.info(f"High relevance ({rag_relevance:.3f}), using precomputed move")
                    best_rag_moves = neighbor_entry.best_moves
                    if best_rag_moves:
                        # Use the stored best move directly
                        final_move_probs = {best_rag_moves[0].get('move'): 1.0}
                else:
                    # Low relevance: Blend RAG with network prior and re-run MCTS
                    logger.info(f"Low relevance ({rag_relevance:.3f}), blending with RAG")
                    
                    # **THIS IS THE KEY RAG-MCTS INTEGRATION**
                    # Blend network policy with RAG retrieval prior
                    blended_prior = self.blend_with_rag(network_policy, neighbor_entry)
                    used_blended_prior = True
                    
                    # Re-run MCTS with blended prior
                    logger.debug("Re-running MCTS with blended prior")
                    final_move_probs, final_value, mcts_stats = self._run_mcts_search(
                        self.game_state.board,
                        moves_history,
                        num_visits=self.config['katago']['visits'],
                        modified_prior=blended_prior,  # Use blended policy!
                    )
                    
                    logger.info(f"MCTS with blended prior complete")
            else:
                # No RAG hits: Consider deep search for novel complex positions
                logger.info("No RAG hits, position is novel")
                if self.config['online_learning']['enabled']:
                    bg_threshold = self.config['online_learning'].get(
                        'background_analysis_threshold', 0.80
                    )
                    
                    if uncertainty >= bg_threshold + 0.1:
                        # Critical uncertainty: Run deep search immediately
                        logger.info("Critical uncertainty, running deep MCTS")
                        final_move_probs, final_value, mcts_stats = self.deep_mcts_search(
                            self.game_state.board,
                            moves_history,
                            modified_prior=None,
                        )
                        used_deep_search = True
                        
                        # Store for future RAG queries
                        self.store_position(
                            position_hash,
                            sym_hash,
                            {'policy': final_move_probs, 'winrate': final_value},
                            uncertainty,
                        )
                    else:
                        # Queue for background analysis
                        self.queue_background_analysis(
                            position_hash,
                            self.game_state.board,
                            moves_history,
                            uncertainty,
                        )
        
        # Step 5: Select best move from MCTS probabilities
        logger.debug("Step 5: Selecting best move")
        if final_move_probs:
            best_move = max(final_move_probs.items(), key=lambda x: x[1])[0]
        else:
            best_move = "pass"
        
        # Check resignation condition
        if self.should_resign(final_value):
            best_move = "resign"
            logger.info(f"Bot resigning at move {self.game_state.move_number}")
        
        # Step 6: Create move decision
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats['total_time_ms'] += elapsed_ms
        self.stats['moves_played'] += 1
        
        # Convert move_probs dict to array for policy field
        policy_array = np.array(list(final_move_probs.values()))
        
        decision = MoveDecision(
            move=best_move,
            policy=policy_array,
            value=final_value,
            winrate=final_value,
            score_lead=0.0,  # TODO: Extract from MCTS stats
            uncertainty=uncertainty,
            rag_queried=rag_queried,
            rag_hit=rag_hit,
            rag_relevance=rag_relevance,
            used_deep_search=used_deep_search,
            time_taken_ms=elapsed_ms,
            visits=self.config['katago']['visits'],
        )
        
        logger.info(
            f"Move {self.game_state.move_number}: {best_move} "
            f"(uncertainty={uncertainty:.3f}, rag_hit={rag_hit}, "
            f"blended={used_blended_prior}, time={elapsed_ms:.1f}ms)"
        )
        
        return decision
    
    def new_game(
        self,
        board_size: int = 19,
        komi: float = 7.5,
        player_color: int = 1,
    ):
        """
        Start a new game.
        
        Args:
            board_size: Size of go board
            komi: Komi value
            player_color: 1 for black, -1 for white
        """
        self.game_state = GameState(
            board=np.zeros((board_size, board_size), dtype=int),
            current_player=1,  # Black plays first
            move_number=0,
            komi=komi,
            history=[],
            captures={1: 0, -1: 0},
            ko_point=None,
        )
        
        # Reset statistics
        self.stats = {
            "moves_played": 0,
            "rag_queries": 0,
            "rag_hits": 0,
            "deep_searches": 0,
            "positions_stored": 0,
            "total_time_ms": 0.0,
            "forced_explorations": 0,
        }
        
        # Reset recursion tracking
        self.current_recursion_depth = 0
        
        # Clear background queue
        self.background_queue.clear()
        
        logger.info(f"New game started: board_size={board_size}, komi={komi}")
    
    def check_database_size(self) -> Tuple[float, bool]:
        """
        Check if database size exceeds limit.
        
        Returns:
            Tuple of (current_size_gb, needs_pruning)
        """
        storage_config = self.config['rag_database']['storage']
        db_path = Path(self.config['rag_database']['database_path'])
        
        if not db_path.exists():
            return 0.0, False
        
        # Get database size in GB
        size_bytes = db_path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)
        
        max_size_gb = storage_config.get('max_size_gb', 0.1)
        needs_pruning = size_gb >= max_size_gb
        
        if needs_pruning:
            logger.warning(
                f"Database size {size_gb:.3f} GB >= limit {max_size_gb} GB, "
                "pruning may be needed"
            )
        
        return size_gb, needs_pruning
    
    def prune_database(self):
        """
        Prune least-used entries from RAG database.
        
        From config rag_database.storage:
        - Remove entries with query_count < min_query_frequency
        - Keep frequently accessed entries
        - Reduce database to target size
        """
        storage_config = self.config['rag_database']['storage']
        
        if not storage_config.get('enable_pruning', True):
            logger.info("Database pruning is disabled")
            return
        
        min_query_freq = storage_config.get('min_query_frequency', 0.5)
        
        # Get all entries from database
        # Note: This is a simplified version; actual implementation would
        # depend on the MemoryIndex API
        logger.info(
            f"Pruning database entries with query frequency < {min_query_freq}"
        )
        
        # Track pruning statistics
        # self.rag_index.prune_by_query_frequency(min_query_freq)
        
        logger.info("Database pruning completed")
    
    def refresh_database_entries(self):
        """
        Refresh frequently-used entries with updated analysis.
        
        From config rag_database.storage:
        - Re-analyze entries accessed > refresh_frequency_threshold times per 100 games
        - Update with fresh analysis to maintain accuracy
        """
        storage_config = self.config['rag_database']['storage']
        
        if not storage_config.get('enable_refresh', True):
            return
        
        refresh_threshold = storage_config.get('refresh_frequency_threshold', 10)
        
        logger.info(
            f"Refreshing entries with access count > {refresh_threshold} per 100 games"
        )
        
        # Note: Actual implementation would iterate through database
        # and re-analyze high-frequency entries
        
        logger.info("Database refresh completed")
    
    def update_query_statistics(self, position_hash: str):
        """
        Update query frequency statistics for database management.
        
        Args:
            position_hash: Hash of queried position
        """
        if not self.config['online_learning'].get('track_query_frequency', True):
            return
        
        # Note: Actual implementation would update the entry's query_count
        # and last_accessed timestamp in the database
        # self.rag_index.increment_query_count(position_hash)
        
        logger.debug(f"Updated query statistics for position {position_hash}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get bot statistics for current game."""
        stats = self.stats.copy()
        if stats['moves_played'] > 0:
            stats['avg_time_per_move_ms'] = stats['total_time_ms'] / stats['moves_played']
            stats['rag_query_rate'] = stats['rag_queries'] / stats['moves_played']
            if stats['rag_queries'] > 0:
                stats['rag_hit_rate'] = stats['rag_hits'] / stats['rag_queries']
        
        # Add database statistics
        db_size_gb, needs_pruning = self.check_database_size()
        stats['database_size_gb'] = db_size_gb
        stats['database_needs_pruning'] = needs_pruning
        stats['background_queue_size'] = len(self.background_queue)
        
        return stats
    
    def shutdown(self):
        """Clean up resources."""
        self._stop_katago()
        logger.info("DataGo bot shut down")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python datago_bot.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    bot = DataGoBot(config_path)
    
    try:
        # Start new game
        bot.new_game()
        
        # Generate some moves
        for _ in range(10):
            decision = bot.generate_move()
            print(f"Move: {decision.move}, Uncertainty: {decision.uncertainty:.3f}")
            
            # Update game state (simplified - actual implementation would handle move execution)
            bot.game_state.move_number += 1
        
        # Print statistics
        print("\nGame Statistics:")
        print(json.dumps(bot.get_statistics(), indent=2))
    
    finally:
        bot.shutdown()
