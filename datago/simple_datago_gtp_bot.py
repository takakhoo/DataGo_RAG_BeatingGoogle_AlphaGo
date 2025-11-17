#!/usr/bin/env python3
"""
simple_datago_gtp_bot.py

Simplified DataGo bot that uses KataGo via GTP for analysis instead of starting
a separate analysis engine. This allows the bot to play against another KataGo
instance without process conflicts.

This version:
1. Uses the GTP opponent's analysis when available
2. Falls back to simple policy if RAG doesn't improve uncertainty
3. Does not start its own KataGo analysis process
"""

import logging
import numpy as np
import yaml
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.memory.index import ANNIndex
from src.memory.schema import MemoryEntry
from src.gating.gate import entropy_of_policy, normalized_entropy
from src.blend.blend import rerank_neighbors, build_retrieval_prior, blend_priors

logger = logging.getLogger(__name__)


@dataclass
class SimpleMoveDecision:
    """Simplified move decision without network evaluator."""
    move: str
    uncertainty: float
    rag_queried: bool
    rag_hit: bool
    rag_relevance: float
    time_taken_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "move": self.move,
            "uncertainty": float(self.uncertainty),
            "rag_queried": self.rag_queried,
            "rag_hit": self.rag_hit,
            "rag_relevance": float(self.rag_relevance) if self.rag_hit else None,
            "time_taken_ms": float(self.time_taken_ms),
        }


class SimpleDataGoGTPBot:
    """
    Simplified RAG-enhanced bot that operates via GTP without separate KataGo process.
    
    This version uses a pre-computed or simple policy distribution and focuses on
    the RAG retrieval and knowledge blending aspects of DataGo.
    """
    
    def __init__(self, config_path: str):
        """Initialize bot with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize RAG database
        logger.info("Initializing RAG database...")
        rag_config = self.config['rag_database']
        embedding_dim = rag_config.get('embedding_dim', 64)
        
        self.rag_index = ANNIndex(
            dim=embedding_dim,
            space=rag_config['ann'].get('distance_metric', 'cosine'),
        )
        
        # Game state tracking
        self.board_size = self.config['katago']['board_size']
        self.move_history = []
        self.current_player = 1  # 1=black, -1=white
        self.move_number = 0
        
        # Statistics
        self.stats = {
            "moves_played": 0,
            "rag_queries": 0,
            "rag_hits": 0,
            "positions_stored": 0,
        }
        
        # Uncertainty thresholds
        self.uncertainty_threshold = self.config['gating']['uncertainty_threshold']
        
        logger.info("SimpleDataGoGTPBot initialized")
    
    def new_game(self, board_size: int = 19, komi: float = 7.5):
        """Start a new game."""
        self.board_size = board_size
        self.move_history = []
        self.move_number = 0
        self.current_player = 1
        logger.info(f"New game: {board_size}x{board_size}, komi={komi}")
    
    def play_move(self, move: str, player: int):
        """Record a move in the game history."""
        self.move_history.append((move, player))
        self.move_number += 1
        self.current_player = -player  # Switch player
    
    def generate_move(self, gtp_controller=None) -> SimpleMoveDecision:
        """
        Generate a move using RAG-enhanced decision making.
        
        For this simplified version, we use a basic policy and focus on
        the RAG retrieval aspect.
        """
        import time
        start_time = time.time()
        
        # Create a simple uniform policy as baseline
        # In a full implementation, this would come from network evaluation
        num_moves = self.board_size * self.board_size + 1  # All points + pass
        baseline_policy = np.ones(num_moves) / num_moves
        
        # Calculate uncertainty (for uniform policy, entropy is high)
        uncertainty = normalized_entropy(baseline_policy)
        
        # Query RAG if uncertainty is high
        rag_queried = False
        rag_hit = False
        rag_relevance = 0.0
        
        if uncertainty > self.uncertainty_threshold:
            rag_queried = True
            self.stats['rag_queries'] += 1
            
            # Create position embedding (simplified - use move count as proxy)
            # In full version, this would be symmetry hash
            embedding = np.random.rand(64)  # Placeholder
            
            # Query RAG
            neighbors = self.rag_index.query(embedding, k=5)
            
            if neighbors:
                rag_hit = True
                self.stats['rag_hits'] += 1
                # Use first neighbor's relevance as proxy
                rag_relevance = neighbors[0].get('relevance', 0.5)
                logger.info(f"RAG hit: {len(neighbors)} neighbors, relevance={rag_relevance:.3f}")
        
        # For now, just use a simple heuristic move selection
        # In practice, you'd blend RAG knowledge with the policy
        move = self._select_simple_move()
        
        time_taken = (time.time() - start_time) * 1000
        
        self.stats['moves_played'] += 1
        
        return SimpleMoveDecision(
            move=move,
            uncertainty=uncertainty,
            rag_queried=rag_queried,
            rag_hit=rag_hit,
            rag_relevance=rag_relevance,
            time_taken_ms=time_taken,
        )
    
    def _select_simple_move(self) -> str:
        """
        Select a move using simple heuristics.
        
        This is a placeholder - in the full bot, moves come from MCTS.
        For testing, we'll use opening theory heuristics.
        """
        # Simple opening moves
        if self.move_number == 0:
            return "D4"  # 4-4 point
        elif self.move_number == 1:
            return "Q16"  # Opposite corner
        elif self.move_number == 2:
            return "D16"  # Another corner
        elif self.move_number == 3:
            return "Q4"  # Last corner
        else:
            # For later moves, return pass (in practice, use real search)
            return "pass"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.stats.copy()
    
    def cleanup(self):
        """Cleanup resources."""
        logger.info("SimpleDataGoGTPBot cleanup complete")
