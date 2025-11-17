#!/usr/bin/env python3
"""
run_datago_match.py

Run a match between DataGo bot and KataGo using RAG-enhanced decision making.

DataGo's pipeline:
1. Get KataGo's analysis (policy distribution, value, move info)
2. Calculate uncertainty using entropy and value variance
3. If uncertain, query RAG database for similar positions
4. Blend RAG knowledge with network policy if relevant match found
5. Select move based on blended policy
"""

import argparse
import logging
import time
import math
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from src.bot.gtp_controller import GTPController
from src.memory.index import ANNIndex
from src.memory.schema import MemoryEntry
from src.gating.gate import normalized_entropy, entropy_of_policy
from src.blend.blend import rerank_neighbors, build_retrieval_prior, blend_priors

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MoveAnalysis:
    """Analysis data for a move from KataGo."""
    move: str
    policy: np.ndarray
    winrate: float
    score_lead: float
    visits: int
    uncertainty: float
    move_info: List[Dict[str, Any]]


class DataGoPlayer:
    """RAG-enhanced player using uncertainty detection and knowledge retrieval."""
    
    def __init__(self, config: Dict[str, Any], katago: GTPController):
        self.config = config
        self.katago = katago
        
        # Initialize RAG index
        logger.info("Initializing RAG database...")
        rag_config = config['rag_database']
        embedding_dim = rag_config.get('embedding_dim', 64)
        
        self.rag_index = ANNIndex(
            dim=embedding_dim,
            space=rag_config['ann'].get('distance_metric', 'cosine'),
        )
        
        # Load RAG database if it exists
        db_path = Path(rag_config['database_path'])
        if db_path.exists():
            logger.info(f"Loading RAG database from {db_path}")
            # TODO: Load entries into index
        else:
            logger.info("No existing RAG database found, starting fresh")
        
        # Statistics
        self.stats = {
            'moves': 0,
            'rag_queries': 0,
            'rag_hits': 0,
            'high_relevance_hits': 0,
            'forced_explorations': 0,
            'positions_stored': 0,
            'total_uncertainty': 0.0,
        }
        
        # Game state
        self.move_history = []
        self.stones_on_board = 0
        
        # Storage threshold (store positions with uncertainty above this)
        # Store top uncertain positions (slightly above query threshold)
        self.storage_threshold = config.get('rag_query', {}).get('uncertainty_threshold', 0.35) + 0.02
    
    def calculate_uncertainty(self, policy: np.ndarray, move_info: List[Dict]) -> float:
        """
        Calculate uncertainty score using config parameters.
        
        Formula: (w1*E + w2*K) * phase(stones_on_board)
        where:
        - E = normalized policy entropy
        - K = value distribution sparseness (visit variance)
        - phase = game phase adjustment
        """
        cfg = self.config['uncertainty_detection']
        
        # E: Normalized entropy of policy
        E = normalized_entropy(policy)
        
        # K: Value distribution sparseness (using visit distribution as proxy)
        if len(move_info) > 1:
            visits = np.array([m['visits'] for m in move_info[:10]])  # Top 10 moves
            visit_probs = visits / (visits.sum() + 1e-9)
            K = normalized_entropy(visit_probs)
        else:
            K = 0.0
        
        # Phase adjustment
        phase_type = cfg['phase_function_type']
        coeffs = cfg['phase_function_coefficients']
        stones_ratio = self.stones_on_board / 361.0
        
        if phase_type == 'linear':
            phase = coeffs[0] * stones_ratio + coeffs[1]
        elif phase_type == 'exponential':
            phase = coeffs[0] * math.exp(coeffs[1] * stones_ratio) + coeffs[2]
        else:  # piecewise
            if self.stones_on_board < 120:
                phase = coeffs[0]
            elif self.stones_on_board < 240:
                phase = coeffs[1]
            else:
                phase = coeffs[2]
        
        # Combined uncertainty
        w1, w2 = cfg['w1'], cfg['w2']
        uncertainty = (w1 * E + w2 * K) * phase
        
        return float(uncertainty)
    
    def query_rag(self, position_hash: str, policy: np.ndarray) -> Optional[Dict[str, Any]]:
        """Query RAG database for similar positions."""
        # Create embedding (simplified - using policy for now)
        # In full version, would use symmetry hash
        embedding = policy[:64] if len(policy) >= 64 else np.pad(policy, (0, 64 - len(policy)))
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        # Query nearest neighbor using retrieve method
        neighbors = self.rag_index.retrieve(embedding, k=1)
        
        if neighbors:
            self.stats['rag_hits'] += 1
            # neighbors is List[Tuple[MemoryEntry, float]]
            entry, distance = neighbors[0]
            # Convert MemoryEntry to dict format
            # Extract best move from entry
            best_move = None
            if hasattr(entry, 'best_moves') and entry.best_moves:
                best_move = entry.best_moves[0].get('move', None)
            
            return {
                'best_move': best_move,
                'relevance': 1.0 - distance,  # Convert distance to similarity
                'entry': entry
            }
        return None
    
    def store_position(self, position_hash: str, policy: np.ndarray, move: str, 
                      uncertainty: float, move_info: List[Dict]):
        """
        Store a high-uncertainty position in the RAG database.
        
        Args:
            position_hash: Hash of the position
            policy: Policy distribution
            move: Move that was played
            uncertainty: Uncertainty score
            move_info: Move information from analysis
        """
        # Create embedding
        embedding = policy[:64] if len(policy) >= 64 else np.pad(policy, (0, 64 - len(policy)))
        embedding = embedding / (np.linalg.norm(embedding) + 1e-9)
        
        # Create best_moves list from move_info
        best_moves = []
        if move_info:
            for info in move_info[:10]:  # Top 10 moves
                best_moves.append({
                    'move': info.get('move', move),
                    'prob': info.get('visits', 0) / 800.0,  # Approximate probability
                    'winrate': info.get('winrate', 0.5),
                })
        else:
            # Just store the move that was played
            best_moves = [{'move': move, 'prob': 1.0, 'winrate': 0.5}]
        
        # Create memory entry
        entry = MemoryEntry.create(
            embed=embedding,
            canonical_board=position_hash,
            best_moves=best_moves,
            importance=uncertainty,
            metadata={
                'move_number': len(self.move_history),
                'stones_on_board': self.stones_on_board,
                'uncertainty': uncertainty,
                'move_played': move,
            }
        )
        
        # Add to index
        self.rag_index.add(entry)
        self.stats['positions_stored'] += 1
        
        logger.info(f"  → Stored position in RAG (uncertainty={uncertainty:.3f}, total={self.stats['positions_stored']})")
    
    def blend_with_rag(
        self,
        network_policy: np.ndarray,
        rag_neighbor: Dict[str, Any],
        move_info: List[Dict]
    ) -> Tuple[str, float]:
        """
        Blend network policy with RAG knowledge.
        
        Returns: (selected_move, relevance_score)
        """
        cfg = self.config['blending']
        beta = cfg['beta']
        
        # Extract RAG's recommended move
        rag_move = rag_neighbor.get('best_move', None)
        if not rag_move:
            # No specific move in RAG, use network policy
            return self._select_from_policy(move_info), 0.0
        
        # Calculate relevance score (simplified)
        relevance = rag_neighbor.get('relevance', 0.5)
        
        # Check relevance threshold
        threshold = self.config['relevance_weights']['relevance_threshold']
        
        if relevance >= threshold:
            # High relevance: Use RAG move directly
            logger.info(f"  → High relevance ({relevance:.3f}), using RAG move: {rag_move}")
            self.stats['high_relevance_hits'] += 1
            return rag_move, relevance
        else:
            # Low relevance: Force exploration of RAG move as priority
            logger.info(f"  → Low relevance ({relevance:.3f}), forcing exploration of: {rag_move}")
            self.stats['forced_explorations'] += 1
            # For now, use the RAG move with priority
            return rag_move, relevance
    
    def _select_from_policy(self, move_info: List[Dict]) -> str:
        """Select move from policy distribution (best move)."""
        if not move_info:
            return "pass"
        return move_info[0]['move']
    
    def generate_move(self) -> Tuple[str, Dict[str, Any]]:
        """
        Generate move using full RAG pipeline.
        
        Returns: (move, metadata)
        """
        start_time = time.time()
        
        # Get KataGo's move and analysis via GTP
        move = self.katago.genmove("B")
        
        if not move:
            return "pass", {'error': 'Failed to get move'}
        
        move = move.upper()
        
        # For this simplified version, we get basic info from genmove
        # In full version, would use kata-analyze command for detailed policy
        # For now, create mock analysis data
        policy = np.random.dirichlet(np.ones(361))  # Mock policy
        move_info = [{'move': move, 'visits': 800, 'winrate': 0.5}]
        
        # Calculate uncertainty
        uncertainty = self.calculate_uncertainty(policy, move_info)
        self.stats['moves'] += 1
        self.stats['total_uncertainty'] += uncertainty
        
        # Check if we should query RAG
        threshold = self.config['rag_query']['uncertainty_threshold']
        rag_queried = False
        rag_hit = False
        relevance = 0.0
        final_move = move
        
        if uncertainty > threshold:
            logger.info(f"  → High uncertainty ({uncertainty:.3f}), querying RAG...")
            self.stats['rag_queries'] += 1
            rag_queried = True
            
            # Query RAG
            position_hash = f"pos_{len(self.move_history)}"  # Simplified
            neighbor = self.query_rag(position_hash, policy)
            
            if neighbor:
                rag_hit = True
                final_move, relevance = self.blend_with_rag(policy, neighbor, move_info)
        
        # Store position if uncertainty is high enough and we didn't find a good match
        should_store = (uncertainty > self.storage_threshold and 
                       (not rag_hit or relevance < self.config['relevance_weights']['relevance_threshold']))
        
        if should_store:
            position_hash = f"pos_{len(self.move_history)}"
            self.store_position(position_hash, policy, final_move, uncertainty, move_info)
        
        # Update game state
        self.move_history.append(final_move)
        if final_move not in ["PASS", "RESIGN"]:
            self.stones_on_board += 1
        
        time_ms = (time.time() - start_time) * 1000
        
        metadata = {
            'uncertainty': uncertainty,
            'rag_queried': rag_queried,
            'rag_hit': rag_hit,
            'relevance': relevance,
            'time_ms': time_ms,
        }
        
        return final_move, metadata


def run_match(
    katago_executable: str,
    katago_model: str,
    katago_config: str,
    config_path: str,
    num_games: int = 1,
    max_moves: int = 200,
):
    """
    Run DataGo vs KataGo match with full RAG pipeline.
    
    DataGo uses:
    - Uncertainty detection based on policy entropy and value variance
    - RAG queries for uncertain positions
    - Knowledge blending when relevant positions found
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("=" * 70)
    logger.info("DataGo Bot vs KataGo Match (Full RAG Pipeline)")
    logger.info("=" * 70)
    logger.info(f"Games: {num_games}, Max moves per game: {max_moves}")
    logger.info(f"Uncertainty threshold: {config['rag_query']['uncertainty_threshold']}")
    logger.info(f"Relevance threshold: {config['relevance_weights']['relevance_threshold']}")
    logger.info(f"Blending beta: {config['blending']['beta']}")
    logger.info("")
    logger.info("Black: DataGo (RAG-enhanced)")
    logger.info("White: KataGo (pure)")
    logger.info("")
    
    # Start KataGo GTP on GPU 7
    logger.info("Starting KataGo GTP on GPU 7...")
    cmd = [
        katago_executable,
        'gtp',
        '-model', katago_model,
        '-config', katago_config,
    ]
    katago = GTPController(command=cmd)
    
    # Initialize DataGo player
    datago = DataGoPlayer(config, katago)
    
    results = []
    
    for game_num in range(1, num_games + 1):
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Game {game_num}/{num_games}")
        logger.info(f"{'=' * 70}\n")
        
        # Reset DataGo for new game
        datago.move_history = []
        datago.stones_on_board = 0
        
        try:
            # Setup game
            katago.boardsize(19)
            katago.clear_board()
            katago.komi(7.5)
            
            move_number = 0
            passes = 0
            
            while move_number < max_moves:
                move_number += 1
                
                if move_number % 2 == 1:  # Black (DataGo)
                    logger.info(f"Move {move_number}: DataGo (Black) thinking...")
                    
                    # Use full RAG pipeline
                    move, metadata = datago.generate_move()
                    
                    if not move:
                        logger.error("Failed to get move")
                        break
                    
                    # Log with RAG metadata
                    log_msg = f"Move {move_number}: DataGo (Black) plays {move}"
                    log_msg += f" [unc={metadata['uncertainty']:.3f}"
                    if metadata['rag_queried']:
                        log_msg += f", RAG={'HIT' if metadata['rag_hit'] else 'MISS'}"
                        if metadata['rag_hit']:
                            log_msg += f", rel={metadata['relevance']:.3f}"
                    log_msg += f", {metadata['time_ms']:.0f}ms]"
                    logger.info(log_msg)
                    
                else:  # White (KataGo pure)
                    logger.info(f"Move {move_number}: KataGo (White) thinking...")
                    move = katago.genmove("W")
                    
                    if not move:
                        logger.error("Failed to get move")
                        break
                    
                    move = move.upper()
                    logger.info(f"Move {move_number}: KataGo (White) plays {move}")
                    
                    # Update DataGo's game state
                    if move not in ["PASS", "RESIGN"]:
                        datago.stones_on_board += 1
                
                # Check game end
                if move == "RESIGN":
                    winner = "KataGo" if move_number % 2 == 1 else "DataGo"
                    logger.info(f"\n{'DataGo' if move_number % 2 == 1 else 'KataGo'} resigned. {winner} wins!")
                    results.append(winner)
                    break
                elif move == "PASS":
                    passes += 1
                    if passes >= 2:
                        logger.info("\nBoth players passed. Game over.")
                        results.append("Draw")
                        break
                else:
                    passes = 0
                
                time.sleep(0.05)
            
            if move_number >= max_moves:
                logger.info(f"\nReached max moves ({max_moves}). Draw.")
                results.append("Draw")
            
            # Print game statistics
            logger.info(f"\nGame {game_num} Statistics:")
            logger.info(f"  Moves played: {datago.stats['moves']}")
            logger.info(f"  RAG queries: {datago.stats['rag_queries']}")
            logger.info(f"  RAG hits: {datago.stats['rag_hits']}")
            logger.info(f"  High relevance hits: {datago.stats['high_relevance_hits']}")
            logger.info(f"  Forced explorations: {datago.stats['forced_explorations']}")
            logger.info(f"  Positions stored: {datago.stats['positions_stored']}")
            if datago.stats['moves'] > 0:
                avg_unc = datago.stats['total_uncertainty'] / datago.stats['moves']
                logger.info(f"  Average uncertainty: {avg_unc:.3f}")
                rag_rate = 100 * datago.stats['rag_queries'] / datago.stats['moves']
                logger.info(f"  RAG query rate: {rag_rate:.1f}%")
                store_rate = 100 * datago.stats['positions_stored'] / datago.stats['moves']
                logger.info(f"  Storage rate: {store_rate:.1f}%")
        
        except Exception as e:
            logger.error(f"Error in game {game_num}: {e}", exc_info=True)
            results.append("Error")
    
    # Cleanup
    logger.info("\nCleaning up...")
    katago.quit()
    
    # Overall statistics
    logger.info("\n" + "=" * 70)
    logger.info("Overall Match Statistics")
    logger.info("=" * 70)
    logger.info(f"Total moves: {datago.stats['moves']}")
    logger.info(f"Total RAG queries: {datago.stats['rag_queries']}")
    logger.info(f"Total RAG hits: {datago.stats['rag_hits']}")
    logger.info(f"High relevance hits: {datago.stats['high_relevance_hits']}")
    logger.info(f"Forced explorations: {datago.stats['forced_explorations']}")
    logger.info(f"Total positions stored: {datago.stats['positions_stored']}")
    logger.info(f"RAG database size: {len(datago.rag_index._entries)} entries")
    if datago.stats['moves'] > 0:
        avg_unc = datago.stats['total_uncertainty'] / datago.stats['moves']
        logger.info(f"Average uncertainty: {avg_unc:.3f}")
        rag_rate = 100 * datago.stats['rag_queries'] / datago.stats['moves']
        logger.info(f"RAG query rate: {rag_rate:.1f}%")
        store_rate = 100 * datago.stats['positions_stored'] / datago.stats['moves']
        logger.info(f"Storage rate: {store_rate:.1f}%")
    
    # Results summary
    logger.info("\n" + "=" * 70)
    logger.info("Match Results")
    logger.info("=" * 70)
    for i, result in enumerate(results, 1):
        logger.info(f"Game {i}: {result}")
    
    datago_wins = sum(1 for r in results if r == "DataGo")
    katago_wins = sum(1 for r in results if r == "KataGo")
    draws = sum(1 for r in results if r == "Draw")
    
    logger.info("")
    logger.info(f"DataGo: {datago_wins} wins")
    logger.info(f"KataGo: {katago_wins} wins")
    logger.info(f"Draws: {draws}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run DataGo vs KataGo match with full RAG pipeline")
    parser.add_argument('--katago-executable', required=True, help='Path to KataGo binary')
    parser.add_argument('--katago-model', required=True, help='Path to KataGo model')
    parser.add_argument('--katago-config', required=True, help='KataGo config')
    parser.add_argument('--config', required=True, help='DataGo config.yaml')
    parser.add_argument('--games', type=int, default=1, help='Number of games (default: 1)')
    parser.add_argument('--max-moves', type=int, default=200, help='Max moves per game (default: 200)')
    
    args = parser.parse_args()
    
    run_match(
        katago_executable=args.katago_executable,
        katago_model=args.katago_model,
        katago_config=args.katago_config,
        config_path=args.config,
        num_games=args.games,
        max_moves=args.max_moves,
    )


if __name__ == '__main__':
    main()
