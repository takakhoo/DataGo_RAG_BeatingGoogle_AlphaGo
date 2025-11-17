#!/usr/bin/env python3
"""
play_datago_vs_katago.py

Play DataGo bot vs KataGo using KataGo's GTP analysis interface.
This script uses KataGo for both players - one as the opponent via GTP,
and one for DataGo's internal analysis via the analysis engine.

Key difference from previous version:
- Uses DIFFERENT KataGo instances on different GPUs to avoid conflicts
- DataGo uses GPU 7 for analysis
- Opponent KataGo uses GPU 6 (or CPU) for GTP play
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.bot.gtp_controller import GTPController
from src.clients.katago_client import KataGoClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleDataGoPlayer:
    """
    Simplified DataGo player that uses KataGo analysis for move generation
    without the full RAG pipeline (for initial testing).
    """
    
    def __init__(self, katago_client: KataGoClient, color: str):
        self.client = katago_client
        self.color = color.upper()
        self.board_size = 19
        self.komi = 7.5
        self.move_history = []
        
    def setup_game(self, board_size: int, komi: float):
        """Initialize a new game."""
        self.board_size = board_size
        self.komi = komi
        self.move_history = []
        logger.info(f"DataGo player setup: {board_size}x{board_size}, komi={komi}")
    
    def play_move(self, move: str, player: str):
        """Record a move."""
        self.move_history.append((move, player))
    
    def generate_move(self) -> str:
        """Generate next move using KataGo analysis."""
        # Create board state string for KataGo
        board_state = self._create_board_state()
        
        # Request analysis with 800 visits (same as opponent)
        result = self.client.analyze_position(
            board_state=board_state,
            board_size=self.board_size,
            komi=self.komi,
            max_visits=800,
        )
        
        # Get best move
        if result and result.move_info:
            best_move = result.move_info[0]['move']
            winrate = result.move_info[0]['winrate']
            visits = result.move_info[0]['visits']
            logger.info(f"DataGo analysis: {best_move} (winrate={winrate:.3f}, visits={visits})")
            return best_move
        else:
            logger.warning("DataGo failed to get analysis, playing pass")
            return "pass"
    
    def _create_board_state(self) -> str:
        """Create board state string for KataGo analysis."""
        # Simple format: list of moves
        moves = []
        for move, player in self.move_history:
            moves.append(f"{player} {move}")
        return " ".join(moves)


def play_match(
    katago_executable: str,
    katago_model: str,
    katago_gtp_config: str,
    katago_analysis_config: str,
    datago_gpu: int = 7,
    opponent_gpu: int = 6,
    datago_color: str = "black",
    num_games: int = 1,
    board_size: int = 19,
    komi: float = 7.5,
    max_moves: int = 300,
):
    """
    Play match between DataGo and KataGo using separate GPUs.
    
    Args:
        katago_executable: Path to KataGo binary
        katago_model: Path to KataGo model
        katago_gtp_config: Path to KataGo GTP config (for opponent)
        katago_analysis_config: Path to KataGo analysis config (for DataGo)
        datago_gpu: GPU for DataGo's KataGo analysis (default 7)
        opponent_gpu: GPU for opponent KataGo GTP (default 6)
        datago_color: "black" or "white"
        num_games: Number of games
        board_size: Board size
        komi: Komi
        max_moves: Maximum moves before draw
    """
    logger.info("=" * 70)
    logger.info("DataGo Bot vs KataGo Match")
    logger.info("=" * 70)
    logger.info(f"Games: {num_games}")
    logger.info(f"DataGo color: {datago_color}, DataGo GPU: {datago_gpu}")
    logger.info(f"KataGo opponent GPU: {opponent_gpu}")
    logger.info(f"Board: {board_size}x{board_size}, Komi: {komi}")
    logger.info("")
    
    # Initialize DataGo's KataGo client (analysis mode, GPU 7)
    logger.info(f"Starting DataGo's KataGo analysis engine on GPU {datago_gpu}...")
    datago_client = KataGoClient(
        katago_executable=katago_executable,
        model_path=katago_model,
        config_path=katago_analysis_config,
        board_size=board_size,
    )
    datago_player = SimpleDataGoPlayer(datago_client, color=datago_color)
    
    # Initialize opponent KataGo (GTP mode, GPU 6 or CPU)
    logger.info(f"Starting opponent KataGo GTP on GPU {opponent_gpu}...")
    opponent_cmd = [
        katago_executable,
        'gtp',
        '-model', katago_model,
        '-config', katago_gtp_config,
        '-override-config', f'openclGpuToUse={opponent_gpu}',
    ]
    opponent_player = GTPController(command=opponent_cmd)
    
    # Assign players by color
    if datago_color.lower() == "black":
        black_player = datago_player
        white_player = opponent_player
        black_name = "DataGo"
        white_name = "KataGo"
    else:
        black_player = opponent_player
        white_player = datago_player
        black_name = "KataGo"
        white_name = "DataGo"
    
    logger.info(f"Black: {black_name}, White: {white_name}")
    logger.info("")
    
    results = []
    
    for game_num in range(1, num_games + 1):
        logger.info(f"{'=' * 70}")
        logger.info(f"Game {game_num}/{num_games}")
        logger.info(f"{'=' * 70}")
        
        try:
            # Setup game
            datago_player.setup_game(board_size, komi)
            opponent_player.boardsize(board_size)
            opponent_player.clear_board()
            opponent_player.komi(komi)
            
            move_number = 0
            passes = 0
            
            # Play the game
            while move_number < max_moves:
                move_number += 1
                
                # Determine current player
                if move_number % 2 == 1:  # Black
                    current_player = black_player
                    current_color = "B"
                    player_name = black_name
                    opponent = white_player
                else:  # White
                    current_player = white_player
                    current_color = "W"
                    player_name = white_name
                    opponent = black_player
                
                # Generate move
                logger.info(f"Move {move_number}: {player_name}({current_color}) thinking...")
                
                if isinstance(current_player, SimpleDataGoPlayer):
                    # DataGo player
                    move = current_player.generate_move()
                else:
                    # KataGo GTP opponent
                    move = current_player.genmove(current_color)
                    if not move:
                        logger.error("KataGo GTP failed to generate move")
                        break
                    move = move.upper()
                
                logger.info(f"Move {move_number}: {player_name}({current_color}) plays {move}")
                
                # Check for pass/resign
                if move.upper() == "RESIGN":
                    winner = white_name if current_color == "B" else black_name
                    logger.info(f"{player_name} resigned. {winner} wins!")
                    results.append(winner)
                    break
                elif move.upper() == "PASS":
                    passes += 1
                    if passes >= 2:
                        logger.info("Both players passed. Game over.")
                        # Could score here, but for simplicity call it a draw
                        results.append("Draw")
                        break
                else:
                    passes = 0
                
                # Send move to opponent
                if isinstance(opponent, SimpleDataGoPlayer):
                    opponent.play_move(move, current_color)
                else:
                    opponent.play(current_color, move)
                
                # Record move for DataGo
                if isinstance(current_player, SimpleDataGoPlayer):
                    current_player.play_move(move, current_color)
                
                time.sleep(0.1)  # Small delay
            
            if move_number >= max_moves:
                logger.info(f"Game reached maximum moves ({max_moves}). Draw.")
                results.append("Draw")
        
        except Exception as e:
            logger.error(f"Error in game {game_num}: {e}", exc_info=True)
            results.append("Error")
        
        logger.info("")
    
    # Cleanup
    logger.info("Cleaning up...")
    datago_client.close()
    opponent_player.quit()
    
    # Print results
    logger.info("=" * 70)
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
    parser = argparse.ArgumentParser(description="Play DataGo bot vs KataGo")
    parser.add_argument('--katago-executable', required=True, help='Path to KataGo binary')
    parser.add_argument('--katago-model', required=True, help='Path to KataGo model')
    parser.add_argument('--katago-gtp-config', required=True, help='KataGo GTP config for opponent')
    parser.add_argument('--katago-analysis-config', required=True, help='KataGo analysis config for DataGo')
    parser.add_argument('--datago-gpu', type=int, default=7, help='GPU for DataGo analysis (default: 7)')
    parser.add_argument('--opponent-gpu', type=int, default=6, help='GPU for opponent GTP (default: 6)')
    parser.add_argument('--datago-color', default='black', choices=['black', 'white'], help='DataGo color')
    parser.add_argument('--games', type=int, default=1, help='Number of games')
    parser.add_argument('--board-size', type=int, default=19, help='Board size')
    parser.add_argument('--komi', type=float, default=7.5, help='Komi')
    parser.add_argument('--max-moves', type=int, default=300, help='Max moves per game')
    
    args = parser.parse_args()
    
    play_match(
        katago_executable=args.katago_executable,
        katago_model=args.katago_model,
        katago_gtp_config=args.katago_gtp_config,
        katago_analysis_config=args.katago_analysis_config,
        datago_gpu=args.datago_gpu,
        opponent_gpu=args.opponent_gpu,
        datago_color=args.datago_color,
        num_games=args.games,
        board_size=args.board_size,
        komi=args.komi,
        max_moves=args.max_moves,
    )


if __name__ == '__main__':
    main()
