#!/usr/bin/env python3
"""
play_vs_katago.py

Simple script to play DataGo bot vs pure KataGo using custom GTP implementation.

Usage:
    python play_vs_katago.py --katago-executable <path> --katago-model <path> --katago-config <path> --games 1
"""

import argparse
import logging
import sys
from pathlib import Path

from src.bot.datago_bot import DataGoBot
from src.bot.gtp_player import GTPPlayer
from src.bot.gtp_controller import GTPController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def play_match(
    config_path: str,
    katago_executable: str,
    katago_model: str,
    katago_config: str,
    datago_color: str = "black",
    num_games: int = 1,
    board_size: int = 19,
    komi: float = 7.5,
):
    """
    Play a match between DataGo bot and pure KataGo.
    
    Args:
        config_path: Path to DataGo config.yaml
        katago_executable: Path to KataGo binary
        katago_model: Path to KataGo model file
        katago_config: Path to KataGo GTP config
        datago_color: "black" or "white"
        num_games: Number of games to play
        board_size: Board size (default 19)
        komi: Komi value (default 7.5)
    """
    logger.info("=" * 70)
    logger.info("DataGo Bot vs KataGo Match")
    logger.info("=" * 70)
    logger.info(f"Games: {num_games}")
    logger.info(f"DataGo playing: {datago_color}")
    logger.info(f"Board size: {board_size}, Komi: {komi}")
    logger.info("")
    
    # Initialize DataGo bot
    logger.info("Initializing DataGo bot...")
    datago_bot = DataGoBot(config_path)
    datago_player = GTPPlayer(datago_bot, color=datago_color)
    
    # Initialize KataGo GTP controller
    logger.info("Starting KataGo opponent...")
    katago_cmd = [
        katago_executable,
        'gtp',
        '-model', katago_model,
        '-config', katago_config,
    ]
    katago_player = GTPController(command=katago_cmd)
    
    # Determine player assignments
    if datago_color.lower() == "black":
        black_player = datago_player
        white_player = katago_player
        black_name = "DataGo"
        white_name = "KataGo"
    else:
        black_player = katago_player
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
            
            katago_player.boardsize(board_size)
            katago_player.clear_board()
            katago_player.komi(komi)
            
            move_number = 0
            game_over = False
            winner = None
            
            # Play the game
            while not game_over:
                move_number += 1
                
                # Determine current player
                if move_number % 2 == 1:  # Odd moves = Black
                    current_player = black_player
                    current_color = "B"
                    player_name = black_name
                    opponent = white_player
                else:  # Even moves = White
                    current_player = white_player
                    current_color = "W"
                    player_name = white_name
                    opponent = black_player
                
                # Generate move
                if isinstance(current_player, GTPPlayer):
                    # DataGo bot
                    decision = current_player.generate_move()
                    move = decision.move
                    logger.info(
                        f"Move {move_number}: {player_name}({current_color}) plays {move} "
                        f"(unc={decision.uncertainty:.3f}, rag_hit={decision.rag_hit})"
                    )
                else:
                    # KataGo via GTP
                    move = current_player.genmove(current_color)
                    if not move:
                        logger.error("KataGo failed to generate move")
                        break
                    move = move.upper()
                    logger.info(f"Move {move_number}: {player_name}({current_color}) plays {move}")
                
                # Check for game end
                if move.upper() in ["RESIGN", "PASS"]:
                    if move.upper() == "RESIGN":
                        winner = white_name if current_color == "B" else black_name
                        logger.info(f"{player_name} resigned. {winner} wins!")
                        game_over = True
                    else:
                        # Check if both passed (simple end condition)
                        logger.info(f"{player_name} passed")
                        # For simplicity, continue playing
                        # A real implementation would check for two consecutive passes
                
                if not game_over:
                    # Play move on both sides
                    if isinstance(current_player, GTPPlayer):
                        current_player.play_move(current_color.lower(), move)
                        opponent.play(current_color, move)
                    else:
                        opponent.play_move(current_color.lower(), move)
                
                # Safety limit
                if move_number > 500:
                    logger.warning("Move limit reached (500 moves)")
                    game_over = True
                    winner = "Draw (move limit)"
            
            results.append({
                'game': game_num,
                'winner': winner,
                'moves': move_number,
            })
            
        except KeyboardInterrupt:
            logger.info("\nMatch interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in game {game_num}: {e}", exc_info=True)
            results.append({
                'game': game_num,
                'winner': 'Error',
                'moves': move_number,
            })
    
    # Cleanup
    logger.info("\nCleaning up...")
    datago_bot.shutdown()
    katago_player.close()
    
    # Print results
    logger.info("\n" + "=" * 70)
    logger.info("MATCH RESULTS")
    logger.info("=" * 70)
    
    datago_wins = sum(1 for r in results if r['winner'] == "DataGo")
    katago_wins = sum(1 for r in results if r['winner'] == "KataGo")
    
    for result in results:
        logger.info(f"Game {result['game']}: {result['winner']} ({result['moves']} moves)")
    
    logger.info("")
    logger.info(f"DataGo: {datago_wins}/{len(results)} wins")
    logger.info(f"KataGo: {katago_wins}/{len(results)} wins")
    logger.info(f"DataGo stats: {datago_bot.get_statistics()}")


def main():
    parser = argparse.ArgumentParser(description="Play DataGo bot vs KataGo")
    
    parser.add_argument(
        '--config',
        default='src/bot/config.yaml',
        help='Path to DataGo config.yaml'
    )
    parser.add_argument(
        '--katago-executable',
        required=True,
        help='Path to KataGo binary'
    )
    parser.add_argument(
        '--katago-model',
        required=True,
        help='Path to KataGo model file'
    )
    parser.add_argument(
        '--katago-config',
        required=True,
        help='Path to KataGo GTP config'
    )
    parser.add_argument(
        '--datago-color',
        choices=['black', 'white'],
        default='black',
        help='Color for DataGo bot (default: black)'
    )
    parser.add_argument(
        '--games',
        type=int,
        default=1,
        help='Number of games to play (default: 1)'
    )
    parser.add_argument(
        '--board-size',
        type=int,
        default=19,
        help='Board size (default: 19)'
    )
    parser.add_argument(
        '--komi',
        type=float,
        default=7.5,
        help='Komi value (default: 7.5)'
    )
    
    args = parser.parse_args()
    
    play_match(
        config_path=args.config,
        katago_executable=args.katago_executable,
        katago_model=args.katago_model,
        katago_config=args.katago_config,
        datago_color=args.datago_color,
        num_games=args.games,
        board_size=args.board_size,
        komi=args.komi,
    )


if __name__ == '__main__':
    main()
