#!/usr/bin/env python3
"""
simple_play_katago.py

Simplified script to test KataGo vs KataGo on GPU 7.
This avoids the complex DataGo bot for initial testing.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.bot.gtp_controller import GTPController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def play_simple_game():
    """Play a simple self-play game with two KataGo instances."""
    
    katago_exe = "/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago"
    model_path = "/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz"
    config_path = "/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg"
    
    logger.info("=" * 70)
    logger.info("KataGo Self-Play Test on GPU 7")
    logger.info("=" * 70)
    
    # Start two KataGo instances
    logger.info("Starting KataGo Black...")
    cmd = [katago_exe, 'gtp', '-model', model_path, '-config', config_path]
    black = GTPController(cmd)
    
    logger.info("Starting KataGo White...")
    white = GTPController(cmd)
    
    # Setup game
    board_size = 19
    komi = 7.5
    
    for player in [black, white]:
        player.boardsize(board_size)
        player.clear_board()
        player.komi(komi)
    
    logger.info(f"\nGame settings: {board_size}x{board_size}, komi={komi}")
    logger.info("Starting game...\n")
    
    # Play game
    move_count = 0
    max_moves = 50  # Limit for testing
    
    while move_count < max_moves:
        move_count += 1
        
        # Black's turn
        logger.info(f"Move {move_count}: Black thinking...")
        black_move = black.genmove('black')
        if not black_move or black_move.lower() in ['resign', 'pass']:
            logger.info(f"Black {black_move}")
            break
        
        logger.info(f"Move {move_count}: Black plays {black_move}")
        white.play('black', black_move)
        
        # White's turn
        move_count += 1
        logger.info(f"Move {move_count}: White thinking...")
        white_move = white.genmove('white')
        if not white_move or white_move.lower() in ['resign', 'pass']:
            logger.info(f"White {white_move}")
            break
        
        logger.info(f"Move {move_count}: White plays {white_move}")
        black.play('white', white_move)
    
    logger.info(f"\nGame finished after {move_count} moves")
    
    # Cleanup
    black.quit()
    white.quit()
    
    logger.info("\n" + "=" * 70)
    logger.info("Test completed successfully!")
    logger.info("=" * 70)


if __name__ == '__main__':
    try:
        play_simple_game()
    except KeyboardInterrupt:
        logger.info("\nGame interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
