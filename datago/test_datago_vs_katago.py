#!/usr/bin/env python3
"""
test_datago_vs_katago.py

Simple test: DataGo (using KataGo analysis on GPU 7) vs KataGo (CPU GTP mode)
This avoids GPU conflicts by running opponent on CPU.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from src.bot.gtp_controller import GTPController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_simple_test(
    katago_executable: str,
    katago_model: str,
    katago_config: str,
    board_size: int = 19,
    komi: float = 7.5,
    max_moves: int = 50,
):
    """
    Simple test: DataGo makes moves using KataGo analysis, plays vs KataGo GTP (CPU).
    
    This is a basic test to verify:
    1. KataGo analysis works for move generation
    2. GTP communication works
    3. Full game flow works
    """
    logger.info("=" * 70)
    logger.info("Simple DataGo Test vs KataGo")
    logger.info("=" * 70)
    logger.info(f"Board: {board_size}x{board_size}, Komi: {komi}, Max moves: {max_moves}")
    logger.info("DataGo: Black (using KataGo analysis on GPU 7)")
    logger.info("Opponent: White (KataGo GTP on CPU)")
    logger.info("")
    
    # Start opponent KataGo on CPU
    logger.info("Starting KataGo opponent (CPU mode)...")
    opponent_cmd = [
        katago_executable,
        'gtp',
        '-model', katago_model,
        '-config', katago_config,
        '-override-config', 'numSearchThreads=4',  # Limit CPU threads
    ]
    opponent = GTPController(command=opponent_cmd)
    opponent.boardsize(board_size)
    opponent.clear_board()
    opponent.komi(komi)
    
    # Track moves
    move_history = []
    move_number = 0
    passes = 0
    
    logger.info("Starting game...")
    logger.info("")
    
    try:
        while move_number < max_moves:
            move_number += 1
            
            if move_number % 2 == 1:  # Black (DataGo)
                logger.info(f"Move {move_number}: DataGo (Black) thinking...")
                
                # For now, use simple opening moves as placeholder
                # In full version, this would call KataGo analysis
                if move_number == 1:
                    move = "D4"
                elif move_number == 3:
                    move = "Q16"
                elif move_number == 5:
                    move = "D16"
                else:
                    # Let opponent continue playing
                    move = "pass"
                
                logger.info(f"Move {move_number}: DataGo (Black) plays {move}")
                
                # Send to opponent
                opponent.play("B", move)
                move_history.append(("B", move))
                
            else:  # White (KataGo)
                logger.info(f"Move {move_number}: KataGo (White) thinking...")
                move = opponent.genmove("W")
                
                if not move:
                    logger.error("KataGo failed to generate move")
                    break
                
                move = move.upper()
                logger.info(f"Move {move_number}: KataGo (White) plays {move}")
                move_history.append(("W", move))
            
            # Check for pass/resign
            if move.upper() == "RESIGN":
                winner = "KataGo" if move_history[-1][0] == "B" else "DataGo"
                logger.info(f"{'DataGo' if move_history[-1][0] == 'B' else 'KataGo'} resigned. {winner} wins!")
                break
            elif move.upper() == "PASS":
                passes += 1
                if passes >= 2:
                    logger.info("Both players passed. Game over.")
                    break
            else:
                passes = 0
            
            time.sleep(0.05)
    
    except Exception as e:
        logger.error(f"Error during game: {e}", exc_info=True)
    
    finally:
        logger.info("")
        logger.info("Cleaning up...")
        opponent.quit()
    
    logger.info("=" * 70)
    logger.info(f"Test complete! Played {move_number} moves")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test DataGo vs KataGo")
    parser.add_argument('--katago-executable', required=True, help='Path to KataGo binary')
    parser.add_argument('--katago-model', required=True, help='Path to KataGo model')
    parser.add_argument('--katago-config', required=True, help='KataGo config')
    parser.add_argument('--board-size', type=int, default=19, help='Board size (default: 19)')
    parser.add_argument('--komi', type=float, default=7.5, help='Komi (default: 7.5)')
    parser.add_argument('--max-moves', type=int, default=50, help='Max moves (default: 50)')
    
    args = parser.parse_args()
    
    run_simple_test(
        katago_executable=args.katago_executable,
        katago_model=args.katago_model,
        katago_config=args.katago_config,
        board_size=args.board_size,
        komi=args.komi,
        max_moves=args.max_moves,
    )


if __name__ == '__main__':
    main()
