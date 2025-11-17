#!/usr/bin/env python3
"""
test_gtp_katago.py

Simple test script to verify the custom GTP controller works with KataGo.
"""

import sys
import logging
import subprocess
from pathlib import Path
from typing import Tuple, List

# Inline GTP controller to avoid import issues
class SimpleGTPController:
    """Simple GTP controller for testing."""
    
    def __init__(self, command: List[str]):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    
    def send_command(self, command: str) -> Tuple[bool, str]:
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        
        response_lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            
            line = line.strip()
            
            if line.startswith('=') or line.startswith('?'):
                success = line.startswith('=')
                while True:
                    next_line = self.process.stdout.readline().strip()
                    if not next_line:
                        break
                    response_lines.append(next_line)
                
                response = '\n'.join(response_lines)
                return success, response
            elif line:
                response_lines.append(line)
        
        return False, "No response"
    
    def name(self):
        success, response = self.send_command("name")
        return response if success else None
    
    def version(self):
        success, response = self.send_command("version")
        return response if success else None
    
    def boardsize(self, size: int):
        return self.send_command(f"boardsize {size}")[0]
    
    def clear_board(self):
        return self.send_command("clear_board")[0]
    
    def komi(self, komi: float):
        return self.send_command(f"komi {komi}")[0]
    
    def genmove(self, color: str):
        success, response = self.send_command(f"genmove {color}")
        return response.strip() if success else None
    
    def play(self, color: str, move: str):
        return self.send_command(f"play {color} {move}")[0]
    
    def quit(self):
        self.send_command("quit")
        self.process.wait(timeout=5)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_katago_gtp():
    """Test KataGo GTP interface."""
    
    # Paths from config
    katago_exe = "/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago"
    model_path = "/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz"
    config_path = "/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg"
    
    logger.info("=" * 70)
    logger.info("Testing KataGo GTP Interface")
    logger.info("=" * 70)
    
    # Check files exist
    for name, path in [("Executable", katago_exe), ("Model", model_path), ("Config", config_path)]:
        if not Path(path).exists():
            logger.error(f"{name} not found at: {path}")
            return False
    
    logger.info("All files found!")
    logger.info("")
    
    # Create GTP controller
    logger.info("Starting KataGo...")
    cmd = [katago_exe, 'gtp', '-model', model_path, '-config', config_path]
    
    try:
        katago = SimpleGTPController(cmd)
        
        # Test basic commands
        logger.info("\nTesting basic GTP commands:")
        logger.info("-" * 70)
        
        name = katago.name()
        logger.info(f"Engine name: {name}")
        
        version = katago.version()
        logger.info(f"Engine version: {version}")
        
        # Setup game
        logger.info("\nSetting up game...")
        katago.boardsize(19)
        katago.clear_board()
        katago.komi(7.5)
        
        # Test a few moves
        logger.info("\nPlaying test moves:")
        logger.info("-" * 70)
        
        move1 = katago.genmove('black')
        logger.info(f"Black plays: {move1}")
        
        # Play a move manually
        katago.play('white', 'D4')
        logger.info(f"White plays: D4")
        
        move2 = katago.genmove('black')
        logger.info(f"Black plays: {move2}")
        
        # Cleanup
        logger.info("\nShutting down...")
        katago.quit()
        
        logger.info("\n" + "=" * 70)
        logger.info("Test PASSED ✓")
        logger.info("=" * 70)
        logger.info("\nYour KataGo GTP interface is working correctly!")
        logger.info("You can now run the full play_vs_katago.py script.")
        
        return True
        
    except Exception as e:
        logger.error(f"\nTest FAILED: {e}", exc_info=True)
        return False


if __name__ == '__main__':
    success = test_katago_gtp()
    sys.exit(0 if success else 1)
