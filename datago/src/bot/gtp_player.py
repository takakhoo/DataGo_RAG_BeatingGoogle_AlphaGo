"""
gtp_player.py

Wrapper for DataGo bot to play via GTP protocol.
Replaces the gomill-based implementation.
"""

from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass

from .datago_bot import DataGoBot, MoveDecision

logger = logging.getLogger(__name__)


@dataclass
class GTPPlayer:
    """Wrapper for DataGo bot to play via GTP protocol."""
    
    bot: DataGoBot
    color: str  # 'black' or 'white'
    
    def __post_init__(self):
        """Initialize the player."""
        self.board_size = 19
        self.komi = 7.5
        self.move_history = []
    
    def setup_game(self, board_size: int, komi: float):
        """
        Setup a new game with the given parameters.
        
        Args:
            board_size: Board size (typically 19)
            komi: Komi value (typically 7.5)
        """
        self.board_size = board_size
        self.komi = komi
        self.move_history = []
        
        # Initialize the bot's game state
        player_color = 1 if self.color.lower() in ['black', 'b'] else -1
        self.bot.new_game(board_size=board_size, komi=komi, player_color=player_color)
        
        logger.info(f"Game setup: {board_size}x{board_size}, komi={komi}")
    
    def generate_move(self) -> MoveDecision:
        """
        Generate a move for this player.
        
        Returns:
            MoveDecision with move and metadata
        """
        # TODO: This needs to interface with the actual DataGo bot
        # For now, this is a placeholder that would need to be integrated
        # with the full bot implementation
        
        logger.info(f"Generating move for {self.color}")
        
        # The DataGoBot would generate a move here
        # This is simplified - real implementation would call bot.generate_move()
        decision = self.bot.generate_move()
        
        return decision
    
    def play_move(self, color: str, move: str):
        """
        Record that a move was played.
        
        Args:
            color: Color that played ('black' or 'white')
            move: Move string (e.g., 'D4', 'pass')
        """
        self.move_history.append((color, move))
        logger.debug(f"Recorded move: {color} {move}")
        
        # Update bot's internal state if needed
        # self.bot.play_move(color, move)
