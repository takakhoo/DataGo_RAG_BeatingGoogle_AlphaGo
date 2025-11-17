"""
network_evaluator.py

Wrapper for getting raw neural network evaluations (policy + value) 
from KataGo without running MCTS search.

This is used by custom_mcts.py to get network priors that can be 
modified before running our own MCTS.
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Dict, Tuple, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)


class KataGoNetworkEvaluator:
    """
    Evaluator that gets raw network policy and value from KataGo.
    
    Uses KataGo's analysis engine with maxVisits=1 to get only the
    neural network evaluation without tree search.
    """
    
    def __init__(
        self,
        katago_executable: str,
        model_path: str,
        config_path: str,
        board_size: int = 19,
    ):
        """
        Initialize network evaluator.
        
        Args:
            katago_executable: Path to KataGo binary
            model_path: Path to KataGo model
            config_path: Path to KataGo config
            board_size: Board size (default 19)
        """
        self.katago_executable = katago_executable
        self.model_path = model_path
        self.config_path = config_path
        self.board_size = board_size
        
        # Start KataGo process
        self.process: Optional[subprocess.Popen] = None
        self._start_katago()
        
        # Cache for position evaluations
        self.eval_cache: Dict[str, Tuple[Dict[str, float], float]] = {}
    
    def _start_katago(self):
        """Start KataGo subprocess."""
        cmd = [
            self.katago_executable,
            'analysis',
            '-model', self.model_path,
            '-config', self.config_path,
        ]
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                universal_newlines=True,
            )
            logger.info(f"KataGo network evaluator started: {' '.join(cmd)}")
        except Exception as e:
            logger.error(f"Failed to start KataGo: {e}")
            raise
    
    def stop(self):
        """Stop KataGo subprocess."""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
            logger.info("KataGo network evaluator stopped")
    
    def evaluate(
        self,
        position_hash: str,
        board_state: np.ndarray,
        moves: list,
        komi: float = 7.5,
        use_cache: bool = True,
    ) -> Tuple[Dict[str, float], float]:
        """
        Evaluate position with neural network only (no MCTS).
        
        Args:
            position_hash: Hash identifying the position
            board_state: Current board state (not used directly, but for reference)
            moves: Move history in GTP format
            komi: Komi value
            use_cache: Whether to use cached evaluations
            
        Returns:
            Tuple of (policy_dict, value)
            - policy_dict: Mapping from move strings to probabilities
            - value: Network's value estimate (winrate)
        """
        # Check cache
        if use_cache and position_hash in self.eval_cache:
            return self.eval_cache[position_hash]
        
        # Query KataGo with maxVisits=1 to get only network evaluation
        query = {
            "id": position_hash,
            "moves": moves,
            "rules": "chinese",
            "komi": komi,
            "boardXSize": self.board_size,
            "boardYSize": self.board_size,
            "maxVisits": 1,  # Only network eval, no MCTS
            "includePolicy": True,
            "includeOwnership": False,
            "includePVVisits": False,
        }
        
        # Send query
        query_str = json.dumps(query) + "\n"
        self.process.stdin.write(query_str)
        self.process.stdin.flush()
        
        # Read response
        response_str = self.process.stdout.readline()
        response = json.loads(response_str)
        
        # Extract policy and value
        policy_dict = self._extract_policy(response)
        value = response.get('rootInfo', {}).get('winrate', 0.5)
        
        # Cache result
        if use_cache:
            self.eval_cache[position_hash] = (policy_dict, value)
        
        return policy_dict, value
    
    def _extract_policy(self, response: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract policy distribution from KataGo response.
        
        Args:
            response: KataGo analysis response
            
        Returns:
            Dictionary mapping move strings to probabilities
        """
        policy_dict = {}
        
        # Get policy from rootInfo or policy field
        if 'policy' in response:
            # Direct policy array (older format)
            policy_array = response['policy']
            # Convert array indices to move strings
            for i, prob in enumerate(policy_array):
                if prob > 0:
                    move = self._index_to_move(i)
                    policy_dict[move] = float(prob)
        
        elif 'moveInfos' in response:
            # Move infos format (standard)
            for move_info in response['moveInfos']:
                move = move_info.get('move')
                # Get prior probability (network policy)
                prior = move_info.get('prior', 0.0)
                if move and prior > 0:
                    policy_dict[move] = float(prior)
        
        # Normalize
        total = sum(policy_dict.values())
        if total > 0:
            policy_dict = {k: v / total for k, v in policy_dict.items()}
        
        return policy_dict
    
    def _index_to_move(self, index: int) -> str:
        """
        Convert policy array index to GTP move string.
        
        Args:
            index: Index in policy array
            
        Returns:
            Move string (e.g., "D4", "Q16", "pass")
        """
        # Pass move is typically at the end
        if index >= self.board_size * self.board_size:
            return "pass"
        
        # Convert index to coordinates
        row = index // self.board_size
        col = index % self.board_size
        
        # Convert to GTP format (A1, B1, ..., skip I)
        col_letters = "ABCDEFGHJKLMNOPQRST"  # Skip 'I'
        col_str = col_letters[col] if col < len(col_letters) else str(col)
        row_str = str(self.board_size - row)  # GTP rows are bottom-up
        
        return f"{col_str}{row_str}"
    
    def clear_cache(self):
        """Clear evaluation cache."""
        self.eval_cache.clear()
        logger.debug("Cleared network evaluation cache")
    
    def __call__(self, position_hash: str) -> Tuple[Dict[str, float], float]:
        """
        Make evaluator callable for use with CustomMCTS.
        
        Note: This simplified interface requires position_hash to contain
        enough information to reconstruct the position. In practice, you'd
        pass additional game state information.
        
        Args:
            position_hash: Position identifier
            
        Returns:
            Tuple of (policy_dict, value)
        """
        # This is a simplified interface - in practice, you'd need to pass
        # the actual board state and move history
        # For now, return cached value or default
        if position_hash in self.eval_cache:
            return self.eval_cache[position_hash]
        
        # Return uniform policy as fallback
        # (In practice, this should never happen - always call evaluate() first)
        logger.warning(f"Position {position_hash} not in cache, returning uniform policy")
        uniform_policy = {"pass": 1.0}
        return uniform_policy, 0.5
