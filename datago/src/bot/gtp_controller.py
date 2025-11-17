"""
gtp_controller.py

Custom GTP (Go Text Protocol) controller for communicating with Go engines.
This replaces the gomill dependency with a simple, lightweight implementation.
"""

import subprocess
import logging
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)


class GTPController:
    """Simple GTP controller for communicating with Go engines."""
    
    def __init__(self, command: List[str]):
        """
        Initialize GTP controller with a command to start the engine.
        
        Args:
            command: Command and arguments to start the engine (e.g., ['katago', 'gtp', ...])
        """
        self.command = command
        self.process = None
        self._start()
    
    def _start(self):
        """Start the GTP engine process."""
        logger.info(f"Starting GTP engine: {' '.join(self.command)}")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    
    def send_command(self, command: str) -> Tuple[bool, str]:
        """
        Send a GTP command and get the response.
        
        Args:
            command: GTP command string
            
        Returns:
            Tuple of (success: bool, response: str)
        """
        if not self.process:
            return False, "Engine not running"
        
        logger.debug(f"GTP > {command}")
        
        try:
            # Send command
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            
            # Read response - capture info lines AND final response
            info_lines = []
            response_lines = []
            success = False
            found_response = False
            
            while not found_response:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                logger.debug(f"GTP < {line}")
                
                # Capture "info" lines (from kata-genmove_analyze etc)
                if line.startswith('info '):
                    info_lines.append(line)
                    continue
                    
                # GTP responses start with = (success) or ? (error)
                if line.startswith('=') or line.startswith('?'):
                    success = line.startswith('=')
                    # Extract response text after = or ?
                    response_text = line[1:].strip()
                    if response_text:
                        response_lines.append(response_text)
                    
                    # Continue reading until blank line
                    while True:
                        next_line = self.process.stdout.readline()
                        if not next_line or not next_line.strip():
                            break
                        stripped = next_line.strip()
                        if stripped:
                            response_lines.append(stripped)
                    
                    found_response = True
                    break
            
            # Combine info lines and response
            all_lines = info_lines + response_lines
            response = '\n'.join(all_lines)
            return success, response
            
        except Exception as e:
            logger.error(f"Error communicating with engine: {e}")
            return False, str(e)
    
    def genmove(self, color: str) -> Optional[str]:
        """
        Generate a move for the given color.
        
        Args:
            color: 'black' or 'white' (or 'b'/'w')
            
        Returns:
            Move string (e.g., 'D4', 'pass', 'resign') or None on error
        """
        color = color.lower()
        if color not in ['black', 'white', 'b', 'w']:
            raise ValueError(f"Invalid color: {color}")
        
        success, response = self.send_command(f"genmove {color}")
        if success:
            return response.strip()
        else:
            logger.error(f"genmove failed: {response}")
            return None
    
    def play(self, color: str, move: str) -> bool:
        """
        Play a move on the board.
        
        Args:
            color: 'black' or 'white' (or 'b'/'w')
            move: Move string (e.g., 'D4', 'pass')
            
        Returns:
            True if successful, False otherwise
        """
        success, response = self.send_command(f"play {color} {move}")
        if not success:
            logger.error(f"play {color} {move} failed: {response}")
        return success
    
    def boardsize(self, size: int) -> bool:
        """Set the board size."""
        success, _ = self.send_command(f"boardsize {size}")
        return success
    
    def clear_board(self) -> bool:
        """Clear the board."""
        success, _ = self.send_command("clear_board")
        return success
    
    def komi(self, komi: float) -> bool:
        """Set the komi value."""
        success, _ = self.send_command(f"komi {komi}")
        return success
    
    def name(self) -> Optional[str]:
        """Get the engine name."""
        success, response = self.send_command("name")
        return response if success else None
    
    def version(self) -> Optional[str]:
        """Get the engine version."""
        success, response = self.send_command("version")
        return response if success else None
    
    def quit(self):
        """Quit the engine gracefully."""
        if self.process:
            try:
                self.send_command("quit")
                self.process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Error during quit: {e}")
                self.process.kill()
    
    def close(self):
        """Close the engine (alias for quit)."""
        self.quit()
    
    def set_max_visits(self, visits: int) -> bool:
        """
        Set KataGo's maxVisits parameter dynamically.
        
        Args:
            visits: Number of visits for MCTS search
            
        Returns:
            True if successful, False otherwise
        """
        success, response = self.send_command(f"kata-set-param maxVisits {visits}")
        if success:
            logger.debug(f"Set maxVisits to {visits}")
            return True
        else:
            logger.warning(f"Failed to set maxVisits: {response}")
            return False
    
    def genmove_analyze(self, color: str = "b") -> Tuple[Optional[str], Optional[Dict]]:
        """
        Run KataGo's kata-genmove_analyze command.
        
        This generates a move AND provides detailed analysis in one command.
        Returns real neural network outputs including policy, value, moveInfos, etc.
        
        Args:
            color: Color to move ('b' or 'w')
            
        Returns:
            Tuple of (move_string, analysis_dict) or (None, None) on error
        """
        # Use kata-genmove_analyze which combines genmove + analysis
        command = f"kata-genmove_analyze {color}"
        success, response = self.send_command(command)
        
        if success and response:
            try:
                # Response format: multiple "info move ..." entries (may be on one long line!)
                # Split by "info move" to separate each move's data
                move_infos = []
                root_info = {}
                
                # Split response by "info move" to get individual move entries
                parts = response.split('info move ')
                
                for part in parts:
                    if not part.strip():
                        continue
                    
                    # Reconstruct the info line
                    line = 'info move ' + part
                    line = line.strip()
                    if not line or not line.startswith('info'):
                        continue
                    
                    # Parse info line: "info move Q16 visits 249 winrate 0.54 ..."
                    parts = line.split()
                    if len(parts) < 4 or parts[1] != 'move':
                        continue
                    
                    # Parse key-value pairs
                    info_dict = {}
                    i = 1  # Start after "info"
                    while i < len(parts):
                        key = parts[i]
                        
                        # Handle "move" specially - next part is the move string
                        if key == 'move' and i + 1 < len(parts):
                            info_dict['move'] = parts[i + 1]
                            i += 2
                            continue
                        
                        # Handle other key-value pairs
                        if i + 1 < len(parts):
                            value = parts[i + 1]
                            try:
                                # Try to parse as number
                                if '.' in value or 'e' in value.lower():
                                    value = float(value)
                                else:
                                    try:
                                        value = int(value)
                                    except ValueError:
                                        pass  # Keep as string
                            except ValueError:
                                pass  # Keep as string
                            info_dict[key] = value
                            i += 2
                        else:
                            i += 1
                    
                    if 'move' in info_dict:
                        move_infos.append(info_dict)
                    
                    # First move info is also the root/best move
                    if not root_info and move_infos:
                        root_info = info_dict.copy()
                
                if move_infos:
                    # Extract best move (first in list)
                    best_move = move_infos[0].get('move')
                    
                    # Build analysis dict in format similar to kata-analyze
                    analysis = {
                        'moveInfos': move_infos,
                        'rootInfo': root_info,
                    }
                    
                    return best_move, analysis
                
            except Exception as e:
                logger.error(f"Error parsing kata-genmove_analyze response: {e}")
        
        logger.warning(f"kata-genmove_analyze failed")
        return None, None
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if self.process and self.process.poll() is None:
            try:
                self.process.kill()
            except:
                pass
