import json
import subprocess
import csv
import os
from pathlib import Path
from typing import Optional, List, Dict

try:
    from sgfmill import boards
except ImportError:
    boards = None


def parse_flagged_positions_csv(csv_path: str, json_dir: str) -> List[Dict]:
    """
    Parse a CSV file containing JSON filenames, load those JSON files from a directory,
    and extract all flagged positions with their moves_history.

    Args:
        csv_path: Path to CSV file where each row contains a JSON filename
        json_dir: Directory path where the JSON game files are stored

    Returns:
        List of dictionaries, each containing:
        - 'game_id': The game identifier
        - 'filename': The JSON filename
        - 'moves_history': List of [player, location] pairs like [["B","Q4"], ["W","D4"], ...]
        - 'position_data': Full flagged position data including uncertainty metrics, children, etc.

    Example:
        >>> positions = parse_flagged_positions_csv("games.csv", "/path/to/rag_data")
        >>> for pos in positions:
        >>>     print(f"Game {pos['game_id']}: {len(pos['moves_history'])} moves")
    """
    all_flagged_positions = []
    json_dir_path = Path(json_dir)

    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        # Skip header if present
        header = next(csv_reader, None)

        for row in csv_reader:
            # Skip empty rows
            if not row:
                continue

            # Get the JSON filename from the first column
            json_filename = row[0].strip()
            
            # Construct full path to JSON file
            json_path = json_dir_path / json_filename
            
            if not json_path.exists():
                print(f"Warning: JSON file not found: {json_path}")
                continue

            try:
                # Load the JSON game file
                with open(json_path, 'r', encoding='utf-8') as f:
                    game_data = json.load(f)

                # Extract game_id and settings
                game_id = game_data.get('game_id', 'unknown')
                settings = game_data.get('settings', {})
                game_rules = settings.get('rules', 'chinese')  # Use actual rules from game
                game_komi = settings.get('komi', 7.5)
                game_board_size = settings.get('board_size', 19)

                # Extract flagged_positions field
                if 'flagged_positions' not in game_data:
                    print(f"Warning: No flagged_positions in {json_filename}")
                    continue

                flagged_positions = game_data['flagged_positions']

                # Extract each flagged position with its full data
                for position in flagged_positions:
                    if 'moves_history' in position:
                        all_flagged_positions.append({
                            'game_id': game_id,
                            'filename': json_filename,
                            'moves_history': position['moves_history'],
                            'position_data': position,  # Include full position data
                            'game_rules': game_rules,  # Store the actual rules used
                            'game_komi': game_komi,
                            'game_board_size': game_board_size
                        })

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON file {json_filename}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing {json_filename}: {e}")
                continue

    return all_flagged_positions


def count_stones_on_board(moves: list, board_size: int = 19) -> dict:
    """
    Count stones on board by replaying moves with capture logic.
    Call this on-demand when retrieving from RAG, not during storage.

    Args:
        moves: List of [player, location] like [["B","Q4"], ["W","D4"]]
        board_size: Board size (default 19)

    Returns:
        {'black': int, 'white': int, 'total': int}
    """
    if boards is None:
        raise ImportError("sgfmill not installed. Run: pip install sgfmill")

    board = boards.Board(board_size)

    for player, location in moves:
        if location.upper() == 'PASS':
            continue

        # Parse KataGo coordinate format (e.g., "Q4")
        col = ord(location[0].upper()) - ord('A')
        if col >= 8:  # Skip 'I' in Go coordinates
            col -= 1
        row = board_size - int(location[1:])

        # Play move (sgfmill handles captures automatically)
        color = 'b' if player == 'B' else 'w'
        board.play(row, col, color)

    # Count stones
    black_count = 0
    white_count = 0
    for r in range(board_size):
        for c in range(board_size):
            stone = board.get(r, c)
            if stone == 'b':
                black_count += 1
            elif stone == 'w':
                white_count += 1

    return {
        'black': black_count,
        'white': white_count,
        'total': black_count + white_count
    }


class GoStateEmbedding:
    """Stores KataGo state embedding for RAG storage."""

    def __init__(self, katago_response: dict, query_info: dict):
        # Stored vectors
        self.state_hash = katago_response['rootInfo']['thisHash']
        self.sym_hash = katago_response['rootInfo']['symHash']
        self.policy = katago_response.get('policy', None)
        self.ownership = katago_response.get('ownership', None)
        self.winrate = katago_response['rootInfo']['winrate']
        self.score_lead = katago_response['rootInfo']['scoreLead']
        self.move_infos = katago_response.get('moveInfos', None)
        self.komi = query_info['komi']
        self.query_id = katago_response['id']
        self.stone_count = (
            katago_response.get('rootInfo', {}).get('stonesOnBoard')
            or query_info.get('stone_count')
            or 0
        )
        self.child_nodes = (
            katago_response.get('child_nodes')
            or katago_response.get('moveInfos')
            or []
        )

        # Temporary fields for storage decision (not saved to DB)
        self.score_stdev = katago_response['rootInfo'].get('scoreStdev', 0)
        self.lcb = katago_response['rootInfo'].get('lcb', 0)

    def to_dict(self):
        """Convert to dictionary for RAG storage."""
        return {
            'sym_hash': self.sym_hash,
            'state_hash': self.state_hash,
            'policy': self.policy,
            'ownership': self.ownership,
            'winrate': self.winrate,
            'score_lead': self.score_lead,
            'move_infos': self.move_infos,
            'komi': self.komi,
            'query_id': self.query_id,
            'stone_count': self.stone_count,
            'child_nodes': self.child_nodes,
        }

class KataGoAnalyzer:
    """Wrapper for KataGo analysis engine"""
    
    def __init__(self, katago_path: str, config_path: str, model_path: str):
        self.katago = subprocess.Popen(
            [katago_path, 'analysis', '-config', config_path, '-model', model_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.query_counter = 0
    
    def analyze_position(self, moves: list, komi: float = 7.5, 
                        rules: str = "chinese",
                        board_size: int = 19,
                        max_visits: Optional[int] = None) -> GoStateEmbedding:
        """
        Analyze a position and return embedding.
        
        Args:
            moves: List of [player, location] like [["B","Q4"], ["W","D4"]]
            komi: Game komi
            rules: Rule set (chinese, japanese, tromp-taylor, etc.)
            board_size: Board size (default 19x19)
            max_visits: Optional visit limit
        
        Returns:
            GoStateEmbedding with all extracted features
        """
        
        query = {
            "id": f"query_{self.query_counter}",
            "moves": moves,
            "rules": rules,
            "komi": komi,
            "boardXSize": board_size,
            "boardYSize": board_size,
            "includePolicy": True,
            "includeOwnership": True,
        }

        if max_visits:
            query["maxVisits"] = max_visits

        self.query_counter += 1

        self.katago.stdin.write(json.dumps(query) + '\n')
        self.katago.stdin.flush()

        response_line = self.katago.stdout.readline()
        response = json.loads(response_line)

        if 'error' in response:
            raise RuntimeError(f"KataGo error: {response['error']}")

        return GoStateEmbedding(response, query)
    
    def close(self):
        self.katago.stdin.close()
        self.katago.wait()


# Example Usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze flagged positions from selfplay games')
    parser.add_argument('--katago-path', default='../../build/katago',
                       help='Path to KataGo executable')
    parser.add_argument('--config', default='../../katago_repo/run/analysis.cfg',
                       help='Path to KataGo analysis config')
    parser.add_argument('--model', default='../../katago_repo/run/kata1-b28c512nbt-s11653980416-d5514111622.bin.gz',
                       help='Path to KataGo model')
    parser.add_argument('--csv', default='rag_files_list.csv',
                       help='Path to CSV file containing JSON filenames')
    parser.add_argument('--json-dir', default='../../build/rag_data',
                       help='Directory containing RAG JSON game files')
    parser.add_argument('--output-dir', default='./rag_output',
                       help='Directory to save output JSON database')
    parser.add_argument('--max-visits', type=int, default=2500,
                       help='Maximum MCTS visits for analysis')
    parser.add_argument('--max-positions', type=int, default=None,
                       help='Maximum number of positions to process (for testing)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    print(f"Initializing KataGo analyzer...")
    print(f"  KataGo: {args.katago_path}")
    print(f"  Config: {args.config}")
    print(f"  Model: {args.model}")
    
    analyzer = KataGoAnalyzer(
        katago_path=args.katago_path,
        config_path=args.config,
        model_path=args.model
    )

    # Configure paths
    csv_path = args.csv
    json_dir = args.json_dir
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Parse CSV and load all flagged positions from JSON files
    print(f"\nLoading flagged positions from CSV: {csv_path}")
    print(f"Reading JSON files from: {json_dir}")
    flagged_positions = parse_flagged_positions_csv(csv_path, json_dir)
    
    total_positions = len(flagged_positions)
    print(f"Found {total_positions} flagged positions")
    
    if args.max_positions:
        flagged_positions = flagged_positions[:args.max_positions]
        print(f"Processing first {len(flagged_positions)} positions (limited by --max-positions)")

    # Process all positions
    all_analyzed_positions = []
    output_path = os.path.join(output_dir, "rag_database.json")
    skipped_count = 0
    
    for idx, position_info in enumerate(flagged_positions):
        game_id = position_info['game_id']
        filename = position_info['filename']
        moves_history = position_info['moves_history']
        position_data = position_info['position_data']
        game_rules = position_info['game_rules']  # Use actual game rules
        game_komi = position_info['game_komi']
        game_board_size = position_info['game_board_size']
        
        print(f"\n{'='*60}")
        print(f"Processing position {idx+1}/{len(flagged_positions)}")
        print(f"Game ID: {game_id}")
        print(f"Filename: {filename}")
        print(f"Move number: {position_data.get('move_number', 'N/A')}")
        print(f"Moves played: {len(moves_history)}")
        
        # Analyze position with KataGo (offline MCTS for depth)
        try:
            embedding = analyzer.analyze_position(
                moves=moves_history,
                komi=game_komi,  # Use actual komi from game
                rules=game_rules,  # Use actual rules from game
                board_size=game_board_size,  # Use actual board size
                max_visits=args.max_visits
            )
        except RuntimeError as e:
            if "Illegal move" in str(e) or "KataGo error" in str(e):
                print(f"⚠️  Skipping position due to error: {e}")
                skipped_count += 1
                continue
            else:
                raise 

        # Build output JSON entry
        output_json = {
            'sym_hash': embedding.sym_hash,
            'state_hash': embedding.state_hash,
            'policy': embedding.policy,
            'ownership': embedding.ownership,
            'winrate': embedding.winrate,
            'score_lead': embedding.score_lead,
            'move_infos': embedding.move_infos,
            'komi': embedding.komi,
            'query_id': embedding.query_id,
            'stone_count': count_stones_on_board(moves_history, board_size=game_board_size),
            'child_nodes': {}
        }
        
        # Add child nodes from original selfplay data
        if 'children' in position_data:
            for child in position_data['children']:
                output_json['child_nodes'][child['move']] = {
                    'value': child.get('value', 0),
                    'prior': child.get('prior', 0),
                    'visits': child.get('visits', 0),
                    'child_sym_hash': child.get('child_sym_hash', '')
                }
        
        all_analyzed_positions.append(output_json)
        
        # Print progress
        if 'uncertainty_metrics' in position_data:
            metrics = position_data['uncertainty_metrics']
            print(f"Original uncertainty - Entropy: {metrics.get('policy_entropy', 'N/A'):.3f}, "
                  f"Variance: {metrics.get('value_variance', 'N/A'):.3f}")
        print(f"Analyzed - Winrate: {embedding.winrate:.3f}, Score Lead: {embedding.score_lead:.2f}")
        
        # Write incrementally after each position
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_analyzed_positions, f, indent=2)
        
        if (idx + 1) % 10 == 0:
            print(f"💾 Saved progress: {len(all_analyzed_positions)} positions ({os.path.getsize(output_path) / 1024 / 1024:.2f} MB)")

    # Final summary
    print(f"\n{'='*60}")
    print(f"✓ Successfully created RAG database with {len(all_analyzed_positions)} positions")
    if skipped_count > 0:
        print(f"  ⚠️  Skipped {skipped_count} positions due to illegal move errors")
    print(f"  File: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    analyzer.close()