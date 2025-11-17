"""
Phase 1: Uncertainty Detection Parameter Tuning
Tunes w1, w2, and phase_function parameters using ground truth database

Expects two separate JSON files (shallow and deep MCTS) with format from claude_instructions.txt:
{
  "sym_hash": "lookup_key",
  "state_hash": "unique_id",
  "policy": [362 floats],
  "ownership": [361 floats],
  "winrate": 0.547,
  "score_lead": 2.3,
  "move_infos": [...],
  "komi": 7.5,
  "query_id": "query_123",
  "stone_count": 85,
  "child_nodes": [...]
}

Hardware: NVIDIA A100
Time Budget: 4-8 hours (supervised learning approach)
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import concurrent.futures
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Optional: sklearn for NDCG metric
try:
    from sklearn.metrics import ndcg_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, using basic metrics only")

# Optional: Weights & Biases for monitoring
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: W&B not available. Using local logging.")


@dataclass
class PositionData:
    """Data for a single position matched between shallow and deep MCTS"""
    id: str  # sym_hash or state_hash
    stones_on_board: int
    
    # Shallow MCTS features
    shallow_policy: np.ndarray
    shallow_winrate: float
    shallow_score_lead: float
    
    # Deep MCTS features (ground truth)
    deep_policy: np.ndarray
    deep_winrate: float
    deep_score_lead: float
    
    # Computed uncertainty indicators (from shallow MCTS)
    policy_entropy: float  # Entropy of shallow policy
    value_entropy: float   # Uncertainty in value (derived from score_lead variance)
    policy_cross_entropy: float  # Cross-entropy between shallow and deep policy
    value_sparseness: float  # Sparseness measure
    
    # Ground truth errors (shallow vs deep)
    policy_kl_divergence: float
    value_absolute_error: float
    combined_error: float
    
    # Optional metadata
    komi: float = 7.5
    query_id: str = ""


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty detection parameters"""
    w1: float  # Weight for policy cross-entropy
    w2: float  # Weight for value distribution sparseness
    phase_function_type: str  # Type of phase function: 'linear', 'exponential', 'piecewise'
    phase_coefficients: List[float]  # Coefficients for the phase function
    config_id: str = ""
    
    def __post_init__(self):
        if not self.config_id:
            coef_str = "_".join([f"{c:.2f}" for c in self.phase_coefficients])
            self.config_id = f"w1_{self.w1:.2f}_w2_{self.w2:.2f}_{self.phase_function_type}_{coef_str}"
    
    def compute_phase_multiplier(self, stones_on_board: int, total_stones: int = 361) -> float:
        """
        Compute phase multiplier based on stones on board.
        
        Args:
            stones_on_board: Number of stones currently on board
            total_stones: Total possible stones (19x19 = 361)
        
        Returns:
            Phase multiplier
        """
        s = stones_on_board / total_stones  # Normalized progress [0, 1]
        
        if self.phase_function_type == 'linear':
            # Linear: phase(s) = a*s + b
            a, b = self.phase_coefficients[0], self.phase_coefficients[1]
            return a * s + b
            
        elif self.phase_function_type == 'exponential':
            # Exponential: phase(s) = a*exp(b*s) + c
            a, b, c = self.phase_coefficients
            return a * np.exp(b * s) + c
            
        elif self.phase_function_type == 'piecewise':
            # Piecewise: different multipliers for early/mid/late game
            early_mult, mid_mult, late_mult = self.phase_coefficients
            if stones_on_board < 120:
                return early_mult
            elif stones_on_board < 240:
                return mid_mult
            else:
                return late_mult
        else:
            raise ValueError(f"Unknown phase function type: {self.phase_function_type}")
    
    def compute_uncertainty(self, policy_entropy: float, value_sparseness: float, 
                          stones_on_board: int, total_stones: int = 361) -> float:
        """
        Compute uncertainty score for a position.
        
        Args:
            policy_entropy: Cross-entropy of policy distribution (E)
            value_sparseness: Sparseness of value distribution (K)
            stones_on_board: Number of stones currently on board
            total_stones: Total possible stones (19x19 = 361)
        
        Returns:
            Uncertainty score
        """
        # Compute phase multiplier based on stones on board
        phase_multiplier = self.compute_phase_multiplier(stones_on_board, total_stones)
        
        # Combined uncertainty score: (w1*E + w2*K) * phase(stones_on_board)
        uncertainty = (self.w1 * policy_entropy + self.w2 * value_sparseness) * phase_multiplier
        return uncertainty


class Phase1Tuner:
    """Tunes Phase 1 parameters using supervised learning on ground truth database"""
    
    def __init__(self, 
                 shallow_db_path: str,
                 deep_db_path: str,
                 output_dir: str = "./tuning_results/phase1_supervised",
                 train_split: float = 0.8):
        """
        Args:
            shallow_db_path: Path to JSON with shallow MCTS results (800 visits)
            deep_db_path: Path to JSON with deep MCTS results (5000+ visits)
            output_dir: Directory to save results
            train_split: Fraction of data to use for training (rest for validation)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_split = train_split
        self.positions_train: List[PositionData] = []
        self.positions_val: List[PositionData] = []
        
        # Load and match positions from both databases
        print(f"Loading shallow MCTS database from {shallow_db_path}...")
        print(f"Loading deep MCTS database from {deep_db_path}...")
        self.load_databases(shallow_db_path, deep_db_path)
        
        # Initialize W&B if available
        if WANDB_AVAILABLE:
            wandb.init(
                project="rag-alphago-phase1",
                name=f"supervised_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "train_size": len(self.positions_train),
                    "val_size": len(self.positions_val),
                    "train_split": train_split
                }
            )
    
    def load_databases(self, shallow_path: str, deep_path: str):
        """Load and match positions from shallow and deep MCTS databases"""
        # Load shallow MCTS results
        with open(shallow_path, 'r') as f:
            shallow_data = json.load(f)
        
        # Load deep MCTS results
        with open(deep_path, 'r') as f:
            deep_data = json.load(f)
        
        # Build dictionaries keyed by sym_hash for matching
        shallow_dict = {}
        if isinstance(shallow_data, list):
            shallow_dict = {entry['sym_hash']: entry for entry in shallow_data}
        elif 'positions' in shallow_data:
            shallow_dict = {entry['sym_hash']: entry for entry in shallow_data['positions']}
        else:
            shallow_dict = {entry['sym_hash']: entry for entry in shallow_data.values()}
        
        deep_dict = {}
        if isinstance(deep_data, list):
            deep_dict = {entry['sym_hash']: entry for entry in deep_data}
        elif 'positions' in deep_data:
            deep_dict = {entry['sym_hash']: entry for entry in deep_data['positions']}
        else:
            deep_dict = {entry['sym_hash']: entry for entry in deep_data.values()}
        
        # Match positions by sym_hash
        print(f"\nMatching positions between databases...")
        print(f"  Shallow entries: {len(shallow_dict)}")
        print(f"  Deep entries: {len(deep_dict)}")
        
        all_positions = []
        matched_count = 0
        
        for sym_hash, shallow_entry in shallow_dict.items():
            if sym_hash in deep_dict:
                deep_entry = deep_dict[sym_hash]
                
                # Extract data
                shallow_policy = np.array(shallow_entry['policy'][:361])  # Only board moves
                deep_policy = np.array(deep_entry['policy'][:361])
                
                # Normalize policies if needed
                if shallow_policy.sum() > 0:
                    shallow_policy = shallow_policy / shallow_policy.sum()
                if deep_policy.sum() > 0:
                    deep_policy = deep_policy / deep_policy.sum()
                
                # Compute features and errors
                features = self._compute_features(shallow_policy, deep_policy, 
                                                 shallow_entry, deep_entry)
                
                position = PositionData(
                    id=sym_hash,
                    stones_on_board=shallow_entry.get('stone_count', 0),
                    shallow_policy=shallow_policy,
                    shallow_winrate=shallow_entry.get('winrate', 0.5),
                    shallow_score_lead=shallow_entry.get('score_lead', 0.0),
                    deep_policy=deep_policy,
                    deep_winrate=deep_entry.get('winrate', 0.5),
                    deep_score_lead=deep_entry.get('score_lead', 0.0),
                    policy_entropy=features['policy_entropy'],
                    value_entropy=features['value_entropy'],
                    policy_cross_entropy=features['policy_cross_entropy'],
                    value_sparseness=features['value_sparseness'],
                    policy_kl_divergence=features['policy_kl_divergence'],
                    value_absolute_error=features['value_absolute_error'],
                    combined_error=features['combined_error'],
                    komi=shallow_entry.get('komi', 7.5),
                    query_id=shallow_entry.get('query_id', '')
                )
                all_positions.append(position)
                matched_count += 1
        
        print(f"  Matched positions: {matched_count}")
        
        if matched_count == 0:
            raise ValueError("No matching positions found between databases! Check sym_hash keys.")
        
        # Split into train/val
        np.random.seed(42)
        np.random.shuffle(all_positions)
        split_idx = int(len(all_positions) * self.train_split)
        
        self.positions_train = all_positions[:split_idx]
        self.positions_val = all_positions[split_idx:]
        
        print(f"\nLoaded {len(all_positions)} positions:")
        print(f"  Training: {len(self.positions_train)}")
        print(f"  Validation: {len(self.positions_val)}")
        
        # Print error statistics
        errors = [p.combined_error for p in all_positions]
        print(f"\nCombined error statistics:")
        print(f"  Mean: {np.mean(errors):.6f}")
        print(f"  Std: {np.std(errors):.6f}")
        print(f"  Min: {np.min(errors):.6f}")
        print(f"  Max: {np.max(errors):.6f}")
        print(f"  Median: {np.median(errors):.6f}")
        print(f"  90th percentile: {np.percentile(errors, 90):.6f}")
        print(f"  95th percentile: {np.percentile(errors, 95):.6f}")
    
    def _compute_features(self, shallow_policy: np.ndarray, deep_policy: np.ndarray,
                         shallow_entry: Dict, deep_entry: Dict) -> Dict:
        """
        Compute uncertainty features and error metrics.
        
        Features for uncertainty detection:
        - policy_entropy: Entropy of shallow policy (high = uncertain)
        - value_entropy: Derived from child node value variance
        - policy_cross_entropy: H(deep, shallow) - measures policy difference
        - value_sparseness: K metric from claude_instructions (sparse = uncertain)
        
        Error metrics (ground truth):
        - policy_kl_divergence: KL(deep || shallow)
        - value_absolute_error: |deep_winrate - shallow_winrate|
        - combined_error: Weighted combination
        """
        eps = 1e-10
        
        # 1. Policy entropy (uncertainty indicator)
        # High entropy = model is uncertain about best move
        policy_entropy = entropy(shallow_policy + eps)
        
        # 2. Value entropy/sparseness (K metric)
        # Extract child node values if available
        child_nodes = shallow_entry.get('child_nodes', [])
        if child_nodes and len(child_nodes) > 1:
            values = np.array([node.get('value', 0.5) for node in child_nodes])
            # K = sparseness metric: higher when values are spread out
            value_mean = np.mean(values)
            value_variance = np.var(values)
            value_sparseness = value_variance  # Simple sparseness measure
            
            # Alternative: entropy of value distribution
            value_range = values.max() - values.min()
            if value_range > 0:
                normalized_values = (values - values.min()) / value_range
                # Add small constant and renormalize to make it a valid distribution
                normalized_values = normalized_values + eps
                normalized_values = normalized_values / normalized_values.sum()
                value_entropy = entropy(normalized_values)
            else:
                value_entropy = 0.0
        else:
            # Fallback: use winrate uncertainty
            value_sparseness = abs(0.5 - shallow_entry.get('winrate', 0.5))
            value_entropy = value_sparseness
        
        # 3. Policy cross-entropy: H(deep, shallow) = -sum(deep * log(shallow))
        # Measures how well shallow predicts deep
        policy_cross_entropy = -np.sum(deep_policy * np.log(shallow_policy + eps))
        
        # 4. Policy KL divergence: KL(deep || shallow) = sum(deep * log(deep/shallow))
        policy_kl_divergence = np.sum(deep_policy * np.log((deep_policy + eps) / (shallow_policy + eps)))
        
        # 5. Value absolute error
        value_absolute_error = abs(deep_entry.get('winrate', 0.5) - shallow_entry.get('winrate', 0.5))
        
        # 6. Combined error (weighted)
        # Weight policy error more heavily since it affects move selection
        combined_error = 0.7 * policy_kl_divergence + 0.3 * value_absolute_error
        
        return {
            'policy_entropy': policy_entropy,
            'value_entropy': value_entropy,
            'policy_cross_entropy': policy_cross_entropy,
            'value_sparseness': value_sparseness,
            'policy_kl_divergence': policy_kl_divergence,
            'value_absolute_error': value_absolute_error,
            'combined_error': combined_error
        }
    
    def evaluate_config(self, config: UncertaintyConfig, 
                       positions: List[PositionData]) -> Dict:
        """
        Evaluate how well a config predicts errors.
        
        Metrics:
        1. Pearson correlation: Linear relationship
        2. Spearman correlation: Rank correlation
        3. NDCG: Ranking quality
        4. Top-K precision: Are high uncertainty positions actually high error?
        """
        # Compute uncertainty scores
        uncertainty_scores = []
        errors = []
        
        for pos in positions:
            uncertainty = config.compute_uncertainty(
                policy_entropy=pos.policy_cross_entropy,  # Use cross-entropy as "E"
                value_sparseness=pos.value_sparseness,    # Use sparseness as "K"
                stones_on_board=pos.stones_on_board
            )
            uncertainty_scores.append(uncertainty)
            errors.append(pos.combined_error)
        
        uncertainty_scores = np.array(uncertainty_scores)
        errors = np.array(errors)
        
        # Compute correlations
        pearson_corr, pearson_p = pearsonr(uncertainty_scores, errors)
        spearman_corr, spearman_p = spearmanr(uncertainty_scores, errors)
        
        # Top-K precision: Among top 10% uncertain, what % are actually top 10% error?
        k = max(1, len(positions) // 10)
        top_k_uncertain_idx = np.argsort(-uncertainty_scores)[:k]
        top_k_error_idx = np.argsort(-errors)[:k]
        top_k_precision = len(set(top_k_uncertain_idx) & set(top_k_error_idx)) / k
        
        # NDCG score if available
        if HAS_SKLEARN and len(positions) > 1:
            try:
                ndcg = ndcg_score(errors.reshape(1, -1), uncertainty_scores.reshape(1, -1))
            except:
                ndcg = 0.0
        else:
            ndcg = 0.0
        
        results = {
            'pearson_correlation': float(pearson_corr),
            'pearson_pvalue': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_pvalue': float(spearman_p),
            'top_k_precision': float(top_k_precision),
            'ndcg_score': float(ndcg),
            'mean_uncertainty': float(np.mean(uncertainty_scores)),
            'std_uncertainty': float(np.std(uncertainty_scores)),
            'mean_error': float(np.mean(errors)),
            'config': asdict(config)
        }
        
        return results
    
    def generate_configs(self) -> List[UncertaintyConfig]:
        """Generate grid of configurations to test"""
        configs = []
        
        # Grid search over w1 values
        w1_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        # Phase function configurations
        # Linear function: phase(s) = a*(s/361) + b
        phase_configs = [
            ('linear', [0.0, 1.0]),    # Constant: always 1.0
            ('linear', [0.3, 0.85]),   # Mild increase: 0.85 (early) to 1.15 (late)
            ('linear', [0.5, 0.75]),   # Moderate increase: 0.75 (early) to 1.25 (late)
            ('linear', [-0.3, 1.15]),  # Mild decrease: 1.15 (early) to 0.85 (late)
            ('linear', [-0.5, 1.25]),  # Moderate decrease: 1.25 (early) to 0.75 (late)
        ]
        
        # Generate all combinations
        for w1 in w1_values:
            w2 = 1.0 - w1  # Normalized weights
            for phase_type, phase_coef in phase_configs:
                config = UncertaintyConfig(
                    w1=w1,
                    w2=w2,
                    phase_function_type=phase_type,
                    phase_coefficients=phase_coef
                )
                configs.append(config)
        
        print(f"Generated {len(configs)} configurations:")
        print(f"  w1 values: {w1_values}")
        print(f"  Phase functions: {len(phase_configs)} variants")
        print(f"  Total: {len(w1_values)} × {len(phase_configs)} = {len(configs)}")
        
        return configs
    
    def grid_search(self) -> Tuple[UncertaintyConfig, Dict]:
        """
        Grid search over discrete parameter values.
        """
        print("\n" + "="*80)
        print("GRID SEARCH: Testing configurations")
        print("="*80)
        
        configs = self.generate_configs()
        
        best_config = None
        best_score = -float('inf')
        all_results = []
        
        for i, config in enumerate(configs):
            print(f"\n[{i+1}/{len(configs)}] Testing: {config.config_id}")
            
            # Evaluate on training set
            train_result = self.evaluate_config(config, self.positions_train)
            
            # Use Pearson correlation as primary metric
            score = train_result['pearson_correlation']
            
            all_results.append({
                'config': asdict(config),
                'train_metrics': train_result,
                'score': score
            })
            
            print(f"  Training Pearson: {score:.4f}")
            print(f"  Training Top-10% precision: {train_result['top_k_precision']:.4f}")
            
            if score > best_score:
                best_score = score
                best_config = config
                print(f"  ✓ New best configuration!")
            
            # Log to W&B
            if WANDB_AVAILABLE:
                wandb.log({
                    'config_id': config.config_id,
                    'train_pearson': score,
                    'train_top_k_precision': train_result['top_k_precision'],
                    'best_score_so_far': best_score
                })
        
        print(f"\n" + "="*80)
        print(f"BEST CONFIGURATION FOUND:")
        print(f"  w1: {best_config.w1:.4f}, w2: {best_config.w2:.4f}")
        print(f"  Phase: {best_config.phase_function_type} {best_config.phase_coefficients}")
        print(f"  Training Pearson correlation: {best_score:.4f}")
        print("="*80)
        
        # Save all results
        results_file = self.output_dir / "grid_search_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved grid search results to: {results_file}")
        
        return best_config, all_results[0]
    
    def optimize_parameters(self) -> UncertaintyConfig:
        """
        Optimize uncertainty parameters using gradient-based optimization.
        
        Objective: Maximize correlation between uncertainty and combined_error
        """
        print("\n" + "="*80)
        print("GRADIENT OPTIMIZATION: Finding optimal parameters")
        print("="*80)
        
        # Extract features and targets from training data
        E = np.array([p.policy_cross_entropy for p in self.positions_train])
        K = np.array([p.value_sparseness for p in self.positions_train])
        S = np.array([p.stones_on_board for p in self.positions_train])
        errors = np.array([p.combined_error for p in self.positions_train])
        
        print(f"Training on {len(self.positions_train)} positions")
        print(f"  E (policy cross-entropy) range: [{E.min():.3f}, {E.max():.3f}]")
        print(f"  K (value sparseness) range: [{K.min():.3f}, {K.max():.3f}]")
        print(f"  Errors range: [{errors.min():.6f}, {errors.max():.6f}]")
        
        # Define objective function
        def objective(params):
            """Negative correlation (minimize to maximize correlation)"""
            w1, a, b = params
            w2 = 1.0 - w1  # Ensure w1 + w2 = 1
            
            # Compute phase multiplier (linear: a*s/361 + b)
            phase = a * (S / 361.0) + b
            
            # Compute uncertainty scores
            uncertainty = (w1 * E + w2 * K) * phase
            
            # Compute correlation
            try:
                corr, _ = pearsonr(uncertainty, errors)
                if np.isnan(corr):
                    return 1e6  # Penalty for invalid correlation
            except:
                return 1e6
            
            # Return negative (minimize)
            return -corr
        
        # Initial guess: equal weights, constant phase
        x0 = [0.5, 0.0, 1.0]  # w1, a, b
        
        # Bounds
        bounds = [
            (0.1, 0.9),   # w1 in [0.1, 0.9]
            (-1.0, 1.0),  # a (phase slope)
            (0.5, 1.5)    # b (phase intercept)
        ]
        
        print("\nRunning L-BFGS-B optimization...")
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'disp': True}
        )
        
        # Extract optimal parameters
        w1_opt = result.x[0]
        w2_opt = 1.0 - w1_opt
        a_opt, b_opt = result.x[1], result.x[2]
        
        optimal_config = UncertaintyConfig(
            w1=w1_opt,
            w2=w2_opt,
            phase_function_type='linear',
            phase_coefficients=[a_opt, b_opt]
        )
        
        print(f"\n" + "="*80)
        print(f"OPTIMIZATION COMPLETE:")
        print(f"  w1 (policy entropy): {w1_opt:.4f}")
        print(f"  w2 (value sparseness): {w2_opt:.4f}")
        print(f"  phase function: {a_opt:.4f}*s + {b_opt:.4f}")
        print(f"  Correlation achieved: {-result.fun:.4f}")
        print("="*80)
        
        return optimal_config
    
    def find_storage_threshold(self, config: UncertaintyConfig,
                               target_percentiles: List[float] = [5, 10, 15, 20, 25]) -> Dict:
        """
        Find optimal storage threshold given the optimized uncertainty function.
        
        Tests different percentiles to find Pareto frontier.
        """
        print("\n" + "="*80)
        print("THRESHOLD ANALYSIS: Finding optimal storage cutoff")
        print("="*80)
        
        # Use validation set
        positions = self.positions_val
        
        # Compute uncertainty for all positions
        uncertainties = []
        errors = []
        for pos in positions:
            unc = config.compute_uncertainty(
                pos.policy_cross_entropy,
                pos.value_sparseness,
                pos.stones_on_board
            )
            uncertainties.append(unc)
            errors.append(pos.combined_error)
        
        uncertainties = np.array(uncertainties)
        errors = np.array(errors)
        
        # For each percentile, compute threshold and metrics
        threshold_analysis = {}
        
        for percentile in target_percentiles:
            # Threshold: store top X% uncertain positions
            threshold = np.percentile(uncertainties, 100 - percentile)
            
            # Positions that would be stored
            stored_mask = uncertainties >= threshold
            num_stored = np.sum(stored_mask)
            
            if num_stored == 0:
                continue
            
            # Average error of stored positions (should be high)
            avg_error_stored = np.mean(errors[stored_mask])
            
            # Average error of NOT stored positions (should be low)
            if np.sum(~stored_mask) > 0:
                avg_error_not_stored = np.mean(errors[~stored_mask])
            else:
                avg_error_not_stored = 0.0
            
            # Benefit: difference in average error
            benefit = avg_error_stored - avg_error_not_stored
            
            # Coverage: what % of high-error positions do we capture?
            high_error_threshold = np.percentile(errors, 100 - percentile)
            high_error_mask = errors >= high_error_threshold
            coverage = np.sum(stored_mask & high_error_mask) / max(1, np.sum(high_error_mask))
            
            threshold_analysis[percentile] = {
                'threshold': float(threshold),
                'num_stored': int(num_stored),
                'storage_rate': float(num_stored / len(positions)),
                'avg_error_stored': float(avg_error_stored),
                'avg_error_not_stored': float(avg_error_not_stored),
                'benefit': float(benefit),
                'coverage': float(coverage)
            }
            
            print(f"\nPercentile {percentile}%:")
            print(f"  Threshold: {threshold:.4f}")
            print(f"  Positions stored: {num_stored} ({100*num_stored/len(positions):.1f}%)")
            print(f"  Avg error (stored): {avg_error_stored:.6f}")
            print(f"  Avg error (not stored): {avg_error_not_stored:.6f}")
            print(f"  Benefit: {benefit:.6f}")
            print(f"  Coverage of high-error: {100*coverage:.1f}%")
        
        # Save analysis
        threshold_file = self.output_dir / "threshold_analysis.json"
        with open(threshold_file, 'w') as f:
            json.dump(threshold_analysis, f, indent=2)
        print(f"\nSaved threshold analysis to: {threshold_file}")
        
        # Plot Pareto frontier
        self.plot_threshold_analysis(threshold_analysis)
        
        return threshold_analysis
    
    def plot_threshold_analysis(self, analysis: Dict):
        """Plot Pareto frontier: storage rate vs benefit"""
        if len(analysis) == 0:
            return
        
        percentiles = sorted(analysis.keys())
        storage_rates = [analysis[p]['storage_rate'] * 100 for p in percentiles]
        benefits = [analysis[p]['benefit'] for p in percentiles]
        coverages = [analysis[p]['coverage'] * 100 for p in percentiles]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Storage rate vs Benefit
        ax1.plot(storage_rates, benefits, 'o-', linewidth=2, markersize=8)
        for i, p in enumerate(percentiles):
            ax1.annotate(f'{p}%', (storage_rates[i], benefits[i]), 
                        textcoords="offset points", xytext=(5,5))
        ax1.set_xlabel('Storage Rate (%)', fontsize=12)
        ax1.set_ylabel('Expected Benefit (Error Reduction)', fontsize=12)
        ax1.set_title('Pareto Frontier: Storage vs Benefit', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Storage rate vs Coverage
        ax2.plot(storage_rates, coverages, 's-', linewidth=2, markersize=8, color='orange')
        for i, p in enumerate(percentiles):
            ax2.annotate(f'{p}%', (storage_rates[i], coverages[i]),
                        textcoords="offset points", xytext=(5,5))
        ax2.set_xlabel('Storage Rate (%)', fontsize=12)
        ax2.set_ylabel('Coverage of High-Error Positions (%)', fontsize=12)
        ax2.set_title('Storage Rate vs Coverage', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / "threshold_pareto_frontier.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Saved Pareto frontier plot to: {plot_file}")
        plt.close()
    
    def run_full_tuning(self, method: str = 'grid'):
        """
        Run complete supervised tuning pipeline.
        
        Args:
            method: 'grid' for grid search or 'optimize' for gradient optimization
        """
        print("\n" + "="*80)
        print("PHASE 1: SUPERVISED UNCERTAINTY TUNING")
        print("="*80)
        print(f"Method: {method.upper()}")
        print(f"Training positions: {len(self.positions_train)}")
        print(f"Validation positions: {len(self.positions_val)}")
        
        # Step 1: Optimize parameters
        if method == 'optimize':
            best_config = self.optimize_parameters()
        else:
            best_config, _ = self.grid_search()
        
        # Step 2: Validate on held-out data
        print("\n" + "="*80)
        print("VALIDATION: Evaluating on held-out data")
        print("="*80)
        val_results = self.evaluate_config(best_config, self.positions_val)
        print(f"Validation Pearson correlation: {val_results['pearson_correlation']:.4f}")
        print(f"Validation Spearman correlation: {val_results['spearman_correlation']:.4f}")
        print(f"Validation Top-10% precision: {val_results['top_k_precision']:.4f}")
        
        # Step 3: Find storage threshold
        threshold_analysis = self.find_storage_threshold(best_config)
        
        # Save final config
        final_config_file = self.output_dir / "best_uncertainty_config.json"
        final_results = {
            'config': asdict(best_config),
            'validation_metrics': val_results,
            'threshold_analysis': threshold_analysis,
            'training_positions': len(self.positions_train),
            'validation_positions': len(self.positions_val),
            'tuning_method': method
        }
        
        with open(final_config_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print("\n" + "="*80)
        print("TUNING COMPLETE")
        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        print(f"Best config: {final_config_file}")
        print(f"\nRecommendation: Use threshold at 10-15% for good balance")
        
        return best_config, threshold_analysis


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1: Supervised Uncertainty Tuning")
    parser.add_argument("--shallow-db", type=str, required=True,
                       help="Path to shallow MCTS JSON database (800 visits)")
    parser.add_argument("--deep-db", type=str, required=True,
                       help="Path to deep MCTS JSON database (5000+ visits)")
    parser.add_argument("--output-dir", type=str, 
                       default="./tuning_results/phase1_supervised")
    parser.add_argument("--method", choices=["grid", "optimize"], default="grid",
                       help="Optimization method: grid (thorough) or optimize (fast)")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Fraction of data for training (rest for validation)")
    
    args = parser.parse_args()
    
    # Create tuner and run
    tuner = Phase1Tuner(
        shallow_db_path=args.shallow_db,
        deep_db_path=args.deep_db,
        output_dir=args.output_dir,
        train_split=args.train_split
    )
    
    best_config, threshold_analysis = tuner.run_full_tuning(method=args.method)
    
    print("\n" + "="*80)
    print("FINAL CONFIGURATION:")
    print(f"  w1: {best_config.w1:.4f}")
    print(f"  w2: {best_config.w2:.4f}")
    print(f"  Phase function: {best_config.phase_function_type}")
    print(f"  Phase coefficients: {best_config.phase_coefficients}")
    print("="*80)


if __name__ == "__main__":
    main()
