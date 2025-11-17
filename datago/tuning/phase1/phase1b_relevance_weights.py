"""
Phase 1b: Relevance Weight Tuning for RAG Similarity Scoring

Tests how policy and value distributions change when non-gamestate features are perturbed
for positions with IDENTICAL sym_hash (same game state, different context).

Key insight: Since sym_hash is identical, the game state is the same. We want to test
how much other features (winrate, score_lead, komi, etc.) affect the optimal policy/value.
If these features are significantly different but policy/value remain similar, they should
have lower weights in the relevance comparison.

The relevance weights from claude_instructions.txt:
    'policy': 0.40,        # Highest - determines exploration
    'winrate': 0.25,       # High - primary utility component  
    'score_lead': 0.10,    # Moderate - secondary utility
    'visit_distribution': 0.15,  # High - shows what MCTS actually preferred
    'stone_count': 0.05,   # Phase context
    'komi': 0.05,          # Must match exactly (binary)

Hardware: NVIDIA A100
Time Budget: 2-4 hours
"""

import os
import json
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Optional: sklearn for metrics
try:
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, using basic metrics only")

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: W&B not available. Using local logging.")


@dataclass
class RAGPosition:
    """Position from RAG database"""
    sym_hash: str
    state_hash: str
    policy: np.ndarray  # (362,) or (361,)
    winrate: float
    score_lead: float
    komi: float
    stone_count: int
    ownership: Optional[np.ndarray] = None
    child_nodes: Optional[List[Dict]] = None
    move_infos: Optional[List[Dict]] = None
    query_id: str = ""


@dataclass
class PositionGroup:
    """Group of positions with identical sym_hash but different contexts"""
    sym_hash: str
    positions: List[RAGPosition]
    
    def get_feature_variance(self, feature: str) -> float:
        """Get variance of a specific feature across positions"""
        if feature == 'winrate':
            values = [p.winrate for p in self.positions]
        elif feature == 'score_lead':
            values = [p.score_lead for p in self.positions]
        elif feature == 'komi':
            values = [p.komi for p in self.positions]
        elif feature == 'stone_count':
            values = [p.stone_count for p in self.positions]
        else:
            return 0.0
        return np.var(values)
    
    def get_policy_variance(self) -> float:
        """Get average pairwise policy distance"""
        policies = [p.policy for p in self.positions]
        if len(policies) < 2:
            return 0.0
        
        distances = []
        for i in range(len(policies)):
            for j in range(i+1, len(policies)):
                # KL divergence
                eps = 1e-10
                kl = np.sum(policies[i] * np.log((policies[i] + eps) / (policies[j] + eps)))
                distances.append(kl)
        return np.mean(distances)
    
    def get_value_variance(self) -> float:
        """Get variance in winrate predictions"""
        winrates = [p.winrate for p in self.positions]
        return np.var(winrates)


@dataclass
class RelevanceWeights:
    """Weights for relevance similarity scoring"""
    policy: float = 0.40
    winrate: float = 0.25
    score_lead: float = 0.10
    visit_distribution: float = 0.15
    stone_count: float = 0.05
    komi: float = 0.05
    
    def __post_init__(self):
        # Normalize weights to sum to 1
        total = (self.policy + self.winrate + self.score_lead + 
                self.visit_distribution + self.stone_count + self.komi)
        if total > 0:
            self.policy /= total
            self.winrate /= total
            self.score_lead /= total
            self.visit_distribution /= total
            self.stone_count /= total
            self.komi /= total
    
    def to_dict(self) -> Dict:
        return {
            'policy': self.policy,
            'winrate': self.winrate,
            'score_lead': self.score_lead,
            'visit_distribution': self.visit_distribution,
            'stone_count': self.stone_count,
            'komi': self.komi
        }
    
    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**d)
    
    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Create from numpy array [policy, winrate, score_lead, visit_dist, stone_count, komi]"""
        return cls(
            policy=arr[0],
            winrate=arr[1],
            score_lead=arr[2],
            visit_distribution=arr[3],
            stone_count=arr[4],
            komi=arr[5]
        )
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimization"""
        return np.array([
            self.policy,
            self.winrate,
            self.score_lead,
            self.visit_distribution,
            self.stone_count,
            self.komi
        ])


@dataclass
class Phase1bMetrics:
    """Metrics for evaluating relevance weights"""
    config_id: str
    weights: Dict
    
    # How well do weights predict policy/value stability?
    policy_stability_correlation: float  # High feature similarity -> low policy change
    value_stability_correlation: float   # High feature similarity -> low value change
    combined_correlation: float
    
    # Sensitivity analysis
    feature_sensitivities: Dict[str, float]  # How much each feature affects predictions
    
    # Statistical significance
    num_groups: int
    num_comparisons: int


class Phase1bTuner:
    """Tunes relevance weights for RAG similarity scoring"""
    
    def __init__(self,
                 rag_database_path: str,
                 output_dir: str = "./tuning_results/phase1b",
                 min_positions_per_group: int = 3):
        """
        Args:
            rag_database_path: Path to RAG database JSON
            output_dir: Directory to save results
            min_positions_per_group: Minimum positions with same sym_hash to form a group
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_positions_per_group = min_positions_per_group
        self.position_groups: List[PositionGroup] = []
        
        # Load and group positions
        print(f"Loading RAG database from {rag_database_path}...")
        self.load_and_group_positions(rag_database_path)
        
        # Initialize W&B if available
        if WANDB_AVAILABLE:
            wandb.init(
                project="rag-alphago-phase1b",
                name=f"relevance_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "num_groups": len(self.position_groups),
                    "min_positions_per_group": min_positions_per_group
                }
            )
    
    def load_and_group_positions(self, db_path: str):
        """Load RAG database and group positions by sym_hash"""
        with open(db_path, 'r') as f:
            data = json.load(f)
        
        # Parse positions
        positions_by_hash = defaultdict(list)
        
        if isinstance(data, list):
            entries = data
        elif 'positions' in data:
            entries = data['positions']
        else:
            entries = list(data.values())
        
        print(f"Loading {len(entries)} positions from database...")
        
        for entry in entries:
            sym_hash = entry.get('sym_hash', '')
            if not sym_hash:
                continue
            
            policy = np.array(entry.get('policy', []))
            if len(policy) > 361:
                policy = policy[:361]  # Only board moves
            
            # Normalize policy
            if policy.sum() > 0:
                policy = policy / policy.sum()
            
            position = RAGPosition(
                sym_hash=sym_hash,
                state_hash=entry.get('state_hash', ''),
                policy=policy,
                winrate=entry.get('winrate', 0.5),
                score_lead=entry.get('score_lead', 0.0),
                komi=entry.get('komi', 7.5),
                stone_count=entry.get('stone_count', 0),
                ownership=np.array(entry.get('ownership', [])) if entry.get('ownership') else None,
                child_nodes=entry.get('child_nodes', []),
                move_infos=entry.get('move_infos', []),
                query_id=entry.get('query_id', '')
            )
            
            positions_by_hash[sym_hash].append(position)
        
        # Create groups with enough positions
        print(f"\nGrouping positions by sym_hash...")
        for sym_hash, positions in positions_by_hash.items():
            if len(positions) >= self.min_positions_per_group:
                group = PositionGroup(sym_hash=sym_hash, positions=positions)
                self.position_groups.append(group)
        
        print(f"Found {len(self.position_groups)} groups with {self.min_positions_per_group}+ positions")
        
        if len(self.position_groups) == 0:
            raise ValueError(
                f"No position groups found with {self.min_positions_per_group}+ identical sym_hashes. "
                f"Need positions with same game state but different contexts (komi, etc.)"
            )
        
        # Print statistics
        group_sizes = [len(g.positions) for g in self.position_groups]
        print(f"\nGroup size statistics:")
        print(f"  Mean: {np.mean(group_sizes):.1f}")
        print(f"  Median: {np.median(group_sizes):.1f}")
        print(f"  Max: {np.max(group_sizes)}")
        print(f"  Total positions in groups: {sum(group_sizes)}")
    
    def compute_feature_similarity(self, pos1: RAGPosition, pos2: RAGPosition,
                                   weights: RelevanceWeights) -> float:
        """
        Compute weighted feature similarity between two positions.
        
        Since sym_hash is identical, we only compare non-gamestate features:
        - winrate, score_lead, komi, stone_count
        - visit_distribution (from child nodes)
        """
        eps = 1e-10
        similarities = {}
        
        # 1. Winrate similarity (normalized difference)
        winrate_diff = abs(pos1.winrate - pos2.winrate)
        similarities['winrate'] = 1.0 - winrate_diff  # Convert to similarity
        
        # 2. Score lead similarity (normalized by typical range ~20)
        score_diff = abs(pos1.score_lead - pos2.score_lead)
        similarities['score_lead'] = 1.0 / (1.0 + score_diff / 20.0)
        
        # 3. Komi similarity (binary: exact match or not)
        similarities['komi'] = 1.0 if pos1.komi == pos2.komi else 0.0
        
        # 4. Stone count similarity (normalized by board size 361)
        stone_diff = abs(pos1.stone_count - pos2.stone_count)
        similarities['stone_count'] = 1.0 - (stone_diff / 361.0)
        
        # 5. Visit distribution similarity (from child nodes)
        if pos1.child_nodes and pos2.child_nodes:
            visits1 = np.array([node.get('visits', 1) for node in pos1.child_nodes])
            visits2 = np.array([node.get('visits', 1) for node in pos2.child_nodes])
            
            # Normalize to distribution
            visits1 = visits1 / (visits1.sum() + eps)
            visits2 = visits2 / (visits2.sum() + eps)
            
            # Pad to same length
            max_len = max(len(visits1), len(visits2))
            if len(visits1) < max_len:
                visits1 = np.pad(visits1, (0, max_len - len(visits1)))
            if len(visits2) < max_len:
                visits2 = np.pad(visits2, (0, max_len - len(visits2)))
            
            # Cosine similarity
            similarities['visit_distribution'] = 1.0 - cosine(visits1, visits2)
        else:
            similarities['visit_distribution'] = 0.5  # Neutral if missing
        
        # 6. Policy similarity (for reference, but gamestate is identical)
        policy_sim = 1.0 - cosine(pos1.policy, pos2.policy)
        similarities['policy'] = policy_sim
        
        # Compute weighted similarity
        weighted_sim = (
            weights.policy * similarities['policy'] +
            weights.winrate * similarities['winrate'] +
            weights.score_lead * similarities['score_lead'] +
            weights.visit_distribution * similarities['visit_distribution'] +
            weights.stone_count * similarities['stone_count'] +
            weights.komi * similarities['komi']
        )
        
        return weighted_sim, similarities
    
    def compute_policy_value_distance(self, pos1: RAGPosition, pos2: RAGPosition) -> Tuple[float, float]:
        """
        Compute how much policy and value differ between positions.
        This is what we want to MINIMIZE when feature similarity is HIGH.
        """
        eps = 1e-10
        
        # Policy distance (KL divergence)
        policy_kl = np.sum(pos1.policy * np.log((pos1.policy + eps) / (pos2.policy + eps)))
        
        # Value distance (absolute winrate difference)
        value_diff = abs(pos1.winrate - pos2.winrate)
        
        return policy_kl, value_diff
    
    def evaluate_weights(self, weights: RelevanceWeights) -> Phase1bMetrics:
        """
        Evaluate how well weights predict policy/value stability.
        
        Hypothesis: If two positions have high feature similarity (according to weights),
        they should have similar policy and value distributions.
        
        We measure correlation between:
        - Feature similarity (high) -> Policy/value distance (low)
        """
        feature_similarities = []
        policy_distances = []
        value_distances = []
        
        # For each group, compare all pairs
        for group in self.position_groups:
            positions = group.positions
            
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    pos1, pos2 = positions[i], positions[j]
                    
                    # Compute feature similarity
                    feat_sim, _ = self.compute_feature_similarity(pos1, pos2, weights)
                    
                    # Compute policy/value distance
                    policy_dist, value_dist = self.compute_policy_value_distance(pos1, pos2)
                    
                    feature_similarities.append(feat_sim)
                    policy_distances.append(policy_dist)
                    value_distances.append(value_dist)
        
        feature_similarities = np.array(feature_similarities)
        policy_distances = np.array(policy_distances)
        value_distances = np.array(value_distances)
        
        # Correlation: high feature sim -> low policy/value distance
        # So we expect NEGATIVE correlation (or positive with inverted distance)
        policy_correlation, _ = spearmanr(-policy_distances, feature_similarities)
        value_correlation, _ = spearmanr(-value_distances, feature_similarities)
        
        # Combined correlation (weighted average)
        combined_correlation = 0.6 * policy_correlation + 0.4 * value_correlation
        
        # Feature sensitivity analysis
        feature_sensitivities = self._compute_feature_sensitivities(weights)
        
        metrics = Phase1bMetrics(
            config_id=f"w_{weights.policy:.2f}_{weights.winrate:.2f}_{weights.score_lead:.2f}",
            weights=weights.to_dict(),
            policy_stability_correlation=policy_correlation,
            value_stability_correlation=value_correlation,
            combined_correlation=combined_correlation,
            feature_sensitivities=feature_sensitivities,
            num_groups=len(self.position_groups),
            num_comparisons=len(feature_similarities)
        )
        
        return metrics
    
    def _compute_feature_sensitivities(self, weights: RelevanceWeights) -> Dict[str, float]:
        """
        Compute how much each feature contributes to predictions.
        
        For each feature, measure how much varying it (while keeping others constant)
        affects the overall similarity score.
        """
        sensitivities = {}
        features = ['winrate', 'score_lead', 'komi', 'stone_count', 'visit_distribution']
        
        for feature in features:
            variances = []
            for group in self.position_groups:
                # Get feature variance within group
                feat_var = group.get_feature_variance(feature)
                variances.append(feat_var)
            
            # Weight by the relevance weight
            weight_val = getattr(weights, feature)
            sensitivity = weight_val * np.mean(variances)
            sensitivities[feature] = sensitivity
        
        return sensitivities
    
    def grid_search(self) -> Tuple[RelevanceWeights, Phase1bMetrics]:
        """
        Grid search over relevance weight combinations.
        
        We'll test variations around the baseline from claude_instructions.txt:
        policy: 0.40, winrate: 0.25, score_lead: 0.10, visit_distribution: 0.15
        """
        print("\n" + "="*80)
        print("GRID SEARCH: Testing weight combinations")
        print("="*80)
        
        # Define search space
        # Since weights are normalized, we vary primary weights
        policy_weights = [0.30, 0.35, 0.40, 0.45, 0.50]
        winrate_weights = [0.15, 0.20, 0.25, 0.30, 0.35]
        score_lead_weights = [0.05, 0.10, 0.15]
        visit_dist_weights = [0.10, 0.15, 0.20]
        
        # Fixed minor weights
        stone_count_weight = 0.05
        komi_weight = 0.05
        
        best_weights = None
        best_metrics = None
        best_correlation = -float('inf')
        
        total_configs = len(policy_weights) * len(winrate_weights) * len(score_lead_weights) * len(visit_dist_weights)
        print(f"Testing {total_configs} configurations...\n")
        
        config_count = 0
        for p_w in policy_weights:
            for wr_w in winrate_weights:
                for sl_w in score_lead_weights:
                    for vd_w in visit_dist_weights:
                        config_count += 1
                        
                        weights = RelevanceWeights(
                            policy=p_w,
                            winrate=wr_w,
                            score_lead=sl_w,
                            visit_distribution=vd_w,
                            stone_count=stone_count_weight,
                            komi=komi_weight
                        )
                        
                        metrics = self.evaluate_weights(weights)
                        
                        if config_count % 50 == 0:
                            print(f"[{config_count}/{total_configs}] Correlation: {metrics.combined_correlation:.4f}")
                        
                        if metrics.combined_correlation > best_correlation:
                            best_correlation = metrics.combined_correlation
                            best_weights = weights
                            best_metrics = metrics
                            print(f"  ✓ New best: {metrics.combined_correlation:.4f}")
                        
                        # Log to W&B
                        if WANDB_AVAILABLE:
                            wandb.log({
                                'policy_weight': p_w,
                                'winrate_weight': wr_w,
                                'score_lead_weight': sl_w,
                                'visit_dist_weight': vd_w,
                                'combined_correlation': metrics.combined_correlation,
                                'policy_correlation': metrics.policy_stability_correlation,
                                'value_correlation': metrics.value_stability_correlation
                            })
        
        print("\n" + "="*80)
        print("GRID SEARCH COMPLETE")
        print("="*80)
        print(f"Best configuration:")
        print(f"  Policy: {best_weights.policy:.3f}")
        print(f"  Winrate: {best_weights.winrate:.3f}")
        print(f"  Score lead: {best_weights.score_lead:.3f}")
        print(f"  Visit distribution: {best_weights.visit_distribution:.3f}")
        print(f"  Stone count: {best_weights.stone_count:.3f}")
        print(f"  Komi: {best_weights.komi:.3f}")
        print(f"\nCombined correlation: {best_metrics.combined_correlation:.4f}")
        print(f"Policy stability correlation: {best_metrics.policy_stability_correlation:.4f}")
        print(f"Value stability correlation: {best_metrics.value_stability_correlation:.4f}")
        
        return best_weights, best_metrics
    
    def optimize_weights(self, initial_weights: Optional[RelevanceWeights] = None) -> Tuple[RelevanceWeights, Phase1bMetrics]:
        """
        Optimize weights using gradient-based optimization.
        """
        print("\n" + "="*80)
        print("OPTIMIZATION: Gradient-based weight tuning")
        print("="*80)
        
        if initial_weights is None:
            # Start from baseline
            initial_weights = RelevanceWeights()
        
        def objective(x):
            """Objective to maximize: combined correlation"""
            # Ensure weights are positive and sum to 1
            x = np.abs(x)
            x = x / x.sum()
            
            weights = RelevanceWeights.from_array(x)
            metrics = self.evaluate_weights(weights)
            
            # Return negative (we minimize)
            return -metrics.combined_correlation
        
        # Initial guess
        x0 = initial_weights.to_array()
        
        # Bounds: all weights positive
        bounds = [(0.01, 0.8) for _ in range(6)]
        
        # Constraint: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        
        print("Starting optimization...")
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 100, 'disp': True}
        )
        
        # Extract best weights
        best_x = result.x
        best_x = best_x / best_x.sum()  # Normalize
        best_weights = RelevanceWeights.from_array(best_x)
        best_metrics = self.evaluate_weights(best_weights)
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Optimized weights:")
        print(f"  Policy: {best_weights.policy:.3f}")
        print(f"  Winrate: {best_weights.winrate:.3f}")
        print(f"  Score lead: {best_weights.score_lead:.3f}")
        print(f"  Visit distribution: {best_weights.visit_distribution:.3f}")
        print(f"  Stone count: {best_weights.stone_count:.3f}")
        print(f"  Komi: {best_weights.komi:.3f}")
        print(f"\nCombined correlation: {best_metrics.combined_correlation:.4f}")
        
        return best_weights, best_metrics
    
    def run_full_tuning(self, method: str = 'grid') -> Tuple[RelevanceWeights, Phase1bMetrics]:
        """
        Run complete Phase 1b tuning process.
        
        Args:
            method: 'grid' for grid search, 'optimize' for gradient optimization
        """
        print("\n" + "="*80)
        print("PHASE 1B: RELEVANCE WEIGHT TUNING")
        print("="*80)
        print(f"Method: {method}")
        print(f"Position groups: {len(self.position_groups)}")
        print(f"Min positions per group: {self.min_positions_per_group}")
        print("="*80)
        
        # Run optimization
        if method == 'optimize':
            best_weights, best_metrics = self.optimize_weights()
        else:
            best_weights, best_metrics = self.grid_search()
        
        # Save results
        results_file = self.output_dir / "phase1b_results.json"
        results = {
            'best_weights': best_weights.to_dict(),
            'metrics': {
                'combined_correlation': best_metrics.combined_correlation,
                'policy_stability_correlation': best_metrics.policy_stability_correlation,
                'value_stability_correlation': best_metrics.value_stability_correlation,
                'feature_sensitivities': best_metrics.feature_sensitivities,
                'num_groups': best_metrics.num_groups,
                'num_comparisons': best_metrics.num_comparisons
            },
            'tuning_method': method,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*80)
        print("PHASE 1B TUNING COMPLETE")
        print("="*80)
        print(f"Results saved to: {results_file}")
        print(f"\nUse these weights in Phase 1c for RAG retrieval")
        
        return best_weights, best_metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 1b: Relevance Weight Tuning")
    parser.add_argument("--rag-database", type=str, required=True,
                       help="Path to RAG database JSON (needs positions with duplicate sym_hash)")
    parser.add_argument("--output-dir", type=str, default="./tuning_results/phase1b")
    parser.add_argument("--method", choices=["grid", "optimize"], default="grid",
                       help="Optimization method: grid (thorough) or optimize (fast)")
    parser.add_argument("--min-group-size", type=int, default=3,
                       help="Minimum positions with same sym_hash to form a group")
    
    args = parser.parse_args()
    
    # Create tuner and run
    tuner = Phase1bTuner(
        rag_database_path=args.rag_database,
        output_dir=args.output_dir,
        min_positions_per_group=args.min_group_size
    )
    
    best_weights, metrics = tuner.run_full_tuning(method=args.method)
    
    print("\n" + "="*80)
    print("FINAL RELEVANCE WEIGHTS:")
    print(f"  policy: {best_weights.policy:.3f}")
    print(f"  winrate: {best_weights.winrate:.3f}")
    print(f"  score_lead: {best_weights.score_lead:.3f}")
    print(f"  visit_distribution: {best_weights.visit_distribution:.3f}")
    print(f"  stone_count: {best_weights.stone_count:.3f}")
    print(f"  komi: {best_weights.komi:.3f}")
    print("="*80)


if __name__ == "__main__":
    main()
