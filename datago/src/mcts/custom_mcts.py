"""
custom_mcts.py

Custom MCTS implementation that can use modified policy priors.
This allows RAG-enhanced policies to be used during tree search.

Unlike KataGo's black-box MCTS, this implementation allows us to:
1. Inject blended priors (network + RAG knowledge)
2. Control search parameters
3. Track and modify the search tree
"""
from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    
    # Position state
    position_hash: str
    parent: Optional[MCTSNode] = None
    children: Dict[str, MCTSNode] = None
    
    # Move that led to this node
    move: Optional[str] = None
    
    # MCTS statistics
    visit_count: int = 0
    total_value: float = 0.0
    prior_probability: float = 0.0
    
    # Network evaluation (cached)
    network_value: Optional[float] = None
    network_policy: Optional[Dict[str, float]] = None
    
    # Game state
    is_terminal: bool = False
    terminal_value: Optional[float] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def q_value(self) -> float:
        """Average action value."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def is_expanded(self) -> bool:
        """Whether this node has been expanded."""
        return len(self.children) > 0 or self.is_terminal


class CustomMCTS:
    """
    Custom MCTS implementation with support for modified priors.
    
    This implements the AlphaGo/AlphaZero style MCTS with PUCT selection,
    but allows injecting modified policy priors (e.g., blended with RAG).
    """
    
    def __init__(
        self,
        network_evaluator,  # Function: position -> (policy_dict, value)
        c_puct: float = 1.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.03,
        dirichlet_epsilon: float = 0.25,
        virtual_loss: float = 3.0,
    ):
        """
        Initialize custom MCTS.
        
        Args:
            network_evaluator: Function that evaluates positions
                               Returns: (policy_dict, value)
            c_puct: Exploration constant (higher = more exploration)
            temperature: Temperature for move selection (1.0 = proportional to visits)
            dirichlet_alpha: Alpha for Dirichlet noise on root
            dirichlet_epsilon: Weight of Dirichlet noise at root
            virtual_loss: Virtual loss to encourage parallel exploration
        """
        self.network_evaluator = network_evaluator
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.virtual_loss = virtual_loss
        
        self.root: Optional[MCTSNode] = None
    
    def search(
        self,
        position_hash: str,
        num_simulations: int,
        modified_prior: Optional[Dict[str, float]] = None,
        current_node: Optional[MCTSNode] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Run MCTS search from a position.
        
        Args:
            position_hash: Hash identifying the position
            num_simulations: Number of MCTS simulations to run
            modified_prior: Optional modified policy prior (e.g., blended with RAG)
            current_node: Optional existing node to continue search from
            
        Returns:
            Tuple of (move_probabilities, estimated_value)
            - move_probabilities: Dict mapping moves to visit-based probabilities
            - estimated_value: Average value from root
        """
        # Create or reuse root node
        if current_node is None:
            self.root = MCTSNode(position_hash=position_hash)
        else:
            self.root = current_node
        
        # Expand root if needed
        if not self.root.is_expanded:
            self._expand_node(self.root, modified_prior)
        
        # Add Dirichlet noise to root priors for exploration
        if self.dirichlet_alpha > 0:
            self._add_dirichlet_noise(self.root)
        
        # Run simulations
        for sim in range(num_simulations):
            node = self.root
            search_path = [node]
            
            # Selection: traverse tree using PUCT
            while node.is_expanded and not node.is_terminal:
                node = self._select_child(node)
                search_path.append(node)
            
            # Expansion and evaluation
            value = 0.0
            if node.is_terminal:
                value = node.terminal_value
            else:
                # Expand node (if not already expanded)
                if not node.is_expanded:
                    self._expand_node(node)
                # Get value from network
                value = node.network_value if node.network_value is not None else 0.0
            
            # Backup: propagate value up the tree
            self._backup(search_path, value)
        
        # Extract move probabilities based on visit counts
        move_probs = self._get_move_probabilities(self.root)
        estimated_value = self.root.q_value
        
        return move_probs, estimated_value
    
    def _expand_node(
        self,
        node: MCTSNode,
        modified_prior: Optional[Dict[str, float]] = None,
    ):
        """
        Expand a node by evaluating it with the network.
        
        Args:
            node: Node to expand
            modified_prior: Optional modified policy (overrides network policy)
        """
        # Get network evaluation
        policy, value = self.network_evaluator(node.position_hash)
        
        # Use modified prior if provided (RAG blending!)
        if modified_prior is not None:
            policy = modified_prior
        
        node.network_policy = policy
        node.network_value = value
        
        # Create child nodes for each legal move
        for move, prior_prob in policy.items():
            if prior_prob > 0:  # Only create nodes for non-zero priors
                child_hash = f"{node.position_hash}_{move}"  # Placeholder
                child = MCTSNode(
                    position_hash=child_hash,
                    parent=node,
                    move=move,
                    prior_probability=prior_prob,
                )
                node.children[move] = child
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select child with highest PUCT value.
        
        PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            node: Parent node
            
        Returns:
            Selected child node
        """
        best_score = -float('inf')
        best_child = None
        
        sqrt_parent_visits = math.sqrt(node.visit_count)
        
        for child in node.children.values():
            # Q value (exploitation)
            q_value = child.q_value
            
            # U value (exploration)
            u_value = (
                self.c_puct * 
                child.prior_probability * 
                sqrt_parent_visits / 
                (1 + child.visit_count)
            )
            
            # PUCT score
            puct_score = q_value + u_value
            
            if puct_score > best_score:
                best_score = puct_score
                best_child = child
        
        return best_child
    
    def _backup(self, search_path: List[MCTSNode], value: float):
        """
        Backup value through the search path.
        
        Args:
            search_path: List of nodes from root to leaf
            value: Value to backup
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value
            # Negate value for opponent's perspective
            value = -value
    
    def _add_dirichlet_noise(self, node: MCTSNode):
        """
        Add Dirichlet noise to root node priors for exploration.
        
        Args:
            node: Root node
        """
        if len(node.children) == 0:
            return
        
        # Generate Dirichlet noise
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(node.children)
        )
        
        # Mix noise with priors
        for i, child in enumerate(node.children.values()):
            original_prior = child.prior_probability
            child.prior_probability = (
                (1 - self.dirichlet_epsilon) * original_prior +
                self.dirichlet_epsilon * noise[i]
            )
    
    def _get_move_probabilities(self, node: MCTSNode) -> Dict[str, float]:
        """
        Get move probabilities based on visit counts.
        
        Args:
            node: Node to extract probabilities from
            
        Returns:
            Dictionary mapping moves to probabilities
        """
        if len(node.children) == 0:
            return {}
        
        # Get visit counts
        visits = {
            move: child.visit_count 
            for move, child in node.children.items()
        }
        
        # Apply temperature
        if self.temperature == 0:
            # Deterministic: pick most visited
            max_visits = max(visits.values())
            probs = {
                move: 1.0 if count == max_visits else 0.0
                for move, count in visits.items()
            }
        else:
            # Temperature-scaled visit counts
            visit_counts = np.array(list(visits.values()))
            if self.temperature == 1.0:
                scaled = visit_counts
            else:
                scaled = visit_counts ** (1.0 / self.temperature)
            
            # Normalize
            total = scaled.sum()
            if total > 0:
                normalized = scaled / total
            else:
                normalized = np.ones_like(scaled) / len(scaled)
            
            probs = {
                move: float(normalized[i])
                for i, move in enumerate(visits.keys())
            }
        
        return probs
    
    def get_principal_variation(self, depth: int = 10) -> List[str]:
        """
        Get the principal variation (most visited path).
        
        Args:
            depth: Maximum depth to traverse
            
        Returns:
            List of moves in the PV
        """
        pv = []
        node = self.root
        
        for _ in range(depth):
            if not node.children:
                break
            
            # Find most visited child
            best_child = max(
                node.children.values(),
                key=lambda c: c.visit_count
            )
            
            pv.append(best_child.move)
            node = best_child
        
        return pv
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MCTS statistics."""
        if self.root is None:
            return {}
        
        return {
            "root_visits": self.root.visit_count,
            "root_value": self.root.q_value,
            "num_children": len(self.root.children),
            "principal_variation": self.get_principal_variation(5),
            "top_moves": sorted(
                [
                    {
                        "move": child.move,
                        "visits": child.visit_count,
                        "q_value": child.q_value,
                        "prior": child.prior_probability,
                    }
                    for child in self.root.children.values()
                ],
                key=lambda x: -x["visits"]
            )[:5],
        }
