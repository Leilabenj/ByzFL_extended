"""
Network Manager for Decentralized Learning

This module provides a simple network manager to coordinate decentralized learning
across multiple nodes. It handles message routing and synchronization for the
first iteration of the decentralized framework.
"""

import torch
import time
from typing import Dict, List, Any, Optional
from collections import defaultdict

from node import Node
from graph import build_graph_G, build_metropolis_W, topology_view


class NetworkManager:
    """
    Manages a network of decentralized learning nodes.
    
    This class coordinates communication between nodes and provides
    a centralized view of the network state for testing and monitoring.
    """
    
    def __init__(self, graph, mixing_matrix, nodes: List[Node]):
        """
        Initialize the network manager.
        
        Parameters
        ----------
        graph : networkx.Graph
            The network topology graph
        mixing_matrix : scipy.sparse.csr_matrix
            The mixing matrix for gossip aggregation
        nodes : list
            List of Node instances
        """
        self.graph = graph
        self.mixing_matrix = mixing_matrix
        self.nodes = {node.node_id: node for node in nodes}
        self.message_buffer = defaultdict(list)  # Buffer for inter-node messages
        
        # Set node_registry for each node to enable neighbor communication
        for node in nodes:
            node.node_registry = self.nodes
        
        # Network state
        self.current_round = 0
        self.global_converged = False
        

    
    def perform_decentralized_round(self) -> Dict[str, Any]:
        """
        Perform one round of decentralized learning following ByzFL pattern.
        
        Returns
        -------
        dict
            Round results including convergence status and metrics
        """
        round_results = {}
        
        # Step 1: All nodes compute gradients (ByzFL Client pattern)
        for node_id, node in self.nodes.items():
            # Compute gradients on local data - this should use each node's unique dataloader
            node.compute_gradients()
            
            # DEBUG: Verify each node has different local loss (indicates different data)
            local_loss = node.get_loss_list()[-1] if node.get_loss_list() else 0.0
            round_results[node_id] = {
                "local_loss": local_loss
            }
        
        # Step 2: Each node performs gossip aggregation on gradients using direct neighbor communication
        for node_id, node in self.nodes.items():
            # Request gradients from neighbors using node communication methods
            neighbor_gradients = node.request_gradient_from_neighbors()
            
            # Perform gossip aggregation on gradients
            if neighbor_gradients:
                aggregated_gradients = node.gossip_aggregate(neighbor_gradients)
                
                # Update model with aggregated gradients after local epoch
                # Each node steps its own scheduler (decentralized approach)
                node.update_model_with_gradients([aggregated_gradients])
                
                # Update current parameters (keep as tensor)
                node.current_parameters = node.get_flat_parameters().detach().clone()
            
            # Check convergence
            node._check_convergence()
            
            round_results[node_id].update({
                "converged": node.converged,
                "parameter_norm": torch.norm(node.current_parameters).item() if node.current_parameters is not None else 0.0
            })
        
        self.current_round += 1
        
        # Check global convergence
        self.global_converged = all(node.converged for node in self.nodes.values())
        
        return {
            "round": self.current_round,
            "global_converged": self.global_converged,
            "node_results": round_results
        }
    
    
    def get_network_state(self) -> Dict[str, Any]:
        """
        Get the current state of the entire network.
        
        Returns
        -------
        dict
            Network state information
        """
        node_states = {}
        for node_id, node in self.nodes.items():
            node_states[node_id] = node.get_node_info()
        
        return {
            "current_round": self.current_round,
            "global_converged": self.global_converged,
            "num_nodes": len(self.nodes),
            "node_states": node_states
        }
    
    def debug_dataloaders(self) -> None:
        """
        Print debug information about each node's dataloader to verify local data distribution.
        """
        print("\n" + "="*60)
        print("DATALOADER DEBUG: Checking if nodes have different local data")
        print("="*60)
        for node_id, node in sorted(self.nodes.items()):
            info = node.get_dataloader_info()
            print(f"\nNode {node_id}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        print("="*60 + "\n")



