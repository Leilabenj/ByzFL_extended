"""
Network Manager for Decentralized Learning

This module provides a simple network manager to coordinate decentralized learning
across multiple nodes. It handles message routing and synchronization for the
first iteration of the decentralized framework.
"""

import numpy as np
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
        
        # Network state
        self.current_round = 0
        self.global_converged = False
        
    def send_message(self, sender_id: int, target_id: int, message: Dict[str, Any]) -> None:
        """
        Send a message from one node to another.
        
        Parameters
        ----------
        sender_id : int
            ID of the sending node
        target_id : int
            ID of the target node
        message : dict
            Message content
        """
        if target_id in self.nodes:
            self.message_buffer[target_id].append(message)
    
    def deliver_messages(self) -> None:
        """
        Deliver all pending messages to their target nodes.
        """
        for target_id, messages in self.message_buffer.items():
            if target_id in self.nodes:
                for message in messages:
                    self.nodes[target_id].receive_message(message)
        self.message_buffer.clear()
    
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
            node.compute_gradients()
            round_results[node_id] = {
                "local_loss": node.get_loss_list()[-1] if node.get_loss_list() else 0.0
            }
        
        # Step 2: Collect gradients from all nodes
        all_gradients = []
        for node_id, node in self.nodes.items():
            gradients = node.get_flat_gradients_with_momentum()
            all_gradients.append(gradients)
        
        # Step 3: Each node performs gossip aggregation on gradients
        for node_id, node in self.nodes.items():
            # Get gradients from neighbors (including self)
            neighbor_gradients = []
            
            # Include self gradients
            neighbor_gradients.append(all_gradients[node_id])
            
            # Get gradients from neighbors (simplified - in practice would be from messages)
            for neighbor_id in node.neighbors:
                if neighbor_id < len(all_gradients):
                    neighbor_gradients.append(all_gradients[neighbor_id])
            
            # Perform gossip aggregation on gradients
            if neighbor_gradients:
                aggregated_gradients = node.gossip_aggregate(neighbor_gradients)
                
                # Update model with aggregated gradients (ByzFL Server pattern)
                node.update_model_with_gradients([aggregated_gradients])
                
                # Update current parameters
                node.current_parameters = node.get_flat_parameters()
            
            # Check convergence
            node._check_convergence()
            
            round_results[node_id].update({
                "converged": node.converged,
                "parameter_norm": np.linalg.norm(node.current_parameters)
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



