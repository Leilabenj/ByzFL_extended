import torch
import numpy as np
import threading
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque

from byzfl.fed_framework import ModelBaseInterface, RobustAggregator
from byzfl.utils.conversion import flatten_dict


class Node(ModelBaseInterface):

    def __init__(self, params):
        # Check for correct types and values in params
        #FROM CLIENT CLASS
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be of type dict, but got {type(params).__name__}")
        if not isinstance(params["loss_name"], str):
            raise TypeError(f"'loss_name' must be of type str, but got {type(params['loss_name']).__name__}")
        if not isinstance(params["nb_labels"], int) or not params["nb_labels"] > 1:
            raise ValueError(f"'nb_labels' must be an integer greater than 1")
        if not isinstance(params["momentum"], float) or not 0 <= params["momentum"] < 1:
            raise ValueError(f"'momentum' must be a float in the range [0, 1)")
        if not isinstance(params["training_dataloader"], torch.utils.data.DataLoader):
            raise TypeError(f"'training_dataloader' must be a DataLoader, but got {type(params['training_dataloader']).__name__}")
        
        #FROM SERVER CLASS
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be of type dict, but got {type(params).__name__}")
        if not isinstance(params["test_loader"], torch.utils.data.DataLoader):
            raise TypeError(f"'test_loader' must be a DataLoader, but got {type(params['test_loader']).__name__}")

        # Initialize Client instance
        super().__init__({
            # Required parameters
            "model_name": params["model_name"],
            "device": params["device"],
            # Optional parameters
            "learning_rate": params.get("learning_rate", None),
            "weight_decay": params.get("weight_decay", None),
            "milestones": params.get("milestones", None),
            "learning_rate_decay": params.get("learning_rate_decay", None),
            "optimizer_name": params.get("optimizer_name", None),
            "optimizer_params": params.get("optimizer_params", {}),
        })

        #FROM CLIENT CLASS

        self.criterion = getattr(torch.nn, params["loss_name"])()
        self.gradient_LF = 0
        self.labelflipping = params["LabelFlipping"]
        self.nb_labels = params["nb_labels"]
        self.momentum = params["momentum"]
        # Initialize momentum gradient as torch.Tensor (matching Client class)
        self.momentum_gradient = torch.zeros_like(
            torch.cat(tuple(
                tensor.view(-1) 
                for tensor in self.model.parameters()
            ))
        )
        self.training_dataloader = params["training_dataloader"]
        self.train_iterator = iter(self.training_dataloader)
        self.store_per_client_metrics = params["store_per_client_metrics"]
        self.loss_list = list()
        self.train_acc_list = list()

        #FROM SERVER CLASS
        self.robust_aggregator = RobustAggregator(params["aggregator_info"], params["pre_agg_list"])
        self.test_loader = params["test_loader"]
        self.validation_loader = params.get("validation_loader")
        if self.validation_loader is not None:
            if not isinstance(params["validation_loader"], torch.utils.data.DataLoader):
                raise TypeError(f"'validation_loader' must be a DataLoader, but got {type(params['validation_loader']).__name__}")

        # DECENTRALIZED COMMUNICATION PARAMETERS
        self.node_id = params["node_id"]
        self.neighbors = params["neighbors"]  # List of neighbor node IDs
        self.mixing_row = params["mixing_row"]  # List of (node_id, weight) tuples
        self.degree = params["degree"]
        
        # Node registry for communication (dict mapping node_id -> Node instance)
        # This allows nodes to communicate with their neighbors
        self.node_registry = params.get("node_registry", None)
        
        # Communication state
        self.message_queue = deque()  # Incoming messages
        self.message_lock = threading.Lock()
        self.round_number = 0
        self.convergence_threshold = params.get("convergence_threshold", 1e-6)
        self.max_rounds = params.get("max_rounds", 100)
        
        # Gossip state
        self.current_parameters = None
        self.previous_parameters = None
        self.converged = False
        
        # Initialize with current model parameters as PyTorch tensor
        flat_params = self.get_flat_parameters()
        if isinstance(flat_params, torch.Tensor):
            self.current_parameters = flat_params.detach().clone()
        else:
            # Convert to tensor if somehow not already a tensor
            self.current_parameters = torch.tensor(flat_params, dtype=torch.float32)


  # ==================== METHODS FROM CLIENT AND SERVER ====================

    def _sample_train_batch(self):
        """
        Description
        -----------
        Retrieves the next batch of data from the training dataloader. If the 
        end of the dataset is reached, the dataloader is reinitialized to start 
        from the beginning.

        Returns
        -------
        tuple
            A tuple containing the input data and corresponding target labels for the current batch.
        """
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.training_dataloader)
            return next(self.train_iterator)

    def compute_gradients(self):
        """
        Description
        -----------
        Computes the gradients of the local model's loss function for the 
        current training batch. If the `LabelFlipping` attack is enabled, 
        gradients for flipped targets are computed and stored separately. 
        Additionally, the training loss and accuracy for the batch are 
        computed and recorded.
        """
        inputs, targets = self._sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        if self.labelflipping:
            self.model.eval()
            targets_flipped = targets.sub(self.nb_labels - 1).mul(-1)
            self._backward_pass(inputs, targets_flipped)
            self.gradient_LF = self.get_dict_gradients()
            self.model.train()

        train_loss_value = self._backward_pass(inputs, targets, train_acc=self.store_per_client_metrics)

        if self.store_per_client_metrics:
            self.loss_list.append(train_loss_value)

        return train_loss_value

    def _backward_pass(self, inputs, targets, train_acc=False):
        """
        Description
        -----------
        Performs a backward pass through the model to compute gradients for 
        the given inputs and targets. Optionally computes training accuracy 
        for the batch.

        Parameters
        ----------
        inputs : torch.Tensor
            The input data for the batch.
        targets : torch.Tensor
            The target labels for the batch.
        train_acc : bool, optional
            If True, computes and stores the training accuracy for the batch. 
            Default is False.

        Returns
        -------
        float
            The loss value for the current batch.
        """
        self.optimizer.zero_grad()  # Zero optimizer state, not just gradients
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss_value = loss.item()
        loss.backward()

        if train_acc:
            # Compute and store train accuracy
            _, predicted = torch.max(outputs.data, 1)
            total = targets.size(0)
            correct = (predicted == targets).sum().item()
            acc = correct / total
            self.train_acc_list.append(acc)

        return loss_value

            

    def get_flat_gradients_with_momentum(self):
        """
        Description
        -----------
        Computes the gradients with momentum applied and returns them as a 
        flat array.

        Returns
        -------
        torch.Tensor
            A flat array containing the gradients with momentum applied.
        """
        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(
            self.get_flat_gradients(),
            alpha=1 - self.momentum
        )
        return self.momentum_gradient

    def get_loss_list(self):
        """
        Description
        -----------
        Retrieves the list of training losses recorded over the course of 
        training.

        Returns
        -------
        list
            A list of float values representing the training losses for each 
            batch.
        """
        return self.loss_list

    def get_train_accuracy(self):
        """
        Description
        -----------
        Retrieves the training accuracy for each batch processed during 
        training.

        Returns
        -------
        list
            A list of float values representing the training accuracy for each 
            batch.
        """
        return self.train_acc_list

    def set_model_state(self, state_dict):
        """
        Description
        -----------
        Updates the state of the model with the provided state dictionary. 
        This method is used to load a saved model state or update 
        the global model in a federated learning context.
        Typically, this method can be used to synchronize clients with the global model.

        Parameters
        ----------
        state_dict : dict
            The state dictionary containing model parameters and buffers.

        Raises
        ------
        TypeError
            If `state_dict` is not a dictionary.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"'state_dict' must be of type dict, but got {type(state_dict).__name__}")
        self.model.load_state_dict(state_dict)

    # ==================== DECENTRALIZED METHODS FOR LOCAL UPDATES AND AGGREGATION ====================

    def request_gradient(self, node_id: int) -> torch.Tensor:
        """
        Request gradient from a specific neighbor node.
        
        Parameters
        ----------
        node_id : int
            The ID of the node from which to request the gradient
            
        Returns
        -------
        torch.Tensor
            The flat gradient vector with momentum from the requested node
            
        Raises
        ------
        ValueError
            If node_id is not a neighbor of this node
        RuntimeError
            If node_registry is not set or node is not found
        """
        # Validate that node_id is a neighbor
        if node_id not in self.neighbors:
            raise ValueError(
                f"Node {node_id} is not a neighbor of node {self.node_id}. "
                f"Current neighbors: {self.neighbors}"
            )
        
        # Check if node_registry is available
        if self.node_registry is None:
            raise RuntimeError(
                "node_registry is not set. Cannot request gradients from neighbors. "
                "Set node_registry in Node initialization (e.g., from NetworkManager)."
            )
        
        # Get the neighbor node from registry
        if node_id not in self.node_registry:
            raise RuntimeError(f"Node {node_id} not found in node registry.")
        
        neighbor_node = self.node_registry[node_id]
        
        # Request and return the gradient with momentum
        return neighbor_node.get_flat_gradients_with_momentum()
    
    def request_gradient_from_neighbors(self) -> List[torch.Tensor]:
        """
        Request gradients from all neighbors.
        
        Returns a list of gradients in the order of self.neighbors.
        Includes self gradient as the first element.
        
        Returns
        -------
        List[torch.Tensor]
            List of gradient vectors: [self_gradient, neighbor1_gradient, neighbor2_gradient, ...]
            The order matches: [self.node_id] + self.neighbors
        """
        neighbor_gradients = []
        
        # Include self gradient first
        self_gradient = self.get_flat_gradients_with_momentum()
        neighbor_gradients.append(self_gradient)
        
        # Request gradients from each neighbor in order
        for neighbor_id in self.neighbors:
            neighbor_gradient = self.request_gradient(neighbor_id)
            neighbor_gradients.append(neighbor_gradient)
        
        return neighbor_gradients
    
    def gossip_aggregate(self, neighbor_parameters: List) -> torch.Tensor:
        """
        Perform gossip-based aggregation using the mixing matrix weights.
        
        Vectorized implementation: W_ni @ params_ni where ni = {self, neighbors}
        Fully tensor-based for efficiency and consistency.
        
        Parameters
        ----------
        neighbor_parameters : list
            List of parameter vectors from neighbors (including self)
            Order: [self_params, neighbor1_params, neighbor2_params, ...]
            
        Returns
        -------
        torch.Tensor
            Aggregated parameter vector
        """
        if not neighbor_parameters:
            return self.current_parameters
        
        # Create mixing weights dictionary for easy lookup
        mixing_weights = dict(self.mixing_row)
        
        # Convert all parameters to PyTorch tensors for consistent operations
        tensor_params = []
        for params in neighbor_parameters:
            if isinstance(params, torch.Tensor):
                tensor_params.append(params.detach().clone())
            elif isinstance(params, np.ndarray):
                tensor_params.append(torch.from_numpy(params).float())
            else:
                # Convert other types to tensor
                tensor_params.append(torch.tensor(params, dtype=torch.float32))
        
        # Ensure all tensors are on the same device
        device = tensor_params[0].device if len(tensor_params) > 0 else self.device
        tensor_params = [p.to(device) for p in tensor_params]
        
        # Build weight vector W_ni matching the order of neighbor_parameters
        # neighbor_parameters order: [self, neighbor1, neighbor2, ...]
        num_nodes_ni = len(tensor_params)
        W_ni = torch.zeros(num_nodes_ni, device=device, dtype=torch.float32)
        
        # Extract weights in the same order as neighbor_parameters
        for i in range(num_nodes_ni):
            if i == 0:  # Self
                node_id = self.node_id
            else:  # Neighbor i-1
                node_id = self.neighbors[i-1] if i-1 < len(self.neighbors) else self.node_id
            W_ni[i] = mixing_weights.get(node_id, 0.0)
        
        # Stack parameters into matrix: shape (num_nodes_ni, param_dim)
        # Each row is a parameter vector from one node
        params_matrix = torch.stack(tensor_params, dim=0)  # Shape: (num_nodes_ni, param_dim)
        
        # Vectorized aggregation: W_ni @ params_matrix
        # This computes sum_i(W_ni[i] * params_matrix[i, :])
        aggregated = torch.matmul(W_ni, params_matrix)  # or W_ni @ params_matrix
        
        return aggregated
    

    
    def update_model_with_gradients(self, gradients):
        """
        Update model using aggregated gradients after local epoch completion.
        In decentralized setting, each node manages its own optimizer and scheduler.
        The scheduler steps after each local epoch update (compute grad + aggregate + update).
        
        Parameters
        ----------
        gradients : list or np.ndarray or torch.Tensor
            Aggregated gradients to apply.
        """
        # Extract the single aggregated gradient (should be tensor from gossip_aggregate)
        if isinstance(gradients, list) and len(gradients) == 1:
            aggregate_gradient = gradients[0]
        else:
            raise ValueError("Expected single aggregated gradient")
        
        # Ensure it's a PyTorch tensor (should already be from gossip_aggregate)
        if not isinstance(aggregate_gradient, torch.Tensor):
            if isinstance(aggregate_gradient, np.ndarray):
                aggregate_gradient = torch.from_numpy(aggregate_gradient).float()
            else:
                aggregate_gradient = torch.tensor(aggregate_gradient, dtype=torch.float32)
        
        # Ensure tensor is on correct device
        aggregate_gradient = aggregate_gradient.to(self.device)
        
        # Set aggregated gradients and step (this advances both optimizer and scheduler)
        self.set_gradients(aggregate_gradient)
        self._step()
    
    def update_model_with_weights(self, weights):
        """
        Update model using aggregated weights.
        This follows the ByzFL Server pattern.
        """
        aggregate_weights = self.aggregate(weights)
        self.set_parameters(aggregate_weights)
    
    def _step(self):
        """
        Execute a single optimization step for the model.
        In decentralized setting, each node manages its own optimizer and scheduler.
        The scheduler steps after each local epoch update.
        """
        self.optimizer.step()
        self.scheduler.step()
    
    def _compute_accuracy(self, data_loader):
        """
        Compute accuracy on a given dataset.
        This follows the ByzFL Server pattern.
        """
        total = 0
        correct = 0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        return correct / total
    
    def compute_test_accuracy(self):
        """
        Compute test accuracy using the test_loader.
        This follows the ByzFL Server pattern.
        """
        return self._compute_accuracy(self.test_loader)
    
    def _check_convergence(self) -> None:
        """
        Check if the node has converged based on parameter changes.
        """
        if self.previous_parameters is not None:
            # Ensure both are tensors for consistent operations
            current = self.current_parameters if isinstance(self.current_parameters, torch.Tensor) else torch.tensor(self.current_parameters, dtype=torch.float32)
            previous = self.previous_parameters if isinstance(self.previous_parameters, torch.Tensor) else torch.tensor(self.previous_parameters, dtype=torch.float32)
            
            # Compute norm using torch
            param_diff = torch.norm(current - previous).item()
            self.converged = param_diff < self.convergence_threshold
        
        # Update previous parameters (ensure it's a tensor)
        if isinstance(self.current_parameters, torch.Tensor):
            self.previous_parameters = self.current_parameters.clone()
        else:
            self.previous_parameters = torch.tensor(self.current_parameters, dtype=torch.float32)
    
    def is_converged(self) -> bool:
        """
        Check if the node has converged.
        
        Returns
        -------
        bool
            True if converged, False otherwise
        """
        return self.converged or self.round_number >= self.max_rounds
    
    def get_node_info(self) -> Dict[str, Any]:
        """
        Get information about the current node state.
        
        Returns
        -------
        dict
            Node information including ID, neighbors, current round, etc.
        """
        return {
            "node_id": self.node_id,
            "neighbors": self.neighbors,
            "degree": self.degree,
            "round_number": self.round_number,
            "converged": self.converged,
            "parameter_norm": torch.norm(self.current_parameters).item() if self.current_parameters is not None else 0.0
        }
    


