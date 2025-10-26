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
        # Initialize momentum gradient as numpy array for consistency
        flat_params = torch.cat(tuple(
            tensor.view(-1) 
            for tensor in self.model.parameters()
        ))
        self.momentum_gradient = np.zeros_like(flat_params.detach().cpu().numpy())
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
        
        # Initialize with current model parameters as numpy array
        flat_params = self.get_flat_parameters()
        if hasattr(flat_params, 'numpy'):  # PyTorch tensor
            self.current_parameters = flat_params.detach().cpu().numpy()
        elif hasattr(flat_params, 'cpu'):  # PyTorch tensor on CPU
            self.current_parameters = flat_params.detach().numpy()
        else:  # Already numpy array
            self.current_parameters = np.array(flat_params)

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
        self.model.zero_grad()
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
    
    def compute_model_update(self, num_rounds):
        """
        Description
        -----------
        Executes multiple rounds of training updates on the model. For each round,
        it samples a batch of training data, performs a backward pass to compute
        gradients, and updates the model parameters. Optionally logs training loss
        and accuracy.

        Parameters
        ----------
        num_rounds : int
            The number of training iterations to perform. Each iteration includes
            sampling a batch, computing the loss and gradients, and updating the model.

        Returns
        -------
        float
            The mean loss across all training rounds.
        """

        losses = np.zeros((num_rounds))
        for i in range(num_rounds):
            inputs, targets = self._sample_train_batch()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            train_loss_value = self._backward_pass(inputs, targets, train_acc=self.store_per_client_metrics)
            losses[i] = train_loss_value
            self.optimizer.step()

            if self.store_per_client_metrics:
                self.loss_list.append(train_loss_value)

        return losses.mean()
            

    def get_flat_flipped_gradients(self):
        """
        Description
        -----------
        Retrieves the gradients computed using flipped targets as a flat array.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            A flat array containing the gradients for the model parameters 
            when trained with flipped targets.
        """
        return flatten_dict(self.gradient_LF)

    def get_flat_gradients_with_momentum(self):
        """
        Description
        -----------
        Computes the gradients with momentum applied and returns them as a 
        flat numpy array.

        Returns
        -------
        np.ndarray
            A flat array containing the gradients with momentum applied.
        """
        # Get current gradients as numpy array
        current_gradients = self.get_flat_gradients()
        if hasattr(current_gradients, 'numpy'):  # PyTorch tensor
            current_gradients = current_gradients.detach().cpu().numpy()
        elif hasattr(current_gradients, 'cpu'):  # PyTorch tensor on CPU
            current_gradients = current_gradients.detach().numpy()
        
        # Apply momentum: momentum_gradient = momentum * momentum_gradient + (1-momentum) * current_gradients
        self.momentum_gradient = self.momentum * self.momentum_gradient + (1 - self.momentum) * current_gradients
        
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

    # ==================== DECENTRALIZED COMMUNICATION METHODS ====================
    
    def send_message(self, target_node_id: int, message: Dict[str, Any]) -> None:
        """
        Send a message to a specific neighbor node.
        
        Parameters
        ----------
        target_node_id : int
            ID of the target neighbor node
        message : dict
            Message containing parameters, round number, etc.
        """
        if target_node_id not in self.neighbors:
            raise ValueError(f"Node {target_node_id} is not a neighbor of node {self.node_id}")
        
        # In a real implementation, this would send over network
        # For now, we'll simulate by directly calling receive_message on target
        # This will be replaced with actual network communication
        pass
    
    def receive_message(self, message: Dict[str, Any]) -> None:
        """
        Receive a message from a neighbor node.
        
        Parameters
        ----------
        message : dict
            Received message containing parameters and metadata
        """
        with self.message_lock:
            self.message_queue.append(message)
    
    def get_pending_messages(self) -> List[Dict[str, Any]]:
        """
        Get all pending messages from the message queue.
        
        Returns
        -------
        list
            List of pending messages
        """
        with self.message_lock:
            messages = list(self.message_queue)
            self.message_queue.clear()
            return messages
    
    def broadcast_parameters(self, round_number: int) -> None:
        """
        Broadcast current model parameters to all neighbors.
        
        Parameters
        ----------
        round_number : int
            Current round number for synchronization
        """
        message = {
            "sender_id": self.node_id,
            "round_number": round_number,
            "parameters": self.current_parameters.copy(),
            "timestamp": time.time()
        }
        
        for neighbor_id in self.neighbors:
            self.send_message(neighbor_id, message)
    
    def gossip_aggregate(self, neighbor_parameters: List) -> np.ndarray:
        """
        Perform gossip-based aggregation using the mixing matrix weights.
        
        Parameters
        ----------
        neighbor_parameters : list
            List of parameter vectors from neighbors (including self)
            
        Returns
        -------
        np.ndarray
            Aggregated parameter vector
        """
        if not neighbor_parameters:
            return self.current_parameters
        
        # Create mixing weights dictionary for easy lookup
        mixing_weights = dict(self.mixing_row)
        
        # Convert all parameters to numpy arrays for consistent operations
        np_params = []
        for params in neighbor_parameters:
            if hasattr(params, 'numpy'):  # PyTorch tensor
                np_params.append(params.detach().cpu().numpy())
            elif hasattr(params, 'cpu'):  # PyTorch tensor on CPU
                np_params.append(params.detach().numpy())
            elif isinstance(params, np.ndarray):  # Already numpy array
                np_params.append(params)
            else:  # Convert other types to numpy array
                np_params.append(np.array(params))
        
        # Initialize aggregated parameters as numpy array
        aggregated = np.zeros_like(np_params[0])
        
        # Weighted average using mixing matrix
        for i, params in enumerate(np_params):
            # For simplicity, assume neighbor_parameters includes self at index 0
            # In practice, you'd need to map neighbor IDs to their parameter indices
            if i == 0:  # Self parameters
                weight = mixing_weights.get(self.node_id, 0.0)
            else:
                # Map to actual neighbor IDs (this is simplified)
                neighbor_id = self.neighbors[i-1] if i-1 < len(self.neighbors) else self.node_id
                weight = mixing_weights.get(neighbor_id, 0.0)
            
            aggregated = aggregated + weight * params
        
        return aggregated
    
    def perform_local_training(self, num_local_epochs: int = 1) -> float:
        """
        Perform local training on the node's data.
        
        Parameters
        ----------
        num_local_epochs : int
            Number of local training epochs
            
        Returns
        -------
        float
            Average training loss
        """
        return self.compute_model_update(num_local_epochs)
    
    def compute_gradients(self):
        """
        Compute gradients for the current training batch.
        This follows the ByzFL Client pattern.
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
    
    
    def aggregate(self, vectors):
        """
        Aggregate input vectors using the configured robust aggregator.
        This follows the ByzFL Server pattern.
        
        Parameters
        ----------
        vectors : list or np.ndarray or torch.Tensor
            A collection of input vectors to be aggregated.
            
        Returns
        -------
        numpy.ndarray or torch.Tensor
            Aggregated output vector, with the same type as the input vectors.
        """
        return self.robust_aggregator.aggregate_vectors(vectors)
    
    def update_model_with_gradients(self, gradients):
        """
        Update model using aggregated gradients.
        This follows the ByzFL Server pattern.
        """
        # Extract the single aggregated gradient (should be numpy array from gossip_aggregate)
        if isinstance(gradients, list) and len(gradients) == 1:
            aggregate_gradient = gradients[0]
        else:
            raise ValueError("Expected single aggregated gradient")
        
        # Ensure it's a numpy array (should already be from gossip_aggregate)
        if not isinstance(aggregate_gradient, np.ndarray):
            if hasattr(aggregate_gradient, 'numpy'):  # PyTorch tensor
                aggregate_gradient = aggregate_gradient.detach().cpu().numpy()
            elif hasattr(aggregate_gradient, 'cpu'):  # PyTorch tensor on CPU
                aggregate_gradient = aggregate_gradient.detach().numpy()
            else:
                aggregate_gradient = np.array(aggregate_gradient)
        
        # Convert numpy array to PyTorch tensor for set_gradients
        aggregate_gradient_tensor = torch.from_numpy(aggregate_gradient).float()
        
        # Set gradients and update model
        self.set_gradients(aggregate_gradient_tensor)
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
        This follows the ByzFL Server pattern.
        """
        self.optimizer.step()
        self.scheduler.step()
    
    def _check_convergence(self) -> None:
        """
        Check if the node has converged based on parameter changes.
        """
        if self.previous_parameters is not None:
            # Ensure both are numpy arrays for consistent operations
            current = np.array(self.current_parameters) if not isinstance(self.current_parameters, np.ndarray) else self.current_parameters
            previous = np.array(self.previous_parameters) if not isinstance(self.previous_parameters, np.ndarray) else self.previous_parameters
            
            param_diff = np.linalg.norm(current - previous)
            self.converged = param_diff < self.convergence_threshold
        
        # Update previous parameters (ensure it's a numpy array)
        if isinstance(self.current_parameters, np.ndarray):
            self.previous_parameters = self.current_parameters.copy()
        else:
            self.previous_parameters = np.array(self.current_parameters)
    
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
            "parameter_norm": np.linalg.norm(self.current_parameters) if self.current_parameters is not None else 0.0
        }

