# Extending ByzFL for Fully Decentralized Learning

## **PERSONAL PROGRESS TRACKER**

**Current Status**: First Iteration en cours : Commencer avec class Node (client + server). Chaque node a son propre model, optimizer et scheduler. Peut aggregate localement apres avoir recu neighbors_grad (gossip_aggr). Network_manager agit encore comme un 'central-like' server qui a acces a tous les grad et les attribue aux voisins. 

Tout semble bien fonctionner (loss decreases, acc increases)
- **A checker** : histoire de momentum..??


- **Next Steps**: Implement local communication between nodes (replace networkmanager)
- **Key Achievements**: 
  - Decentralized Node class with ByzFL compatibility
  - Graph-based topology with Metropolis-Hastings mixing
  - Gossip-based aggregation implemented
  - Tensor/Numpy consistency issues resolved



Iteration 2 : Still no byzclients. First ensure communication between nodes is efficient and working.

---

This project enhances the ByzFL library by enabling fully decentralized federated learning without a central server. It implements peer-to-peer communication, gossip-based aggregation, and robust consensus mechanisms to ensure resilience against Byzantine failures.

## First Iteration: Core Communication & Gossip-Based Aggregation

The first iteration focuses on establishing the fundamental decentralized learning mechanisms while maintaining compatibility with ByzFL's robust aggregation methods. Note: This iteration does not handle Byzantine nodes yet or communication between nodes - it focuses purely on establishing the class Node (client+server) and gossip-based aggregation among honest nodes.

### Key Features Implemented

#### 1. **Decentralized Node Architecture**
- **Hybrid Node Design**: Each node combines ByzFL's `Client` and `Server` functionality
- **ByzFL Compatibility**: Inherits from `ModelBaseInterface` and uses ByzFL's training methods
- **Local Training**: Nodes perform local training using `compute_gradients()` and `get_flat_gradients()`
- **Gossip Aggregation**: Nodes aggregate gradients using `update_model_with_gradients()`

#### 2. **Graph-Based Topology**
- **Random Regular Graphs**: Uses NetworkX to generate k-regular graph topologies
- **Metropolis-Hastings Mixing Matrix**: Ensures convergence to consensus through doubly-stochastic weights
- **Spectral Analysis**: Provides network diagnostics including spectral gap and convergence properties
- **Topology Views**: Each node maintains a local view of its neighborhood

#### 3. **Core Communication Layer**
- Not present --> NetworkManager

#### 4. **Gossip-Based Aggregation**
- **Mixing Matrix Integration**: Uses Metropolis-Hastings weights for gradient aggregation
- **Local Robust Aggregation**: Each node aggregates using weights from W (mixing weights matrix)
- **ByzFL Pattern Compliance**: Follows the exact ByzFL workflow: `compute_gradients()` → `get_flat_gradients()` → `update_model_with_gradients()`



### Learning Process Flow

1. **Initialization**: All nodes start with identical models from node 0
2. **Local Training**: Each node computes gradients on its local data
3. **Gradient Collection**: Network manager collects gradients from all nodes
4. **Gossip Aggregation**: Each node aggregates gradients from its neighbors using mixing weights
5. **Model Update**: Nodes update their models with aggregated gradients
6. **Convergence Check**: Monitor parameter changes for convergence
7. **Repeat**: Continue until convergence or maximum rounds

### Usage Example

```python
# Import decentralized components
from byzfl.decentralized_framework import Node, NetworkManager
from byzfl.decentralized_framework.graph import build_graph_G, build_metropolis_W
from byzfl import DataDistributor

# Setup decentralized learning
nb_nodes = 6
graph_degree = 3
nb_training_steps = 100

# Create network topology
graph = build_graph_G(graph_degree, nb_nodes)
mixing_matrix = build_metropolis_W(graph)

# Distribute MNIST data among nodes
data_distributor = DataDistributor({
    "data_distribution_name": "dirichlet_niid",
    "distribution_parameter": 0.5,
    "nb_honest": nb_nodes,
    "data_loader": train_loader,
    "batch_size": 32,
})
node_dataloaders = data_distributor.split_data()

# Create decentralized nodes
decentralized_nodes = []
for i in range(nb_nodes):
    node_params = {
        "model_name": "cnn_mnist",
        "device": "cpu",
        "optimizer_name": "SGD",
        "learning_rate": 0.01,
        "loss_name": "NLLLoss",
        "training_dataloader": node_dataloaders[i],
        "momentum": 0.9,
        "nb_labels": 10,
        "node_id": i,
        "neighbors": [],  # Set by network manager
        "mixing_row": [],  # Set by network manager
        "degree": 0,  # Set by network manager
        "aggregator_info": {"name": "Average", "parameters": {}},
        "pre_agg_list": [],
    }
    node = Node(node_params)
    decentralized_nodes.append(node)

# Create network manager
network_manager = NetworkManager(graph, mixing_matrix, decentralized_nodes)

# Initialize shared model
initial_model = decentralized_nodes[0].get_dict_parameters()
for node in decentralized_nodes:
    node.set_model_state(initial_model)

# Run decentralized learning
for training_step in range(nb_training_steps):
    round_result = network_manager.perform_decentralized_round()
    
    if training_step % 20 == 0:
        avg_acc = np.mean([node.compute_test_accuracy() for node in decentralized_nodes])
        print(f"Step {training_step}: Average Accuracy = {avg_acc:.4f}")
    
    if round_result['global_converged']:
        print(f"Converged after {training_step + 1} steps!")
        break
```

### Current Limitations (First Iteration)

1. **No Byzantine Node Handling**: This iteration assumes all nodes are honest - Byzantine attacks are not implemented yet
2. **Centralized Coordination**: Network manager acts as a central coordinator with global knowledge
3. **No True P2P Communication**: Message passing methods are placeholders
4. **Synchronous Learning**: All nodes must be synchronized
5. **Simple Aggregation**: Uses basic averaging instead of ByzFL's robust aggregators
6. **Static Topology**: Network structure doesn't change during learning

### Future Iterations Roadmap

- **Iteration 2**: True peer-to-peer communication (and asynchronous learning?)
- **Iteration 3**: Byzantine robustness integration with ByzFL's robust aggregators

### Files Structure

```
byzfl/decentralized_framework/
├── node.py              # Decentralized node implementation
├── network_manager.py    # Network coordination and management
├── graph.py             # Graph topology and mixing matrix utilities
└── test.ipynb          # Complete example with MNIST dataset (workflow)
```

