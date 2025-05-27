# src/connectome_analysis/models/gnn_models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, Any

class BrainGCN(torch.nn.Module):
    """Graph Convolutional Network for brain connectomes"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.out_channels = out_channels # Expose output channels

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x) # Add activation after second conv
        return x

class BrainGAT(torch.nn.Module):
    """Graph Attention Network for brain connectivity analysis"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1) # Output layer typically has 1 head
        self.out_channels = out_channels # Expose output channels

    def forward(self, data: Data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x) # Add activation after second conv
        return x

class ConnectomeClassifier(torch.nn.Module):
    """Wrapper for graph-level classification tasks"""
    def __init__(self, gnn_model: torch.nn.Module, num_classes: int):
        super().__init__()
        self.gnn_model = gnn_model
        # Assuming the GNN model outputs node embeddings, we need a pooling layer
        # and a final classification head.
        # The GNN model's output channels should match the input to the linear layer.
        # We'll add a placeholder linear layer here. The actual input size will depend
        # on the GNN model's output and the pooling method.
        # For now, let's assume the GNN model outputs a fixed-size representation per graph after pooling.
        # This requires the GNN model's forward method to handle pooling internally or
        # we need to add pooling here. Let's add global mean pooling here for flexibility.
        
        # Placeholder for the size of the GNN output after pooling.
        # This needs to be determined based on the actual GNN model and pooling.
        # For now, let's assume the GNN model's last layer has 'out_channels' and we use global_mean_pool.
        # The input size to the linear layer would then be 'out_channels'.
        # This requires the gnn_model to have an 'out_channels' attribute or similar.
        # Let's refine this: the GNN model should output node embeddings, and this classifier
        # will handle the pooling and final classification.
        
        # The GNN model should have an 'out_channels' attribute after initialization
        # which represents the dimension of the node embeddings it outputs.
        gnn_output_dim = int(gnn_model.out_channels) # type: ignore # Explicitly cast to int
        if gnn_output_dim is None: # This check is now redundant but kept for safety
            raise ValueError("GNN model must have an 'out_channels' attribute defined.")

        self.classifier_head = nn.Linear(gnn_output_dim, num_classes)

    def forward(self, data: Batch):
        # The GNN model processes the batch of graphs and outputs node embeddings
        x = self.gnn_model(data)

        # Apply global pooling to get a graph-level representation
        # data.batch is required for global pooling on a Batch object
        x = global_mean_pool(x, data.batch) # type: ignore

        # Final classification layer
        x = self.classifier_head(x)
        return x # Return logits, CrossEntropyLoss will handle softmax

# Example usage (requires PyTorch Geometric Data objects)
if __name__ == '__main__':
    print("GNN models module initialized.")
    print("Requires PyTorch Geometric Data objects and training pipeline for full functionality.")

    # Example of how you might use it with dummy data
    # try:
    #     # Dummy data: a simple graph with 3 nodes, 2 features per node
    #     edge_index = torch.tensor([[0, 1, 1, 2],
    #                                [1, 0, 2, 1]], dtype=torch.long)
    #     x = torch.randn(3, 2) # 3 nodes, 2 features per node
    #     data = Data(x=x, edge_index=edge_index)

    #     # Create a batch of dummy data
    #     data_list = [data, data] # Batch of 2 identical graphs
    #     batch = Batch.from_data_list(data_list)

    #     # Example GCN model
    #     in_channels = data.num_node_features
    #     hidden_channels = 16
    #     gnn_out_channels = 32
    #     num_classes = 2 # Binary classification

    #     gcn_model = BrainGCN(in_channels, hidden_channels, gnn_out_channels)
    #     classifier = ConnectomeClassifier(gcn_model, num_classes)

    #     # Forward pass
    #     out = classifier(batch)
    #     print("\nDummy GCN Classifier Output Shape:", out.shape) # Should be [batch_size, num_classes]

    #     # Example GAT model
    #     gat_model = BrainGAT(in_channels, hidden_channels, gnn_out_channels, heads=2)
    #     classifier_gat = ConnectomeClassifier(gat_model, num_classes)

    #     # Forward pass
    #     out_gat = classifier_gat(batch)
    #     print("Dummy GAT Classifier Output Shape:", out_gat.shape) # Should be [batch_size, num_classes]

# Dictionary to map GNN model names to their classes
GNN_MODEL_CLASSES = {
    'BrainGCN': BrainGCN,
    'BrainGAT': BrainGAT,
}

def create_gnn_model(model_name: str, config: Dict[str, Any]) -> torch.nn.Module:
    """
    Factory function to create a GNN model instance.

    Args:
        model_name: The name of the GNN model to create (e.g., 'BrainGCN').
        config: A dictionary containing configuration parameters for the GNN model.
                Expected keys: 'in_channels', 'hidden_channels', 'out_channels', 'num_classes'.
                For BrainGAT, 'heads' is also expected.

    Returns:
        An instance of a GNN model (e.g., BrainGCN, BrainGAT) wrapped in ConnectomeClassifier.

    Raises:
        ValueError: If an unknown model name is provided or required parameters are missing.
    """
    if model_name not in GNN_MODEL_CLASSES:
        raise ValueError(f"Unknown GNN model: {model_name}. "
                         f"Available GNN models are: {list(GNN_MODEL_CLASSES.keys())}")

    # Extract common parameters and assert their types
    in_channels: int = config.get('in_channels') # type: ignore
    hidden_channels: int = config.get('hidden_channels') # type: ignore
    out_channels: int = config.get('out_channels') # type: ignore
    num_classes: int = config.get('num_classes') # type: ignore

    # This check is still useful for runtime validation, even with type ignores
    if any(p is None for p in [in_channels, hidden_channels, out_channels, num_classes]):
        raise ValueError(f"Missing required parameters for GNN model '{model_name}'. "
                         f"Expected: 'in_channels', 'hidden_channels', 'out_channels', 'num_classes'.")

    gnn_base_model: torch.nn.Module
    if model_name == 'BrainGCN':
        gnn_base_model = BrainGCN(in_channels, hidden_channels, out_channels)
    elif model_name == 'BrainGAT':
        heads: int = config.get('heads', 1) # type: ignore # Default to 1 head if not specified
        gnn_base_model = BrainGAT(in_channels, hidden_channels, out_channels, heads)
    else:
        # This case should ideally not be reached due to the initial check
        raise ValueError(f"Unhandled GNN model type: {model_name}")

    # Wrap the GNN base model in the ConnectomeClassifier
    return ConnectomeClassifier(gnn_base_model, num_classes)

# Example usage (requires PyTorch Geometric Data objects)
if __name__ == '__main__':
    print("GNN models module initialized.")
    print("Requires PyTorch Geometric Data objects and training pipeline for full functionality.")

    # Example of how to use the factory function with dummy data
    try:
        # Dummy data: a simple graph with 3 nodes, 2 features per node
        edge_index = torch.tensor([[0, 1, 1, 2],
                                   [1, 0, 2, 1]], dtype=torch.long)
        x = torch.randn(3, 2) # 3 nodes, 2 features per node
        data = Data(x=x, edge_index=edge_index)

        # Create a batch of dummy data
        data_list = [data, data] # Batch of 2 identical graphs
        batch = Batch.from_data_list(data_list) # type: ignore

        # Example config for GCN
        gcn_config = {
            'in_channels': data.num_node_features,
            'hidden_channels': 16,
            'out_channels': 32,
            'num_classes': 2
        }
        # Create GCN model using the factory function
        gcn_classifier = create_gnn_model('BrainGCN', gcn_config)

        # Forward pass
        out_gcn = gcn_classifier(batch)
        print("\nDummy GCN Classifier Output Shape:", out_gcn.shape) # Should be [batch_size, num_classes]

        # Example config for GAT
        gat_config = {
            'in_channels': data.num_node_features,
            'hidden_channels': 16,
            'out_channels': 32,
            'heads': 2,
            'num_classes': 2
        }
        # Create GAT model using the factory function
        gat_classifier = create_gnn_model('BrainGAT', gat_config)

        # Forward pass
        out_gat = gat_classifier(batch)
        print("Dummy GAT Classifier Output Shape:", out_gat.shape) # Should be [batch_size, num_classes]

    except Exception as e:
        print(f"Error during dummy example execution: {e}")
        print("This is expected as data loading and training are not implemented here.")
