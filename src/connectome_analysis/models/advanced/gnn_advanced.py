# src/connectome_analysis/models/gnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from typing import Dict, Any

class GraphPooling(nn.Module):
    """Graph pooling mechanisms for graph-level representations"""
    def __init__(self, method: str = 'mean'):
        super().__init__()
        self.method = method

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.method == 'mean':
            return global_mean_pool(x, batch)
        elif self.method == 'max':
            return global_max_pool(x, batch)
        # Add attention pooling later if needed
        else:
            raise ValueError(f"Unsupported pooling method: {self.method}")

class BrainGCN(torch.nn.Module):
    """Graph Convolutional Network for brain connectome classification"""
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int, dropout: float, num_classes: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(in_channels, hidden_dim))
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.pooling = GraphPooling(method='mean') # Default to mean pooling
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.pooling(x, batch)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(-1) # Sigmoid for binary classification

class BrainGAT(torch.nn.Module):
    """Graph Attention Network for brain connectome classification"""
    def __init__(self, in_channels: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float, num_classes: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.conv_layers = nn.ModuleList()
        # First layer
        self.conv_layers.append(GATConv(in_channels, hidden_dim, heads=num_heads, dropout=dropout))
        # Subsequent layers
        for _ in range(num_layers - 1):
            # Note: GATConv output features are hidden_dim * num_heads
            self.conv_layers.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout))

        self.pooling = GraphPooling(method='mean') # Default to mean pooling
        # Final linear layer input size is hidden_dim * num_heads from the last GAT layer
        self.fc = nn.Linear(hidden_dim * num_heads, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
            # Dropout is already applied within GATConv

        x = self.pooling(x, batch)
        x = self.fc(x)
        return torch.sigmoid(x).squeeze(-1) # Sigmoid for binary classification

# Example usage (requires PyTorch Geometric Data object)
if __name__ == '__main__':
    print("GNN models module initialized.")
    print("Requires PyTorch Geometric Data objects and training pipeline for full functionality.")

    # Example of how you might use it with dummy data
    # try:
    #     # Dummy data: a simple graph
    #     edge_index = torch.tensor([[0, 1, 1, 2],
    #                                [1, 0, 2, 1]], dtype=torch.long)
    #     # Dummy node features (e.g., degree, centrality)
    #     x = torch.randn(3, 16) # 3 nodes, 16 features per node
    #     # Dummy batch vector
    #     batch = torch.tensor([0, 0, 0], dtype=torch.long) # All nodes belong to the same graph

    #     data = Data(x=x, edge_index=edge_index, batch=batch)

    #     # Example config for GCN
    #     gcn_config = {
    #         'in_channels': 16,
    #         'hidden_dim': 32,
    #         'num_layers': 2,
    #         'dropout': 0.5,
    #         'num_classes': 1
    #     }
    #     gcn_model = BrainGCN(**gcn_config)

    #     # Example config for GAT
    #     gat_config = {
    #         'in_channels': 16,
    #         'hidden_dim': 32,
    #         'num_layers': 2,
    #         'num_heads': 4,
    #         'dropout': 0.5,
    #         'num_classes': 1
    #     }
    #     gat_model = BrainGAT(**gat_config)

    #     # Forward pass with dummy data
    #     gcn_output = gcn_model(data)
    #     print(f"Dummy GCN output shape: {gcn_output.shape}")

    #     gat_output = gat_model(data)
    #     print(f"Dummy GAT output shape: {gat_output.shape}")

    # except Exception as e:
    #     print(f"Error during dummy example execution: {e}")
    #     print("This is expected as data loading is not implemented here.")
