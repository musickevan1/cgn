# src/connectome_analysis/models/graph_utils.py

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Optional, Dict, Any

def connectivity_to_pyg_data(connectivity_matrix: np.ndarray, label: int, node_features: Optional[np.ndarray] = None) -> Data:
    """
    Convert a connectivity matrix and optional node features to a PyTorch Geometric Data object.

    Args:
        connectivity_matrix: A square numpy array representing the connectivity matrix.
        label: The graph-level label.
        node_features: Optional numpy array of node features (shape: num_nodes x num_features).

    Returns:
        A PyTorch Geometric Data object.
    """
    if connectivity_matrix.shape[0] != connectivity_matrix.shape[1]:
        raise ValueError("Connectivity matrix must be square.")

    num_nodes = connectivity_matrix.shape[0]

    # Convert connectivity matrix to edge index and edge attributes
    # We'll treat the connectivity matrix as an adjacency matrix.
    # For weighted graphs, edge_attr would be the connection strength.
    # For unweighted graphs, edge_attr could be None or a tensor of ones.
    # Let's extract edges where connection strength is non-zero.
    row, col = np.where(connectivity_matrix != 0)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(connectivity_matrix[row, col], dtype=torch.float)

    # Convert node features to torch tensor
    if node_features is not None:
        if node_features.shape[0] != num_nodes:
            raise ValueError("Number of node features must match number of nodes in connectivity matrix.")
        x = torch.tensor(node_features, dtype=torch.float)
    else:
        # If no node features are provided, use a placeholder (e.g., identity matrix or ones)
        # Using ones as a simple placeholder
        x = torch.ones((num_nodes, 1), dtype=torch.float) # Placeholder: 1 feature per node

    # Convert label to torch tensor
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

class BrainGraphDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset for brain graph data.
    """
    def __init__(self, connectomes: List[np.ndarray], labels: List[int], atlas_info: Optional[Dict[str, Any]] = None, transform=None, pre_transform=None):
        """
        Args:
            connectomes: A list of numpy arrays, where each array is a connectivity matrix.
            labels: A list of labels corresponding to each connectome.
            atlas_info: Optional dictionary containing information about the brain atlas (e.g., node features).
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
        """
        self.connectomes = connectomes
        self.labels = labels
        self.atlas_info = atlas_info
        self.node_features = self._get_node_features() # Extract node features if available
        super().__init__(None, transform, pre_transform)

    def _get_node_features(self) -> Optional[np.ndarray]:
        """Extract node features from atlas_info if available."""
        # This is a placeholder. The actual implementation depends on the structure
        # of atlas_info and what node features are available (e.g., coordinates,
        # anatomical properties).
        if self.atlas_info and 'node_features' in self.atlas_info:
            # Assuming 'node_features' is a numpy array in atlas_info
            return np.array(self.atlas_info['node_features'])
        return None

    def len(self) -> int:
        return len(self.connectomes)

    def get(self, idx: int) -> Data:
        connectivity_matrix = self.connectomes[idx]
        label = self.labels[idx]
        
        # Use the same node features for all graphs in this dataset instance
        node_features = self.node_features

        # Convert the connectivity matrix and label to a PyTorch Geometric Data object
        data = connectivity_to_pyg_data(connectivity_matrix, label, node_features)

        return data

# The prompt mentioned a function create_brain_graph_dataset,
# but implementing a custom Dataset class is the standard PyTorch Geometric way
# to handle datasets. The BrainGraphDataset class fulfills this purpose.
# If a simple function is still desired, it could be a wrapper around the Dataset creation.
def create_brain_graph_dataset(connectomes: List[np.ndarray], labels: List[int], atlas_info: Optional[Dict[str, Any]] = None) -> BrainGraphDataset:
     """
     Helper function to create a BrainGraphDataset instance.
     """
     return BrainGraphDataset(connectomes, labels, atlas_info)


if __name__ == '__main__':
    print("Graph utilities module initialized.")
    print("Requires connectivity matrices and labels for full functionality.")

    # Example usage with dummy data
    # try:
    #     # Dummy connectivity matrix (3 nodes)
    #     dummy_conn_matrix = np.array([[0, 1, 0.5],
    #                                   [1, 0, 0],
    #                                   [0.5, 0, 0]])
    #     dummy_label = 1 # Example label

    #     # Convert to PyG Data object
    #     pyg_data = connectivity_to_pyg_data(dummy_conn_matrix, dummy_label)
    #     print("\nDummy PyG Data object:", pyg_data)
    #     print("  x:", pyg_data.x)
    #     print("  edge_index:", pyg_data.edge_index)
    #     print("  edge_attr:", pyg_data.edge_attr)
    #     print("  y:", pyg_data.y)

    #     # Dummy dataset creation
    #     dummy_connectomes = [dummy_conn_matrix, dummy_conn_matrix] # List of matrices
    #     dummy_labels = [1, 0] # List of labels
    #     dummy_atlas_info = {'node_features': np.random.rand(3, 5)} # Dummy node features

    #     brain_dataset = create_brain_graph_dataset(dummy_connectomes, dummy_labels, dummy_atlas_info)
    #     print("\nDummy BrainGraphDataset created with", len(brain_dataset), "graphs.")
    #     print("First graph in dataset:", brain_dataset[0])

    # except Exception as e:
    #     print(f"Error during dummy example execution: {e}")
    #     print("This is expected as real data loading is not implemented here.")
