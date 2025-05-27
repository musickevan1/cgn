import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from torch_geometric.data import Data as PyGData

def compute_gradient_saliency_map(
    model: torch.nn.Module,
    data: PyGData,
    target_class: Optional[int] = None
) -> torch.Tensor:
    """
    Computes a gradient-based saliency map for a GNN model.
    This method calculates the gradient of the model's output (for a target class)
    with respect to the input features (node features or edge features).

    Args:
        model (torch.nn.Module): The trained GNN model.
        data (PyGData): The input graph data (PyG Data object).
        target_class (Optional[int]): The class index for which to compute the saliency.
                                      If None, computes for the predicted class.

    Returns:
        torch.Tensor: Saliency map (gradients) with respect to input features.
                      Shape will match the input features (e.g., [num_nodes, num_features]).
    """
    model.eval()
    data = data.clone() # Clone to avoid modifying original data

    # Ensure input features require gradients
    if hasattr(data, 'x') and data.x is not None:
        data.x.requires_grad_(True)
        input_tensor = data.x
    elif hasattr(data, 'edge_attr') and data.edge_attr is not None:
        data.edge_attr.requires_grad_(True)
        input_tensor = data.edge_attr
    else:
        raise ValueError("Input data must have 'x' (node features) or 'edge_attr' (edge features).")

    # Forward pass
    output = model(data)

    # Select target class for gradient computation
    if target_class is None:
        target_class = output.argmax(dim=-1).item() # Use predicted class
    
    # Compute gradient of the output with respect to the input features
    model.zero_grad()
    if output.ndim > 1: # If output is a batch of predictions
        output_scalar = output[0, target_class] # Assuming batch size 1 for simplicity
    else: # If output is a single prediction (e.g., for a single graph)
        output_scalar = output[target_class]

    output_scalar.backward()

    # Return gradients
    if hasattr(data, 'x') and data.x is not None:
        x_features = data.x  # Assign to a new variable after None check
        if x_features.grad is None:
            raise RuntimeError("Gradients for node features (data.x) are None. Ensure requires_grad_(True) was set and backward() was called.")
        return x_features.grad
    elif hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_features = data.edge_attr  # Assign to a new variable after None check
        if edge_features.grad is None:
            raise RuntimeError("Gradients for edge attributes (data.edge_attr) are None. Ensure requires_grad_(True) was set and backward() was called.")
        return edge_features.grad
    else:
        # This should not be reached if input validation (line 29) worked correctly,
        # as it ensures either data.x or data.edge_attr is present.
        raise RuntimeError("No valid feature tensor with gradients found. Unexpected state.")

def visualize_saliency_map_on_graph(
    saliency_map: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    node_labels: Optional[List[str]] = None,
    title: str = "Saliency Map on Graph",
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Visualizes a saliency map on a graph.
    This is a conceptual visualization and might require a graph drawing library like NetworkX
    and specific plotting capabilities.

    Args:
        saliency_map (torch.Tensor): Saliency scores for nodes or edges.
                                     If node-level, shape [num_nodes, num_features].
                                     If edge-level, shape [num_edges, num_edge_features].
        edge_index (torch.Tensor): Edge indices (2, num_edges).
        num_nodes (int): Total number of nodes in the graph.
        node_labels (Optional[List[str]]): Labels for nodes.
        title (str): Title of the visualization.
        figsize (Tuple[int, int]): Figure size.
    """
    print("Warning: `visualize_saliency_map_on_graph` is a conceptual placeholder. Requires graph plotting library.")
    # Example conceptual visualization:
    # import networkx as nx
    # G = nx.Graph()
    # G.add_nodes_from(range(num_nodes))
    #
    # # Normalize saliency map for visualization (e.g., to 0-1 range)
    # if saliency_map.ndim == 2: # Node-level saliency, average across features
    #     node_saliency = saliency_map.abs().mean(dim=1).cpu().numpy()
    # elif saliency_map.ndim == 1: # Node-level saliency, single feature
    #     node_saliency = saliency_map.abs().cpu().numpy()
    # else: # Edge-level saliency, average across edge features
    #     edge_saliency = saliency_map.abs().mean(dim=1).cpu().numpy()
    #
    # if 'node_saliency' in locals():
    #     node_colors = plt.cm.Reds(node_saliency / node_saliency.max())
    #     nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    # elif 'edge_saliency' in locals():
    #     # Need to map edge_saliency back to edges in G
    #     edge_colors = plt.cm.Blues(edge_saliency / edge_saliency.max())
    #     edge_widths = edge_saliency * 5 # Scale for visibility
    #     # Add edges with colors and widths
    #
    # plt.figure(figsize=figsize)
    # pos = nx.spring_layout(G) # or other layout
    # nx.draw_networkx_labels(G, pos, labels={i: label for i, label in enumerate(node_labels)} if node_labels else None)
    # plt.title(title)
    # plt.axis('off')
    # plt.show()
