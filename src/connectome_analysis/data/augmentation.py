import numpy as np
import torch
from typing import Union, Tuple, Optional, cast

def add_gaussian_noise(
    connectome: Union[np.ndarray, torch.Tensor],
    mean: float = 0.0,
    std: float = 0.01
) -> Union[np.ndarray, torch.Tensor]:
    """
    Adds Gaussian noise to a connectome.

    Args:
        connectome (Union[np.ndarray, torch.Tensor]): The input connectome (adjacency matrix).
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        Union[np.ndarray, torch.Tensor]: The connectome with added Gaussian noise.
    """
    if isinstance(connectome, np.ndarray):
        noise = np.random.normal(mean, std, connectome.shape)
        return connectome + noise
    elif isinstance(connectome, torch.Tensor):
        noise = torch.randn(connectome.shape, device=connectome.device) * std + mean
        return connectome + noise
    else:
        raise TypeError("Connectome must be a NumPy array or a PyTorch tensor.")

def flip_connections(
    connectome: Union[np.ndarray, torch.Tensor],
    p: float = 0.01
) -> Union[np.ndarray, torch.Tensor]:
    """
    Randomly flips (inverts) a percentage of connections in a connectome.
    Assumes connectome values are between 0 and 1 (e.g., correlation coefficients).
    Flipping means `value -> 1 - value`.

    Args:
        connectome (Union[np.ndarray, torch.Tensor]): The input connectome (adjacency matrix).
        p (float): Probability of flipping a connection (between 0 and 1).

    Returns:
        Union[np.ndarray, torch.Tensor]: The connectome with flipped connections.
    """
    if not (0 <= p <= 1):
        raise ValueError("Probability 'p' must be between 0 and 1.")

    if isinstance(connectome, np.ndarray):
        mask = np.random.rand(*connectome.shape) < p
        augmented_connectome = np.copy(connectome)
        augmented_connectome[mask] = 1.0 - augmented_connectome[mask]
        return augmented_connectome
    elif isinstance(connectome, torch.Tensor):
        mask = torch.rand(connectome.shape, device=connectome.device) < p
        augmented_connectome = connectome.clone()
        augmented_connectome[mask] = 1.0 - augmented_connectome[mask]
        return augmented_connectome
    else:
        raise TypeError("Connectome must be a NumPy array or a PyTorch tensor.")

def permute_nodes(
    connectome: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
    """
    Randomly permutes the nodes (rows and columns) of a connectome.
    If labels are provided, they are permuted consistently.

    Args:
        connectome (Union[np.ndarray, torch.Tensor]): The input connectome (adjacency matrix).
        labels (Union[np.ndarray, torch.Tensor], optional): Corresponding labels for each node.

    Returns:
        Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
            The permuted connectome and permuted labels (if provided).
    """
    num_nodes = connectome.shape[0]
    permutation = np.random.permutation(num_nodes)

    if isinstance(connectome, np.ndarray):
        permuted_connectome = connectome[permutation, :][:, permutation]
    elif isinstance(connectome, torch.Tensor):
        # Ensure permutation is a torch.Tensor for consistent indexing
        permutation_tensor = torch.from_numpy(permutation).to(connectome.device)
        permuted_connectome = connectome[permutation_tensor, :][:, permutation_tensor]
    else:
        raise TypeError("Connectome must be a NumPy array or a PyTorch tensor.")

    permuted_labels: Optional[Union[np.ndarray, torch.Tensor]]
    if labels is not None:
        if isinstance(labels, np.ndarray):
            permuted_labels = labels[permutation]
        elif isinstance(labels, torch.Tensor):
            # Ensure permutation is a torch.Tensor for consistent indexing
            permutation_tensor = torch.from_numpy(permutation).to(labels.device)
            permuted_labels = labels[permutation_tensor]
        else:
            raise TypeError("Labels must be a NumPy array or a PyTorch tensor.")
    else:
        permuted_labels = None

    return permuted_connectome, cast(Optional[Union[np.ndarray, torch.Tensor]], permuted_labels)
