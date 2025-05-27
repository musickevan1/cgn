import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from typing import Dict, Any, Optional

# Import ConnectomeClassifier for the factory function
from src.connectome_analysis.models.gnn_models import ConnectomeClassifier

class PositionalEncoding(nn.Module):
    """
    Simple positional encoding for graph nodes.
    Can be extended for more complex graph-aware positional encodings.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape `[num_nodes_in_batch, d_model]`
        """
        # Add positional encoding to the input features based on the node index within the batch.
        # This assumes a global positional encoding across the entire batch of nodes.
        # For graph transformers, a more sophisticated graph-aware PE (e.g., Laplacian PE) is often used.
        # For now, this simple sequential PE is applied.
        # self.pe has shape [1, max_len, d_model]
        # x has shape [num_nodes_in_batch, d_model]
        # We need to add self.pe[0, :num_nodes_in_batch, :] to x
        return x + self.pe[0, :x.size(0)] # type: ignore

class BrainGraphTransformer(nn.Module):
    """
    Transformer model for brain graph analysis.
    Processes node features and uses a Transformer Encoder.
    """
    def __init__(self, in_channels: int, d_model: int, nhead: int, num_encoder_layers: int,
                 dim_feedforward: int, dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Linear projection for input features to match d_model
        self.input_projection = nn.Linear(in_channels, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Output layer (for graph-level classification after pooling)
        # This will be handled by ConnectomeClassifier, but we need to expose out_channels
        self.out_channels = d_model # The output dimension of the transformer encoder per node

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: A PyTorch Geometric Data object.
                  Expected to have `x` (node features) and `batch` (for batching).
        Returns:
            Node embeddings after transformer processing, shape `[num_nodes_in_batch, d_model]`
        """
        x, batch = data.x, data.batch

        # Project input features to d_model
        x = self.input_projection(x) # Shape: [num_nodes_in_batch, d_model]

        # Add positional encoding
        # Reshape x to [batch_size, num_nodes_per_graph, d_model] for positional encoding
        # This requires knowing the number of nodes per graph, which is tricky with Batch objects.
        # For simplicity, let's assume a fixed max_nodes or handle padding.
        # A more robust approach would be to use `to_dense_batch` or iterate over graphs.
        # For now, let's apply PE per node, assuming the PE is based on global node index within the batch.
        # This is a simplification and might need refinement for true graph positional encodings.
        x = self.positional_encoding(x) # Shape: [num_nodes_in_batch, d_model]

        # Transformer expects input shape (batch_size, sequence_length, d_model)
        # We need to convert the batched graph data into this format.
        # This is a common challenge with PyG and Transformers.
        # A simple way is to use `to_dense_batch` if all graphs have similar node counts,
        # or pad/unpad manually. For now, let's assume `x` is already a sequence of nodes
        # and the `batch` tensor can be used to reconstruct graph-wise sequences.
        # This requires a custom batching or unbatching logic.

        # For a simple initial implementation, let's assume the transformer encoder
        # can process the flattened node features and rely on the global pooling
        # in ConnectomeClassifier to aggregate. This is a strong simplification.
        # A more correct approach would involve:
        # 1. Unbatching `x` into a list of `[num_nodes_i, d_model]` tensors.
        # 2. Padding them to `[max_nodes, d_model]` and creating a mask.
        # 3. Passing to transformer encoder.
        # 4. Unpadding and re-batching.

        # Given the current `ConnectomeClassifier` expects node embeddings,
        # we will pass the node embeddings from the transformer encoder.
        # The `transformer_encoder` expects `(sequence_length, batch_size, d_model)` if `batch_first=False`
        # or `(batch_size, sequence_length, d_model)` if `batch_first=True`.
        # Our `x` is `[num_nodes_in_batch, d_model]`. We need to reshape it.
        # This is where the `batch` tensor from PyG `Data` object is crucial.

        # Let's assume for now that the transformer encoder can operate on the node features directly
        # and we will rely on global pooling to aggregate. This is a very simplified view
        # and might not fully leverage the transformer's sequence modeling capabilities on graphs.
        # A more appropriate way would be to use a graph-aware transformer (e.g., Graphormer, SAN).
        # For this initial implementation, we'll treat nodes as a sequence within the batch.

        # Reshape x for transformer: [num_nodes_in_batch, d_model] -> [batch_size, max_nodes_per_graph, d_model]
        # This requires `torch_geometric.utils.to_dense_batch`
        # from torch_geometric.utils import to_dense_batch
        # x_dense, mask = to_dense_batch(x, batch) # x_dense: [batch_size, max_nodes, d_model]
        # x_transformed = self.transformer_encoder(x_dense, src_key_padding_mask=~mask)
        # x_unbatched = x_transformed[mask] # Get back to [num_nodes_in_batch, d_model]

        # For simplicity, let's assume a single graph or a batch where transformer can operate on nodes directly.
        # This is a placeholder and needs proper graph-to-sequence handling.
        # For now, we'll just pass x directly, which means the transformer treats all nodes in the batch as one sequence.
        # This is NOT ideal for graph transformers but allows initial setup.
        # The `batch_first=True` in TransformerEncoderLayer means input is (batch, seq, feature)
        # Our `x` is (num_nodes_in_batch, d_model). We need to add a dummy batch dimension if processing as one sequence.
        # Or, if we want to process each graph in the batch as a separate sequence, we need `to_dense_batch`.

        # Let's use a simplified approach for now, assuming `x` is already structured for the transformer.
        # This will likely need to be revisited for a proper graph transformer.
        # For a true graph transformer, we'd need to convert the PyG Batch object into a format
        # suitable for the standard TransformerEncoder (e.g., padding, masks).
        # For now, let's just pass x through, which means it's treating all nodes in the batch as one long sequence.
        # This is a temporary simplification.
        
        # The `transformer_encoder` expects `(batch_size, sequence_length, d_model)`.
        # Our `x` is `(num_nodes_in_batch, d_model)`.
        # We need to convert `x` to a batched sequence.
        # This is where `torch_geometric.utils.to_dense_batch` is typically used.
        # However, `to_dense_batch` is not imported here.
        # For a minimal working example, let's assume `x` is already in the correct shape
        # or that we are processing a single graph.
        # This is a significant simplification and will need to be addressed for proper batching.

        # For the purpose of getting the structure in place, let's assume `x` is already
        # `[batch_size, num_nodes_per_graph, d_model]` and `batch` is not directly used here.
        # This means the `data` object passed to `forward` should already be pre-processed.
        # This is a temporary workaround.

        # A more robust approach for PyG Data objects:
        # from torch_geometric.utils import to_dense_batch
        # x_dense, mask = to_dense_batch(x, batch)
        # x_transformed_dense = self.transformer_encoder(x_dense, src_key_padding_mask=~mask)
        # x_transformed = x_transformed_dense[mask] # Convert back to sparse node representation

        # For now, let's just pass x directly, assuming it's a single sequence of nodes.
        # This will require the input `data.x` to be already shaped correctly for the transformer.
        # This is a placeholder for proper graph-to-sequence conversion.
        
        # The simplest way to make it run without complex batching logic here is to treat
        # the entire batch of nodes as a single sequence. This is not a true graph transformer
        # but allows the structure to be built.
        # Reshape x to (1, num_nodes_in_batch, d_model) if batch_first=True
        # x = x.unsqueeze(0) # Add dummy batch dimension if processing as one sequence
        # x = self.transformer_encoder(x)
        # x = x.squeeze(0) # Remove dummy batch dimension

        # Let's assume the input `x` is already `[num_nodes_in_batch, d_model]`
        # and the transformer encoder can handle it. This is a simplification.
        # The `batch` argument is not used in the transformer encoder itself, but for pooling.
        # The `ConnectomeClassifier` will handle the pooling.

        # For a proper transformer, we need to handle the batch dimension correctly.
        # Let's assume `x` is already `[batch_size, sequence_length, d_model]`
        # and the `data.batch` is used for pooling later.
        # This means the `input_projection` should output `[num_nodes_in_batch, d_model]`.
        # The `positional_encoding` also outputs `[num_nodes_in_batch, d_model]`.
        # The `transformer_encoder` expects `[batch_size, sequence_length, d_model]`.
        # So, we need to convert `[num_nodes_in_batch, d_model]` to `[batch_size, max_nodes_per_graph, d_model]`
        # using `to_dense_batch`.

        # Let's import `to_dense_batch` and use it.
        from torch_geometric.utils import to_dense_batch

        x_dense, mask = to_dense_batch(x, batch) # x_dense: [batch_size, max_nodes, d_model]
        x_transformed_dense = self.transformer_encoder(x_dense, src_key_padding_mask=~mask)
        x_transformed = x_transformed_dense[mask] # Convert back to sparse node representation

        return x_transformed # Return node embeddings

# Factory function for BrainGraphTransformer
def create_transformer_model(model_name: str, config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create a BrainGraphTransformer instance wrapped in ConnectomeClassifier.

    Args:
        model_name: The name of the transformer model to create (e.g., 'BrainGraphTransformer').
        config: A dictionary containing configuration parameters for the transformer model.
                Expected keys: 'in_channels', 'd_model', 'nhead', 'num_encoder_layers',
                'dim_feedforward', 'dropout', 'num_classes'.

    Returns:
        An instance of BrainGraphTransformer wrapped in ConnectomeClassifier.

    Raises:
        ValueError: If an unknown model name is provided or required parameters are missing.
    """
    if model_name != 'BrainGraphTransformer':
        raise ValueError(f"Unknown transformer model: {model_name}. "
                         f"Only 'BrainGraphTransformer' is supported.")

    in_channels = config.get('in_channels')
    d_model = config.get('d_model')
    nhead = config.get('nhead')
    num_encoder_layers = config.get('num_encoder_layers')
    dim_feedforward = config.get('dim_feedforward')
    dropout = config.get('dropout', 0.1)
    num_classes = config.get('num_classes')

    if any(p is None for p in [in_channels, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes]):
        raise ValueError(f"Missing required parameters for BrainGraphTransformer. "
                         f"Expected: 'in_channels', 'd_model', 'nhead', 'num_encoder_layers', "
                         f"'dim_feedforward', 'num_classes'.")

    transformer_base_model = BrainGraphTransformer(
        in_channels=int(in_channels), # type: ignore
        d_model=int(d_model), # type: ignore
        nhead=int(nhead), # type: ignore
        num_encoder_layers=int(num_encoder_layers), # type: ignore
        dim_feedforward=int(dim_feedforward), # type: ignore
        dropout=float(dropout), # type: ignore
        num_classes=int(num_classes) # type: ignore
    )

    # Wrap the transformer base model in the ConnectomeClassifier
    # The ConnectomeClassifier expects the base model to have an 'out_channels' attribute
    return ConnectomeClassifier(transformer_base_model, int(num_classes)) # type: ignore

if __name__ == '__main__':
    print("Transformer model module initialized.")

    # Dummy data for testing
    num_nodes_per_graph = 116
    num_graphs = 4
    in_channels = 16 # Example node features
    num_classes = 2

    # Create dummy PyG Data objects
    data_list = []
    for i in range(num_graphs):
        x = torch.randn(num_nodes_per_graph, in_channels)
        edge_index = torch.randint(0, num_nodes_per_graph, (2, num_nodes_per_graph * 2))
        y = torch.tensor([i % num_classes], dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))

    batch = Batch.from_data_list(data_list)

    # Example config for BrainGraphTransformer
    transformer_config = {
        'in_channels': in_channels,
        'd_model': 64,
        'nhead': 4,
        'num_encoder_layers': 2,
        'dim_feedforward': 128,
        'dropout': 0.1,
        'num_classes': num_classes
    }

    # Create transformer model using the factory function
    transformer_classifier = create_transformer_model('BrainGraphTransformer', transformer_config)

    # Forward pass
    out = transformer_classifier(batch)
    print("\nDummy BrainGraphTransformer Classifier Output Shape:", out.shape) # Should be [batch_size, num_classes]
