import unittest
import torch
from torch_geometric.data import Data, Batch
from src.connectome_analysis.models.transformer.transformer_model import BrainGraphTransformer, create_transformer_model

class TestTransformerModel(unittest.TestCase):

    def setUp(self):
        self.in_channels = 16
        self.d_model = 32
        self.nhead = 4
        self.num_encoder_layers = 2
        self.dim_feedforward = 64
        self.dropout = 0.1
        self.num_classes = 2
        self.num_nodes_per_graph = 116
        self.num_graphs = 4

        # Create dummy PyG Data objects
        self.data_list = []
        for i in range(self.num_graphs):
            x = torch.randn(self.num_nodes_per_graph, self.in_channels)
            edge_index = torch.randint(0, self.num_nodes_per_graph, (2, self.num_nodes_per_graph * 2))
            y = torch.tensor([i % self.num_classes], dtype=torch.long)
            self.data_list.append(Data(x=x, edge_index=edge_index, y=y))

        self.batch = Batch.from_data_list(self.data_list)

        self.transformer_config = {
            'in_channels': self.in_channels,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'num_classes': self.num_classes
        }

    def test_brain_graph_transformer_forward(self):
        """Test the forward pass of BrainGraphTransformer."""
        model = BrainGraphTransformer(
            in_channels=self.in_channels,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            num_classes=self.num_classes
        )
        output = model(self.batch)
        # The output should be node embeddings, so shape is [total_num_nodes, d_model]
        self.assertEqual(output.shape, (self.num_nodes_per_graph * self.num_graphs, self.d_model))

    def test_create_transformer_model_factory(self):
        """Test the create_transformer_model factory function."""
        classifier_model = create_transformer_model('BrainGraphTransformer', self.transformer_config)
        self.assertIsInstance(classifier_model, torch.nn.Module)
        # Test forward pass through the classifier model
        output = classifier_model(self.batch)
        # The output should be graph-level predictions, so shape is [num_graphs, num_classes]
        self.assertEqual(output.shape, (self.num_graphs, self.num_classes))

    def test_create_transformer_model_invalid_name(self):
        """Test create_transformer_model with an invalid model name."""
        with self.assertRaises(ValueError):
            create_transformer_model('InvalidTransformer', self.transformer_config)

    def test_create_transformer_model_missing_params(self):
        """Test create_transformer_model with missing parameters."""
        invalid_config = self.transformer_config.copy()
        del invalid_config['in_channels'] # Remove a required parameter
        with self.assertRaises(ValueError):
            create_transformer_model('BrainGraphTransformer', invalid_config)

if __name__ == '__main__':
    unittest.main()
