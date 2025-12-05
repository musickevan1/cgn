import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch import nn

class GNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        lr: float = 0.001,
        weight_decay: float = 5e-4
    ):
        super().__init__()
        self.save_hyperparameters_dict = {
            "lr": lr,
            "weight_decay": weight_decay
        }

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels)) # Last conv layer

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        for conv in self.convs:
            x = conv(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 2. Readout layer (aggregation)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)

        return x
