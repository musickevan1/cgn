import torch
from src.connectome_analysis.training.lightning_modules import BaseLightningModule
from torch_geometric.data import Data

class GNNLightningModule(BaseLightningModule):
    def __init__(self, model, num_classes=2, learning_rate=1e-3):
        super().__init__(model, num_classes, learning_rate)
        self.save_hyperparameters(ignore=['model'])

    def forward(self, data: Data):
        # GNN models typically take a Data object from PyTorch Geometric
        return self.model(data)

    def _common_step(self, batch: Data, batch_idx):
        # Assuming batch is a PyTorch Geometric Data object
        # Data object typically has batch.x (node features), batch.edge_index, batch.y (labels)
        logits = self.forward(batch)
        labels = batch.y.long() if isinstance(batch.y, torch.Tensor) else torch.tensor(batch.y, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels, logits
