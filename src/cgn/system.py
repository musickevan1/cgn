import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Dict

class CGNSystem(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Save model hyperparameters if they exist
        if hasattr(model, "save_hyperparameters_dict"):
            self.lr = model.save_hyperparameters_dict["lr"]
            self.weight_decay = model.save_hyperparameters_dict["weight_decay"]
        else:
            self.lr = 1e-3
            self.weight_decay = 0.0

    def forward(self, x, edge_index, batch):
        return self.model(x, edge_index, batch)

    def training_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        preds = out.argmax(dim=1)
        acc = (preds == batch.y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        out = self(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        preds = out.argmax(dim=1)
        acc = (preds == batch.y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer
