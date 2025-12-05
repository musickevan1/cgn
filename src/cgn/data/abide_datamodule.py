import lightning.pytorch as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import Optional, List
import numpy as np

class ABIDEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int = 32,
        num_workers: int = 0,
        use_synthetic: bool = False,
        node_feature_dim: int = 1,
        num_nodes: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset: Optional[List[Data]] = None
        self.val_dataset: Optional[List[Data]] = None
        self.test_dataset: Optional[List[Data]] = None

    def setup(self, stage: Optional[str] = None):
        if self.hparams.use_synthetic:
            self._setup_synthetic()
        else:
            raise NotImplementedError("Real ABIDE data loading not yet implemented. Use use_synthetic=True.")

    def _setup_synthetic(self):
        # Generate random graphs to mimic functional connectivity matrices
        # In a real scenario, nodes = regions of interest (ROIs), edges = correlation

        num_samples = 200
        self.train_dataset = [self._generate_sample() for _ in range(int(0.8 * num_samples))]
        self.val_dataset = [self._generate_sample() for _ in range(int(0.1 * num_samples))]
        self.test_dataset = [self._generate_sample() for _ in range(int(0.1 * num_samples))]

    def _generate_sample(self) -> Data:
        # Create a fully connected graph (dense adjacency) or sparse k-NN
        num_nodes = self.hparams.num_nodes
        x = torch.randn(num_nodes, self.hparams.node_feature_dim)

        # Simple random edge index (sparse for efficiency in this mock)
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 5))

        # Binary target: 0 (Control) or 1 (ASD)
        y = torch.tensor([np.random.randint(0, 2)], dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
