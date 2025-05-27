import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Optional, List, Tuple, Union
import torch
import numpy as np
from torch_geometric.data import Data # Import Data object

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        # This method should be overridden by subclasses to load and split data
        pass

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not set. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not set. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not set. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

class ConnectomeDataset(Dataset):
    def __init__(self, data: Union[np.ndarray, List[Data]], labels: np.ndarray):
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        if isinstance(data, list) and all(isinstance(d, Data) for d in data):
            self.data = data # Store list of PyG Data objects directly
            self.is_geometric = True
        elif isinstance(data, np.ndarray):
            self.data = torch.tensor(data, dtype=torch.float32)
            self.is_geometric = False
        else:
            raise TypeError("Data must be a numpy array or a list of PyTorch Geometric Data objects.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.is_geometric:
            # For geometric data, return the Data object directly
            return self.data[idx]
        else:
            # For tensor data, return features and labels
            return self.data[idx], self.labels[idx]

class ConnectomeDataModule(BaseDataModule):
    def __init__(
        self,
        features: Union[np.ndarray, List[Data]], # Allow List[Data]
        labels: np.ndarray,
        site_labels: Optional[np.ndarray] = None,
        split_type: str = "stratified_kfold",
        n_splits: int = 5,
        fold_idx: int = 0,
        batch_size: int = 32,
        num_workers: int = 0,
        random_state: Optional[int] = None
    ):
        super().__init__(batch_size, num_workers)
        self.features = features
        self.labels = labels
        self.site_labels = site_labels
        self.split_type = split_type
        self.n_splits = n_splits
        self.fold_idx = fold_idx
        self.random_state = random_state

    def setup(self, stage: Optional[str] = None):
        from src.connectome_analysis.data.dataset_splits import get_data_split_indices

        splits = get_data_split_indices(
            labels=self.labels,
            site_labels=self.site_labels,
            split_type=self.split_type,
            n_splits=self.n_splits,
            random_state=self.random_state
        )

        if not splits:
            raise ValueError("No splits generated. Check your data and split parameters.")

        if self.fold_idx >= len(splits):
            raise IndexError(f"Fold index {self.fold_idx} out of bounds for {len(splits)} splits.")

        train_idx, val_idx = splits[self.fold_idx]

        # For simplicity, using the same validation set as test set for now.
        # In a full pipeline, a separate test set would be held out.
        
        # Handle slicing based on feature type
        if isinstance(self.features, np.ndarray):
            train_features = self.features[train_idx]
            val_features = self.features[val_idx]
        elif isinstance(self.features, list) and all(isinstance(d, Data) for d in self.features):
            # For list of Data objects, select by index
            train_features = [self.features[i] for i in train_idx]
            val_features = [self.features[i] for i in val_idx]
        else:
            raise TypeError("Features must be a numpy array or a list of PyTorch Geometric Data objects.")

        self.train_dataset = ConnectomeDataset(train_features, self.labels[train_idx])
        self.val_dataset = ConnectomeDataset(val_features, self.labels[val_idx])
        self.test_dataset = ConnectomeDataset(val_features, self.labels[val_idx])
