import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from abc import ABC, abstractmethod
from typing import Tuple, Any

class BaseTxDataset(Dataset, ABC):
    """Abstract Base Class enforcing dataset contract design pattern."""
    
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

class CustomTransactionDataset(BaseTxDataset):
    """Encapsulated PyTorch Dataset for scaled tabular features with defensive checks."""
    def __init__(self, X: Any, y: Any):
        try:
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        except Exception as e:
            raise TypeError(f"Failed to convert inputs into PyTorch Tensors: {str(e)}")

        if len(self.X) != len(self.y):
            raise ValueError(f"Feature length ({len(self.X)}) does not match label length ({len(self.y)})")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        return self.X[idx], self.y[idx]

def create_dataloaders(
    X_train: Any, 
    y_train: Any, 
    X_test: Any, 
    y_test: Any, 
    batch_size: int = 64,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Factory function producing optimized train/test PyTorch DataLoaders with class balancing."""
    train_dataset = CustomTransactionDataset(X_train, y_train)
    test_dataset = CustomTransactionDataset(X_test, y_test)

    # 1. Compute class distribution weights for training samples
    class_sample_count = torch.tensor([
        (train_dataset.y == 0).sum().item(), 
        (train_dataset.y == 1).sum().item()
    ])
    class_weights = 1.0 / class_sample_count.float()
    
    # 2. Assign individual sample weights based on label
    sample_weights = torch.tensor([class_weights[int(t)] for t in train_dataset.y])

    # 3. Instantiate WeightedRandomSampler (shuffle must be False when using sampler)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, test_loader
