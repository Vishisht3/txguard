import torch
from torch.utils.data import Dataset, DataLoader
class CustomTransactionDataset(Dataset):
  
  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)
    
  def __len__(self):
    return len(self.X)
  
  def __getitem__(self, idx: int):
    return self.X[idx], self.y[idx]

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size: int=64):
  train_dataset = CustomTransactionDataset(X_train, y_train)
  test_dataset = CustomTransactionDataset(X_test, y_test)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader
