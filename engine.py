import torch
from typing import Dict, List
import copy

class EarlyStopping:
    """Monitors test loss and triggers termination when loss plateaus."""
    def __init__(self, patience: int = 3, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    train_loss = 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_logits = model(X).squeeze(-1)
        loss = loss_fn(y_logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(dataloader)

def test_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, device: torch.device) -> float:
    model.eval()
    test_loss = 0.0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logits = model(X).squeeze(-1)
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
    return test_loss / len(dataloader)

def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, epochs: int, device: torch.device, patience: int = 3) -> Dict[str, List[float]]:
    results = {"train_loss": [], "test_loss": []}
    early_stopper = EarlyStopping(patience=patience)

    for epoch in range(epochs):
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss = test_step(model, test_dataloader, loss_fn, device)
        
        print(f"Epoch: {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        
        early_stopper(test_loss, model)
        if early_stopper.early_stop:
            print(f"[INFO] Early stopping triggered at epoch {epoch+1}. Restoring optimal weights.")
            model.load_state_dict(early_stopper.best_state_dict)
            break
            
    return results
