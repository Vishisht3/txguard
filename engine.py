import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
from typing import Dict, List, Optional, Tuple
import copy

class EnginePipelineError(Exception):
    """Custom exception raised for exceptions during engine training/evaluation execution."""
    pass

class CostSensitiveFocalLoss(nn.Module):
    """Cost-sensitive Focal Loss heavily penalizing False Negatives (missed fraud)."""
    def __init__(self, alpha: float = 0.75, gamma: float = 4.0, fn_weight: float = 8.0):
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in range [0, 1], got {alpha}")
        self.alpha = alpha
        self.gamma = gamma
        self.fn_weight = fn_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        alpha_t = self.alpha * targets * self.fn_weight + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss
        return loss.mean()

class EarlyStoppingPRAUC:
    """Monitors validation PR-AUC score and restores optimal model state dict."""
    def __init__(self, patience: int = 5, delta: float = 0.001):
        self.patience = patience
        self.delta = delta
        self.best_pr_auc = -1.0
        self.counter = 0
        self.early_stop = False
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None

    def __call__(self, val_pr_auc: float, model: torch.nn.Module) -> None:
        if val_pr_auc > self.best_pr_auc + self.delta:
            self.best_pr_auc = val_pr_auc
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_step_hnm(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    hard_neg_threshold: float = 0.70
) -> float:
    model.train()
    train_loss = 0.0
    
    for X, y in dataloader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        y_logits = model(X).squeeze(-1)
        loss = loss_fn(y_logits, y)
        
        with torch.no_grad():
            probs = torch.sigmoid(y_logits)
            hard_negs = (y == 0) & (probs > hard_neg_threshold)
            
        if hard_negs.sum() > 0:
            hn_loss = F.binary_cross_entropy_with_logits(y_logits[hard_negs], y[hard_negs])
            loss = loss + 0.5 * hn_loss

        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
            
        train_loss += loss.item()
        
    return train_loss / len(dataloader)

def test_eval(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    loss_fn: nn.Module, 
    device: torch.device
) -> Tuple[float, float]:
    model.eval()
    test_loss = 0.0
    all_targets: List[float] = []
    all_probs: List[float] = []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_logits = model(X).squeeze(-1)
            loss = loss_fn(y_logits, y)
            test_loss += loss.item()
            
            probs = torch.sigmoid(y_logits)
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = test_loss / len(dataloader)
    precision, recall, _ = precision_recall_curve(all_targets, all_probs)
    pr_auc_score = auc(recall, precision)
    
    return avg_loss, pr_auc_score

def train(
    model: nn.Module, 
    train_dataloader: torch.utils.data.DataLoader, 
    test_dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: nn.Module, 
    epochs: int, 
    device: torch.device, 
    patience: int = 5,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Dict[str, List[float]]:
    results: Dict[str, List[float]] = {"train_loss": [], "test_loss": [], "test_pr_auc": []}
    early_stopper = EarlyStoppingPRAUC(patience=patience)

    try:
        for epoch in range(epochs):
            train_loss = train_step_hnm(model, train_dataloader, loss_fn, optimizer, device, scheduler=scheduler)
            test_loss, val_pr_auc = test_eval(model, test_dataloader, loss_fn, device)
            
            print(f"Epoch: {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Val PR-AUC: {val_pr_auc:.4f}")
            results["train_loss"].append(train_loss)
            results["test_loss"].append(test_loss)
            results["test_pr_auc"].append(val_pr_auc)
            
            early_stopper(val_pr_auc, model)
            if early_stopper.early_stop:
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}. Restoring optimal weights.")
                if early_stopper.best_state_dict is not None:
                    model.load_state_dict(early_stopper.best_state_dict)
                break

        if not early_stopper.early_stop and early_stopper.best_state_dict is not None:
            model.load_state_dict(early_stopper.best_state_dict)
                
    except Exception as e:
        raise EnginePipelineError(f"Fatal error encountered during engine training: {str(e)}") from e
        
    return results
