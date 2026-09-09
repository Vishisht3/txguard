import torch
from torch import nn

class ModelArchitectureError(Exception):
    """Custom exception raised for architectural or tensor dimensional mismatches."""
    pass

class RiskScorerModel(nn.Module):
    """Deep Tabular MLP featuring Batch Normalization and Dropout Regularization."""
    def __init__(self, input_features: int, hidden_units: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        
        if input_features <= 0:
            raise ModelArchitectureError(f"input_features must be positive, got {input_features}")
            
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(in_features=hidden_units, out_features=hidden_units // 2),
            nn.BatchNorm1d(hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(in_features=hidden_units // 2, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ModelArchitectureError(f"Expected 2D tensor (batch_size, features), got shape {x.shape}")
        return self.layer_stack(x)
