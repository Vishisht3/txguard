import torch
from torch import nn

class ModelArchitectureError(Exception):
    """Custom exception raised for architectural or tensor dimensional mismatches."""
    pass

class ResidualBlock(nn.Module):
    """Residual block providing direct gradient skip connections for tabular features."""
    def __init__(self, dim: int, dropout_rate: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.Mish()
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.act(self.bn(self.fc(x))))

class RiskScorerModel(nn.Module):
    """Advanced Deep Tabular MLP with Residual Connections and Mish activations."""
    def __init__(self, input_features: int, hidden_units: int = 64, dropout_rate: float = 0.3):
        super().__init__()
        
        if input_features <= 0:
            raise ModelArchitectureError(f"input_features must be positive, got {input_features}")
            
        self.input_layer = nn.Sequential(
            nn.Linear(input_features, hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.Mish()
        )
        self.res_block1 = ResidualBlock(hidden_units, dropout_rate=dropout_rate)
        
        self.transition = nn.Sequential(
            nn.Linear(hidden_units, hidden_units // 2),
            nn.BatchNorm1d(hidden_units // 2),
            nn.Mish()
        )
        self.res_block2 = ResidualBlock(hidden_units // 2, dropout_rate=dropout_rate)
        
        self.output_layer = nn.Linear(hidden_units // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ModelArchitectureError(f"Expected 2D tensor (batch_size, features), got shape {x.shape}")
        
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.transition(x)
        x = self.res_block2(x)
        return self.output_layer(x)
