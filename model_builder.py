import torch
from torch import nn

class RiskScorerModel(nn.Module):
    def __init__(self, input_features: int, hidden_units: int = 64):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_units, out_features=hidden_units // 2),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units // 2, out_features=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_stack(x)
