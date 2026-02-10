"""Classification head for antibody developability prediction."""

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Linear probe baseline: single linear layer for binary classification."""

    def __init__(self, input_dim: int = 1280) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)
