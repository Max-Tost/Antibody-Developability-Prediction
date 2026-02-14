"""Classification heads for antibody developability prediction.

Two architectures:
- LinearClassifier: single-layer baseline (linear probe)
- MLPClassifier: two-layer MLP with dropout
"""

import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Linear probe baseline: single linear layer for binary classification."""

    def __init__(self, input_dim: int = 1280) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class MLPClassifier(nn.Module):
    """Two-layer MLP for binary classification.

    Architecture: Linear → ReLU → Dropout → Linear
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_classifier(
    model_type: str = "mlp",
    input_dim: int = 1280,
    hidden_dim: int = 256,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory function to create a classifier from config.

    Args:
        model_type: One of 'mlp' or 'linear'.
        input_dim: Dimensionality of ESM-2 embeddings.
        hidden_dim: Hidden layer size (MLP only).
        dropout: Dropout rate (MLP only).
    """
    if model_type == "linear":
        return LinearClassifier(input_dim=input_dim)
    elif model_type == "mlp":
        return MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp' or 'linear'.")
