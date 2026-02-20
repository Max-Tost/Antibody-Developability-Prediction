"""Training loop with evaluation for antibody developability prediction."""

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.models.classifier import build_classifier

logger = logging.getLogger(__name__)


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Compute binary classification metrics."""
    return {
        "auc_roc": roc_auc_score(labels, probabilities),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0

    for embeddings, labels in dataloader:
        embeddings, labels = embeddings.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, dict[str, float]]:
    """Evaluate model, return loss and metrics."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for embeddings, labels in dataloader:
        embeddings, labels = embeddings.to(device), labels.to(device)

        logits = model(embeddings)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    predictions = (all_probs >= 0.5).astype(int)

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, predictions, all_probs)

    return avg_loss, metrics
