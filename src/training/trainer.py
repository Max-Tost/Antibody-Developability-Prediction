"""Training loop with k-fold cross-validation and evaluation."""

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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from src.data.dataset import AntibodyDataset
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


def train_fold(
    fold: int,
    train_dataset: Subset,
    val_dataset: Subset,
    cfg: dict,
    device: str,
) -> dict[str, float]:
    """Train and evaluate a single fold."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )

    model = build_classifier(
        model_type=cfg["model"]["type"],
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_metrics = {}

    for epoch in range(cfg["training"]["max_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            best_metrics = val_metrics.copy()

    logger.info(f"Fold {fold}: best AUC = {best_auc:.4f}")
    return best_metrics


def run_kfold_training(cfg: dict) -> None:
    """Run k-fold cross-validation training pipeline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = AntibodyDataset(
        csv_path=cfg["data"]["train_csv"],
        embeddings_dir=cfg["data"]["embeddings_dir"],
    )

    labels = dataset.df["label"].values

    kfold = StratifiedKFold(
        n_splits=cfg["training"]["n_folds"],
        shuffle=True,
        random_state=cfg["training"]["random_seed"],
    )

    fold_metrics: list[dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(labels, labels)):
        logger.info(f"--- Fold {fold + 1}/{cfg['training']['n_folds']} ---")
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        metrics = train_fold(fold + 1, train_subset, val_subset, cfg, device)
        fold_metrics.append(metrics)

    # Aggregate results across folds
    aggregated = {}
    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        aggregated[f"mean_{key}"] = np.mean(values)
        aggregated[f"std_{key}"] = np.std(values)

    logger.info("=== Cross-validation results ===")
    for key in ["auc_roc", "precision", "recall", "f1"]:
        mean = aggregated[f"mean_{key}"]
        std = aggregated[f"std_{key}"]
        logger.info(f"  {key}: {mean:.4f} ± {std:.4f}")
