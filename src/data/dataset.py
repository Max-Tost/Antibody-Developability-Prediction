"""PyTorch Dataset for antibody developability prediction.

Loads precomputed ESM-2 embeddings and corresponding binary labels.
"""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class AntibodyDataset(Dataset):
    """Dataset that pairs cached ESM-2 embeddings with binary labels.

    Expects:
        - A CSV file with columns: 'sequence', 'label', 'source'
        - A directory of cached embeddings as .pt files (one per sequence,
          named by index or hash)
    """

    def __init__(self, csv_path: str, embeddings_dir: str) -> None:
        self.df = pd.read_csv(csv_path)
        self.embeddings_dir = Path(embeddings_dir)

        # Verify embeddings exist
        self._verify_embeddings()

    def _verify_embeddings(self) -> None:
        """Check that all required embedding files are present."""
        missing = []
        for idx in range(len(self.df)):
            emb_path = self.embeddings_dir / f"{idx}.pt"
            if not emb_path.exists():
                missing.append(idx)
        if missing:
            raise FileNotFoundError(
                f"Missing embeddings for {len(missing)} sequences. "
                f"Run extract_embeddings.py first."
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.load(
            self.embeddings_dir / f"{idx}.pt",
            weights_only=True,
        )
        label = torch.tensor(self.df.iloc[idx]["label"], dtype=torch.float32)
        return embedding, label
