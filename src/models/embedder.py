"""ESM-2 protein language model embedding extraction.

Extracts fixed-size embeddings from antibody VH sequences using a pretrained
ESM-2 model from HuggingFace. Embeddings are mean-pooled over sequence length
and cached to disk for reuse.
"""

import logging
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ESM2Embedder:
    """Extract mean-pooled embeddings from ESM-2.

    Args:
        model_name: HuggingFace model identifier (e.g. facebook/esm2_t33_650M_UR50D).
        device: Torch device for inference.
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("Model loaded")

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Embed a single amino acid sequence.

        Returns a 1D tensor of shape (embedding_dim,) — mean-pooled over
        all sequence positions.
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024,
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, embed_dim)

        # Mean pool over all positions
        attention_mask = inputs["attention_mask"]
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
        embedding = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        return embedding.squeeze(0).cpu()

    def embed_dataset(
        self,
        csv_path: str,
        output_dir: str,
        batch_log_interval: int = 50,
    ) -> None:
        """Embed all sequences from a CSV and cache as individual .pt files.

        Args:
            csv_path: Path to CSV with a 'sequence' column.
            output_dir: Directory to save embedding .pt files.
            batch_log_interval: Log progress every N sequences.
        """
        df = pd.read_csv(csv_path)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for idx, row in df.iterrows():
            emb_file = out_path / f"{idx}.pt"
            if emb_file.exists():
                continue

            embedding = self.embed_sequence(row["sequence"])
            torch.save(embedding, emb_file)

            if (idx + 1) % batch_log_interval == 0:
                logger.info(f"Embedded {idx + 1}/{len(df)} sequences")

        logger.info(f"All {len(df)} embeddings saved to {out_path}")
